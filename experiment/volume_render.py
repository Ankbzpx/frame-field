import argparse
from glob import glob
import json
import os
from typing import Tuple

from common import fibonacci_sphere, normalize, Timer
from config import Config
from config_utils import config_latent, config_model
from eval_jax import batch_call, voxel_infer
import model_jax

import equinox as eqx
import jax
from jax import jit, numpy as jnp, vmap
from jax2torch import jax2torch
import nerfacc
import numpy as np
from PIL import Image
from pyrr import Matrix44
import torch
from torch2jax import j2t, t2j

from icecream import ic
import polyscope as ps


# S-Density in NeuS, biased
# FIXME: s should be a learned parameter
@jit
def s_density(x, s=100):
    return s * jnp.exp(-s * x) / jnp.pow(1 + jnp.exp(-s * x), 2)


def batch_call_pytorch(
    func,
    input,
    tmp_cpu=False,
    num_out_args=1,
    out_map_func=lambda x: [x],
    group_size=256**2,
):
    n_iters = len(input) // group_size

    if tmp_cpu:
        device = input.device

    if n_iters == 0:
        output = func(input)
        output = out_map_func(output)
    else:
        output = {}
        for i in range(num_out_args):
            output[i] = None

        for input_batch in torch.split(input, group_size):
            output_ = func(input_batch)
            output_ = out_map_func(output_)

            for i in range(num_out_args):
                if tmp_cpu:
                    output[i] = (
                        output_[i].detach().cpu()
                        if output[i] is None
                        else torch.concat([output[i], output_[i].detach().cpu()])
                    )
                else:
                    output[i] = (
                        output_[i]
                        if output[i] is None
                        else torch.concat([output[i], output_[i]])
                    )

        output = list(output.values())

    if tmp_cpu:
        output = [output[i].to(device) for i in range(num_out_args)]

    if num_out_args == 1:
        output = output[0]

    return output


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="*", help="Path to pointcloud files."
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default="data/sdf",
        help="Path to pointcloud folder.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/octa.json", help="Path to config file."
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--vis", action="store_true", help="Visualize")
    parser.add_argument("--skip", action="store_true", help="Skip existing output")
    args = parser.parse_args()

    if args.model is not None:
        tag = ""
        model_list = args.model
    else:
        # TODO; Maybe not hard coded
        tag = "_".join(args.model_folder.split("/")[-2:])
        model_list = sorted(glob(os.path.join(args.model_folder, "*.ply")))

    for model in model_list:
        sdf_paths = [model]
        config = json.load(open(args.config))
        config["sdf_paths"] = sdf_paths

        cfg_name = args.config.split("/")[-1].split(".")[0]
        model_name = model.split("/")[-1].split(".")[0]
        name = model_name
        print(name)

        cfg = Config(**config)
        cfg.name = name
        cfg.out_dir = os.path.join(cfg.out_dir, cfg_name, tag)
        cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, cfg_name, tag)

        if args.skip:
            out_file = os.path.join(cfg.out_dir, f"{model_name}.obj")
            if os.path.exists(out_file):
                continue

        model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)

        latents, latent_dim = config_latent(cfg)
        model = config_model(cfg, model_key, latent_dim)
        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
        )

        tokens = "0_1_0".split("_")
        # Interpolate latent
        i = int(tokens[0])
        j = int(tokens[1])
        t = float(tokens[2])
        latent = (1 - t) * latents[i] + t * latents[j]

        @jit
        def infer_sdf(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model(x, z)[:, 0]

        # jit causes "Results do not match the reference. This is likely a bug/unexpected loss of precision.", but why?
        # @jit
        def infer_normal(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model.call_grad(x, z)

        def grad_out_map_func(out):
            (sdf_, _), VN_ = out
            return sdf_, VN_

        # TODO: debug density
        # grid_res = 64
        # sdf, _ = voxel_infer(infer_sdf, grid_res=grid_res)
        # density = s_density(sdf)
        # ps.init()
        # ps.register_volume_grid("voxel", (grid_res, grid_res, grid_res),
        #                         (-1., -1., -1.),
        #                         (1., 1., 1.)).add_scalar_quantity("density", density[..., 0])
        # ps.show()
        # exit()

        infer_sdf_pytorch = jax2torch(infer_sdf)
        infer_normal_pytorch = jax2torch(infer_normal)
        s_density_pytorch = jax2torch(s_density)

        def sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
        ) -> torch.Tensor:
            """Define how to query density for the estimator."""
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sdf = batch_call_pytorch(infer_sdf_pytorch, positions, True)
            sigmas = s_density_pytorch(sdf)
            return sigmas

        def rgb_sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Query rgb and density values from a user-defined radiance field."""
            t_origins = rays_o[ray_indices]  # (n_samples, 3)
            t_dirs = rays_d[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sdf, normal = batch_call_pytorch(
                infer_normal_pytorch, positions, True, 2, grad_out_map_func
            )
            normal = normal / (torch.linalg.norm(normal, dim=-1, keepdim=True) + 1e-8)
            sigmas = s_density_pytorch(sdf)
            return normal, sigmas  # (n_samples, 3), (n_samples,)

        def occ_fn(x: torch.Tensor):
            sdf = batch_call(infer_sdf, t2j(x))
            return torch.abs(j2t(sdf)) < 0.1

        res = 256
        xx, yy = np.meshgrid(np.arange(res), np.arange(res))
        xx = 2 * xx / (res - 1) - 1
        yy = 2 * yy / (res - 1) - 1
        zz = -np.ones_like(xx)
        pts_view = np.stack([xx, yy, zz], -1).reshape(-1, 3)

        num_view = 8
        # For simplicity, let's assume the fov is 90 degree
        camera_centers = 2 * fibonacci_sphere(num_view + 2)[1:-1]
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0, 1, 0])

        timer = Timer()
        roi_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]).cuda()
        estimator = nerfacc.OccGridEstimator(roi_aabb, res).cuda()
        estimator.eval()
        estimator._update(0, occ_fn)

        timer.log("Update occ")

        for i in range(num_view):
            # pyrr stores glMatrix, which is column order
            eye = camera_centers[i]
            view = np.array(Matrix44.look_at(eye, target, up)).T
            R = view[:3, :3]
            t = view[:3, 3]
            pts_world = (pts_view - t[None, :]) @ R

            rays_o = np.repeat(eye[None, ...], res**2, axis=0)
            rays_d = pts_world - eye[None, ...]
            rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

            with torch.no_grad():
                # Apparently, I cannot j2t due to memory management
                rays_o = torch.from_numpy(rays_o).float().cuda()
                rays_d = torch.from_numpy(rays_d).float().cuda()

                ray_indices, t_starts, t_ends = estimator.sampling(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    sigma_fn=sigma_fn,
                    near_plane=0.1,
                    far_plane=4.0,
                    early_stop_eps=1e-4,
                    alpha_thre=1e-2,
                )

                timer.log(f"Sample_{i}")

                color, opacity, depth, extras = nerfacc.rendering(
                    t_starts,
                    t_ends,
                    ray_indices,
                    n_rays=rays_o.shape[0],
                    rgb_sigma_fn=rgb_sigma_fn,
                )

                timer.log(f"Render_{i}")

                color = 0.5 * (color + 1)
                color_img = color.reshape(res, res, 3).detach().cpu().numpy()
                color_img = np.uint8(color_img * 255)
                Image.fromarray(color_img).save(f"test_{i}.png")
