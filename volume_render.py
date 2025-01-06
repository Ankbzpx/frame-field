import equinox as eqx
import jax
from jax import numpy as jnp, vmap, jit
import model_jax
from eval_jax import voxel_infer, batch_call

import json
from glob import glob
import argparse
from common import fibonacci_sphere, normalize
from config import Config
from config_utils import config_model, config_latent, config_training_data
import os
import numpy as np

import torch
from torch2jax import j2t
from jax2torch import jax2torch
import nerfacc

from pyrr import Matrix44
import polyscope as ps
from icecream import ic


# S-Density in NeuS, biased
# FIXME: How to specify a reasonable variance?
@jit
def s_density(x, s=100):
    return s * jnp.exp(-s * x) / jnp.pow(1 + jnp.exp(-s * x), 2)


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        nargs='*',
                        help='Path to pointcloud files.')
    parser.add_argument('--model_folder',
                        type=str,
                        default='data/sdf',
                        help='Path to pointcloud folder.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/octa.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    parser.add_argument('--skip',
                        action='store_true',
                        help='Skip existing output')
    args = parser.parse_args()

    # For simplicity, let's assume the fov is 90 degree
    camera_centers = 2 * fibonacci_sphere(10)[1:-1]
    target = np.array([0., 0., 0.])
    up = np.array([0, 1, 0])

    # pyrr stores glMatrix, which is column order
    eye = camera_centers[0]
    view = np.array(Matrix44.look_at(eye, target, up)).T
    R = view[:3, :3]
    t = view[:3, 3]

    res = 64
    xx, yy = np.meshgrid(np.arange(res), np.arange(res))
    xx = 2 * xx / (res - 1) - 1
    yy = 2 * yy / (res - 1) - 1
    zz = -np.ones_like(xx)
    pts_view = np.stack([xx, yy, zz], -1).reshape(-1, 3)
    pts_world = (pts_view - t[None, :]) @ R

    rays_o = np.repeat(eye[None, ...], res**2, axis=0)
    rays_d = pts_world - eye[None, ...]
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # Apparently, I cannot j2t due to memory management
    rays_o = torch.from_numpy(rays_o).float().cuda()
    rays_d = torch.from_numpy(rays_d).float().cuda()

    if args.model is not None:
        tag = ''
        model_list = args.model
    else:
        # TODO; Maybe not hard coded
        tag = '_'.join(args.model_folder.split('/')[-2:])
        model_list = sorted(glob(os.path.join(args.model_folder, '*.ply')))

    for model in model_list:
        sdf_paths = [model]
        config = json.load(open(args.config))
        config['sdf_paths'] = sdf_paths

        cfg_name = args.config.split('/')[-1].split('.')[0]
        model_name = model.split('/')[-1].split('.')[0]
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

        model_key, data_key = jax.random.split(
            jax.random.PRNGKey(cfg.training.seed), 2)

        latents, latent_dim = config_latent(cfg)
        model = config_model(cfg, model_key, latent_dim)
        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)

        tokens = '0_1_0'.split('_')
        # Interpolate latent
        i = int(tokens[0])
        j = int(tokens[1])
        t = float(tokens[2])
        latent = (1 - t) * latents[i] + t * latents[j]

        @jit
        def infer_sdf(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model(x, z)[:, 0]

        # FIXME: Convert to cpu to save vram
        def infer_sdf_batched(x):
            return batch_call(infer_sdf, x)

        # TODO: debug density
        grid_res = 64
        # sdf, _ = voxel_infer(infer_sdf, grid_res=grid_res)
        # density = s_density(sdf)
        # ps.init()
        # ps.register_volume_grid("voxel", (grid_res, grid_res, grid_res),
        #                         (-1., -1., -1.),
        #                         (1., 1., 1.)).add_scalar_quantity("density", density[..., 0])
        # ps.show()
        # exit()

        infer_sdf_torch = jax2torch(infer_sdf_batched)
        s_density_torch = jax2torch(s_density)

        def sigma_fn(t_starts: torch.Tensor, t_ends: torch.Tensor,
                     ray_indices: torch.Tensor) -> torch.Tensor:
            """ Define how to query density for the estimator."""
            t_origins = rays_o[ray_indices]    # (n_samples, 3)
            t_dirs = rays_d[ray_indices]    # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sdf = infer_sdf_torch(positions)
            sigmas = s_density_torch(sdf)
            return sigmas

        def occ_fn(x: torch.Tensor):
            sdf = infer_sdf_torch(x)
            return sdf < 0

        roi_aabb = torch.tensor([-1., -1., -1., 1., 1., 1.]).cuda()
        estimator = nerfacc.OccGridEstimator(roi_aabb, grid_res).cuda()

        estimator._update(0, occ_fn)
        ray_indices, t_starts, t_ends = estimator.sampling(rays_o=rays_o,
                                                           rays_d=rays_d,
                                                           sigma_fn=sigma_fn,
                                                           near_plane=0.1,
                                                           far_plane=4.0,
                                                           early_stop_eps=1e-4,
                                                           alpha_thre=1e-2)
        ic(ray_indices)
        ic(t_starts)
        ic(t_ends)
