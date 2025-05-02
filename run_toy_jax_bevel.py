import argparse
import json
import os
import pickle
import random

from common import aabb_compute, normalize, normalize_aabb, Timer, vis_oct_field
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, load_sdf
from eval_jax import extract_surface
from loss import align_sh4_functional_grad, cosine_similarity, eikonal
import model_jax
from sh_representation import (
    eulerXYZ_to_R3,
    proj_sh4_to_R3,
    rot6d_to_R3,
    rot6d_to_sh4_zonal,
    rotvec_to_R3,
    rotvec_to_sh4_expm,
)

import equinox as eqx
import igl
import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array, PyTree
import numpy as np
import optax
import scipy.spatial
import torch

# https://github.com/google/jax/issues/3382
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import trimesh


multiprocessing.set_start_method("forkserver", force=True)


from icecream import ic
import polyscope as ps


class ToyDataset(Dataset):
    def __init__(self, cfg: Config, latents, case_tag):
        super().__init__()

        n_models = len(cfg.sdf_paths)
        assert n_models > 0

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps

        # Working on numpy array
        latents = np.array(latents)

        def sample_sdf_data(sdf_path, latent):
            sdf_data = load_sdf(sdf_path)

            sdf_sample_path = os.path.join(
                os.path.dirname(sdf_path), f"{case_tag}_sample.ply"
            )
            samples_close_sur = load_sdf(sdf_sample_path)["samples_on_sur"]

            V_center, scale, _ = aabb_compute(sdf_data["samples_on_sur"])
            samples_on_sur = (sdf_data["samples_on_sur"] - V_center) / scale
            sdf_data["samples_on_sur"] = samples_on_sur

            samples_close_sur = (samples_close_sur - V_center) / scale
            sdf_data["samples_close_sur"] = samples_close_sur

            # Reference: https://github.com/bearprin/Neural-Singular-Hessian/blob/ca7da0ce5d0c680393f1091ac8a6eafbe32248b4/surface_reconstruction/recon_dataset.py#L49
            # Use max distance among 51 closet points to approximate close neighbor
            kd_tree = scipy.spatial.KDTree(samples_close_sur)
            dists, _ = kd_tree.query(samples_close_sur, k=51, workers=-1)
            sigmas = dists[:, -1:]
            sdf_data["sigmas"] = sigmas
            sdf_data["latent"] = latent
            return sdf_data

        self.sdf_data_list = [
            sample_sdf_data(*args) for args in zip(cfg.sdf_paths, latents)
        ]

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        # VERY IMPORTANT: By default pytorch does not reset numpy seed for each __getitem__ call
        #   It means even if I fix the batch index in training loop, the results will still be different
        def sample_data(
            samples_on_sur, normals_on_sur, samples_close_sur, sigmas, latent
        ):
            idx_permute = np.random.permutation(len(samples_on_sur))
            idx = idx_permute[: self.n_samples]

            samples_on_sur = samples_on_sur[idx]

            if len(normals_on_sur) > 0:
                normals_on_sur = normals_on_sur[idx]

            samples_off_sur = np.random.uniform(-1, 1, size=(len(samples_on_sur), 3))

            idx = np.random.choice(np.arange(len(sigmas)), len(samples_on_sur))
            sigmas = sigmas[idx]
            samples_close_sur = samples_close_sur[idx]
            samples_close_sur = samples_close_sur + sigmas * np.random.randn(
                len(samples_on_sur), 3
            )

            latent = np.repeat(latent[None, ...], len(samples_on_sur), axis=0)

            return {
                "samples_on_sur": samples_on_sur.astype(np.float32),
                "normals_on_sur": normals_on_sur.astype(np.float32),
                "samples_off_sur": samples_off_sur.astype(np.float32),
                "samples_close_sur": samples_close_sur.astype(np.float32),
                "latent": latent.astype(np.float32),
            }

        sdf_data_samples_frag = [
            sample_data(**sdf_data) for sdf_data in self.sdf_data_list
        ]

        sdf_data = {}
        for key in sdf_data_samples_frag[0].keys():
            sdf_data[key] = np.hstack([frag[key] for frag in sdf_data_samples_frag])

        return sdf_data


def train(cfg: Config, model: model_jax.MLP, data, enable, reg_aux):
    optim, opt_state = config_optim(cfg, model)

    # Let's not complicate things
    smooth_schedule = optax.constant_schedule(cfg.loss_cfg.smooth)
    align_schedule = optax.constant_schedule(cfg.loss_cfg.align)

    regularize_schedule = optax.linear_schedule(
        0,
        cfg.loss_cfg.regularize,
        int(0.2 * cfg.training.n_steps),
        int(cfg.loss_cfg.regularize_begin * cfg.training.n_steps),
    )

    if not os.path.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(
        model: model_jax.MLP,
        samples_on_sur: Array,
        normals_on_sur: Array,
        samples_off_sur: Array,
        samples_close_sur: Array,
        latent: Array,
        loss_cfg: LossConfig,
        step_count: int,
    ):
        smooth_weight = smooth_schedule(step_count)
        align_weight = align_schedule(step_count)
        regularize_weight = regularize_schedule(step_count)

        param_func = lambda x: x
        proj_func = proj_sh4_to_R3

        # The python if is determined at tracing time. jax.lax.cond helps reduce computation when weight is scheduled to be 0
        if enable and loss_cfg.smooth > 0:

            def eval_smooth(samples):
                return model.call_jac_param(samples, latent, param_func)

            jac_on, ((pred_on_sur_sdf, aux_on), pred_normals_on_sur) = eval_smooth(
                samples_on_sur
            )
            jac_off, ((pred_off_sur_sdf, aux_off), pred_normals_off_sur) = eval_smooth(
                samples_off_sur
            )
        else:
            (pred_on_sur_sdf, aux_on), pred_normals_on_sur = model.call_grad(
                samples_on_sur, latent
            )
            (pred_off_sur_sdf, aux_off), pred_normals_off_sur = model.call_grad(
                samples_off_sur, latent
            )

        if reg_aux:
            (pred_close_sur_sdf, aux_close), pred_normals_close_sur = model.call_grad(
                samples_close_sur, latent
            )

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        loss_normal = (
            loss_cfg.normal
            * (1 - vmap(cosine_similarity)(pred_normals_on_sur, normals_on_sur)).mean()
        )
        loss_off = loss_cfg.off_sur * jnp.exp(-1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_eikonal = (
            loss_cfg.eikonal
            * vmap(eikonal)(
                jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
            ).mean()
        )
        loss = loss_mse + loss_off + loss_eikonal + loss_normal
        loss_dict = {
            "loss_mse": loss_mse,
            "loss_off": loss_off,
            "loss_eikonal": loss_eikonal,
            "loss_normal": loss_normal,
        }

        if enable and loss_cfg.align > 0:

            def eval_align_loss(normal, aux):
                sh4_align = vmap(param_func)(aux)
                loss_align = align_sh4_functional_grad(sh4_align, normal)
                return loss_align

            sample_weight = jax.lax.stop_gradient(
                jnp.exp(-1e2 * jnp.abs(pred_on_sur_sdf))
            )
            normal_align = jax.lax.stop_gradient(jnp.vstack([pred_normals_on_sur]))
            aux_align = jnp.vstack([aux_on])
            loss_align = (
                align_weight
                * (sample_weight * eval_align_loss(normal_align, aux_align)).mean()
            )
            loss += loss_align
            loss_dict["loss_align"] = loss_align

        if enable and loss_cfg.regularize > 0:

            def eval_reg_loss(normal, aux):
                sh4_align = vmap(param_func)(aux)
                loss_reg = align_sh4_functional_grad(sh4_align, normal)
                return loss_reg

            if reg_aux:
                normal_reg = jnp.vstack([pred_normals_on_sur, pred_normals_close_sur])
                aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on, aux_close]))
            else:
                normal_reg = jnp.vstack([pred_normals_on_sur])
                aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on]))

            loss_reg = regularize_weight * eval_reg_loss(normal_reg, aux_reg).mean()
            loss += loss_reg
            loss_dict["loss_reg"] = loss_reg

        if enable and loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f")

            sh4_jac = jnp.vstack([jac_on, jac_off])
            sh4_norm = jnp.linalg.norm(jnp.vstack([aux_on, aux_off]), axis=-1)
            loss_smooth = smooth_weight * (eval_smooth_loss(sh4_jac) / sh4_norm).mean()
            loss += loss_smooth
            loss_dict["loss_smooth"] = loss_smooth

        loss_dict["loss_total"] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(
        model: model_jax.MLP, opt_state: PyTree, batch: PyTree, loss_cfg: LossConfig
    ):
        # FIXME: The static index is risky--it depends on the order of optax.chain
        step_count = opt_state[0].count
        grads, loss_dict = loss_func(
            model, **batch, loss_cfg=loss_cfg, step_count=step_count
        )
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    pbar = tqdm(range(cfg.training.n_steps))

    data_iter = iter(data)
    for _ in pbar:
        batch = next(data_iter)
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)
        model, opt_state, loss_dict = make_step(model, opt_state, batch, cfg.loss_cfg)

        if np.isnan(loss_dict["loss_total"]):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        pbar.set_postfix(loss_dict)

    eqx.tree_serialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
    )

    return model


def eval(
    cfg: Config,
    model: model_jax.MLP,
    latent,
    enable,
    vis=False,
    grid_res=256,
):
    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    timer = Timer()

    V, F, _ = extract_surface(infer, grid_res=grid_res)

    timer.log("Extract surface")

    @jit
    def infer_aux(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 1:]

    def vis_oct(samples):
        aux = infer_aux(samples)
        Rs = proj_sh4_to_R3(aux)
        return vis_oct_field(Rs, samples, 0.64 / grid_res)

    V_octa, F_octa = vis_oct(V)

    if vis:
        ps.init()
        ps.register_surface_mesh(cfg.name, V, F)
        ps.register_surface_mesh("Vis", V_octa, F_octa)
        ps.show()

    output_dir = os.path.join(os.path.dirname(cfg.out_dir), "toy")
    os.makedirs(output_dir, exist_ok=True)
    igl.write_triangle_mesh(os.path.join(output_dir, f"{cfg.name}.obj"), V, F)
    igl.write_triangle_mesh(
        os.path.join(output_dir, f"{cfg.name}_octa.obj"), V_octa, F_octa
    )

    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/toy.json", help="Path to config file."
    )
    parser.add_argument("--case", type=str, default="bevel", help="Toy case.")
    parser.add_argument("--enable", action="store_true", help="Enable our loss")
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--vis", action="store_true", help="Visualize")
    parser.add_argument("--smooth", type=float, default=1.0, help="Octa smoothness")
    parser.add_argument("--reg_aux", action="store_true", help="Regularize aux samples")
    args = parser.parse_args()

    name = args.case
    sdf_paths = [f"data/sdf/{name}.ply"]
    config = json.load(open(args.config))
    config["sdf_paths"] = sdf_paths
    config["loss_cfg"]["smooth"] = args.smooth

    cfg_name = (
        f"{name}_{args.smooth}"
        + ("_aux" if args.reg_aux else "")
        + ("_ours" if args.enable else "_vanilla")
    )

    cfg = Config(**config)
    cfg.name = cfg_name
    cfg.out_dir = os.path.join(cfg.out_dir, cfg_name)
    cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, cfg_name)

    model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)
    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    if args.eval:
        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
        )
    else:
        dataset = ToyDataset(cfg, latents, name.split("_")[0])
        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )
        model = train(cfg, model, dataloader, args.enable, args.reg_aux)

    eval(cfg, model, jnp.zeros((0,)), args.enable, args.vis)
