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
    def __init__(self, cfg: Config, data, latent):
        super().__init__()

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps

        # Working on numpy array
        self.latent = np.array(latent[0])
        self.samples_on_sur = data["x_on"]
        self.normals_on_sur = data["n_on"]

        self.samples_close_sur = data["x_close"]
        self.normals_close_sur = data["n_close"]
        self.sdf_close_sur = data["sdf"]

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        idx = np.random.choice(np.arange(len(self.samples_on_sur)), self.n_samples)
        samples_on_sur = self.samples_on_sur[idx]
        normals_on_sur = self.normals_on_sur[idx]

        samples_close_sur = self.samples_close_sur[idx]
        normals_close_sur = self.normals_close_sur[idx]
        sdf_close_sur = self.sdf_close_sur[idx]

        latent = np.repeat(self.latent[None, ...], len(samples_on_sur), axis=0)

        return {
            "samples_on_sur": samples_on_sur,
            "normals_on_sur": normals_on_sur,
            "samples_close_sur": samples_close_sur,
            "normals_close_sur": normals_close_sur,
            "sdf_close_sur": sdf_close_sur,
            "latent": latent,
        }


def train(cfg: Config, model: model_jax.MLP, data, beta):
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
        samples_close_sur: Array,
        normals_close_sur: Array,
        sdf_close_sur: Array,
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
        if loss_cfg.smooth > 0:

            def eval_smooth(samples):
                return model.call_jac_param(samples, latent, param_func)

            jac_on, ((_, aux_on), _) = eval_smooth(samples_on_sur)
            jac_close, ((_, aux_close), _) = eval_smooth(samples_close_sur)
        else:
            (_, aux_on), _ = model.call_grad(samples_on_sur, latent)
            (_, aux_close), _ = model.call_grad(samples_close_sur, latent)

        loss = 0
        loss_dict = {}

        sample_weight = jax.lax.stop_gradient(jnp.exp(-beta * jnp.abs(sdf_close_sur)))
        sample_weight = jnp.concat([jnp.ones_like(sample_weight), sample_weight])

        if loss_cfg.align > 0:

            def eval_align_loss(normal, aux):
                sh4_align = vmap(param_func)(aux)
                loss_align = align_sh4_functional_grad(sh4_align, normal)
                return loss_align

            normal_align = jax.lax.stop_gradient(
                jnp.vstack([normals_on_sur, normals_close_sur])
            )
            aux_align = jnp.vstack([aux_on, aux_close])
            loss_align = (
                align_weight
                * (sample_weight * eval_align_loss(normal_align, aux_align)).mean()
            )
            loss += loss_align
            loss_dict["loss_align"] = loss_align

        if loss_cfg.regularize > 0:

            def eval_reg_loss(normal, aux):
                sh4_align = vmap(param_func)(aux)
                loss_reg = align_sh4_functional_grad(sh4_align, normal)
                return loss_reg

            normal_reg = jnp.vstack([normals_on_sur, normals_close_sur])
            aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on, aux_close]))

            loss_reg = (
                regularize_weight
                * (sample_weight * eval_reg_loss(normal_reg, aux_reg)).mean()
            )
            loss += loss_reg
            loss_dict["loss_reg"] = loss_reg

        if loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f")

            loss_smooth = (
                smooth_weight
                * (
                    sample_weight * eval_smooth_loss(jnp.vstack([jac_on, jac_close]))
                ).mean()
            )
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
    V,
    latent,
    vis=False,
    grid_res=256,
):
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
        ps.register_surface_mesh("Vis", V_octa, F_octa)
        ps.show()

    output_dir = os.path.join(os.path.dirname(cfg.out_dir), "toy")
    os.makedirs(output_dir, exist_ok=True)
    igl.write_triangle_mesh(
        os.path.join(output_dir, f"{cfg.name}_octa.obj"), V_octa, F_octa
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/toy.json", help="Path to config file."
    )
    parser.add_argument("--case", type=str, default="bevel", help="Toy case.")
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--vis", action="store_true", help="Visualize")
    parser.add_argument("--smooth", type=float, default=1.0, help="Octa smoothness")
    parser.add_argument("--beta", type=float, default=100.0, help="Density variance")
    args = parser.parse_args()

    name = args.case
    config = json.load(open(args.config))
    config["sdf_paths"] = [f"data/toy/{name}.npz"]
    config["loss_cfg"]["smooth"] = args.smooth

    cfg_name = f"{name}_{args.smooth}_{args.beta}"

    cfg = Config(**config)
    cfg.name = cfg_name
    cfg.out_dir = os.path.join(cfg.out_dir, cfg_name)
    cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, cfg_name)

    data = np.load(cfg.sdf_paths[0])

    model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)
    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    dataset = ToyDataset(cfg, data, latents)

    if args.eval:
        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
        )
    else:
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
        model = train(cfg, model, dataloader, args.beta)

    eval(cfg, model, dataset.samples_on_sur, jnp.zeros((0,)), args.vis)
