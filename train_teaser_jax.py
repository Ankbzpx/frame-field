import argparse
import copy
import json
import os
import random

from common import aabb_compute, normalize, vis_oct_field
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, load_sdf, SDFDataset
from eval_jax import batch_call, eval, extract_surface
from loss import (
    align_basis_explicit,
    align_sh4_explicit,
    align_sh4_explicit_cosine,
    align_sh4_functional_grad,
    eikonal,
)
import model_jax
from sh_representation import (
    eulerXYZ_to_R3,
    proj_sh4_sdp,
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
import matplotlib
import numpy as np
import optax
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from icecream import ic
import polyscope as ps


jax.config.update("jax_default_matmul_precision", "tensorfloat32")

matplotlib.use("Agg")


def sample_plane(dim, skip_sides=False):
    axis = np.linspace(-1, 1, dim)
    if skip_sides:
        axis = axis[1:-1]
    xy = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    z = np.zeros(len(xy))
    xyz = np.hstack([xy, z[:, None]])
    return xyz


def eval_iter(cfg: Config, model, latent, tag):
    cfg = copy.copy(cfg)
    cfg.name = f"{cfg.name}_{tag}"
    cfg.out_dir = os.path.join(cfg.out_dir, "debug_iters")
    eval(cfg, model, latent, grid_res=256, save_octa=True)


def train(cfg: Config, model: model_jax.MLP, data, input_samples):
    writer = SummaryWriter(logdir=os.path.join("checkpoints/runs"))
    optim, opt_state = config_optim(cfg, model)

    # Let's not complicate things
    # smooth_schedule = optax.constant_schedule(cfg.loss_cfg.smooth)
    align_schedule = optax.constant_schedule(cfg.loss_cfg.align)
    lip_schedule = optax.constant_schedule(cfg.loss_cfg.lip)

    smooth_schedule = optax.linear_schedule(
        0,
        cfg.loss_cfg.smooth,
        1,
        int(cfg.loss_cfg.smooth_begin * cfg.training.n_steps),
    )
    align_schedule = optax.linear_schedule(
        0,
        cfg.loss_cfg.align,
        1,
        int(cfg.loss_cfg.align_begin * cfg.training.n_steps),
    )
    regularize_schedule = optax.linear_schedule(
        0,
        cfg.loss_cfg.regularize,
        int(0.2 * cfg.training.n_steps),
        int(cfg.loss_cfg.regularize_begin * cfg.training.n_steps),
    )
    hessian_schedule = optax.linear_schedule(
        cfg.loss_cfg.hessian,
        cfg.loss_cfg.hessian_annealing * cfg.loss_cfg.hessian,
        int(0.1 * cfg.training.n_steps),
    )
    digs_schedule = optax.linear_schedule(
        cfg.loss_cfg.digs, cfg.loss_cfg.digs_annealing, int(0.1 * cfg.training.n_steps)
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
        hessian_weight = hessian_schedule(step_count)
        lip_weight = lip_schedule(step_count)
        digs_weight = digs_schedule(step_count)

        # Map network output to sh4 parameterization
        if loss_cfg.rot6d:
            param_func = rot6d_to_sh4_zonal
            proj_func = vmap(rot6d_to_R3)
        elif loss_cfg.rotvec:
            # Needs second order differentiable
            param_func = rotvec_to_sh4_expm
            proj_func = vmap(rotvec_to_R3)
        else:
            param_func = lambda x: x
            proj_func = proj_sh4_to_R3

        # The python if is determined at tracing time. jax.lax.cond helps reduce computation when weight is scheduled to be 0
        if loss_cfg.smooth > 0:

            def eval_smooth(samples):
                return model.call_jac_param(samples, latent, param_func)

            jac_on, ((pred_on_sur_sdf, aux_on), pred_normals_on_sur) = eval_smooth(
                samples_on_sur
            )
        else:
            (pred_on_sur_sdf, aux_on), pred_normals_on_sur = model.call_grad(
                samples_on_sur, latent
            )
        (pred_off_sur_sdf, _), pred_normals_off_sur = model.mlps[0].call_grad(
            samples_off_sur, latent
        )

        # **IMPORTANT** This wrapper is necessary, as jax.lax.cond assumes all callables are python functions (which the equinox module functions are not)
        # More see: https://github.com/patrick-kidger/equinox/issues/119
        def eval_hessian(samples):
            return model.call_hessian(samples, latent)

        if loss_cfg.hessian > 0:
            hessian_close = jax.lax.cond(
                hessian_weight > 0,
                eval_hessian,
                lambda x: jnp.empty((len(x), 3, 3)),
                samples_close_sur,
            )

        if loss_cfg.digs > 0:
            hessian_off = jax.lax.cond(
                digs_weight > 0,
                eval_hessian,
                lambda x: jnp.empty((len(x), 3, 3)),
                samples_off_sur,
            )

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(-1e2 * jnp.abs(pred_off_sur_sdf)).mean()

        if loss_cfg.off_eikonal:
            loss_eikonal = (
                loss_cfg.eikonal
                * vmap(eikonal)(
                    jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
                ).mean()
            )
        else:
            loss_eikonal = loss_cfg.eikonal * vmap(eikonal)(pred_normals_on_sur).mean()

        loss = loss_mse + loss_off + loss_eikonal
        loss_dict = {
            "loss_mse": loss_mse,
            "loss_off": loss_off,
            "loss_eikonal": loss_eikonal,
        }

        sample_weight = jax.lax.stop_gradient(jnp.exp(-1e2 * jnp.abs(pred_on_sur_sdf)))

        def eval_align_loss(normal, aux):
            if loss_cfg.explicit_basis or loss_cfg.rot6d:
                basis_align = proj_func(aux)
                loss_align = align_basis_explicit(basis_align, normal)
            else:
                sh4_align = vmap(param_func)(aux)
                loss_align = align_sh4_functional_grad(sh4_align, normal)

            return loss_align

        if loss_cfg.align > 0:
            normal_align = jax.lax.stop_gradient(jnp.vstack([pred_normals_on_sur]))
            aux_align = jnp.vstack([aux_on])
            loss_align = (
                align_weight
                * (
                    sample_weight
                    * jax.lax.cond(
                        align_weight > 0,
                        eval_align_loss,
                        lambda x, y: jnp.zeros(len(sample_weight)),
                        *(normal_align, aux_align),
                    )
                ).mean()
            )
            loss += loss_align
            loss_dict["loss_align"] = loss_align

        if loss_cfg.regularize > 0:
            normal_reg = jnp.vstack([pred_normals_on_sur])
            aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on]))
            loss_reg = (
                regularize_weight
                * (
                    sample_weight
                    * jax.lax.cond(
                        regularize_weight > 0,
                        eval_align_loss,
                        lambda x, y: jnp.zeros(len(sample_weight)),
                        *(normal_reg, aux_reg),
                    )
                ).mean()
            )

            loss += loss_reg
            loss_dict["loss_reg"] = loss_reg

        if loss_cfg.lip > 0:
            loss_lip = lip_weight * model.get_aux_loss()
            loss += loss_lip
            loss_dict["loss_lip"] = loss_lip

        if loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f")

            sh4_jac = jnp.vstack([jac_on])
            loss_smooth = (
                smooth_weight
                * (
                    sample_weight
                    * jax.lax.cond(
                        smooth_weight > 0,
                        eval_smooth_loss,
                        lambda x: jnp.zeros(len(sample_weight)),
                        sh4_jac,
                    )
                ).mean()
            )
            loss += loss_smooth
            loss_dict["loss_smooth"] = loss_smooth

        if loss_cfg.hessian > 0:

            def eval_hessian_loss(hessian):
                return 0.5 * jnp.abs(vmap(jnp.linalg.det)(hessian)).mean()

            loss_hessian = hessian_weight * jax.lax.cond(
                hessian_weight > 0, eval_hessian_loss, lambda x: 0.0, hessian_close
            )
            loss += loss_hessian
            loss_dict["loss_hessian"] = loss_hessian

        if loss_cfg.digs > 0:

            def eval_digs_loss(hessian):
                return jnp.clip(jnp.abs(vmap(jnp.trace)(hessian)), 0.1, 50).mean()

            loss_digs = digs_weight * jax.lax.cond(
                digs_weight > 0, eval_digs_loss, lambda x: 0.0, hessian_off
            )
            loss += loss_digs
            loss_dict["loss_digs"] = loss_digs

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

    loss_history = {}
    pbar = tqdm(range(cfg.training.n_steps), dynamic_ncols=True)

    data_iter = iter(data)
    for iteration in pbar:
        batch = next(data_iter)
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)
        model, opt_state, loss_dict = make_step(model, opt_state, batch, cfg.loss_cfg)

        if np.isnan(loss_dict["loss_total"]):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(cfg.training.n_steps)
            loss_history[key][iteration] = loss_dict[key]

        writer.add_scalars(f"{cfg.name}", loss_dict, iteration)
        pbar.set_postfix({"loss_total": loss_dict["loss_total"]})

        align_end_iter = int((0.2 + cfg.loss_cfg.align_begin) * cfg.training.n_steps)
        if iteration == align_end_iter:
            eqx.tree_serialise_leaves(
                os.path.join(cfg.checkpoints_dir, f"{cfg.name}_align.eqx"), model
            )

        regularize_end_iter = int(
            (0.2 + cfg.loss_cfg.regularize_begin) * cfg.training.n_steps
        )
        if iteration == regularize_end_iter:
            eqx.tree_serialise_leaves(
                os.path.join(cfg.checkpoints_dir, f"{cfg.name}_regularize.eqx"), model
            )

        if iteration % cfg.training.eval_every == 0 and iteration != 0:
            eval_latent = jnp.empty((0,))
            eval_iter(cfg, model, eval_latent, iteration)

        # @jit
        # def infer_grad(x):
        #     z = jnp.empty((len(x), 0))
        #     (sdf_, aux_), VN_ = model.call_grad(x, z)
        #     return sdf_, aux_, VN_

        # sdf_input, q_input, vn_input = batch_call(infer_grad, input_samples, 3, lambda x: x)

        # sdf_input = np.asarray(sdf_input)
        # q_input = np.asarray(q_input)
        # vn_input = np.asarray(vn_input)

        # data_dump = {
        #     "sdf": sdf_input,
        #     "q": q_input,
        #     "vn": vn_input
        # }
        # tag = str(iteration).zfill(6)
        # np.savez(f"tmp/{tag}.npz", **data_dump)

    eqx.tree_serialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}_final.eqx"), model
    )

    return model


def apply_T(T, x):
    A = T[:3, :3]
    t = T[:3, 3][None, :]
    return x @ A.T + t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--vis_tag", type=str, help="Visualization tag.")
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split("/")[-1].split(".")[0]

    model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    tag = args.vis_tag
    if tag is not None:
        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"{cfg.name}_{tag}.eqx"), model
        )

        gt_path = cfg.sdf_paths[0]
        gt_path = gt_path.replace(gt_path.split("/")[-2], "gt")
        V_gt, F_gt = igl.read_triangle_mesh(os.path.expandvars(gt_path))

        sdf_data = load_sdf(cfg.sdf_paths[0])
        sur_sample = sdf_data["samples_on_sur"]
        pc_center, pc_scale, _ = aabb_compute(sur_sample)
        T_normalize = np.eye(4)
        T_normalize[:3, :3] *= pc_scale
        T_normalize[:3, 3] = pc_center

        @jit
        def infer_sdf(x):
            z = jnp.empty((0,))[None, ...].repeat(len(x), 0)
            return model.mlps[0](x, z)

        @jit
        def infer_octa(x):
            z = jnp.empty((0,))[None, ...].repeat(len(x), 0)
            return model.mlps[1](x, z)

        V, F, _ = extract_surface(infer_sdf, grid_res=512)
        V = apply_T(T_normalize, V)

        R_plane = eulerXYZ_to_R3(
            np.deg2rad(76.9152), np.deg2rad(19.3204), np.deg2rad(36.4993)
        )
        s_plane = np.array([0.143765, 0.143765, 0.143765])
        t_plane = np.array([1.21634, 0.253752, 1.40811])
        T_plane = np.eye(4)
        T_plane[:3, :3] = np.diag(s_plane) @ R_plane
        T_plane[:3, 3] = t_plane

        R_obj = eulerXYZ_to_R3(
            np.deg2rad(9.8842), np.deg2rad(-29.596), np.deg2rad(252.193)
        )
        s_obj = np.array([1.377, 1.377, 1.377])
        t_obj = np.array([1.1543, 0.014629, 0.794958])
        T_obj = np.eye(4)
        T_obj[:3, :3] = np.diag(s_obj) @ R_obj
        T_obj[:3, 3] = t_obj

        T_relative = np.linalg.inv(T_obj) @ T_plane

        V_plane = np.array(
            [
                [-1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [1, -1, 0],
            ]
        )
        V_plane = apply_T(T_relative, V_plane)
        F_plane = np.array([[0, 1, 2], [0, 2, 3]])

        image_res = 512
        samples_image = sample_plane(image_res)
        samples_image = apply_T(T_relative, samples_image)
        sdf = infer_sdf(apply_T(jnp.linalg.inv(T_normalize), samples_image)).reshape(
            -1,
        )

        octa_res = 32
        samples_octa = sample_plane(octa_res, True)
        samples_octa = apply_T(T_relative, samples_octa)
        octa = infer_octa(apply_T(jnp.linalg.inv(T_normalize), samples_octa))
        octa = proj_sh4_sdp(octa)
        Rs = proj_sh4_to_R3(octa)
        V_octa, F_octa = vis_oct_field(Rs, samples_octa, 0.06 / octa_res)

        V = apply_T(T_obj, V)
        V_gt = apply_T(T_obj, V_gt)
        V_plane = apply_T(T_obj, V_plane)
        samples_image = apply_T(T_obj, samples_image)
        V_octa = apply_T(T_obj, V_octa)

        np.save(f"output/{tag}.npy", sdf)
        igl.write_triangle_mesh(
            f"output/{tag}_octa.obj", np.float64(V_octa), np.int64(F_octa)
        )
        igl.write_triangle_mesh(f"output/{tag}.obj", np.float64(V), np.int64(F))
        igl.write_triangle_mesh(
            "output/plane.obj", np.float64(V_plane), np.int64(F_plane)
        )

        # ps.init()
        # ps.register_surface_mesh("extract", V, F)
        # ps.register_surface_mesh("gt", V_gt, F_gt)
        # ps.register_surface_mesh("pl", V_plane, F_plane)
        # ps.register_point_cloud("samples_image", samples_image).add_scalar_quantity(
        #     "sdf", sdf
        # )
        # ps.register_surface_mesh("octa", V_octa, F_octa)
        # ps.show()

    else:
        np.random.seed(0)
        dataset = SDFDataset(cfg, latents)

        g = torch.Generator()
        g.manual_seed(0)

        # https://github.com/google/jax/issues/3382
        import torch.multiprocessing as multiprocessing

        multiprocessing.set_start_method("forkserver", force=True)

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

        input_samples = dataset.sdf_data_list[0]["samples_on_sur"]

        train(cfg, model, dataloader, input_samples)
