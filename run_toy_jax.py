import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
import igl
import json
import argparse

import model_jax
from config import Config, LossConfig
from config_utils import config_model, config_latent, config_toy_training_data
from common import vis_oct_field, Timer, normalize
from eval_jax import extract_surface
from sh_representation import (proj_sh4_to_R3, rot6d_to_R3, eulerXYZ_to_R3,
                               rotvec_to_R3)
import os
import optax
from jaxtyping import PyTree, Array
from sh_representation import (rotvec_to_sh4_expm, rotvec_to_R3, rot6d_to_R3,
                               rot6d_to_sh4_zonal, proj_sh4_to_R3)
from loss import (eikonal, align_sh4_explicit, align_sh4_functional,
                  align_sh4_explicit_cosine, align_basis_explicit,
                  align_basis_functional, cosine_similarity,
                  double_well_potential, align_sh4_explicit_l2)
from tqdm import tqdm

import polyscope as ps
from icecream import ic


def train(cfg: Config, model: model_jax.MLP, data):
    # For faster convergence
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        cfg.training.lr,
        peak_value=5 * cfg.training.lr,
        warmup_steps=100,
        decay_steps=cfg.training.n_steps)
    optim = optax.adam(lr_scheduler)
    opt_state = optim.init(eqx.filter([model], eqx.is_array))

    smooth_schedule = optax.constant_schedule(cfg.loss_cfg.smooth)
    align_schedule = optax.linear_schedule(
        0, cfg.loss_cfg.align, 1,
        int(cfg.loss_cfg.align_begin * cfg.training.n_steps))
    regularize_schedule = optax.linear_schedule(
        0, cfg.loss_cfg.regularize, int(0.2 * cfg.training.n_steps),
        int(cfg.loss_cfg.regularize_begin * cfg.training.n_steps))

    if not os.path.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model: model_jax.MLP, samples_on_sur: Array,
                  normals_on_sur: Array, samples_off_sur: Array, latent: Array,
                  loss_cfg: LossConfig, step_count: int):

        smooth_weight = smooth_schedule(step_count)
        align_weight = align_schedule(step_count)
        regularize_weight = regularize_schedule(step_count)

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
            jac_on, ((pred_on_sur_sdf, aux_on),
                     pred_normals_on_sur) = model.call_jac_param(
                         samples_on_sur, latent, param_func)
            jac_off, ((pred_off_sur_sdf, aux_off),
                      pred_normals_off_sur) = model.call_jac_param(
                          samples_off_sur, latent, param_func)
        else:
            (pred_on_sur_sdf, aux_on), pred_normals_on_sur = model.call_grad(
                samples_on_sur, latent)
            (pred_off_sur_sdf, aux_off), pred_normals_off_sur = model.call_grad(
                samples_off_sur, latent)

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(
            -1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_eikonal = loss_cfg.eikonal * vmap(eikonal)(
            pred_normals_on_sur).mean()
        loss = loss_mse + loss_off + loss_eikonal
        loss_dict = {
            'loss_mse': loss_mse,
            'loss_off': loss_off,
            'loss_eikonal': loss_eikonal
        }

        if loss_cfg.align > 0:

            def eval_align_loss(normal, aux):
                if loss_cfg.explicit_basis or loss_cfg.rot6d:
                    basis_align = proj_func(aux)
                    loss_align = align_basis_explicit(basis_align, normal)
                else:
                    sh4_align = vmap(param_func)(aux)
                    # **VERY IMPORTANT** Enforcing unit norm matters here because our samples are very sparse and separated by a large gap
                    #   If the two connected components have very distinct norm, the shs of frames in between (the gap) will be interpolated
                    #   resulting in non-trivial rotation. It is rarely the issue for sufficient dense input
                    loss_align = align_sh4_explicit_cosine(
                        sh4_align,
                        normal) + 0.1 * vmap(eikonal)(sh4_align).mean()

                return loss_align

            sample_weight = jax.lax.stop_gradient(
                jnp.exp(-1e2 * jnp.abs(pred_on_sur_sdf)))
            normal_align = jax.lax.stop_gradient(
                jnp.vstack([pred_normals_on_sur]))
            aux_align = jnp.vstack([aux_on])
            loss_align = align_weight * (sample_weight * eval_align_loss(
                normal_align, aux_align)).mean()
            loss += loss_align
            loss_dict['loss_align'] = loss_align

        if loss_cfg.regularize > 0:

            def eval_reg_loss(normal, aux):
                if loss_cfg.explicit_basis or loss_cfg.rot6d:
                    basis_reg = proj_func(aux)
                    loss_reg = align_basis_explicit(basis_reg, normal).mean()
                else:
                    sh4_align = vmap(param_func)(aux)
                    sh4_align = vmap(normalize)(sh4_align)
                    loss_reg = align_sh4_explicit(sh4_align, normal).mean()
                    # loss_reg = align_sh4_explicit_l2(sh4_align, normal).mean()
                    # loss_reg = align_sh4_explicit_cosine(sh4_align,
                    #                                      normal).mean()
                    # loss_reg = align_sh4_functional(sh4_align, normal).mean()

                return loss_reg

            # We need additional samples given how sparse our on-manifold samples are
            normal_reg = jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
            aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on, aux_off]))
            loss_reg = regularize_weight * eval_reg_loss(normal_reg, aux_reg)
            loss += loss_reg
            loss_dict['loss_reg'] = loss_reg

        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * model.get_aux_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        if loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, 'f').mean()

            sh4_jac = jnp.vstack([jac_on, jac_off])
            loss_smooth = smooth_weight * eval_smooth_loss(sh4_jac)
            loss += loss_smooth
            loss_dict['loss_smooth'] = loss_smooth

        loss_dict['loss_total'] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(model: model_jax.MLP, opt_state: PyTree, batch: PyTree,
                  loss_cfg: LossConfig):
        # FIXME: The static index is risky--it depends on the order of optax.chain
        step_count = opt_state[0].count
        grads, loss_dict = loss_func(model,
                                     **batch,
                                     loss_cfg=loss_cfg,
                                     step_count=step_count)
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    pbar = tqdm(range(cfg.training.n_steps))

    data_iter = iter(data)
    for iteration in pbar:
        batch = next(data_iter)
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)
        model, opt_state, loss_dict = make_step(model, opt_state, batch,
                                                cfg.loss_cfg)

        if np.isnan(loss_dict['loss_total']):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        pbar.set_postfix({"loss_total": loss_dict['loss_total']})

    eqx.tree_serialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)

    return model


def eval(cfg: Config,
         model: model_jax.MLP,
         latent,
         samples_sup,
         samples_interp,
         out_dir,
         vis=False):

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    timer = Timer()

    V, F, _ = extract_surface(infer)

    timer.log('Extract surface')

    @jit
    def infer_aux(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 1:]

    def vis_oct(samples):
        aux = infer_aux(samples)
        if cfg.loss_cfg.rot6d:
            Rs = vmap(rot6d_to_R3)(aux[:, :6])
        elif cfg.loss_cfg.rotvec:
            Rs = vmap(rotvec_to_R3)(aux[:, :3])
        else:
            sh4 = aux[:, :9]
            # rotvec = proj_sh4_to_rotvec(sh4)
            # Rs = vmap(rotvec_to_R3)(rotvec)
            Rs = proj_sh4_to_R3(sh4)

        return vis_oct_field(Rs, samples, 0.01)

    V_vis_sup, F_vis_sup = vis_oct(samples_sup)
    V_vis_interp, F_vis_interp = vis_oct(samples_interp)

    # Compensate small rotation
    R = eulerXYZ_to_R3(np.pi / 6, np.pi / 3, np.pi / 4)
    V = np.float64(V @ R)
    V_vis_sup = np.float64(V_vis_sup @ R)
    V_vis_interp = np.float64(V_vis_interp @ R)

    if vis:
        ps.init()
        ps.register_surface_mesh("mesh", V, F)
        ps.register_surface_mesh("Oct frames supervise", V_vis_sup, F_vis_sup)
        ps.register_surface_mesh("Oct frames interpolation", V_vis_interp,
                                 F_vis_interp)
        ps.show()

    # V_cube = np.vstack([V_vis_sup, V_vis_interp])
    # F_cube = np.vstack([F_vis_sup, F_vis_interp + F_vis_sup.max() + 1])

    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_mc.obj", V, F)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_sup.obj", V_vis_sup,
                            F_vis_sup)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_interp.obj", V_vis_interp,
                            F_vis_interp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/toy.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    args = parser.parse_args()

    # 1, 2, 3, 4
    # 150, 135, 120, 90, 60, 45, 30
    for gap in [4]:
        for theta in [90]:
            name = f"crease_{gap}_{theta}"

            config = json.load(open(args.config))
            # Placeholder
            config['sdf_paths'] = ['/']
            cfg = Config(**config)
            cfg.name = name

            toy_sample = np.load(f"data/toy/crease_{gap}_{theta}.npz")
            samples_sup = toy_sample['samples_sup']
            samples_vn_sup = toy_sample['samples_vn_sup']
            samples_interp = toy_sample['samples_interp']

            model_key, data_key = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), 2)

            latents, latent_dim = config_latent(cfg)
            model = config_model(cfg, model_key, latent_dim)

            if args.eval:
                model: model_jax.MLP = eqx.tree_deserialise_leaves(
                    os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)
            else:
                data = config_toy_training_data(cfg, samples_sup,
                                                samples_vn_sup, latents)
                model = train(cfg, model, data)

            eval(cfg, model, jnp.zeros((0,)), samples_sup, samples_interp,
                 "output/toy", args.vis)

            # exit()
