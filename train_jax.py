import numpy as np
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
import optax
from jaxtyping import PyTree, Array
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os

import model_jax
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data_pytorch
from sh_representation import (rotvec_to_sh4_expm, rotvec_to_R3, rot6d_to_R3,
                               rot6d_to_sh4_zonal, proj_sh4_to_R3)
from loss import (eikonal, align_sh4_explicit, align_basis_explicit,
                  align_basis_functional)

matplotlib.use('Agg')


def train(cfg: Config,
          model: model_jax.MLP,
          data,
          checkpoints_folder,
          total_steps=None):
    optim, opt_state = config_optim(cfg, model)

    if total_steps is None:
        total_steps = cfg.training.n_steps

    smooth_schedule = optax.constant_schedule(cfg.loss_cfg.smooth)
    align_schedule = optax.constant_schedule(cfg.loss_cfg.align)
    regularize_schedule = optax.polynomial_schedule(
        1e-4, cfg.loss_cfg.regularize, 0.5, int(0.4 * cfg.training.n_steps),
        int(0.2 * cfg.training.n_steps))
    hessian_schedule = optax.polynomial_schedule(
        cfg.loss_cfg.hessian, 1e-4, 0.5, int(0.4 * cfg.training.n_steps),
        int(0.2 * cfg.training.n_steps))

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model: model_jax.MLP, samples_on_sur: Array,
                  normals_on_sur: Array, samples_off_sur: Array,
                  samples_close_sur: Array, latent: Array, loss_cfg: LossConfig,
                  step_count: int):

        smooth_weight = smooth_schedule(step_count)
        align_weight = align_schedule(step_count)
        regularize_weight = regularize_schedule(step_count)
        hessian_weight = hessian_schedule(step_count)

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

        # The scheduled weight is jit traced value--I'm not allowed to compare it with a scalar
        if not loss_cfg.grid and loss_cfg.smooth > 0:
            jac_on, out_on = model.call_jac_param(samples_on_sur, latent,
                                                  param_func)
            jac_off, out_off = model.call_jac_param(samples_off_sur, latent,
                                                    param_func)
        else:
            out_on = model.call_grad(samples_on_sur, latent)
            out_off = model.call_grad(samples_off_sur, latent)

        if loss_cfg.hessian > 0:
            hessian_close, out_close = model.call_hessian_aux(
                samples_close_sur, latent)
        else:
            out_close = model.call_grad(samples_close_sur, latent)

        ((pred_on_sur_sdf, aux_on), pred_normals_on_sur) = out_on
        ((pred_off_sur_sdf, aux_off), pred_normals_off_sur) = out_off
        ((pred_close_sur_sdf, aux_close), pred_normals_close_sur) = out_close

        # normal_pred = jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
        # aux_pred = jnp.vstack([aux_on, aux_off])

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
            normal_align = jax.lax.stop_gradient(
                jnp.vstack([pred_normals_on_sur]))
            aux_align = jnp.vstack([aux_on])

            if loss_cfg.explicit_basis:
                basis_align = proj_func(aux_align)
                loss_align = align_basis_explicit(basis_align, normal_align)
            elif loss_cfg.rot6d:
                basis_align = proj_func(aux_align)
                loss_align = align_basis_functional(basis_align, normal_align)
            else:
                sh4_align = vmap(param_func)(aux_align)
                loss_align = align_sh4_explicit(sh4_align, normal_align)

            loss_align = align_weight * loss_align
            loss += loss_align
            loss_dict['loss_align'] = loss_align

        # The same as align. Duplicate for weight scheduling
        if loss_cfg.regularize > 0:
            normal_reg = jnp.vstack([pred_normals_close_sur])
            aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_close]))

            if loss_cfg.explicit_basis:
                basis_reg = proj_func(aux_reg)
                loss_reg = align_basis_explicit(basis_reg, normal_reg)
            elif loss_cfg.rot6d:
                basis_reg = proj_func(aux_reg)
                loss_reg = align_basis_functional(basis_reg, normal_reg)
            else:
                sh4_align = vmap(param_func)(aux_reg)
                loss_reg = align_sh4_explicit(sh4_align, normal_reg)

            loss_reg = regularize_weight * loss_reg
            loss += loss_reg
            loss_dict['loss_reg'] = loss_reg

        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * model.get_aux_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        if loss_cfg.smooth > 0:
            if loss_cfg.grid:
                loss_smooth = smooth_weight * model.get_aux_loss()
            else:
                sh4_jac = jnp.vstack([jac_on, jac_off])
                sh4_jac_norm = vmap(jnp.linalg.norm, in_axes=(0, None))(sh4_jac,
                                                                        'f')
                loss_smooth = smooth_weight * sh4_jac_norm.mean()
            loss += loss_smooth
            loss_dict['loss_smooth'] = loss_smooth

        if loss_cfg.hessian > 0:
            loss_hessian = hessian_weight * 0.5 * jnp.abs(
                vmap(jnp.linalg.det)(hessian_close)).mean()
            loss += loss_hessian
            loss_dict['loss_hessian'] = loss_hessian

        loss_dict['loss_total'] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(model: model_jax.MLP, opt_state: PyTree, batch: PyTree,
                  loss_cfg: LossConfig):
        # FIXME: The static index is risky--it depends on the order of optax.chain
        step_count = opt_state[-1].count
        grads, loss_dict = loss_func(model,
                                     **batch,
                                     loss_cfg=loss_cfg,
                                     step_count=step_count)
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    loss_history = {}
    pbar = tqdm(range(total_steps))

    for iteration in pbar:
        batch = next(iter(data))
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)
        model, opt_state, loss_dict = make_step(model, opt_state, batch,
                                                cfg.loss_cfg)

        if np.isnan(loss_dict['loss_total']):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(total_steps)
            loss_history[key][iteration] = loss_dict[key]

        loss_dict_log = jax.tree.map(lambda x: f"{x:.4}", loss_dict)
        pbar.set_postfix({
            "loss_total": loss_dict_log['loss_total'],
            "loss_mse": loss_dict_log['loss_mse'],
        })

        # TODO: Use better plot such as tensorboardX
        # Loss plot
        # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
        if iteration % cfg.training.plot_every == 0:
            plt.close(1)
            plt.figure(1)
            plt.semilogy(loss_history['loss_total'][:iteration])
            plt.title('Reconstruction loss')
            plt.grid()
            plt.savefig(f"{checkpoints_folder}/{cfg.name}_loss_history.jpg")

    eqx.tree_serialise_leaves(f"{checkpoints_folder}/{cfg.name}.eqx", model)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    model_key, data_key = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    # Debug
    # total_steps = 1001
    total_steps = None

    data = config_training_data_pytorch(cfg, latents, total_steps=total_steps)

    train(cfg, model, data, 'checkpoints')
