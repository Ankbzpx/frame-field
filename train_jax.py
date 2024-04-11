import numpy as np
import jax
from jax import vmap, numpy as jnp, jit, grad
import equinox as eqx
import optax
from typing import Callable
from jaxtyping import PyTree, Array
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os
import torch

import model_jax
from common import normalize
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data, config_training_data_pytorch
from sh_representation import (rotvec_to_sh4, rotvec_to_sh4_expm, rotvec_to_R3,
                               rotvec_n_to_z, rotvec_to_R9, project_n, sh4_z_4,
                               rot6d_to_R3, rot6d_to_sh4_zonal, proj_sh4_to_R3,
                               proj_sh4_to_rotvec, R3_to_sh4_zonal,
                               oct_polynomial_zonal_unit_norm,
                               oct_polynomial_sh4_unit_norm)
from loss import cosine_similarity, eikonal, double_well_potential

import polyscope as ps
from icecream import ic

matplotlib.use('Agg')


def train(cfg: Config, model: model_jax.MLP, data, checkpoints_folder):
    optim, opt_state = config_optim(cfg, model)

    # total_steps = cfg.training.n_epochs * cfg.training.n_steps
    total_steps = 10000

    if cfg.loss_cfg.smooth > 0:
        smooth_schedule = optax.polynomial_schedule(1e-1 * cfg.loss_cfg.smooth,
                                                    1e1 * cfg.loss_cfg.smooth,
                                                    0.5, total_steps,
                                                    cfg.training.warmup_steps)
    else:
        smooth_schedule = optax.constant_schedule(0)

    if cfg.loss_cfg.align > 0:
        align_schedule = optax.polynomial_schedule(1e-2 * cfg.loss_cfg.align,
                                                   cfg.loss_cfg.align, 0.5,
                                                   total_steps,
                                                   cfg.training.warmup_steps)
    else:
        align_schedule = optax.constant_schedule(0)

    # if cfg.loss_cfg.hessian > 0:
    #     hessian_schedule = optax.polynomial_schedule(cfg.loss_cfg.hessian,
    #                                                  1e-4, 0.5,
    #                                                  int(0.4 * total_steps),
    #                                                  int(0.2 * total_steps))
    # else:
    #     hessian_schedule = optax.constant_schedule(0)

    hessian_schedule = optax.constant_schedule(cfg.loss_cfg.hessian)

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
        elif loss_cfg.hessian > 0:
            hessian_on, out_on = model.call_hessian_aux(samples_on_sur, latent)
            hessian_close = model.call_hessian(samples_close_sur, latent)
            out_off = model.call_grad(samples_off_sur, latent)
        else:
            out_on = model.call_grad(samples_on_sur, latent)
            out_off = model.call_grad(samples_off_sur, latent)

        ((pred_on_sur_sdf, aux), pred_normals_on_sur) = out_on
        ((pred_off_sur_sdf, aux_off), pred_normals_off_sur) = out_off

        normal_pred = jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
        aux_pred = jnp.vstack([aux, aux_off])

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
            normal_align = normal_pred
            aux_align = aux_pred

            if loss_cfg.explicit_basis:
                basis_align = proj_func(aux_align)
                # Normalization matters here because we want the dot product to be either 0 or 1
                dps = jnp.einsum('bij,bi->bj', basis_align,
                                 vmap(normalize)(normal_align))
                loss_align = loss_cfg.align * double_well_potential(
                    jnp.abs(dps)).sum(-1).mean()
            else:
                if loss_cfg.rot6d:
                    basis_align = proj_func(aux_align)
                    poly_val = vmap(oct_polynomial_zonal_unit_norm)(
                        jax.lax.stop_gradient(normal_align), basis_align)
                    loss_align = loss_cfg.align * jnp.abs(1 - poly_val).mean()
                else:
                    sh4_align = vmap(param_func)(aux_align)
                    R9_zn = vmap(rotvec_to_R9)(
                        vmap(rotvec_n_to_z)(normal_align))

                    norm_scale = jnp.sqrt(7 / 12 +
                                          loss_cfg.xy_scale**2 * 5 / 12)
                    sh4_n = vmap(project_n,
                                 in_axes=(0, 0, None))(sh4_align, R9_zn,
                                                       loss_cfg.xy_scale)
                    # Its projection on n should match itself
                    loss_align = align_weight * (1 - vmap(cosine_similarity)
                                                 (sh4_align, sh4_n)).mean()
                    # The balance of align and unit norm weighting matters for flowline, even if the unit norm can mostly be satisfied near the end
                    loss_sh4_norm = loss_cfg.unit_norm * vmap(
                        eikonal, in_axes=(0, None))(sh4_align,
                                                    norm_scale).mean()
                    loss += loss_sh4_norm
                    loss_dict['loss_sh4_norm'] = loss_sh4_norm

            loss += loss_align
            loss_dict['loss_align'] = loss_align

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
            # loss_hessian = hessian_weight * 0.5 * (
            #     jnp.abs(vmap(jnp.linalg.det)(hessian_on)).mean() +
            #     jnp.abs(vmap(jnp.linalg.det)(hessian_close)).mean())
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
        step_count = opt_state.inner_states['standard'].inner_state[0].count
        grads, loss_dict = loss_func(model,
                                     **batch,
                                     loss_cfg=loss_cfg,
                                     step_count=step_count)
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    loss_history = {}
    pbar = tqdm(range(cfg.training.n_steps))

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
        pbar.set_postfix(loss_dict_log)

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
    cfg.training.n_steps = 501

    data = config_training_data_pytorch(cfg, latents)

    train(cfg, model, data, 'checkpoints')
