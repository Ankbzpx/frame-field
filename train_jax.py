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
from common import normalize
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data_pytorch
from sh_representation import (rotvec_to_sh4_expm, rotvec_to_R3, rot6d_to_R3,
                               rot6d_to_sh4_zonal, proj_sh4_to_R3)
from loss import (eikonal, align_sh4_explicit, align_sh4_functional,
                  align_sh4_explicit_cosine, align_basis_explicit,
                  align_basis_functional)
from eval_jax import eval
from tensorboardX import SummaryWriter
import copy
from icecream import ic

matplotlib.use('Agg')


def eval_iter(cfg: Config, model, latent, iter):
    cfg = copy.copy(cfg)
    cfg.name = f"{cfg.name}_{iter}"
    cfg.out_dir = os.path.join(cfg.out_dir, 'debug_iters')
    eval(cfg, model, latent, grid_res=256)


def train(cfg: Config, model: model_jax.MLP, data):
    writer = SummaryWriter(logdir=os.path.join('checkpoints/runs'))
    optim, opt_state = config_optim(cfg, model)

    smooth_schedule = optax.constant_schedule(cfg.loss_cfg.smooth)
    align_schedule = optax.linear_schedule(
        0, cfg.loss_cfg.align, 1,
        int(cfg.loss_cfg.align_begin * cfg.training.n_steps))
    lip_schedule = optax.linear_schedule(
        0, cfg.loss_cfg.lip, 1,
        int(cfg.loss_cfg.align_begin * cfg.training.n_steps))
    regularize_schedule = optax.linear_schedule(
        0, cfg.loss_cfg.regularize, int(0.2 * cfg.training.n_steps),
        int(cfg.loss_cfg.regularize_begin * cfg.training.n_steps))
    hessian_schedule = optax.linear_schedule(
        cfg.loss_cfg.hessian,
        cfg.loss_cfg.hessian_annealing * cfg.loss_cfg.hessian,
        int(0.1 * cfg.training.n_steps))
    digs_schedule = optax.linear_schedule(cfg.loss_cfg.digs,
                                          cfg.loss_cfg.digs_annealing,
                                          int(0.1 * cfg.training.n_steps))

    if not os.path.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)

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

            def eval_grad(samples):
                return jnp.empty(
                    (len(samples), 9, 3)), model.call_grad(samples, latent)

            jac_on, ((pred_on_sur_sdf, aux_on),
                     pred_normals_on_sur) = jax.lax.cond(
                         smooth_weight > 0, eval_smooth, eval_grad,
                         samples_on_sur)
            jac_off, ((pred_off_sur_sdf, _),
                      _) = jax.lax.cond(smooth_weight > 0, eval_smooth,
                                        eval_grad, samples_off_sur)
        else:
            (pred_on_sur_sdf, aux_on), pred_normals_on_sur = model.call_grad(
                samples_on_sur, latent)
            pred_off_sur_sdf = model(samples_off_sur, latent)[:, 0]

        if loss_cfg.hessian > 0:
            hessian_close = jax.lax.cond(hessian_weight > 0, model.call_hessian,
                                         lambda x, y: jnp.empty((len(x), 3, 3)),
                                         *(samples_close_sur, latent))

        if loss_cfg.digs > 0:
            hessian_off = jax.lax.cond(digs_weight > 0, model.call_hessian,
                                       lambda x, y: jnp.empty((len(x), 3, 3)),
                                       *(samples_off_sur, latent))

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
                    loss_align = align_sh4_explicit_cosine(sh4_align, normal)

                return loss_align

            sample_weight = jax.lax.stop_gradient(
                jnp.exp(-1e2 * jnp.abs(pred_on_sur_sdf)))
            normal_align = jax.lax.stop_gradient(
                jnp.vstack([pred_normals_on_sur]))
            aux_align = jnp.vstack([aux_on])
            loss_align = align_weight * (sample_weight * jax.lax.cond(
                align_weight > 0, eval_align_loss, lambda x, y: jnp.zeros(
                    len(sample_weight)), *(normal_align, aux_align))).mean()
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

                return loss_reg

            normal_reg = jnp.vstack([pred_normals_on_sur])
            aux_reg = jax.lax.stop_gradient(jnp.vstack([aux_on]))
            loss_reg = regularize_weight * jax.lax.cond(
                regularize_weight > 0, eval_reg_loss, lambda x, y: 0.,
                *(normal_reg, aux_reg))
            loss += loss_reg
            loss_dict['loss_reg'] = loss_reg

        if loss_cfg.lip > 0:
            loss_lip = lip_weight * model.get_aux_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        if loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, 'f')

            sh4_jac = jnp.vstack([jac_on, jac_off])
            loss_smooth = smooth_weight * jax.lax.cond(
                smooth_weight > 0, eval_smooth_loss, lambda x: 0., sh4_jac)
            loss += loss_smooth
            loss_dict['loss_smooth'] = loss_smooth

        if loss_cfg.hessian > 0:

            def eval_hessian_loss(hessian):
                return 0.5 * jnp.abs(vmap(jnp.linalg.det)(hessian)).mean()

            loss_hessian = hessian_weight * jax.lax.cond(
                hessian_weight > 0, eval_hessian_loss, lambda x: 0.,
                hessian_close)
            loss += loss_hessian
            loss_dict['loss_hessian'] = loss_hessian

        if loss_cfg.digs > 0:

            def eval_digs_loss(hessian):
                return jnp.clip(jnp.abs(vmap(jnp.trace)(hessian)), 0.1,
                                50).mean()

            loss_digs = digs_weight * jax.lax.cond(
                digs_weight > 0, eval_digs_loss, lambda x: 0., hessian_off)
            loss += loss_digs
            loss_dict['loss_digs'] = loss_digs

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

    loss_history = {}
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

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(cfg.training.n_steps)
            loss_history[key][iteration] = loss_dict[key]

        writer.add_scalars(f'{cfg.name}', loss_dict, iteration)
        pbar.set_postfix({"loss_total": loss_dict['loss_total']})

        # TODO: Use better plot such as tensorboardX
        # Loss plot
        # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
        if iteration % cfg.training.eval_every == 0 and iteration != 0:
            eval_latent = jnp.empty((0,))
            eval_iter(cfg, model, eval_latent, iteration)

    eqx.tree_serialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)

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

    data = config_training_data_pytorch(cfg, latents)

    train(cfg, model, data)
