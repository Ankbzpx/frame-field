import numpy as np
import jax
from jax import vmap, numpy as jnp, jit, grad
import equinox as eqx
from jaxtyping import PyTree, Array
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os

import model_jax
from common import normalize, vis_oct_field
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data
from sh_representation import rotvec_to_sh4, rotvec_n_to_z, rotvec_to_R9, project_n, rot6d_to_R3, rot6d_to_sh4_zonal, oct_polynomial_sh4, proj_sh4_to_R3, oct_polynomial_zonal_unit_norm, oct_polynomial_sh4_unit_norm
from loss import cosine_similarity, eikonal, double_well_potential

import polyscope as ps
from icecream import ic

matplotlib.use('Agg')


def train(cfg: Config):
    model_key, data_key = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    optim, opt_state = config_optim(cfg, model)
    data = config_training_data(cfg, data_key, latents)

    total_steps = cfg.training.n_epochs * cfg.training.n_steps
    checkpoints_folder = 'checkpoints'
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model: model_jax.MLP, samples_on_sur: list[Array],
                  normals_on_sur: list[Array], samples_off_sur: list[Array],
                  sdf_off_sur: list[Array], latent: list[Array],
                  loss_cfg: LossConfig):

        if loss_cfg.smooth > 0:
            param_func = rot6d_to_sh4_zonal if loss_cfg.rot6d else lambda x: x
            jac, out_on = model.call_jac_param(samples_on_sur, latent,
                                               param_func)
            jac_off, out_off = model.call_jac_param(samples_off_sur, latent,
                                                    param_func)
        else:
            out_on = model.call_grad(samples_on_sur, latent)
            out_off = model.call_grad(samples_off_sur, latent)

        ((pred_on_sur_sdf, aux), pred_normals_on_sur) = out_on
        ((pred_off_sur_sdf, aux_off), pred_normals_off_sur) = out_off

        normal_pred = jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])

        if loss_cfg.match_all_level_set:
            normal_align = jnp.where(loss_cfg.allow_gradient, normal_pred,
                                     jax.lax.stop_gradient(normal_pred))
            aux_align = jnp.vstack([aux, aux_off])
        elif loss_cfg.match_zero_level_set:
            normal_align = jnp.where(loss_cfg.allow_gradient,
                                     pred_normals_on_sur,
                                     jax.lax.stop_gradient(pred_normals_on_sur))
            aux_align = aux
        else:
            normal_align = normals_on_sur
            aux_align = aux

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        # loss_sdf = jnp.abs(pred_off_sur_sdf - sdf_off_sur).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(
            -1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = loss_cfg.normal * (1 - vmap(cosine_similarity)(
            pred_normals_on_sur, normals_on_sur)).mean()
        loss_eikonal = loss_cfg.eikonal * vmap(eikonal)(normal_pred).mean()

        loss = loss_mse + loss_off + loss_normal + loss_eikonal

        loss_dict = {
            'loss_mse': loss_mse,
            'loss_off': loss_off,
            'loss_normal': loss_normal,
            'loss_eikonal': loss_eikonal
        }

        if loss_cfg.rot6d:
            basis = vmap(rot6d_to_R3)(aux)
            dps = jnp.einsum('bij,bi->bj', basis,
                             jax.lax.stop_gradient(normal_align))
            if loss_cfg.fix_basis:
                # We assume the first axis is normal
                loss_align = loss_cfg.align * (
                    (1 - dps[:, 0]).mean() + jnp.abs(dps[:, 1]).mean())
            else:

                loss_align = loss_cfg.align * double_well_potential(
                    jnp.abs(dps)).sum(-1).mean()
            loss += loss_align
            loss_dict['loss_align'] = loss_align
        else:
            # Alignment
            R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal_align))
            sh4_n = vmap(project_n)(aux_align, R9_zn)
            loss_align = loss_cfg.align * (1 - vmap(cosine_similarity)
                                           (aux_align, sh4_n)).mean()
            loss += loss_align
            loss_dict['loss_align'] = loss_align

            # project_n does not penalize the norm
            loss_unit_norm = loss_cfg.unit_norm * vmap(eikonal)(
                aux_align).mean()
            loss += loss_unit_norm
            loss_dict['loss_unit_norm'] = loss_unit_norm

        if loss_cfg.regularize > 0:
            if loss_cfg.rot6d:
                basis = vmap(rot6d_to_R3)(jax.lax.stop_gradient(aux_off))
                if loss_cfg.use_basis:
                    dps = jnp.einsum('bij,bi->bj', basis,
                                     vmap(normalize)(pred_normals_off_sur))
                    if loss_cfg.fix_basis:
                        loss_regularize = loss_cfg.regularize * (
                            (1 - dps[:, 0]).mean() + jnp.abs(dps[:, 1]).mean())
                    else:
                        loss_regularize = loss_cfg.regularize * double_well_potential(
                            jnp.abs(dps)).sum(-1).mean()
                else:
                    dps = vmap(oct_polynomial_zonal_unit_norm)(
                        pred_normals_off_sur, basis)

                    loss_regularize = loss_cfg.regularize * (1 - dps).mean()

            else:
                if loss_cfg.use_basis:
                    # This is unavoidably expensive
                    basis = proj_sh4_to_R3(jax.lax.stop_gradient(aux_off),
                                           max_iter=50)
                    dps = jnp.einsum('bij,bi->bj', basis,
                                     vmap(normalize)(pred_normals_off_sur))
                    loss_regularize = loss_cfg.regularize * double_well_potential(
                        jnp.abs(dps)).sum(-1).mean()
                else:
                    dps = vmap(oct_polynomial_sh4_unit_norm)(
                        pred_normals_off_sur, jax.lax.stop_gradient(aux_off))

                    loss_regularize = loss_cfg.regularize * (1 - dps).mean()

            loss += loss_regularize
            loss_dict['loss_regularize'] = loss_regularize

        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * model.get_aux_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        if loss_cfg.smooth > 0:
            if loss_cfg.match_all_level_set:
                sh4_jac = jnp.vstack([jac, jac_off])
            else:
                sh4_jac = jac

            sh4_jac_norm = vmap(jnp.linalg.norm, in_axes=[0, None])(sh4_jac,
                                                                    'f')
            loss_smooth = loss_cfg.smooth * sh4_jac_norm.mean()
            loss += loss_smooth
            loss_dict['loss_smooth'] = loss_smooth

        loss_dict['loss_total'] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(model: model_jax.MLP, opt_state: PyTree, batch: PyTree,
                  loss_cfg: LossConfig):

        grads, loss_dict = loss_func(model, **batch, loss_cfg=loss_cfg)
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    loss_history = {}
    pbar = tqdm(range(total_steps))
    for epoch in pbar:
        batch_id = epoch % cfg.training.n_steps
        batch = jax.tree_map(lambda x: x[batch_id], data)
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
            loss_history[key][epoch] = loss_dict[key]

        pbar.set_postfix({
            "loss": loss_dict['loss_total'],
            "loss_regularize": loss_dict['loss_regularize']
        })

        # TODO: Better plot such as using tensorboardX
        # Loss plot
        # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
        if epoch % cfg.training.plot_every == 0:
            plt.close(1)
            plt.figure(1)
            plt.semilogy(loss_history['loss_total'][:epoch])
            plt.title('Reconstruction loss')
            plt.grid()
            plt.savefig(f"{checkpoints_folder}/{cfg.name}_loss_history.jpg")

    eqx.tree_serialise_leaves(f"{checkpoints_folder}/{cfg.name}.eqx", model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    train(cfg)
