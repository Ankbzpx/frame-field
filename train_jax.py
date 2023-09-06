import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
import equinox as eqx
from jaxtyping import PyTree, Array
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
import os

import model_jax
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data
from sh_representation import rotvec_to_sh4, rotvec_n_to_z, rotvec_to_R9, project_n, rot6d_to_sh4_zonal

import polyscope as ps
from icecream import ic

matplotlib.use('Agg')


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)

    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


@jit
def eikonal(x):
    return jnp.abs(jnp.linalg.norm(x) - 1)


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
            val, jac = model.call_jac(
                samples_on_sur, latent,
                rot6d_to_sh4_zonal if loss_cfg.rot6d else lambda x: x)
            pred_on_sur_sdf = val[:, 0]
            aux = val[:, 1:]
            sh4 = aux[:, :9]
            pred_normals_on_sur = jac[:, 0]

            val_off, jac_off = model.call_jac(
                samples_off_sur, latent,
                rot6d_to_sh4_zonal if loss_cfg.rot6d else lambda x: x)
            pred_off_sur_sdf = val_off[:, 0]
            aux_off = val_off[:, 1:]
            sh4_off = aux_off[:, :9]
            pred_normals_off_sur = jac_off[:, 0]
        else:
            (pred_on_sur_sdf, aux), pred_normals_on_sur = model.call_grad(
                samples_on_sur, latent)

            sh4 = vmap(rot6d_to_sh4_zonal)(
                aux[:, :6]) if loss_cfg.rot6d else aux[:, :9]

            (pred_off_sur_sdf, aux_off), pred_normals_off_sur = model.call_grad(
                samples_off_sur, latent)

            sh4_off = vmap(rot6d_to_sh4_zonal)(
                aux_off[:, :6]) if loss_cfg.rot6d else aux_off[:, :9]

        normal_pred = jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])

        if loss_cfg.match_all_level_set:
            normal_align = jnp.where(loss_cfg.allow_gradient, normal_pred,
                                     jax.lax.stop_gradient(normal_pred))
            sh4_align = jnp.vstack([sh4, sh4_off])
        elif loss_cfg.match_zero_level_set:
            normal_align = jnp.where(loss_cfg.allow_gradient,
                                     pred_normals_on_sur,
                                     jax.lax.stop_gradient(pred_normals_on_sur))
            sh4_align = sh4
        else:
            normal_align = normals_on_sur
            sh4_align = sh4

        # Alignment
        R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal_align))
        sh4_n = vmap(project_n)(sh4_align, R9_zn)
        loss_align = loss_cfg.align * (1 - vmap(cosine_similarity)
                                       (sh4_align, sh4_n)).mean()
        # project_n does not penalize the norm
        loss_unit_norm = loss_cfg.unit_norm * vmap(eikonal)(sh4_align).mean()

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        # loss_sdf = jnp.abs(pred_off_sur_sdf - sdf_off_sur).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(
            -1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = loss_cfg.normal * (1 - vmap(cosine_similarity)(
            pred_normals_on_sur, normals_on_sur)).mean()
        loss_eikonal = loss_cfg.eikonal * vmap(eikonal)(normal_pred).mean()

        loss = loss_mse + loss_off + loss_normal + loss_eikonal + loss_align + loss_unit_norm

        loss_dict = {
            'loss_mse': loss_mse,
            'loss_off': loss_off,
            'loss_normal': loss_normal,
            'loss_eikonal': loss_eikonal,
            'loss_align': loss_align,
            'loss_unit_norm': loss_unit_norm
        }

        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * model.get_aux_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        if loss_cfg.smooth > 0:
            if loss_cfg.match_all_level_set:
                sh4_grad = jnp.vstack([jac[:, 1:10], jac_off[:, 1:10]])
            else:
                sh4_grad = jac[:, 1:10]

            sh4_grad_norm = vmap(jnp.linalg.norm, in_axes=[0, None])(sh4_grad,
                                                                     'f')
            loss_smooth = loss_cfg.smooth * sh4_grad_norm.mean()
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

        pbar.set_postfix({"loss": loss_dict['loss_total']})

        # TODO: better plot like using tensorboardX
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
