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
from sh_representation import rotvec_to_sh4, rotvec_n_to_z, rotvec_to_R9, project_n, rot6d_to_sh4_zonal, oct_polynomial_sh4, proj_sh4_to_R3

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


@jit
def closest_rep_vec(v, sh4, max_iter=10):
    v = jax.lax.stop_gradient(v)
    sh4 = jax.lax.stop_gradient(sh4)

    def proj_sh4(i, v):
        return normalize(grad(oct_polynomial_sh4)(v, sh4))

    return jax.lax.fori_loop(0, max_iter, proj_sh4, v)


@jit
def double_well_potential(x):
    return 16 * (x - 0.5)**4 - 8 * (x - 0.5)**2 + 1


def train(cfg: Config):
    model_key, data_key = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    # model: model_jax.MLP = eqx.tree_deserialise_leaves(
    #     f"checkpoints/{cfg.name}.eqx", model)

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

        param_func = rot6d_to_sh4_zonal if loss_cfg.rot6d else lambda x: x

        if loss_cfg.smooth > 0:
            jac, out_on = model.call_jac_param(samples_on_sur, latent,
                                               param_func)
            jac_off, out_off = model.call_jac_param(samples_off_sur, latent,
                                                    param_func)
        else:
            out_on = model.call_grad_param(samples_on_sur, latent, param_func)
            out_off = model.call_grad_param(samples_off_sur, latent, param_func)

        ((pred_on_sur_sdf, sh4), pred_normals_on_sur) = out_on
        ((pred_off_sur_sdf, sh4_off), pred_normals_off_sur) = out_off

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

        if loss_cfg.regularize > 0:
            basis = proj_sh4_to_R3(jax.lax.stop_gradient(sh4_off), max_iter=50)

            # V_vis, F_vis = vis_oct_field(basis, samples_off_sur, 0.005)
            # ps.init()
            # ps.register_surface_mesh('cubes', V_vis, F_vis)
            # pc = ps.register_point_cloud('samples_off_sur', samples_off_sur, radius=1e-4)
            # pc.add_vector_quantity('pred_normals_off_sur', jax.lax.stop_gradient(pred_normals_off_sur))
            # ps.show()
            # exit()

            dps = jnp.einsum('bij,bi->bj', basis,
                             vmap(normalize)(pred_normals_off_sur))
            loss_regularize = loss_cfg.regularize * double_well_potential(
                jnp.abs(dps)).sum(-1).mean()

            # rep_vec = vmap(closest_rep_vec)(pred_normals_off_sur, sh4_off)
            # loss_regularize = loss_cfg.regularize * (
            #     1 -
            #     vmap(cosine_similarity)(pred_normals_off_sur, rep_vec)).mean()

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

        pbar.set_postfix({"loss": loss_dict['loss_total']})

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
