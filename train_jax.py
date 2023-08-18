import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
import optax
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
from practical_3d_frame_field_generation import rotvec_n_to_z, rotvec_to_R9, rotvec_to_sh4

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
    sdf_data = np.load(cfg.sdf_path)
    samples_on_sur = sdf_data['samples_on_sur']
    normals_on_sur = sdf_data['normals_on_sur']
    samples_off_sur = sdf_data['samples_off_sur']
    sdf_off_sur = sdf_data['sdf_off_sur']

    model_key, data_key = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed), 2)

    # preload data in memory to speedup training
    idx = jax.random.choice(data_key, jnp.arange(len(samples_on_sur)),
                            (cfg.training.n_steps, cfg.training.n_samples))
    data = {
        'samples_on_sur': samples_on_sur[idx],
        'normals_on_sur': normals_on_sur[idx],
        'samples_off_sur': samples_off_sur[idx],
        'sdf_off_sur': sdf_off_sur[idx]
    }

    model: model_jax.MLP = getattr(model_jax, cfg.mlp_type)(**cfg.mlp_cfg,
                                                            key=model_key)

    total_steps = cfg.training.n_epochs * cfg.training.n_steps
    # peak_value 1e-4 for Siren or it blows up
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        cfg.training.lr,
        peak_value=cfg.training.lr_peak,
        warmup_steps=1000,
        decay_steps=total_steps)
    optim = optax.adam(learning_rate=lr_scheduler)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    checkpoints_folder = 'checkpoints'
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # Loss plot
    # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
    loss_history = np.zeros(total_steps)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_func(model: model_jax.MLP, samples_on_sur: list[Array],
                  normals_on_sur: list[Array], samples_off_sur: list[Array],
                  sdf_off_sur: list[Array], loss_cfg: LossConfig):

        if loss_cfg.smooth > 0:
            val, jac = model.call_jac(samples_on_sur)
            pred_on_sur_sdf = val[:, 0]
            aux = val[:, 1:]
            sh9 = aux[:, :9]
            pred_normals_on_sur = jac[:, 0]
        else:
            (pred_on_sur_sdf,
             aux), pred_normals_on_sur = model.call_grad(samples_on_sur)
            sh9 = aux[:, :9]

        (pred_off_sur_sdf,
         _), pred_normals_off_sur = model.call_grad(samples_off_sur)

        # Alignment
        normal_align = jnp.where(loss_cfg.match_sdf_normal,
                                 jax.lax.stop_gradient(pred_normals_on_sur),
                                 normals_on_sur)
        R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal_align))
        sh9_n = jnp.einsum('nji,ni->nj', R9_zn, sh9)
        loss_twist = loss_cfg.twist * jnp.abs(
            (sh9_n[:, 0]**2 + sh9_n[:, 8]**2) - 5 / 12).mean()
        loss_align = loss_cfg.align * (
            1 - vmap(cosine_similarity, in_axes=[0, None])
            (sh9_n[:, 1:8], jnp.array([0, 0, 0,
                                       np.sqrt(7 / 12), 0, 0, 0]))).mean()

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        # loss_sdf = jnp.abs(pred_off_sur_sdf - sdf_off_sur).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(
            -1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = loss_cfg.normal * (1 - vmap(cosine_similarity)(
            pred_normals_on_sur, normals_on_sur)).mean()
        loss_eikonal = loss_cfg.eikonal * 0.5 * (
            vmap(eikonal)(pred_normals_on_sur).mean() +
            vmap(eikonal)(pred_normals_off_sur).mean())

        loss = loss_mse + loss_off + loss_normal + loss_eikonal + loss_twist + loss_align

        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * model.get_aux_loss()
            loss += loss_lip

        if loss_cfg.smooth > 0:
            loss_smooth = loss_cfg.smooth * vmap(
                jnp.linalg.norm, in_axes=[0, None])(jac[:, 1:10], 'f').mean()
            loss += loss_smooth

        if loss_cfg.rot > 0:
            sh9_est = vmap(rotvec_to_sh4)(aux[:, 9:])
            loss_rot = loss_cfg.rot * (1 - vmap(cosine_similarity)(
                sh9_est, jax.lax.stop_gradient(sh9)).mean())
            loss += loss_rot

        return loss

    @eqx.filter_jit
    def make_step(model: model_jax.MLP, opt_state: PyTree, batch: PyTree,
                  loss_cfg: LossConfig):
        loss_value, grads = loss_func(model, **batch, loss_cfg=loss_cfg)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    pbar = tqdm(range(total_steps))
    for epoch in pbar:
        batch_id = epoch % cfg.training.n_steps
        batch = jax.tree_map(lambda x: x[batch_id], data)
        model, opt_state, loss_value = make_step(model, opt_state, batch,
                                                 cfg.loss_cfg)
        loss_history[epoch] = loss_value
        pbar.set_postfix({"loss": loss_value})

        if epoch % 1000 == 0:
            plt.close(1)
            plt.figure(1)
            plt.semilogy(loss_history[:epoch])
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
