import numpy as np

import jax
from jax import vmap, numpy as jnp, jit
import optax
import equinox as eqx
from jaxtyping import PyTree, Array

from model_jax import StandardMLP
from practical_3d_frame_field_generation import rotvec_to_z, rotvec_to_R9

import polyscope as ps
from icecream import ic

from tqdm import tqdm
import matplotlib.pyplot as plt
import os


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)

    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


@jit
def eikonal(x):
    return jnp.abs(jnp.linalg.norm(x) - 1)


if __name__ == '__main__':

    sdf_data = np.load('data/sdf/fandisk.npz')
    samples_on_sur = sdf_data['samples_on_sur']
    normals_on_sur = sdf_data['normals_on_sur']
    samples_off_sur = sdf_data['samples_off_sur']
    sdf_off_sur = sdf_data['sdf_off_sur']

    # ps.init()
    # vis_on_sur = ps.register_point_cloud('samples_on_sur', samples_on_sur, point_render_mode='quad')
    # vis_on_sur.add_vector_quantity('normals_on_sur', normals_on_sur, enabled=True)
    # vis_off_sur = ps.register_point_cloud('samples_off_sur', samples_off_sur, point_render_mode='quad')
    # vis_off_sur.add_scalar_quantity('sdf_off_sur', sdf_off_sur, enabled=True)
    # ps.show()
    # exit()

    n_epochs = 100000
    n_samples_per_epoch = 1024
    model_key, data_key = jax.random.split(jax.random.PRNGKey(0), 2)

    # preload data in memory for speedup
    idx = jax.random.choice(data_key, jnp.arange(len(samples_on_sur)),
                            (n_epochs * n_samples_per_epoch,))

    samples_on_sur = samples_on_sur[idx].reshape(n_epochs, n_samples_per_epoch,
                                                 -1)
    normals_on_sur = normals_on_sur[idx].reshape(n_epochs, n_samples_per_epoch,
                                                 -1)
    samples_off_sur = samples_off_sur[idx].reshape(n_epochs,
                                                   n_samples_per_epoch, -1)
    sdf_off_sur = sdf_off_sur[idx].reshape(n_epochs, n_samples_per_epoch, -1)

    mlp_cfg = {
        'in_features': 3,
        'hidden_features': 256,
        'hidden_layers': 8,
        'out_features': 10,
    }
    model = StandardMLP(**mlp_cfg, key=model_key, activation='elu')
    lr = 1e-4
    lr_scheduler = optax.warmup_cosine_decay_schedule(lr,
                                                      peak_value=5 * lr,
                                                      warmup_steps=1000,
                                                      decay_steps=n_epochs)
    optim = optax.adam(learning_rate=lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    checkpoints_folder = 'checkpoints'
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # Loss plot
    # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
    loss_history = np.zeros(n_epochs)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_func(model: eqx.Module, samples_on_sur: list[Array],
                  normals_on_sur: list[Array], samples_off_sur: list[Array],
                  sdf_off_sur: list[Array]):
        (pred_on_sur_sdf,
         sh9), pred_normals_on_sur = model.call_grad(samples_on_sur)
        (pred_off_sur_sdf,
         _), pred_normals_off_sur = model.call_grad(samples_off_sur)

        # Alignment
        R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_to_z)(normals_on_sur))
        sh9_n = jnp.einsum('nji,ni->nj', R9_zn, sh9)
        loss_twist = jnp.abs((sh9_n[:, 0]**2 + sh9_n[:, 8]**2) - 5 / 12).mean()
        loss_align = jnp.abs(
            sh9_n[:, 1:8] -
            jnp.array([0, 0, 0, np.sqrt(7 / 12), 0, 0, 0])[None, :]).mean()

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = jnp.abs(pred_on_sur_sdf).mean()
        # loss_sdf = jnp.abs(pred_off_sur_sdf - sdf_off_sur).mean()
        loss_off = jnp.exp(-1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = (1 - jnp.abs(
            vmap(cosine_similarity)(pred_normals_on_sur,
                                    normals_on_sur))).mean()
        loss_eikonal = 0.5 * (vmap(eikonal)(pred_normals_on_sur).mean() +
                              vmap(eikonal)(pred_normals_off_sur).mean())

        loss = 3e3 * loss_mse + 1e2 * loss_off + 1e2 * loss_normal + 5e1 * loss_eikonal + 1e2 * (
            loss_twist + loss_align)
        return loss

    @eqx.filter_jit
    def make_step(model: eqx.Module, opt_state: PyTree,
                  samples_on_sur: list[Array], normals_on_sur: list[Array],
                  samples_off_sur: list[Array], sdf_off_sur: list[Array]):
        loss_value, grads = loss_func(model, samples_on_sur, normals_on_sur,
                                      samples_off_sur, sdf_off_sur)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        batch_id = epoch % n_epochs
        model, opt_state, loss_value = make_step(model, opt_state,
                                                 samples_on_sur[batch_id],
                                                 normals_on_sur[batch_id],
                                                 samples_off_sur[batch_id],
                                                 sdf_off_sur[batch_id])
        loss_history[epoch] = loss_value
        pbar.set_postfix({"loss": loss_value})

        if epoch % 1000 == 0:
            plt.close(1)
            plt.figure(1)
            plt.semilogy(loss_history[:epoch])
            plt.title('Reconstruction loss')
            plt.grid()
            plt.savefig(f"{checkpoints_folder}/fandisk_loss_history.jpg")

    eqx.tree_serialise_leaves(f"{checkpoints_folder}/fandisk.eqx", model)
