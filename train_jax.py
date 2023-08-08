import numpy as np

import jax
from jax import vmap, numpy as jnp, jit
import optax
import equinox as eqx
from jaxtyping import PyTree, Array

from model_jax import StandardMLP

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
    pc = np.load('data/sdf/fandisk.npy')
    coords = pc[:, :3]
    normals = pc[:, 3:]

    # [0, 1]
    coords -= np.mean(coords, axis=0, keepdims=True)
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)

    # [-0.95, 0.95]
    coords -= 0.5
    coords *= 1.9

    n_epochs = 200000
    n_samples_per_epoch = 512
    model_key, data_key = jax.random.split(jax.random.PRNGKey(0), 2)

    # preload data in memory for speedup
    idx = jax.random.choice(data_key, jnp.arange(len(coords)),
                            (n_epochs * n_samples_per_epoch,))

    sample_on_sur = coords[idx].reshape(n_epochs, n_samples_per_epoch, -1)
    sample_normals = normals[idx].reshape(n_epochs, n_samples_per_epoch, -1)

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
    def loss_func(model: eqx.Module, coords_on_sur: list[Array],
                  coords_off_sur: list[Array], normals_on_sur: list[Array]):
        (pred_on_sur_sdf,
         sh9), pred_normals_on_sur = model.call_grad(coords_on_sur)
        (pred_off_sur_sdf,
         _), pred_normals_off_sur = model.call_grad(coords_off_sur)

        loss_mse = jnp.abs(pred_on_sur_sdf).mean()
        loss_off = jnp.exp(-1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = (1 - jnp.abs(
            vmap(cosine_similarity)(pred_normals_on_sur,
                                    normals_on_sur))).mean()
        loss_eikonal = 0.5 * (vmap(eikonal)(pred_normals_on_sur).mean() +
                              vmap(eikonal)(pred_normals_off_sur).mean())

        loss = 3e3 * loss_mse + 1e2 * loss_off + 1e2 * loss_normal + 5e1 * loss_eikonal
        return loss

    @eqx.filter_jit
    def make_step(model: eqx.Module, opt_state: PyTree,
                  coords_on_sur: list[Array], coords_off_sur: list[Array],
                  normals_on_sur: list[Array]):
        loss_value, grads = loss_func(model, coords_on_sur, coords_off_sur,
                                      normals_on_sur)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:

        batch_id = epoch % n_epochs

        _, subkey = jax.random.split(data_key)
        sample_off_sur_batch = jax.random.uniform(subkey,
                                                  (n_samples_per_epoch, 3))

        model, opt_state, loss_value = make_step(model, opt_state,
                                                 sample_on_sur[batch_id],
                                                 sample_off_sur_batch,
                                                 sample_normals[batch_id])
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
