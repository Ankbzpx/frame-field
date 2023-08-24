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
from sh_representation import rotvec_to_sh4, rotvec_to_sh4_expm, rotvec_n_to_z, rotvec_to_R9

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

    total_steps = cfg.training.n_epochs * cfg.training.n_steps

    lr_scheduler_standard = optax.warmup_cosine_decay_schedule(
        cfg.training.lr,
        peak_value=5 * cfg.training.lr,
        warmup_steps=500,
        decay_steps=total_steps)

    # During experiment, large lr (i.e. 5e-4 for batch size of 1024) easily blows up Siren. So special care must be taken...
    lr_scheduler_siren = optax.warmup_cosine_decay_schedule(
        0.2 * cfg.training.lr,
        peak_value=cfg.training.lr,
        warmup_steps=500,
        decay_steps=total_steps)

    if len(cfg.mlp_types) == 1:
        model: model_jax.MLP = getattr(model_jax,
                                       cfg.mlp_types[0])(**cfg.mlp_cfgs[0],
                                                         key=model_key)
        optim = optax.adam(learning_rate=lr_scheduler_siren if cfg.
                           mlp_types[0] == 'Siren' else lr_scheduler_standard)
        opt_state = optim.init(eqx.filter([model], eqx.is_array))

    else:
        model: model_jax.MLP = model_jax.MLPComposer(model_key, cfg.mlp_types,
                                                     cfg.mlp_cfgs)

        # Reference: https://github.com/patrick-kidger/equinox/issues/79
        param_spec = jax.tree_map(lambda _: "standard", model)
        # I think `is_siren` treat `SineLayer` as a leaf, but tree_leaves still return other leaves during traversal, hence needs to call `is_siren` again
        is_siren = lambda x: hasattr(x, "omega_0")
        where_siren_W = lambda m: tuple(x.W for x in jax.tree_util.tree_leaves(
            m, is_leaf=is_siren) if is_siren(x))
        where_siren_b = lambda m: tuple(x.b for x in jax.tree_util.tree_leaves(
            m, is_leaf=is_siren) if is_siren(x))
        param_spec = eqx.tree_at(where_siren_W,
                                 param_spec,
                                 replace_fn=lambda _: "siren")
        param_spec = eqx.tree_at(where_siren_b,
                                 param_spec,
                                 replace_fn=lambda _: "siren")

        # Workaround Callable
        optim = optax.multi_transform(
            {
                "standard": optax.adam(lr_scheduler_standard),
                "siren": optax.adam(lr_scheduler_siren),
            }, [param_spec])
        opt_state = optim.init(eqx.filter([model], eqx.is_array))

    sdf_data = np.load(cfg.sdf_path)
    samples_on_sur = sdf_data['samples_on_sur']
    normals_on_sur = sdf_data['normals_on_sur']
    samples_off_sur = sdf_data['samples_off_sur']
    sdf_off_sur = sdf_data['sdf_off_sur']

    # preload data in memory to speedup training
    idx = jax.random.choice(data_key, jnp.arange(len(samples_on_sur)),
                            (cfg.training.n_steps, cfg.training.n_samples))
    data = {
        'samples_on_sur': samples_on_sur[idx],
        'normals_on_sur': normals_on_sur[idx],
        'samples_off_sur': samples_off_sur[idx],
        'sdf_off_sur': sdf_off_sur[idx]
    }

    checkpoints_folder = 'checkpoints'
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model: model_jax.MLP, samples_on_sur: list[Array],
                  normals_on_sur: list[Array], samples_off_sur: list[Array],
                  sdf_off_sur: list[Array], loss_cfg: LossConfig):

        if loss_cfg.smooth > 0:
            val, jac = model.call_jac(samples_on_sur)
            pred_on_sur_sdf = val[:, 0]
            aux = val[:, 1:]
            sh4 = aux[:, :9]
            pred_normals_on_sur = jac[:, 0]

            val_off, jac_off = model.call_jac(samples_off_sur)
            pred_off_sur_sdf = val_off[:, 0]
            aux_off = val_off[:, 1:]
            sh4_off = aux_off[:, :9]
            pred_normals_off_sur = jac_off[:, 0]
        else:
            (pred_on_sur_sdf,
             aux), pred_normals_on_sur = model.call_grad(samples_on_sur)
            sh4 = aux[:, :9]

            (pred_off_sur_sdf,
             aux_off), pred_normals_off_sur = model.call_grad(samples_off_sur)
            sh4_off = aux_off[:, :9]

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
        sh4_n = jnp.einsum('nji,ni->nj', R9_zn, sh4_align)
        loss_twist = loss_cfg.twist * jnp.abs(
            (sh4_n[:, 0]**2 + sh4_n[:, 8]**2) - 5 / 12).mean()
        loss_align = loss_cfg.align * (
            1 - vmap(cosine_similarity, in_axes=[0, None])
            (sh4_n[:, 1:8], jnp.array([0, 0, 0,
                                       np.sqrt(7 / 12), 0, 0, 0]))).mean()

        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
        # loss_sdf = jnp.abs(pred_off_sur_sdf - sdf_off_sur).mean()
        loss_off = loss_cfg.off_sur * jnp.exp(
            -1e2 * jnp.abs(pred_off_sur_sdf)).mean()
        loss_normal = loss_cfg.normal * (1 - vmap(cosine_similarity)(
            pred_normals_on_sur, normals_on_sur)).mean()
        loss_eikonal = loss_cfg.eikonal * vmap(eikonal)(normal_pred).mean()

        loss = loss_mse + loss_off + loss_normal + loss_eikonal + loss_twist + loss_align

        loss_dict = {
            'loss_mse': loss_mse,
            'loss_off': loss_off,
            'loss_normal': loss_normal,
            'loss_eikonal': loss_eikonal,
            'loss_twist': loss_twist,
            'loss_align': loss_align
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

        if loss_cfg.rot > 0:
            # It's rare to reach singularity in practice. Also, expm is 3 times slower...
            sh4_est = vmap(rotvec_to_sh4)(aux[:, 9:])
            # sh4_est = vmap(rotvec_to_sh4_expm)(aux[:, 9:])
            loss_rot = loss_cfg.rot * (1 - vmap(cosine_similarity)(
                sh4_est, jax.lax.stop_gradient(sh4)).mean())
            loss += loss_rot
            loss_dict['loss_rot'] = loss_rot

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
        pbar.set_postfix({"loss": loss_dict['loss_total']})

        if np.isnan(loss_dict['loss_total']):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(total_steps)
            loss_history[key][epoch] = loss_dict[key]

        # TODO: better plot like using tensorboardX
        # Loss plot
        # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/main_lipmlp.py#L44
        if epoch % 500 == 0:
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
