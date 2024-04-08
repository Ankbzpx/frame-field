import equinox as eqx
import numpy as np
import jax
from jax import numpy as jnp, vmap, jit
import model_jax
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_training_data_param, config_optim
from jaxtyping import PyTree, Array
from common import normalize
from sh_representation import (rotvec_to_sh4_expm, rot6d_to_sh4_zonal,
                               R3_to_sh4_zonal, rot6d_to_R3, rotvec_to_R3,
                               proj_sh4_to_R3)
from loss import cosine_similarity, double_well_potential

from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import optax

import os
import json

import polyscope as ps
from icecream import ic

ParamMLP: model_jax.MLP = model_jax.Siren


# "Unsupervised Deep Learning for Structured Shape Matching" by Roufosse et al.
@jit
def orthogonality(J):
    return jnp.linalg.norm(J.T @ J - jnp.eye(3))


# "Self-supervised Learning of Implicit Shape Representation with Dense Correspondence for Deformable Objects" by Zhang et al.
@jit
def neg_det_penalty(J):
    return jax.nn.relu(-jnp.linalg.det(J))


@jit
def fit_jac_rot6d(J):
    # J is row-wise
    a0 = J[0]
    a1 = J[1]
    a2 = J[2]

    b0 = normalize(a0)
    b1 = normalize(a1 - jnp.dot(b0, a1) * b0)
    b2 = jnp.cross(b0, b1)

    J_fit = jnp.array([b0, b1, b2]).T

    loss_fit = (1 - cosine_similarity(a1, jax.lax.stop_gradient(b1))) + (
        1 - cosine_similarity(a2, jax.lax.stop_gradient(b2))) + orthogonality(J)

    return J_fit, loss_fit


def train(cfg: Config, model: model_jax.MLP, model_octa: model_jax.MLP, data,
          checkpoints_folder, inverse: bool):
    optim, opt_state = config_optim(cfg, model)

    total_steps = cfg.training.n_epochs * cfg.training.n_steps

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # TODO: Put these to config files
    # Prioritize positive determinant over conformal
    align_weight_init = 1e2
    # align_schedule = optax.polynomial_schedule(align_weight_init,
    #                                            1e1 * align_weight_init, 0.5,
    #                                            total_steps,
    #                                            cfg.training.warmup_steps)
    align_schedule = optax.constant_schedule(align_weight_init)

    orth_weight_init = 5e1
    # orth_schedule = optax.polynomial_schedule(1e1 * orth_weight_init,
    #                                           orth_weight_init, 0.5,
    #                                           total_steps,
    #                                           cfg.training.warmup_steps)
    orth_schedule = optax.constant_schedule(orth_weight_init)

    normal_weight_init = 1e2
    normal_schedule = optax.polynomial_schedule(1e-2 * normal_weight_init,
                                                normal_weight_init, 0.5,
                                                total_steps,
                                                cfg.training.warmup_steps)
    # normal_schedule = optax.constant_schedule(normal_weight_init)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model: model_jax.MLP, model_octa: model_jax.MLP,
                  samples_on_sur: Array, close_samples_mask: Array,
                  samples_off_sur: Array, latent: Array, loss_cfg: LossConfig,
                  step_count: int):

        align_weight = align_schedule(step_count)
        orth_weight = orth_schedule(step_count)
        normal_weight = normal_schedule(step_count)

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

        # We no longer care which is on, which is off
        samples = jnp.vstack([samples_on_sur, samples_off_sur])
        latent = jnp.vstack([latent, latent])

        # Forward f: cart -> param
        #   \int_V \| \nabla f(cart) - X(cart) \|^2
        #
        # Inverse f: param -> cart
        #   \int_V \| \nabla f(param) - inv(X(f(param))) \|^2
        # (sh4) => \int_V \| (\nabla f(param)).T - X(f(param)) \|^2
        # (rot6d) => \int_V \| \nabla f(param) - (X(f(param))).T \|^2

        # Fix (0, 0, 0)
        boundary_pt = jnp.zeros(3)
        loss_boundary = jnp.abs(
            model.single_call(boundary_pt, latent[0]) - boundary_pt).mean()

        @jit
        def param_infer(sample, latent):
            # Octahedral supervision
            J, samples_param = model.single_call_jac(sample, latent)
            sample_cart = samples_param if inverse else sample
            sdf, aux = model_octa.single_call_split(sample_cart, latent)
            return sdf, (aux, J)

        (_, (aux, J)), normal_param = vmap(
            eqx.filter_value_and_grad(param_infer, has_aux=True))(samples,
                                                                  latent)
        aux = jax.lax.stop_gradient(aux)

        if loss_cfg.rot6d:
            basis = proj_func(aux)
            # J is rowwise
            basis = basis if inverse else jnp.transpose(basis, (0, 2, 1))

            # I don't think we need to care about octahedral matching for now
            loss_align = align_weight * vmap(jnp.linalg.norm)(J - basis).mean()
            loss_orth = orth_weight * vmap(orthogonality)(J).mean()
        else:
            # Use fitted J because the original one is not orthogonal such that
            # - Its inverse is not its transpose
            # - Minimize sh4 difference norm loses it meaning
            J, loss_fit = vmap(fit_jac_rot6d)(J)
            J = jnp.transpose(J, (0, 2, 1)) if inverse else J

            sh4 = vmap(param_func)(aux)

            sh4_align = vmap(R3_to_sh4_zonal)(J)
            loss_align = align_weight * vmap(
                jnp.linalg.norm)(sh4 - sh4_align).mean()

            loss_orth = orth_weight * loss_fit.mean()

        loss = loss_boundary + loss_orth + loss_align

        loss_dict = {
            'loss_boundary': loss_boundary,
            'loss_orth': loss_orth,
            'loss_align': loss_align
        }

        if inverse:
            # Normal in parameterization space should match canonical basis
            dps = jnp.einsum('ij,bi->bj', jnp.eye(3),
                             vmap(normalize)(normal_param))
            loss_normal = normal_weight * double_well_potential(
                jnp.abs(dps)).sum(-1).mean()

            loss += loss_normal
            loss_dict['loss_normal'] = loss_normal

        loss_dict['loss_total'] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(model: model_jax.MLP, model_octa: model_jax.MLP,
                  opt_state: PyTree, batch: PyTree, loss_cfg: LossConfig):
        step_count = opt_state.inner_states['standard'].inner_state[-1].count
        grads, loss_dict = loss_func(model,
                                     model_octa,
                                     **batch,
                                     loss_cfg=loss_cfg,
                                     step_count=step_count)
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    loss_history = {}
    pbar = tqdm(range(total_steps))
    for epoch in pbar:
        batch_id = epoch % cfg.training.n_steps
        batch = jax.tree_map(lambda x: x[batch_id], data)
        model, opt_state, loss_dict = make_step(model, model_octa, opt_state,
                                                batch, cfg.loss_cfg)

        if np.isnan(loss_dict['loss_total']):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(total_steps)
            loss_history[key][epoch] = loss_dict[key]

        pbar.set_postfix(loss_dict)

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

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--inverse',
                        action='store_true',
                        help='Inverse parameterization')
    args = parser.parse_args()

    name = args.config.split('/')[-1].split('.')[0]
    suffix = "param_inv" if args.inverse else "param"

    cfg = Config(**json.load(open(args.config)))
    cfg.name = f'{name}_{suffix}'

    model_key, data_key = jax.random.split(
        jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model_octa = config_model(cfg, jax.random.PRNGKey(0), latent_dim)
    model_octa: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{name}.eqx", model_octa)

    mlp_cfg = cfg.mlp_cfgs[0]
    mlp_type = cfg.mlp_types[0]

    # 1 general vector field
    mlp_cfg['out_features'] = 3
    model = ParamMLP(**mlp_cfg, key=model_key)

    # 3 scalar fields
    # model = ParamMLP(mlp_types=[mlp_type] * 3,
    #                  mlp_cfgs=[mlp_cfg] * 3,
    #                  key=model_key)

    # Debug
    # cfg.training.n_steps = 101

    data = config_training_data_param(cfg, data_key, latents)

    train(cfg, model, model_octa, data, 'checkpoints', args.inverse)
