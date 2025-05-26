import argparse
import copy
import json
import os

from common import normalize
from config import Config, LossConfig
from config_utils import config_latent, config_model, config_optim, config_training_data
from eval_jax import eval
from loss import (
    align_basis_explicit,
    align_sh4_explicit,
    align_sh4_explicit_cosine,
    align_sh4_functional_grad,
    eikonal,
    eval_ma,
)
import model_jax
from sh_representation import (
    proj_sh4_to_R3,
    rot6d_to_R3,
    rot6d_to_sh4_zonal,
    rotvec_to_R3,
    rotvec_to_sh4_expm,
)

import equinox as eqx
import jax
from jax import numpy as jnp, vmap
from jaxtyping import Array, PyTree
import matplotlib
import numpy as np
import optax
from tensorboardX import SummaryWriter
from tqdm import tqdm


matplotlib.use("Agg")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def eval_iter(cfg: Config, model, latent, tag, udf):
    cfg = copy.copy(cfg)
    cfg.name = f"{cfg.name}_{tag}"
    cfg.out_dir = os.path.join(cfg.out_dir, "debug_iters")
    eval(cfg, model, latent, grid_res=256, udf=udf)


def train(cfg: Config, model: model_jax.MLP, data, udf):
    writer = SummaryWriter(logdir=os.path.join("checkpoints/runs"))
    optim, opt_state = config_optim(cfg, model)

    tau_schedule = optax.linear_schedule(
        0,
        1,
        int(cfg.loss_cfg.octa_duration * cfg.training.n_steps),
        int(cfg.loss_cfg.octa_begin * cfg.training.n_steps),
    )
    # Use separate schedulers here as they have different annealed weights
    hessian_schedule = optax.linear_schedule(
        cfg.loss_cfg.hessian,
        cfg.loss_cfg.hessian_annealing,
        int(cfg.loss_cfg.init_duration * cfg.training.n_steps),
    )
    digs_schedule = optax.linear_schedule(
        cfg.loss_cfg.digs,
        cfg.loss_cfg.digs_annealing,
        int(cfg.loss_cfg.init_duration * cfg.training.n_steps),
    )

    if not os.path.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(
        model: model_jax.MLP,
        samples_on_sur: Array,
        normals_on_sur: Array,
        samples_off_sur: Array,
        samples_close_sur: Array,
        latent: Array,
        loss_cfg: LossConfig,
        step_count: int,
    ):
        tau = tau_schedule(step_count)
        align_weight = tau * loss_cfg.align
        smooth_weight = tau * loss_cfg.smooth
        lip_weight = tau * loss_cfg.lip

        hessian_weight = hessian_schedule(step_count)
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

        if udf:
            samples_all = jnp.vstack([samples_on_sur, samples_close_sur])
            latents_all = jnp.vstack([latent, latent])
            hessians_all = model.mlps[0].call_hessian(samples_all, latents_all)
            sh4_jac, aux_align = model.mlps[1].call_jac(samples_all, latents_all)

            (udf_on, _), pred_normals_on_sur = model.mlps[0].call_grad(
                samples_on_sur, latent
            )
            (udf_close, _), pred_normals_close_sur = model.mlps[0].call_grad(
                samples_on_sur, latent
            )
            df_align = jnp.concatenate([udf_on, udf_close])
            normal_align = jnp.vstack([pred_normals_on_sur, pred_normals_close_sur])

            loss_di = loss_cfg.w_di * jnp.abs(udf_on).mean()
            loss_neu = (
                loss_cfg.w_neu * vmap(jnp.linalg.norm)(pred_normals_on_sur).mean()
            )
            loss_ma = loss_cfg.w_ma * vmap(eval_ma)(hessians_all).mean().mean()
            loss_off = loss_cfg.w_off * jnp.exp(-5e2 * jnp.abs(udf_close)).mean()

            loss = loss_di + loss_neu + loss_ma + loss_off
            loss_dict = {
                "loss_di": loss_di,
                "loss_neu": loss_neu,
                "loss_ma": loss_ma,
                "loss_off": loss_off,
            }
        else:
            # The python if is determined at tracing time. jax.lax.cond helps reduce computation when weight is scheduled to be 0
            if loss_cfg.smooth > 0:
                sh4_jac, aux_align = model.mlps[1].call_jac(samples_on_sur, latent)
            else:
                aux_align = model.mlps[1](samples_on_sur, latent)

            (pred_on_sur_sdf, _), pred_normals_on_sur = model.mlps[0].call_grad(
                samples_on_sur, latent
            )

            df_align = pred_on_sur_sdf
            normal_align = pred_normals_on_sur

            (pred_off_sur_sdf, _), pred_normals_off_sur = model.mlps[0].call_grad(
                samples_off_sur, latent
            )

            if loss_cfg.hessian > 0:
                hessian_close = model.mlps[0].call_hessian(samples_close_sur, latent)

            if loss_cfg.digs > 0:
                hessian_off = model.mlps[0].call_hessian(samples_off_sur, latent)

            # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
            loss_mse = loss_cfg.on_sur * jnp.abs(pred_on_sur_sdf).mean()
            loss_off = (
                loss_cfg.off_sur
                * jnp.exp(-loss_cfg.beta * jnp.abs(pred_off_sur_sdf)).mean()
            )
            if loss_cfg.off_eikonal:
                loss_eikonal = (
                    loss_cfg.eikonal
                    * vmap(eikonal)(
                        jnp.vstack([pred_normals_on_sur, pred_normals_off_sur])
                    ).mean()
                )
            else:
                loss_eikonal = (
                    loss_cfg.eikonal * vmap(eikonal)(pred_normals_on_sur).mean()
                )

            loss = loss_mse + loss_off + loss_eikonal
            loss_dict = {
                "loss_mse": loss_mse,
                "loss_off": loss_off,
                "loss_eikonal": loss_eikonal,
            }

        if udf:
            sample_weight = jax.lax.stop_gradient(
                jnp.exp(-loss_cfg.beta * jnp.sqrt(jnp.abs(df_align / 1000)))
            )
        else:
            sample_weight = jax.lax.stop_gradient(
                jnp.exp(-loss_cfg.beta * jnp.abs(df_align))
            )

        if loss_cfg.align > 0:

            def eval_align_loss(normal, aux):
                if loss_cfg.explicit_basis or loss_cfg.rot6d:
                    basis_align = proj_func(aux)
                    align_measure = align_basis_explicit(basis_align, normal)
                else:
                    sh4_align = vmap(param_func)(aux)
                    align_measure = align_sh4_functional_grad(sh4_align, normal)

                return (sample_weight * align_measure).mean()

            loss_align = align_weight * jax.lax.cond(
                align_weight > 0,
                eval_align_loss,
                lambda x, y: 0.0,
                *(normal_align, aux_align),
            )
            loss += loss_align
            loss_dict["loss_align"] = loss_align

        if loss_cfg.smooth > 0:

            def eval_smooth_loss(jac):
                smooth_measure = vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f")
                return (sample_weight * smooth_measure).mean()

            loss_smooth = smooth_weight * jax.lax.cond(
                smooth_weight > 0, eval_smooth_loss, lambda x: 0.0, sh4_jac
            )
            loss += loss_smooth
            loss_dict["loss_smooth"] = loss_smooth

        if loss_cfg.lip > 0:
            loss_lip = lip_weight * model.get_aux_loss()
            loss += loss_lip
            loss_dict["loss_lip"] = loss_lip

        if loss_cfg.hessian > 0:

            def eval_hessian_loss(hessian):
                return 0.5 * jnp.abs(vmap(jnp.linalg.det)(hessian)).mean()

            loss_hessian = hessian_weight * jax.lax.cond(
                hessian_weight > 0, eval_hessian_loss, lambda x: 0.0, hessian_close
            )
            loss += loss_hessian
            loss_dict["loss_hessian"] = loss_hessian

        if loss_cfg.digs > 0:

            def eval_digs_loss(hessian):
                return jnp.clip(jnp.abs(vmap(jnp.trace)(hessian)), 0.1, 50).mean()

            loss_digs = digs_weight * jax.lax.cond(
                digs_weight > 0, eval_digs_loss, lambda x: 0.0, hessian_off
            )
            loss += loss_digs
            loss_dict["loss_digs"] = loss_digs

        loss_dict["loss_total"] = loss

        return loss, loss_dict

    @eqx.filter_jit
    def make_step(
        model: model_jax.MLP, opt_state: PyTree, batch: PyTree, loss_cfg: LossConfig
    ):
        # FIXME: The static index is risky--it depends on the order of optax.chain
        step_count = opt_state[0].count
        grads, loss_dict = loss_func(
            model, **batch, loss_cfg=loss_cfg, step_count=step_count
        )
        updates, opt_state = optim.update([grads], opt_state, [model])
        model = eqx.apply_updates([model], updates)[0]
        return model, opt_state, loss_dict

    loss_history = {}
    pbar = tqdm(range(cfg.training.n_steps), dynamic_ncols=True)

    data_iter = iter(data)
    for iteration in pbar:
        batch = next(data_iter)
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)
        model, opt_state, loss_dict = make_step(model, opt_state, batch, cfg.loss_cfg)

        if np.isnan(loss_dict["loss_total"]):
            print("NaN occurred!")
            print(loss_dict)
            exit()

        for key in loss_dict.keys():
            # preallocate
            if key not in loss_history:
                loss_history[key] = np.zeros(cfg.training.n_steps)
            loss_history[key][iteration] = loss_dict[key]

        writer.add_scalars(f"{cfg.name}", loss_dict, iteration)
        pbar.set_postfix({"loss_total": loss_dict["loss_total"]})

        if iteration == int(cfg.loss_cfg.octa_begin * cfg.training.n_steps):
            eval_latent = jnp.empty((0,))
            eval_iter(cfg, model, eval_latent, "init", udf=udf)
        # elif iteration % cfg.training.eval_every == 0 and iteration != 0:
        #     eval_latent = jnp.empty((0,))
        #     eval_iter(cfg, model, eval_latent, iteration)

    eqx.tree_serialise_leaves(
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split("/")[-1].split(".")[0]

    model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)

    latents, latent_dim = config_latent(cfg)
    model = config_model(cfg, model_key, latent_dim)

    data = config_training_data(cfg, latents, udf=cfg.udf)

    train(cfg, model, data, udf=cfg.udf)
