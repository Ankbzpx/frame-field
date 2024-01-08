import numpy as np
import jax
from jax import numpy as jnp
import optax
import equinox as eqx

import model_jax
from config import Config

import polyscope as ps
from icecream import ic


def config_latent(cfg: Config):
    n_models = len(cfg.sdf_paths)
    # Compute (n_models - 1) dim simplex vertices as latent
    # Reference: https://mathoverflow.net/a/184585
    q, _ = jnp.linalg.qr(jnp.ones((n_models, 1)), mode="complete")
    return q[:, 1:], n_models - 1


# Note: this function will update cfg
def config_model(cfg: Config, model_key, latent_dim) -> model_jax.MLP:
    if len(cfg.mlp_types) == 1:
        cfg.mlps[0].in_features += latent_dim
        return getattr(model_jax, cfg.mlp_types[0])(**cfg.mlp_cfgs[0],
                                                    key=model_key)

    else:
        if cfg.loss_cfg.tangent:
            MultiMLP = model_jax.MLPComposerCurl
        else:
            MultiMLP = model_jax.MLPComposer

        for mlp_cfg in cfg.mlps:
            mlp_cfg.in_features += latent_dim

        return MultiMLP(
            model_key,
            cfg.mlp_types,
            cfg.mlp_cfgs,
        )


def config_optim(cfg: Config, model: model_jax.MLP):
    total_steps = cfg.training.n_epochs * cfg.training.n_steps
    # lr_scheduler_standard = optax.constant_schedule(cfg.training.lr_multiplier *
    #                                                 cfg.training.lr)
    lr_scheduler_standard = optax.warmup_cosine_decay_schedule(
        cfg.training.lr_multiplier * cfg.training.lr,
        peak_value=cfg.training.lr_peak_multiplier * cfg.training.lr,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=total_steps)

    # Large lr (i.e. 5e-4 for batch size of 1024) could blow up Siren. So special care must be taken...
    # lr_scheduler_siren = optax.constant_schedule(
    #     cfg.training.lr_multiplier_siren * cfg.training.lr)
    lr_scheduler_siren = optax.warmup_cosine_decay_schedule(
        cfg.training.lr_multiplier_siren * cfg.training.lr,
        peak_value=cfg.training.lr_peak_multiplier_siren * cfg.training.lr,
        warmup_steps=cfg.training.warmup_steps,
        decay_steps=total_steps)

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

    return optim, opt_state


def eval_data_scale(cfg: Config):

    def cal_scale(sdf_path):
        sdf_data = dict(np.load(sdf_path))

        # Assume centered
        return jnp.max(sdf_data['samples_on_sur'], axis=0)

    scale = jnp.stack([cal_scale(sdf_path) for sdf_path in cfg.sdf_paths
                      ]).max(axis=0)
    return scale


def progressive_sample_off_surf(cfg: Config,
                                data_key,
                                samples_on_sur,
                                sample_bound,
                                close_scale,
                                scaler_factor=[1.0, 1.0, 1.0],
                                ratio=0.25):
    sample_size = cfg.training.n_samples
    # Progressive sample
    scale = scaler_factor[0] * close_scale * np.ones(cfg.training.n_steps)
    scale[cfg.training.n_steps // 3:] = scaler_factor[1] * close_scale
    scale[int(2 * cfg.training.n_steps / 3):] = scaler_factor[2] * close_scale

    close_sample_size = int(ratio * sample_size)
    free_sample_size = sample_size - close_sample_size

    close_samples = scale[:, None, None] * jax.random.normal(
        data_key, (cfg.training.n_steps, close_sample_size,
                   3)) + samples_on_sur[:, :close_sample_size]
    close_samples = jnp.clip(close_samples, -0.9999, 0.9999)

    free_samples = jax.random.uniform(
        data_key, (cfg.training.n_steps, free_sample_size, 3),
        minval=-sample_bound,
        maxval=sample_bound)

    samples_off_sur = jnp.concatenate([close_samples, free_samples], axis=1)

    close_samples_mask = jnp.concatenate(
        [jnp.ones(close_sample_size),
         jnp.zeros(free_sample_size)])
    close_samples_mask = close_samples_mask[None, :].repeat(
        cfg.training.n_steps, axis=0)

    return {
        'samples_off_sur': samples_off_sur,
        'sdf_off_sur': jnp.zeros((cfg.training.n_steps, sample_size)),
        'close_samples_mask': close_samples_mask
    }


# preload data in memory to speedup training
def config_training_data(cfg: Config, data_key, latents):
    n_models = len(cfg.sdf_paths)
    assert n_models > 0
    sample_size = cfg.training.n_samples // n_models

    # Here we uniform sample region slight larger than and of same aspect ratio of aabb
    sample_bound = 1.25 * eval_data_scale(cfg)

    def sample_sdf_data(sdf_path, latent):
        sdf_data = dict(np.load(sdf_path))

        def random_batch(x):
            # Clamp num of input samples for ablation
            input_sample_size = jnp.minimum(len(x),
                                            cfg.training.n_input_samples)
            idx = jax.random.choice(data_key, jnp.arange(input_sample_size),
                                    (cfg.training.n_steps, sample_size))
            return x[idx]

        data = jax.tree_map(lambda x: random_batch(x), sdf_data)

        data.update(
            progressive_sample_off_surf(cfg,
                                        data_key,
                                        data['samples_on_sur'],
                                        sample_bound,
                                        close_scale=cfg.training.close_scale))
        data['latent'] = latent[None, None,
                                ...].repeat(cfg.training.n_steps,
                                            axis=0).repeat(sample_size, axis=1)
        return data

    data_frags = [
        sample_sdf_data(*args) for args in zip(cfg.sdf_paths, latents)
    ]
    data = {}
    for key in data_frags[0].keys():
        data[key] = jnp.hstack([frag[key] for frag in data_frags])

    return data


# preload data in memory to speedup training
def config_training_data_param(cfg: Config, data_key, latents):
    n_models = len(cfg.sdf_paths)
    assert n_models > 0
    sample_size = cfg.training.n_samples // n_models

    # Here we uniform sample region slight larger than and of same aspect ratio of aabb
    sample_bound = 1.25 * eval_data_scale(cfg)

    def sample_sdf_data(sdf_path, latent):
        sdf_data = dict(np.load(sdf_path))

        # Don't need normals
        del sdf_data['normals_on_sur']

        def random_batch(x):
            total_sample_size = len(x)
            idx = jax.random.choice(data_key, jnp.arange(total_sample_size),
                                    (cfg.training.n_steps, sample_size))
            return x[idx]

        data = jax.tree_map(lambda x: random_batch(x), sdf_data)
        data.update(
            progressive_sample_off_surf(cfg,
                                        data_key,
                                        data['samples_on_sur'],
                                        sample_bound,
                                        close_scale=cfg.training.close_scale,
                                        ratio=0.0))

        del data['sdf_off_sur']

        data['latent'] = latent[None, None,
                                ...].repeat(cfg.training.n_steps,
                                            axis=0).repeat(sample_size, axis=1)
        return data

    data_frags = [
        sample_sdf_data(*args) for args in zip(cfg.sdf_paths, latents)
    ]
    data = {}
    for key in data_frags[0].keys():
        data[key] = jnp.hstack([frag[key] for frag in data_frags])

    return data


def config_toy_training_data(cfg: Config, data_key, sur_samples,
                             sur_normal_samples, latents, gap):
    sample_size = cfg.training.n_samples
    idx = jax.random.choice(data_key, jnp.arange(len(sur_samples)),
                            (cfg.training.n_steps, sample_size))
    samples_on_sur = sur_samples[idx]

    data = {
        'samples_on_sur': samples_on_sur,
        'normals_on_sur': sur_normal_samples[idx]
    }
    data.update(
        progressive_sample_off_surf(cfg,
                                    data_key,
                                    samples_on_sur,
                                    1.0,
                                    close_scale=gap * 1e-2))
    data['latent'] = latents[None, None, 0].repeat(cfg.training.n_steps,
                                                   axis=0).repeat(sample_size,
                                                                  axis=1)

    return data
