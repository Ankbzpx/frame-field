import numpy as np
import jax
from jax import numpy as jnp
import optax
import equinox as eqx

import model_jax
from config import Config


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
        if cfg.conditioning:
            # Do NOT modify config inside eqx.Module, because the __init__ will be called twice
            cfg.mlps[1].in_features += cfg.mlps[0].in_features
            MultiMLP = model_jax.MLPComposerCondition
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
    lr_scheduler_standard = optax.warmup_cosine_decay_schedule(
        cfg.training.lr,
        peak_value=5 * cfg.training.lr,
        warmup_steps=500,
        decay_steps=total_steps)

    # Large lr (i.e. 5e-4 for batch size of 1024) could blow up Siren. So special care must be taken...
    lr_scheduler_siren = optax.warmup_cosine_decay_schedule(
        0.2 * cfg.training.lr,
        peak_value=cfg.training.lr,
        warmup_steps=500,
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


# preload data in memory to speedup training
def config_training_data(cfg: Config, data_key, latents):
    n_models = len(cfg.sdf_paths)
    assert n_models > 0
    sample_size = cfg.training.n_samples // n_models

    def sample_sdf_data(sdf_path, latent):
        sdf_data = dict(np.load(sdf_path))
        total_sample_size = len(sdf_data['samples_on_sur'])
        idx = jax.random.choice(data_key, jnp.arange(total_sample_size),
                                (cfg.training.n_steps, sample_size))

        data = jax.tree_map(lambda x: x[idx], sdf_data)
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
