import numpy as np
import jax
from jax import numpy as jnp
import optax
import equinox as eqx
import torch
from torch.utils.data import Dataset, DataLoader
import random
# https://github.com/google/jax/issues/3382
import torch.multiprocessing as multiprocessing

multiprocessing.set_start_method('spawn')

import model_jax
from config import Config
from common import normalize_aabb

import open3d as o3d
import scipy.spatial

import polyscope as ps
from icecream import ic


def config_latent(cfg: Config):
    n_models = len(cfg.sdf_paths)
    # Compute (n_models - 1) dim simplex vertices as latent
    # Reference: https://mathoverflow.net/a/184585
    q, _ = jnp.linalg.qr(jnp.ones((n_models, 1)), mode="complete")
    return q[:, 1:], n_models - 1


# IMPORTANT: this function will update cfg
def config_model(cfg: Config, model_key, latent_dim) -> model_jax.MLP:
    if len(cfg.mlp_types) == 1:
        cfg.mlps[0].in_features += latent_dim
        return getattr(model_jax, cfg.mlp_types[0])(**cfg.mlp_cfgs[0],
                                                    key=model_key)

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
    # Matters for querying the step count
    lr_scheduler = optax.constant_schedule(cfg.training.lr)

    optim = optax.adam(lr_scheduler)
    opt_state = optim.init(eqx.filter([model], eqx.is_array))

    return optim, opt_state


# Estimate aabb scaling
def eval_data_scale(cfg: Config):

    def cal_scale(sdf_path):
        sdf_data = dict(np.load(sdf_path))

        # Assume centered
        return jnp.max(sdf_data['samples_on_sur'], axis=0)

    scale = jnp.stack([cal_scale(sdf_path) for sdf_path in cfg.sdf_paths
                      ]).max(axis=0)
    return scale


# Default matching training sample size
def progressive_sample_off_surf(cfg: Config,
                                data_key,
                                samples_on_sur,
                                sample_bound,
                                close_sample_sigma,
                                close_sample_ratio=0.25,
                                sigma_scaler_factor=[1.0, 1.0, 1.0]):
    sample_size = cfg.training.n_samples
    # Progressive sample
    sigma = sigma_scaler_factor[0] * close_sample_sigma * np.ones(
        cfg.training.n_steps)
    sigma[cfg.training.n_steps //
          3:] = sigma_scaler_factor[1] * close_sample_sigma
    sigma[int(2 * cfg.training.n_steps /
              3):] = sigma_scaler_factor[2] * close_sample_sigma

    close_sample_size = int(close_sample_ratio * sample_size)
    free_sample_size = sample_size - close_sample_size

    close_samples = sigma[:, None, None] * jax.random.normal(
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


def load_sdf(sdf_path):
    if sdf_path.split('.')[-1] == 'ply':
        pc_o3d = o3d.io.read_point_cloud(sdf_path)
        sdf_data = {
            'samples_on_sur': np.asarray(pc_o3d.points),
            'normals_on_sur': np.asarray(pc_o3d.normals)
        }
    else:
        sdf_data = dict(np.load(sdf_path))
    return sdf_data


class SDFDataset(Dataset):

    def __init__(self, cfg: Config, latents):
        super().__init__()

        n_models = len(cfg.sdf_paths)
        assert n_models > 0

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps

        # Working on numpy array
        latents = np.array(latents)

        def sample_sdf_data(sdf_path, latent):
            sdf_data = load_sdf(sdf_path)
            samples_on_sur = normalize_aabb(sdf_data['samples_on_sur'])
            sdf_data['samples_on_sur'] = samples_on_sur

            # Reference: https://github.com/bearprin/Neural-Singular-Hessian/blob/ca7da0ce5d0c680393f1091ac8a6eafbe32248b4/surface_reconstruction/recon_dataset.py#L49
            # Use max distance among 51 closet points to approximate close neighbor
            kd_tree = scipy.spatial.KDTree(samples_on_sur)
            dists, _ = kd_tree.query(samples_on_sur, k=51, workers=-1)
            sigmas = dists[:, -1:]
            sdf_data['sigmas'] = sigmas
            sdf_data['latent'] = latent
            return sdf_data

        self.sdf_data_list = [
            sample_sdf_data(*args) for args in zip(cfg.sdf_paths, latents)
        ]

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):

        def sample_data(samples_on_sur, normals_on_sur, sigmas, latent):
            idx_permute = np.random.permutation(len(samples_on_sur))
            idx = idx_permute[:self.n_samples]

            samples_on_sur = samples_on_sur[idx]
            sigmas = sigmas[idx]

            if len(normals_on_sur) > 0:
                normals_on_sur = normals_on_sur[idx]

            samples_off_sur = np.random.uniform(-1,
                                                1,
                                                size=(len(samples_on_sur), 3))

            samples_close_sur = samples_on_sur + sigmas * np.random.randn(
                len(samples_on_sur), 3)

            latent = np.repeat(latent[None, ...], len(samples_on_sur), axis=0)

            return {
                "samples_on_sur": samples_on_sur.astype(np.float32),
                "normals_on_sur": normals_on_sur.astype(np.float32),
                "samples_off_sur": samples_off_sur.astype(np.float32),
                "samples_close_sur": samples_close_sur.astype(np.float32),
                "latent": latent.astype(np.float32)
            }

        sdf_data_samples_frag = [
            sample_data(**sdf_data) for sdf_data in self.sdf_data_list
        ]

        sdf_data = {}
        for key in sdf_data_samples_frag[0].keys():
            sdf_data[key] = np.hstack(
                [frag[key] for frag in sdf_data_samples_frag])

        return sdf_data


def config_training_data_pytorch(cfg: Config, latents):
    np.random.seed(0)
    dataset = SDFDataset(cfg, latents)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return dataloader


# preload data in memory to speedup training
def config_training_data(cfg: Config, data_key, latents):
    n_models = len(cfg.sdf_paths)
    assert n_models > 0
    sample_size = cfg.training.n_samples // n_models

    def sample_sdf_data(sdf_path, latent):
        sdf_data = load_sdf(sdf_path)
        samples_on_sur = normalize_aabb(sdf_data['samples_on_sur'])
        sdf_data['samples_on_sur'] = samples_on_sur

        # Reference: https://github.com/bearprin/Neural-Singular-Hessian/blob/ca7da0ce5d0c680393f1091ac8a6eafbe32248b4/surface_reconstruction/recon_dataset.py#L49
        # Use max distance among 51 closet points to approximate close neighbor
        kd_tree = scipy.spatial.KDTree(samples_on_sur)
        dists, _ = kd_tree.query(samples_on_sur, k=51, workers=-1)
        sigmas = dists[:, -1:]
        sdf_data['sigmas'] = sigmas

        if cfg.training.n_input_samples != -1:
            # Clamp num of input samples
            input_sample_size = jnp.minimum(len(samples_on_sur),
                                            cfg.training.n_input_samples)
            sdf_data = jax.tree_map(lambda x: x[:input_sample_size], sdf_data)

        # Share the index to preserve correspondence
        idx = jax.random.choice(data_key, jnp.arange(len(samples_on_sur)),
                                (cfg.training.n_steps, sample_size))
        data = jax.tree_map(lambda x: x[idx], sdf_data)

        data['samples_close_sur'] = data[
            'samples_on_sur'] + data['sigmas'] * jax.random.normal(
                data_key, (cfg.training.n_steps, cfg.training.n_samples, 3))
        data['samples_off_sur'] = jax.random.uniform(
            data_key, (cfg.training.n_steps, cfg.training.n_samples, 3),
            minval=-1,
            maxval=1)
        data['latent'] = latent[None, None,
                                ...].repeat(cfg.training.n_steps,
                                            axis=0).repeat(sample_size, axis=1)
        del data['sigmas']
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
        sdf_data = load_sdf(sdf_path)

        # Don't need normals
        del sdf_data['normals_on_sur']

        def random_batch(x):
            total_sample_size = len(x)
            idx = jax.random.choice(data_key, jnp.arange(total_sample_size),
                                    (cfg.training.n_steps, sample_size))
            return x[idx]

        data = jax.tree_map(lambda x: random_batch(x), sdf_data)
        data.update(
            progressive_sample_off_surf(
                cfg,
                data_key,
                data['samples_on_sur'],
                sample_bound,
                close_scale=cfg.training.close_scale,
        # Disable close surface sample because the sampling is performed in parameterization space
        # --we do not know if they will be mapped to close surface samples in original space
                close_sample_ratio=0.0))

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
    normals_on_sur = sur_normal_samples[idx]
    samples_off_sur = jax.random.uniform(
        data_key, (cfg.training.n_steps, cfg.training.n_samples, 3),
        minval=-1.0,
        maxval=1.0)
    latent = latents[None, None, 0].repeat(cfg.training.n_steps,
                                           axis=0).repeat(sample_size, axis=1)

    data = {
        'samples_on_sur': samples_on_sur,
        'normals_on_sur': normals_on_sur,
        'samples_off_sur': samples_off_sur,
        'samples_close_sur': None,
        'latent': latent
    }

    return data
