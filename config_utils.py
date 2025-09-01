import os
import random

from common import normalize_aabb
from config import Config
import model_jax

import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np
import open3d as o3d
import optax
import scipy.spatial
import torch
from torch.utils.data import DataLoader, Dataset

from icecream import ic
import polyscope as ps


# IMPORTANT: this function will update cfg
def config_model(cfg: Config, model_key) -> model_jax.MLP:
    if len(cfg.mlp_types) == 1:
        return getattr(model_jax, cfg.mlp_types[0])(**cfg.mlp_cfgs[0], key=model_key)

    else:
        MultiMLP = model_jax.MLPComposer

        return MultiMLP(
            model_key,
            cfg.mlp_types,
            cfg.mlp_cfgs,
        )


def config_optim(cfg: Config, model: model_jax.MLP):
    if cfg.training.schedule_free:
        optim = optax.contrib.schedule_free_adamw(cfg.training.lr)
    else:
        optim = optax.adam(cfg.training.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    return optim, opt_state


def load_sdf(sdf_path):
    if sdf_path.split(".")[-1] == "ply":
        pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(sdf_path))
        sdf_data = {
            "samples_on_sur": np.asarray(pc_o3d.points),
            "normals_on_sur": np.asarray(pc_o3d.normals),
        }
    else:
        sdf_data = dict(np.load(sdf_path))
    return sdf_data


class DFDataset(Dataset):
    def __init__(self, cfg: Config, udf):
        super().__init__()

        n_models = len(cfg.sdf_paths)
        assert n_models > 0

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps
        self.udf = udf

        def sample_sdf_data(sdf_path):
            sdf_data = load_sdf(sdf_path)
            samples_on_sur = normalize_aabb(sdf_data["samples_on_sur"])
            sdf_data["samples_on_sur"] = samples_on_sur

            # Reference: https://github.com/bearprin/Neural-Singular-Hessian/blob/ca7da0ce5d0c680393f1091ac8a6eafbe32248b4/surface_reconstruction/recon_dataset.py#L49
            # Use max distance among 51 closet points to approximate close neighbor
            kd_tree = scipy.spatial.KDTree(samples_on_sur)
            dists, _ = kd_tree.query(samples_on_sur, k=51, workers=-1)
            sigmas = dists[:, -1:]
            sdf_data["sigmas"] = sigmas
            return sdf_data

        self.sdf_data_list = [sample_sdf_data(sdf_path) for sdf_path in cfg.sdf_paths]

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        # VERY IMPORTANT: By default pytorch does not reset numpy seed for each __getitem__ call
        #   It means even if I fix the batch index in training loop, the results will still be different
        def sample_data(samples_on_sur, normals_on_sur, sigmas):
            idx_permute = np.random.permutation(len(samples_on_sur))
            idx = idx_permute[: self.n_samples]
            samples_on_sur = samples_on_sur[idx]

            if len(normals_on_sur) > 0:
                normals_on_sur = normals_on_sur[idx]

            samples_off_sur = np.random.uniform(-1, 1, size=(len(samples_on_sur), 3))

            if self.udf:
                samples_close_sur = samples_on_sur + 0.01 * np.random.randn(
                    len(samples_on_sur), 3
                )
            else:
                sigmas = sigmas[idx]
                samples_close_sur = samples_on_sur + sigmas * np.random.randn(
                    len(samples_on_sur), 3
                )

            return {
                "samples_on_sur": samples_on_sur.astype(np.float32),
                "normals_on_sur": normals_on_sur.astype(np.float32),
                "samples_off_sur": samples_off_sur.astype(np.float32),
                "samples_close_sur": samples_close_sur.astype(np.float32),
            }

        sdf_data_samples_frag = [
            sample_data(**sdf_data) for sdf_data in self.sdf_data_list
        ]

        sdf_data = {}
        for key in sdf_data_samples_frag[0].keys():
            sdf_data[key] = np.hstack([frag[key] for frag in sdf_data_samples_frag])

        return sdf_data


def config_training_data(cfg: Config, with_jax=True):
    np.random.seed(0)
    dataset = DFDataset(cfg, udf=cfg.udf)

    g = torch.Generator()
    g.manual_seed(0)

    if with_jax:
        # https://github.com/google/jax/issues/3382
        import torch.multiprocessing as multiprocessing

        multiprocessing.set_start_method("forkserver", force=True)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            generator=g,
        )

    return dataloader


class ToyDataset(Dataset):
    def __init__(self, cfg: Config, samples_on_sur, normals_on_sur):
        super().__init__()

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps

        # Working on numpy array
        self.samples_on_sur = samples_on_sur
        self.normals_on_sur = normals_on_sur

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        idx = np.random.choice(np.arange(len(self.samples_on_sur)), self.n_samples)
        samples_on_sur = self.samples_on_sur[idx]

        if len(self.normals_on_sur) > 0:
            normals_on_sur = self.normals_on_sur[idx]

        samples_off_sur = np.random.uniform(-1, 1, size=(len(samples_on_sur), 3))

        return {
            "samples_on_sur": samples_on_sur,
            "normals_on_sur": normals_on_sur,
            "samples_off_sur": samples_off_sur,
        }


def config_toy_training_data(cfg: Config, samples_on_sur, normals_on_sur, latents):
    dataset = ToyDataset(cfg, samples_on_sur, normals_on_sur, latents)

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
