import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))

from config import Config

from common_pytorch import normalize_aabb
import numpy as np
import open3d as o3d
import scipy.spatial
import torch
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, cfg: Config):
        super().__init__()

        n_models = len(cfg.sdf_paths)
        assert n_models > 0

        self.n_samples = cfg.training.n_samples
        self.n_steps = cfg.training.n_steps

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


def config_training_data(cfg: Config):
    np.random.seed(0)
    dataset = DFDataset(cfg)

    g = torch.Generator()
    g.manual_seed(0)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        generator=g,
    )

    return dataloader
