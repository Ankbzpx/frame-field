import numpy as np
from torch.utils.data import Dataset

from icecream import ic
import polyscope as ps


# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/dataio.py#L390
class PointCloud(Dataset):

    def __init__(self, pc_path, pc_sample_size):
        super().__init__()

        pc = np.load(pc_path)
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

        self.coords = coords
        self.normals = normals
        self.pc_sample_size = pc_sample_size

    def __len__(self):
        return len(self.coords) // self.pc_sample_size

    def __getitem__(self, index):

        rand_indices = np.random.choice(len(self.coords), self.pc_sample_size)

        coords = np.vstack([
            self.coords[rand_indices],
            np.random.uniform(-1, 1, size=(self.pc_sample_size, 3))
        ]).astype(np.float32)

        normals = np.vstack(
            [self.normals[rand_indices],
             np.zeros((self.pc_sample_size, 3))]).astype(np.float32)

        on_surface = np.concatenate(
            [np.ones((self.pc_sample_size,)),
             np.zeros((self.pc_sample_size,))]).astype(bool)

        return {'coords': coords, 'normals': normals, 'on_surface': on_surface}
