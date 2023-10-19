import numpy as np
import igl
from common import normalize, normalize_aabb
from joblib import Parallel, delayed
import multiprocessing
from glob import glob
from tqdm import tqdm
import os

from icecream import ic
import polyscope as ps


class SDFSampler:

    def __init__(self,
                 model_path,
                 normalize=True,
                 surface_ratio=0.6,
                 close_ratio=0.3,
                 sigma=5e-2):
        V, F = igl.read_triangle_mesh(model_path)
        if normalize:
            V = normalize_aabb(V)

        self.V = V
        self.F = F
        self.surface_ratio = surface_ratio
        self.close_ratio = close_ratio
        self.sigma = sigma

    def sample_sdf_igl(self, x):
        return igl.signed_distance(x, self.V, self.F)[0]

    def sample_importance(self, sample_size, multiplier=10., beta=1.5):

        sample_size_full = int(sample_size * multiplier)
        n_surface = int(sample_size_full * self.surface_ratio)
        n_close = int(sample_size_full * self.close_ratio)
        n_free = sample_size_full - (n_surface + n_close)

        bary, f_id = igl.random_points_on_mesh(n_surface, self.V, self.F)
        surface_samples = np.sum(bary[..., None] * self.V[self.F[f_id]], 1)

        degen_n = normalize(np.array([1., 1., 1.]))[None, ...]
        FN = igl.per_face_normals(self.V, self.F, np.float64(degen_n))

        surface_samples += self.sigma * np.random.normal(size=(n_surface,
                                                               1)) * FN[f_id]

        bary, f_id = igl.random_points_on_mesh(n_close, self.V, self.F)

        close_samples = np.sum(
            bary[..., None] * self.V[self.F[f_id]],
            1) + 2. * self.sigma * np.random.normal(size=(n_close, 3))

        free_samples = np.random.uniform(low=-1.0, high=1.0, size=(n_free, 3))

        # Reference: https://github.com/nmwsharp/neural-implicit-queries/blob/c17e4b54f216cefb02d00ddba25c4f15b9873278/src/geometry.py#LL43C1-L43C1
        samples_full = np.vstack([surface_samples, close_samples, free_samples])
        dist_sq, _, _ = igl.point_mesh_squared_distance(samples_full, self.V,
                                                        self.F)
        weight = np.exp(-beta * np.sqrt(dist_sq))
        weight = weight / np.sum(weight)

        sample_indices = np.random.choice(np.arange(sample_size_full),
                                          size=sample_size,
                                          p=weight,
                                          replace=False)
        samples = samples_full[sample_indices]
        sdf_vals, _, _ = igl.signed_distance(np.array(samples), self.V, self.F)

        return samples, np.array(sdf_vals)

    def sample_surface(self, sample_size):
        bary, f_id = igl.random_points_on_mesh(sample_size, self.V, self.F)
        surface_samples = np.sum(bary[..., None] * self.V[self.F[f_id]], 1)

        z = normalize(np.array([1, 1, 1]))
        FN = igl.per_face_normals(self.V, self.F, np.float64(z[None, :]))

        return surface_samples, FN[f_id]

    def sample_dense(self, res=512):
        line = np.linspace(-1.0, 1.0, res)
        samples = np.stack(np.meshgrid(line, line, line), -1).reshape(-1, 3)

        splits = len(samples) // 100000
        sdf_vals = Parallel(
            n_jobs=multiprocessing.cpu_count() - 2, backend='multiprocessing')(
                delayed(self.sample_sdf_igl)(sample_split)
                for sample_split in np.array_split(samples, splits, axis=0))

        sdf_vals = np.concatenate(sdf_vals)
        return samples, np.array(sdf_vals)


if __name__ == '__main__':
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to input model.')
    parser.add_argument('--subfolder',
                        type=str,
                        default='',
                        help='Subfolder path.')
    parser.add_argument('--sample_size',
                        type=int,
                        default=10000,
                        help='Number of samples.')
    args = parser.parse_args()

    model_base_folder_path = 'data/mesh'
    subfolder = args.subfolder

    if args.model_path is not None:
        subfolder = '/'.join(args.model_path.split('/')[2:-1])
        model_path_list = [args.model_path]
    else:
        model_folder_path = os.path.join(model_base_folder_path, subfolder)
        model_path_list = sorted(
            glob(os.path.join(model_folder_path, '*.obj')) +
            glob(os.path.join(model_folder_path, '*.ply')))

    sample_size = args.sample_size

    if args.model_path is not None:
        sdf_base_path = os.path.join('data/sdf', subfolder)
    else:
        sdf_base_path = os.path.join('data/sdf', subfolder, str(sample_size))

    if not os.path.exists(sdf_base_path):
        os.makedirs(sdf_base_path)

    for model_path in tqdm(model_path_list):
        model_name = model_path.split('/')[-1].split('.')[0]
        model_out_path = os.path.join(sdf_base_path, f'{model_name}.npz')

        sampler = SDFSampler(model_path)
        samples_on_sur, normals_on_sur = sampler.sample_surface(sample_size)

        np.savez(model_out_path,
                 samples_on_sur=samples_on_sur,
                 normals_on_sur=normals_on_sur)
