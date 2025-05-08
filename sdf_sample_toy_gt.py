import pickle

from common import normalize_aabb

import igl
import jax
from jax import jit, numpy as jnp, vmap
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import trimesh

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    # case = "trough"
    case = "stair"

    mesh: trimesh.Trimesh = trimesh.load(f"data/toy/{case}.obj")
    V = mesh.vertices
    F = mesh.faces
    V = normalize_aabb(V)
    FN = igl.per_face_normals(V, F, np.array([1.0, 0.0, 0.0])[None, :])

    x_on, fid = trimesh.sample.sample_surface_even(trimesh.Trimesh(V, F), 100000)
    n_on = FN[fid]

    # 1. Sample uniformly
    x = np.random.uniform(-1.0, 1.0, (10000, 3))

    # 2. Compute SDF and CDF
    u = igl.signed_distance(x, V, F)[0]
    p = np.exp(-50 * np.abs(u))
    p /= p.sum()

    cdf = np.cumsum(p)

    # 3. Resample
    idx = np.searchsorted(cdf, np.random.uniform(0.0, 1.0, 100000))
    x_close = x[idx] + 0.1 * np.random.randn(100000, 3)
    u, _, _, n_close = igl.signed_distance(x_close, V, F, return_normals=True)

    np.savez(
        f"data/toy/{case}.npz",
        x_on=x_on,
        n_on=n_on,
        x_close=x_close,
        n_close=n_close,
        sdf=u,
    )
