import math
import os
import time

from mesh_helper import OBJMesh, write_obj

import igl
from jax import jit, numpy as jnp, vmap
import numpy as np
import scipy.sparse

from icecream import ic
import polyscope as ps


# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


def normalize_aabb(V, scale=0.9):
    V_center, scale, _ = aabb_compute(V, scale)
    return (V - V_center) / scale


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
def vis_oct_field(R3s, V, size):
    V_cube = np.array(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
        ]
    )

    F_cube = np.array(
        [
            [7, 6, 2],
            [2, 3, 7],
            [0, 4, 5],
            [5, 1, 0],
            [0, 2, 6],
            [6, 4, 0],
            [7, 3, 1],
            [1, 5, 7],
            [3, 2, 0],
            [0, 1, 3],
            [4, 6, 7],
            [7, 5, 4],
        ]
    )

    NV = len(V)
    F_vis = (
        np.repeat(F_cube[None, ...], NV, 0)
        + (len(V_cube) * np.arange(NV))[:, None, None]
    ).reshape(-1, 3)

    V_vis = (V[:, None, :] + np.einsum("nij,bj->nbi", R3s, size * V_cube)).reshape(
        -1, 3
    )

    return V_vis, F_vis


def ps_register_curve_network(name, V, E, **wargs):
    V, E = rm_unref_vertices(V, E)
    ps.register_curve_network(name, V, E, **wargs)


# Replace entries in sparse matrix by coefficient weighted identity blocks
def unroll_identity_block(A, dim):
    H, W = A.shape
    A_coo = scipy.sparse.coo_array(A)
    A_unroll_row = ((dim * A_coo.row)[..., None] + np.arange(dim)[None, ...]).reshape(
        -1
    )
    A_unroll_col = ((dim * A_coo.col)[..., None] + np.arange(dim)[None, ...]).reshape(
        -1
    )
    A_unroll_data = np.repeat(A_coo.data, dim)

    return scipy.sparse.csc_array(
        (A_unroll_data, (A_unroll_row, A_unroll_col)), shape=(dim * H, dim * W)
    )


def unpack_stiffness(L):
    V_cot_adj_coo = scipy.sparse.coo_array(L)
    # We don't need diagonal
    valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
    E_i = V_cot_adj_coo.col[valid_entries_mask]
    E_j = V_cot_adj_coo.row[valid_entries_mask]
    E_weight = V_cot_adj_coo.data[valid_entries_mask]
    return E_i, E_j, E_weight


# scipy.sparse.block_diag is extremely slow for some reason..
def block_diag(As):
    b, n, m = As.shape
    col_idx = (np.tile(np.arange(m), n)[None, :] + (m * np.arange(b))[:, None]).reshape(
        -1,
    )
    row_idx = (
        np.repeat(np.arange(n), m)[None, :] + (n * np.arange(b))[:, None]
    ).reshape(
        -1,
    )
    data = As.reshape(
        -1,
    )
    return scipy.sparse.csc_array((data, (row_idx, col_idx)), shape=(n * b, m * b))


# Remove unreference vertices and assign new vertex indices
def rm_unref_vertices(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(
        F.flatten(), return_index=True, return_inverse=True
    )
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_unique][V_map]

    return V, F


def surface_vertex_topology(V, F):
    E = np.stack(
        [
            np.stack([F[:, 0], F[:, 1]], -1),
            np.stack([F[:, 1], F[:, 2]], -1),
            np.stack([F[:, 2], F[:, 0]], -1),
        ],
        1,
    ).reshape(-1, 2)

    # Use row-wise unique to filter boundary and nonmanifold vertices
    E_row_sorted = np.sort(E, axis=1)
    _, ue_inv, ue_count = np.unique(
        E_row_sorted, axis=0, return_counts=True, return_inverse=True
    )

    V_boundary = np.full((len(V)), False)
    V_boundary[list(np.unique(E[(ue_count == 1)[ue_inv]][:, 0]))] = True

    V_nonmanifold = np.full((len(V)), False)
    V_nonmanifold[list(np.unique(E[(ue_count > 2)[ue_inv]][:, 0]))] = True

    return E, V_boundary, V_nonmanifold


# Remove isolated components except for the most promising one
def filter_components(V, F, VN):
    A = igl.adjacency_matrix(F)
    # K is the size of each component
    (n_c, C, K) = igl.connected_components(A)

    confidence = 2

    if n_c > 1:
        # Purely heuristic
        idx_top3 = np.argsort(K)[::-1][:3]

        # Remove unreasonable small components
        idx_top3 = idx_top3[K[idx_top3] / K[idx_top3[0]] > 0.15]

        def validate_VN(k):
            vid = np.argwhere(C == k).reshape(
                -1,
            )
            V_filter = V[vid]
            mass_center = V_filter.mean(axis=0)
            dps = jnp.einsum(
                "bi,bi->b", vmap(normalize)(V[vid] - mass_center[None, :]), VN[vid]
            )

            ratio = np.sum(dps > 0) / (K[k] // 2)

            return np.linalg.norm(mass_center), ratio

        min_dist = 1.0
        confidence = 0
        idx_min = idx_top3[0]

        for idx in idx_top3:
            dist, ratio = validate_VN(idx)
            if dist < min_dist and (ratio > 1.0 or dist < 0.1):
                min_dist = dist
                confidence = ratio
                idx_min = idx

        idx = idx_min

        VF, NI = igl.vertex_triangle_adjacency(F, F.max() + 1)
        FV = np.split(VF, NI[1:-1])

        V_filter = np.argwhere(C != idx_min).reshape(
            -1,
        )
        F_filter = np.unique(np.concatenate([FV[vid] for vid in V_filter]))
        F = np.delete(F, F_filter, axis=0)

        V, F = rm_unref_vertices(V, F)

    return V, F, confidence > 1.5


class Timer:
    def __init__(self):
        self.reset()

    def log(self, msg):
        cur_time = time.time()
        print(f"{msg}: {cur_time - self.start_time}")
        self.start_time = cur_time

    def reset(self):
        self.start_time = time.time()


def write_triangle_mesh_VC(save_path, V, F, VC):
    mesh = OBJMesh(V, F)
    mesh.vertex_colors = VC
    write_obj(save_path, mesh)


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    return xyz


# Copied from: https://marcalexa.github.io/superfibonacci/
def super_fibonacci(n):
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041

    Q = np.empty(shape=(n, 4), dtype=float)

    for i in range(n):
        s = i + 0.5
        r = np.sqrt(s / n)
        R = np.sqrt(1.0 - s / n)
        alpha = 2.0 * np.pi * s / phi
        beta = 2.0 * np.pi * s / psi
        Q[i, 0] = r * np.sin(alpha)
        Q[i, 1] = r * np.cos(alpha)
        Q[i, 2] = R * np.sin(beta)
        Q[i, 3] = R * np.cos(beta)

    return Q
