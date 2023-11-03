import numpy as np
import igl
from jax import jit, numpy as jnp, vmap
import scipy.sparse
import os
import time

import polyscope as ps
from icecream import ic

# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def normalize_aabb(V, scale=0.95):
    V = np.copy(V)

    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    V -= V_center
    scale = (V_aabb_max - V_center).max() / scale
    V /= scale

    return V


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
def vis_oct_field(R3s, V, size):
    V_cube = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
                       [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]])

    F_cube = np.array([[7, 6, 2], [2, 3, 7], [0, 4, 5], [5, 1, 0], [0, 2, 6],
                       [6, 4, 0], [7, 3, 1], [1, 5, 7], [3, 2, 0], [0, 1, 3],
                       [4, 6, 7], [7, 5, 4]])

    NV = len(V)
    F_vis = (np.repeat(F_cube[None, ...], NV, 0) +
             (len(V_cube) * np.arange(NV))[:, None, None]).reshape(-1, 3)

    V_vis = (V[:, None, :] +
             np.einsum('nij,bj->nbi', R3s, size * V_cube)).reshape(-1, 3)

    return V_vis, F_vis


def ps_register_curve_network(name, V, E, **wargs):
    V, E = rm_unref_vertices(V, E)
    ps.register_curve_network(name, V, E, **wargs)


def ps_register_basis(name,
                      Rs,
                      V,
                      length=0.02,
                      radius=0.0015,
                      enabled=True,
                      **wargs):
    pc = ps.register_point_cloud(name, V, radius=1e-4, enabled=enabled, **wargs)

    pc.add_vector_quantity('x',
                           Rs[..., 0],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)
    pc.add_vector_quantity('y',
                           Rs[..., 1],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)
    pc.add_vector_quantity('z',
                           Rs[..., 2],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)


# Replace entries in sparse matrix by coefficient weighted identity blocks
def unroll_identity_block(A, dim):
    H, W = A.shape
    A_coo = scipy.sparse.coo_array(A)
    A_unroll_row = ((dim * A_coo.row)[..., None] +
                    np.arange(dim)[None, ...]).reshape(-1)
    A_unroll_col = ((dim * A_coo.col)[..., None] +
                    np.arange(dim)[None, ...]).reshape(-1)
    A_unroll_data = np.repeat(A_coo.data, dim)

    return scipy.sparse.coo_array((A_unroll_data, (A_unroll_row, A_unroll_col)),
                                  shape=(dim * H, dim * W)).tocsc()


def unpack_stiffness(L):
    V_cot_adj_coo = scipy.sparse.coo_array(L)
    # We don't need diagonal
    valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
    E_i = V_cot_adj_coo.col[valid_entries_mask]
    E_j = V_cot_adj_coo.row[valid_entries_mask]
    E_weight = V_cot_adj_coo.data[valid_entries_mask]
    return E_i, E_j, E_weight


# Remove unreference vertices and assign new vertex indices
def rm_unref_vertices(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(F.flatten(),
                                                         return_index=True,
                                                         return_inverse=True)
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_unique][V_map]

    return V, F


def surface_vertex_topology(V, F):
    E = np.stack([
        np.stack([F[:, 0], F[:, 1]], -1),
        np.stack([F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 2], F[:, 0]], -1)
    ], 1).reshape(-1, 2)

    # Use row-wise unique to filter boundary and nonmanifold vertices
    E_row_sorted = np.sort(E, axis=1)
    _, ue_inv, ue_count = np.unique(E_row_sorted,
                                    axis=0,
                                    return_counts=True,
                                    return_inverse=True)

    V_boundary = np.full((len(V)), False)
    V_boundary[list(np.unique(E[(ue_count == 1)[ue_inv]][:, 0]))] = True

    V_nonmanifold = np.full((len(V)), False)
    V_nonmanifold[list(np.unique(E[(ue_count > 2)[ue_inv]][:, 0]))] = True

    return E, V_boundary, V_nonmanifold


# Remove isolated components except for the most promising one
def filter_components(V, F, VN):
    A = igl.adjacency_matrix(F)
    (n_c, C, K) = igl.connected_components(A)

    if n_c > 1:
        # Purely heuristic
        idx_top3 = np.argsort(K)[::-1][:3]

        def validate_VN(k):
            vid = np.argwhere(C == k).reshape(-1,)
            V_filter = V[vid]
            mass_center = V_filter.mean(axis=0)
            dps = jnp.einsum('bi,bi->b',
                             vmap(normalize)(V[vid] - mass_center[None, :]),
                             VN[vid])
            valid = np.sum(dps > 0) > K[k] // 2

            return np.linalg.norm(mass_center) if valid else np.inf

        idx = idx_top3[np.argmin([validate_VN(idx) for idx in idx_top3])]

        VF, NI = igl.vertex_triangle_adjacency(F, F.max() + 1)
        FV = np.split(VF, NI[1:-1])

        V_filter = np.argwhere(C != idx).reshape(-1,)
        F_filter = np.unique(np.concatenate([FV[vid] for vid in V_filter]))
        F = np.delete(F, F_filter, axis=0)
        V, F = rm_unref_vertices(V, F)

    return V, F


class Timer:

    def __init__(self):
        self.reset()

    def log(self, msg):
        cur_time = time.time()
        print(f"{msg}: {cur_time - self.start_time}")
        self.start_time = cur_time

    def reset(self):
        self.start_time = time.time()


def gen_idx_0(xx, yy, zz):
    idx_0 = np.concatenate([
        (xx[::2, ::2, ::2] + yy[::2, ::2, ::2] + zz[::2, ::2, ::2]).reshape(-1),
        (xx[1:-1:2, 1:-1:2, ::2] + yy[1:-1:2, 1:-1:2, ::2] +
         zz[1:-1:2, 1:-1:2, ::2]).reshape(-1),
        (xx[2::2, 1::2, 1:-1:2] + yy[2::2, 1::2, 1:-1:2] +
         zz[2::2, 1::2, 1:-1:2]).reshape(-1),
        (xx[1::2, 2::2, 1:-1:2] + yy[1::2, 2::2, 1:-1:2] +
         zz[1::2, 2::2, 1:-1:2]).reshape(-1)
    ])
    idx_1 = np.concatenate([
        (xx[1::2, ::2, ::2] + yy[1::2, ::2, ::2] +
         zz[1::2, ::2, ::2]).reshape(-1),
        (xx[2:-1:2, 1:-1:2, ::2] + yy[2:-1:2, 1:-1:2, ::2] +
         zz[2:-1:2, 1:-1:2, ::2]).reshape(-1),
        (xx[1:-1:2, 1::2, 1:-1:2] + yy[1:-1:2, 1::2, 1:-1:2] +
         zz[1:-1:2, 1::2, 1:-1:2]).reshape(-1),
        (xx[::2, 2::2, 1:-1:2] + yy[::2, 2::2, 1:-1:2] +
         zz[::2, 2::2, 1:-1:2]).reshape(-1)
    ])
    idx_2 = np.concatenate([(xx[1::2, 1::2, ::2] + yy[1::2, 1::2, ::2] +
                             zz[1::2, 1::2, ::2]).reshape(-1),
                            (xx[2:-1:2, 2:-1:2, ::2] + yy[2:-1:2, 2:-1:2, ::2] +
                             zz[2:-1:2, 2:-1:2, ::2]).reshape(-1),
                            (xx[1:-1:2, ::2, 1:-1:2] + yy[1:-1:2, ::2, 1:-1:2] +
                             zz[1:-1:2, ::2, 1:-1:2]).reshape(-1),
                            (xx[::2, 1:-1:2, 1:-1:2] + yy[::2, 1:-1:2, 1:-1:2] +
                             zz[::2, 1:-1:2, 1:-1:2]).reshape(-1)])
    idx_3 = np.concatenate([
        (xx[::2, 1::2, ::2] + yy[::2, 1::2, ::2] +
         zz[::2, 1::2, ::2]).reshape(-1),
        (xx[1:-1:2, 2:-1:2, ::2] + yy[1:-1:2, 2:-1:2, ::2] +
         zz[1:-1:2, 2:-1:2, ::2]).reshape(-1),
        (xx[2::2, ::2, 1:-1:2] + yy[2::2, ::2, 1:-1:2] +
         zz[2::2, ::2, 1:-1:2]).reshape(-1),
        (xx[1::2, 1:-1:2, 1:-1:2] + yy[1::2, 1:-1:2, 1:-1:2] +
         zz[1::2, 1:-1:2, 1:-1:2]).reshape(-1)
    ])
    idx_4 = np.concatenate([
        (xx[::2, ::2, 1::2] + yy[::2, ::2, 1::2] +
         zz[::2, ::2, 1::2]).reshape(-1),
        (xx[1:-1:2, 1:-1:2, 1::2] + yy[1:-1:2, 1:-1:2, 1::2] +
         zz[1:-1:2, 1:-1:2, 1::2]).reshape(-1),
        (xx[2::2, 1::2, 2:-1:2] + yy[2::2, 1::2, 2:-1:2] +
         zz[2::2, 1::2, 2:-1:2]).reshape(-1),
        (xx[1::2, 2::2, 2:-1:2] + yy[1::2, 2::2, 2:-1:2] +
         zz[1::2, 2::2, 2:-1:2]).reshape(-1)
    ])
    idx_5 = np.concatenate([
        (xx[1::2, ::2, 1::2] + yy[1::2, ::2, 1::2] +
         zz[1::2, ::2, 1::2]).reshape(-1),
        (xx[2:-1:2, 1:-1:2, 1::2] + yy[2:-1:2, 1:-1:2, 1::2] +
         zz[2:-1:2, 1:-1:2, 1::2]).reshape(-1),
        (xx[1:-1:2, 1::2, 2:-1:2] + yy[1:-1:2, 1::2, 2:-1:2] +
         zz[1:-1:2, 1::2, 2:-1:2]).reshape(-1),
        (xx[::2, 2::2, 2:-1:2] + yy[::2, 2::2, 2:-1:2] +
         zz[::2, 2::2, 2:-1:2]).reshape(-1)
    ])
    idx_6 = np.concatenate([
        (xx[1::2, 1::2, 1::2] + yy[1::2, 1::2, 1::2] +
         zz[1::2, 1::2, 1::2]).reshape(-1),
        (xx[2:-1:2, 2:-1:2, 1::2] + yy[2:-1:2, 2:-1:2, 1::2] +
         zz[2:-1:2, 2:-1:2, 1::2]).reshape(-1),
        (xx[1:-1:2, ::2, 2:-1:2] + yy[1:-1:2, ::2, 2:-1:2] +
         zz[1:-1:2, ::2, 2:-1:2]).reshape(-1),
        (xx[::2, 1:-1:2, 2:-1:2] + yy[::2, 1:-1:2, 2:-1:2] +
         zz[::2, 1:-1:2, 2:-1:2]).reshape(-1)
    ])
    idx_7 = np.concatenate([
        (xx[::2, 1::2, 1::2] + yy[::2, 1::2, 1::2] +
         zz[::2, 1::2, 1::2]).reshape(-1),
        (xx[1:-1:2, 2:-1:2, 1::2] + yy[1:-1:2, 2:-1:2, 1::2] +
         zz[1:-1:2, 2:-1:2, 1::2]).reshape(-1),
        (xx[2::2, ::2, 2:-1:2] + yy[2::2, ::2, 2:-1:2] +
         zz[2::2, ::2, 2:-1:2]).reshape(-1),
        (xx[1::2, 1:-1:2, 2:-1:2] + yy[1::2, 1:-1:2, 2:-1:2] +
         zz[1::2, 1:-1:2, 2:-1:2]).reshape(-1)
    ])
    return np.vstack([
        np.stack([idx_4, idx_0, idx_7, idx_5], -1),
        np.stack([idx_1, idx_0, idx_5, idx_2], -1),
        np.stack([idx_6, idx_2, idx_5, idx_7], -1),
        np.stack([idx_3, idx_0, idx_2, idx_7], -1),
        np.stack([idx_0, idx_2, idx_5, idx_7], -1)
    ])


def gen_idx_1(xx, yy, zz):
    idx_0 = np.concatenate([(xx[1:-1:2, ::2, ::2] + yy[1:-1:2, ::2, ::2] +
                             zz[1:-1:2, ::2, ::2]).reshape(-1),
                            (xx[::2, 1:-1:2, ::2] + yy[::2, 1:-1:2, ::2] +
                             zz[::2, 1:-1:2, ::2]).reshape(-1),
                            (xx[1::2, 1::2, 1:-1:2] + yy[1::2, 1::2, 1:-1:2] +
                             zz[1::2, 1::2, 1:-1:2]).reshape(-1),
                            (xx[2::2, 2::2, 1:-1:2] + yy[2::2, 2::2, 1:-1:2] +
                             zz[2::2, 2::2, 1:-1:2]).reshape(-1)])
    idx_1 = np.concatenate([
        (xx[2::2, ::2, ::2] + yy[2::2, ::2, ::2] +
         zz[2::2, ::2, ::2]).reshape(-1),
        (xx[1::2, 1:-1:2, ::2] + yy[1::2, 1:-1:2, ::2] +
         zz[1::2, 1:-1:2, ::2]).reshape(-1),
        (xx[::2, 1::2, 1:-1:2] + yy[::2, 1::2, 1:-1:2] +
         zz[::2, 1::2, 1:-1:2]).reshape(-1),
        (xx[1:-1:2, 2::2, 1:-1:2] + yy[1:-1:2, 2::2, 1:-1:2] +
         zz[1:-1:2, 2::2, 1:-1:2]).reshape(-1)
    ])
    idx_2 = np.concatenate([
        (xx[2::2, 1::2, ::2] + yy[2::2, 1::2, ::2] +
         zz[2::2, 1::2, ::2]).reshape(-1),
        (xx[1::2, 2::2, ::2] + yy[1::2, 2::2, ::2] +
         zz[1::2, 2::2, ::2]).reshape(-1),
        (xx[::2, ::2, 1:-1:2] + yy[::2, ::2, 1:-1:2] +
         zz[::2, ::2, 1:-1:2]).reshape(-1),
        (xx[1:-1:2, 1:-1:2, 1:-1:2] + yy[1:-1:2, 1:-1:2, 1:-1:2] +
         zz[1:-1:2, 1:-1:2, 1:-1:2]).reshape(-1)
    ])
    idx_3 = np.concatenate([
        (xx[1:-1:2, 1::2, ::2] + yy[1:-1:2, 1::2, ::2] +
         zz[1:-1:2, 1::2, ::2]).reshape(-1),
        (xx[::2, 2::2, ::2] + yy[::2, 2::2, ::2] +
         zz[::2, 2::2, ::2]).reshape(-1),
        (xx[1::2, ::2, 1:-1:2] + yy[1::2, ::2, 1:-1:2] +
         zz[1::2, ::2, 1:-1:2]).reshape(-1),
        (xx[2::2, 1:-1:2, 1:-1:2] + yy[2::2, 1:-1:2, 1:-1:2] +
         zz[2::2, 1:-1:2, 1:-1:2]).reshape(-1)
    ])
    idx_4 = np.concatenate([(xx[1:-1:2, ::2, 1::2] + yy[1:-1:2, ::2, 1::2] +
                             zz[1:-1:2, ::2, 1::2]).reshape(-1),
                            (xx[::2, 1:-1:2, 1::2] + yy[::2, 1:-1:2, 1::2] +
                             zz[::2, 1:-1:2, 1::2]).reshape(-1),
                            (xx[1::2, 1::2, 2:-1:2] + yy[1::2, 1::2, 2:-1:2] +
                             zz[1::2, 1::2, 2:-1:2]).reshape(-1),
                            (xx[2::2, 2::2, 2:-1:2] + yy[2::2, 2::2, 2:-1:2] +
                             zz[2::2, 2::2, 2:-1:2]).reshape(-1)])
    idx_5 = np.concatenate([
        (xx[2::2, ::2, 1::2] + yy[2::2, ::2, 1::2] +
         zz[2::2, ::2, 1::2]).reshape(-1),
        (xx[1::2, 1:-1:2, 1::2] + yy[1::2, 1:-1:2, 1::2] +
         zz[1::2, 1:-1:2, 1::2]).reshape(-1),
        (xx[::2, 1::2, 2:-1:2] + yy[::2, 1::2, 2:-1:2] +
         zz[::2, 1::2, 2:-1:2]).reshape(-1),
        (xx[1:-1:2, 2::2, 2:-1:2] + yy[1:-1:2, 2::2, 2:-1:2] +
         zz[1:-1:2, 2::2, 2:-1:2]).reshape(-1)
    ])
    idx_6 = np.concatenate([
        (xx[2::2, 1::2, 1::2] + yy[2::2, 1::2, 1::2] +
         zz[2::2, 1::2, 1::2]).reshape(-1),
        (xx[1::2, 2::2, 1::2] + yy[1::2, 2::2, 1::2] +
         zz[1::2, 2::2, 1::2]).reshape(-1),
        (xx[::2, ::2, 2:-1:2] + yy[::2, ::2, 2:-1:2] +
         zz[::2, ::2, 2:-1:2]).reshape(-1),
        (xx[1:-1:2, 1:-1:2, 2:-1:2] + yy[1:-1:2, 1:-1:2, 2:-1:2] +
         zz[1:-1:2, 1:-1:2, 2:-1:2]).reshape(-1)
    ])
    idx_7 = np.concatenate([
        (xx[1:-1:2, 1::2, 1::2] + yy[1:-1:2, 1::2, 1::2] +
         zz[1:-1:2, 1::2, 1::2]).reshape(-1),
        (xx[::2, 2::2, 1::2] + yy[::2, 2::2, 1::2] +
         zz[::2, 2::2, 1::2]).reshape(-1),
        (xx[1::2, ::2, 2:-1:2] + yy[1::2, ::2, 2:-1:2] +
         zz[1::2, ::2, 2:-1:2]).reshape(-1),
        (xx[2::2, 1:-1:2, 2:-1:2] + yy[2::2, 1:-1:2, 2:-1:2] +
         zz[2::2, 1:-1:2, 2:-1:2]).reshape(-1)
    ])
    return np.vstack([
        np.stack([idx_5, idx_1, idx_4, idx_6], -1),
        np.stack([idx_7, idx_3, idx_6, idx_4], -1),
        np.stack([idx_0, idx_1, idx_3, idx_4], -1),
        np.stack([idx_2, idx_1, idx_6, idx_3], -1),
        np.stack([idx_4, idx_6, idx_1, idx_3], -1)
    ])


def tet_from_grid(res, grid_min=-1.0, grid_max=1.0):
    axis = np.linspace(grid_min, grid_max, res)

    V = np.stack(np.meshgrid(axis, axis, axis), -1).reshape(-1, 3)
    xx, yy, zz = np.indices((res, res, res))
    res_x, res_y, res_z = xx.shape
    yy = res_x * yy
    zz = res_y * res_x * zz

    T0 = gen_idx_0(xx, yy, zz)
    T1 = gen_idx_1(xx, yy, zz)
    T = np.vstack([T0, T1])
    return V, T


# TODO: Nasty workaround, should find a better way to do it
def voxel_tet_from_grid_scale(res, grid_scale):
    max_scale = (grid_scale / grid_scale.min(keepdims=True)).max()
    res_scale = np.round(max_scale * res).astype(int)
    if res_scale % 2 == 1:
        res_scale += 1

    V, T = tet_from_grid(res_scale)
    V, T = crop_tets(V, T, grid_scale)

    return V, T

    # # Risky...
    # unit_size = 1.0 / (res_scale // 2)
    # grid_res = (2 * np.round(grid_scale_safe / unit_size)).astype(int)
    # assert len(V) == np.prod(grid_res)

    # return V, T, grid_res


def crop_tets(V, T, grid_scale):
    V_mask = np.ones(len(V)).astype(bool)
    grid_scale_safe = grid_scale / grid_scale.max(keepdims=True)
    V_mask[np.abs(V[:, 0]) > grid_scale_safe[0]] = False
    V_mask[np.abs(V[:, 1]) > grid_scale_safe[1]] = False
    V_mask[np.abs(V[:, 2]) > grid_scale_safe[2]] = False
    T_mask = V_mask[T].sum(axis=1) == 4
    V, T = rm_unref_vertices(V, T[T_mask])
    V = V * grid_scale.max()
    return V, T
