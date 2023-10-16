import numpy as np
import igl
from jax import jit, numpy as jnp, vmap
import scipy.sparse
import os

import polyscope as ps
from icecream import ic

# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def normalize_aabb(V):
    V = np.copy(V)
    # [0, 1]
    V -= np.mean(V, axis=0, keepdims=True)
    V_max = np.amax(V)
    V_min = np.amin(V)
    V = (V - V_min) / (V_max - V_min)

    # [-0.95, 0.95]
    V -= 0.5
    V *= 1.9
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


def ps_register_curve_network(name, V, E):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(E,
                                                         return_index=True,
                                                         return_inverse=True)
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    ps.register_curve_network(name, V[V_unique][V_map],
                              V_map_inv[V_unique_idx_inv].reshape(E.shape))


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
