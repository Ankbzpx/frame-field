import numpy as np
import igl
from jax import jit, numpy as jnp
import scipy.sparse
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def vis_oct_field(R3s, V, T, scale=0.1):
    V_cube = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
                       [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]])

    F_cube = np.array([[7, 6, 2], [2, 3, 7], [0, 4, 5], [5, 1, 0], [0, 2, 6],
                       [6, 4, 0], [7, 3, 1], [1, 5, 7], [3, 2, 0], [0, 1, 3],
                       [4, 6, 7], [7, 5, 4]])

    NV = len(V)
    F_vis = (np.repeat(F_cube[None, ...], NV, 0) +
             (len(V_cube) * np.arange(NV))[:, None, None]).reshape(-1, 3)

    size = scale * igl.avg_edge_length(V, T)
    V_vis = (V[:, None, :] +
             np.einsum('nij,bj->nbi', R3s, size * V_cube)).reshape(-1, 3)

    return V_vis, F_vis


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