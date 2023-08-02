import numpy as np
import igl
from jax import jit, numpy as jnp
import scipy.sparse


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


# Replace entries in sparse matrix as weighted identity
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
