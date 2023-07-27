import igl
import numpy as np
import jax
from jax import Array, vmap, jit, numpy as jnp, value_and_grad
from functools import partial

# Facilitate vscode intellisense
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from scipy.spatial.transform import Rotation as R
from extrinsically_smooth_direction_field import normalize
import optax

import polyscope as ps
from icecream import ic

# Supplementary of https://dl.acm.org/doi/abs/10.1145/3366786
# yapf: disable
Lx = jnp.array([[0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(2), 0],
                [0, 0, 0, 0, 0, 0, -jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, -3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, -jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, jnp.sqrt(10), 0, 0, 0, 0, 0],
                [0, 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0]])

Ly = jnp.array([[0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                [-jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, -jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [0, 0, -3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -jnp.sqrt(10), 0, 0, 0],
                [0, 0, 0, 0, jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, 0, 0, 3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, 0, 0, jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, 0, 0, jnp.sqrt(2), 0]])

Lz = jnp.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 4.],
    [0, 0, 0, 0, 0, 0, 0, 3., 0],
    [0, 0, 0, 0, 0, 0, 2., 0, 0],
    [0, 0, 0, 0, 0, 1., 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1., 0, 0, 0, 0, 0],
    [0, 0, -2., 0, 0, 0, 0, 0, 0],
    [0, -3., 0, 0, 0, 0, 0, 0, 0],
    [-4., 0, 0, 0, 0, 0, 0, 0, 0],
])

# jax.scipy.linalg.expm(jnp.pi / 2 *Lx)
R_x90 = jnp.array(
    [[0, 0, 0, 0, 0, jnp.sqrt(7 / 2) / 2, 0, -1 / (2 * jnp.sqrt(2)), 0],
     [0, -3 / 4, 0, jnp.sqrt(7) / 4, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1 / (2 * jnp.sqrt(2)), 0, jnp.sqrt(7 / 2) / 2, 0],
     [0, jnp.sqrt(7) / 4, 0, 3 / 4, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 3 / 8, 0, jnp.sqrt(5) / 4, 0, jnp.sqrt(35) / 8],
     [-jnp.sqrt(7 / 2) / 2, 0, -1 / (2 * jnp.sqrt(2)), 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, jnp.sqrt(5) / 4, 0, 1 / 2, 0, -jnp.sqrt(7) / 4],
     [1 / (2 * jnp.sqrt(2)), 0, -jnp.sqrt(7 / 2) / 2, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, jnp.sqrt(35) / 8, 0, -jnp.sqrt(7) / 4, 0, 1 / 8]])

sh4_canonical = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, jnp.sqrt(5 / 12)])

# jax.scipy.linalg.expm(theta *Lz)
def R_z(theta):
    return jnp.array([
        [jnp.cos(4 * theta), 0, 0, 0, 0, 0, 0, 0, jnp.sin(4 * theta)],
        [0, jnp.cos(3 * theta), 0, 0, 0, 0, 0, jnp.sin(3 * theta), 0],
        [0, 0, jnp.cos(2 * theta), 0, 0, 0, jnp.sin(2 * theta), 0, 0],
        [0, 0, 0, jnp.cos(theta), 0, jnp.sin(theta), 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, -jnp.sin(theta), 0, jnp.cos(theta), 0, 0, 0],
        [0, 0, -jnp.sin(2 * theta), 0, 0, 0, jnp.cos(2 * theta), 0, 0],
        [0, -jnp.sin(3 * theta), 0, 0, 0, 0, 0, jnp.cos(3 * theta), 0],
        [-jnp.sin(4 * theta), 0, 0, 0, 0, 0, 0, 0, jnp.cos(4 * theta)],
    ])
# yapf: enable


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
@jit
def rotvec_to_z(n):
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(normalize(n), z)
    angle = jnp.arctan2(jnp.linalg.norm(axis), normalize(n)[2])
    return angle * normalize(axis)


@jit
def rotvec_to_R3(rotvec):
    A = jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                   [-rotvec[1], rotvec[0], 0]])

    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_R9(rotvec):
    A = rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz
    return jax.scipy.linalg.expm(A)


# TODO: Adjust the threshold since the input may not be valid SH4 coefficients
@jit
def proj_sh4_to_rotvec(sh4_target, lr=1e-2, min_loss=1e-4, max_iter=1000):
    # Should I use different key for each function call?
    key = jax.random.PRNGKey(0)
    rotvec = jax.random.normal(key, (3,))

    optimizer = optax.adam(lr)
    params = {'rotvec': rotvec}
    opt_state = optimizer.init(params)

    state = {"loss": 100., "iter": 0, "opt_state": opt_state, "params": params}

    @jit
    def loss_func(params):
        return jnp.power(
            rotvec_to_R9(params['rotvec']) @ sh4_canonical - sh4_target,
            2).mean()

    @jit
    def condition_func(state):
        return (state["loss"] > min_loss) & (state["iter"] < max_iter)

    @jit
    def body_func(state):
        loss, grads = value_and_grad(loss_func)(state["params"])
        updates, state["opt_state"] = optimizer.update(grads,
                                                       state["opt_state"])
        state["params"] = optax.apply_updates(state["params"], updates)

        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)
    return state["params"]["rotvec"]


def vis_oct_field(rotvecs, V, T, scale=0.1):
    V_cube = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
                       [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]])

    F_cube = np.array([[7, 6, 2], [2, 3, 7], [0, 4, 5], [5, 1, 0], [0, 2, 6],
                       [6, 4, 0], [7, 3, 1], [1, 5, 7], [3, 2, 0], [0, 1, 3],
                       [4, 6, 7], [7, 5, 4]])

    NV = len(V)
    F_vis = (np.repeat(F_cube[None, ...], NV, 0) +
             (len(V_cube) * np.arange(NV))[:, None, None]).reshape(-1, 3)

    size = scale * igl.avg_edge_length(V, T)

    R3s = vmap(rotvec_to_R3)(rotvecs)
    V_vis = (V[:, None, :] +
             np.einsum('nij,bj->nbi', R3s, size * V_cube)).reshape(-1, 3)

    return V_vis, F_vis

if __name__ == '__main__':
    V, T, _ = igl.read_off('data/tet/join.off')
    F = igl.boundary_facets(T)

    # boundary_facets gives opposite orientation for some reason
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)
    boundary_vid = np.unique(F)
    VN = igl.per_vertex_normals(V, F)[boundary_vid]

    NV = len(V)
    NB = len(boundary_vid)

    L = igl.cotmatrix(V, T)
    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_to_z)(VN))

    sh0 = jnp.array([jnp.sqrt(5 / 12), 0, 0, 0, 0, 0, 0, 0, 0])
    sh4 = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, 0])
    sh8 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, jnp.sqrt(5 / 12)])

    # R9_zn a = sh4 + c0 sh0 + c1 sh8
    # => a = R9_zn.T @ sh4 + c0 R9_zn.T @ sh0 + c1 R9_zn.T @ sh8
    sh0_n = jnp.einsum('bji,j->bi', R9_zn, sh0)
    sh4_n = jnp.einsum('bji,j->bi', R9_zn, sh4)
    sh8_n = jnp.einsum('bji,j->bi', R9_zn, sh8)

    # Build system
    # NV x 9 + NB x 2, have to unroll...
    A_tl = scipy.sparse.block_diag([L] * 9)
    A_tr = scipy.sparse.csr_matrix((NV * 9, NB * 2))
    boundary_scaling = 100.
    A_bl = scipy.sparse.coo_array(
        (boundary_scaling * np.ones(9 * NB), (np.arange(9 * NB),
                           ((9 * boundary_vid)[..., None] +
                            np.arange(9)[None, ...]).reshape(-1))),
        shape=(9 * NB, 9 * NV)).tocsc()
    A_br = scipy.sparse.block_diag(boundary_scaling * np.stack([sh0_n, sh8_n], -1))
    A = scipy.sparse.vstack(
        [scipy.sparse.hstack([A_tl, A_tr]),
         scipy.sparse.hstack([A_bl, A_br])])
    b = np.concatenate([np.zeros((NV * 9,)), boundary_scaling * sh4_n.reshape(-1,)])

    # A @ x = b
    # => (A.T @ A) @ x = A.T @ b
    x, _ = scipy.sparse.linalg.cg(A.T @ A, A.T @ b)
    sh4_opt = x[:NV * 9].reshape(NV, 9)

    rotvecs = vmap(proj_sh4_to_rotvec)(sh4_opt)

    V_vis, F_vis = vis_oct_field(rotvecs, V, T)
    
    ps.init()
    ps.register_volume_mesh("tet", V, T)
    ps.register_surface_mesh("Oct", V_vis, F_vis)
    verts_vis = ps.register_point_cloud('v', V[boundary_vid], radius=1e-4)
    verts_vis.add_vector_quantity("VN", VN, enabled=True)
    ps.show()
