import igl
import numpy as np
import jax
from jax import vmap, jit, numpy as jnp, value_and_grad
from jax.experimental import sparse
from jaxopt import LBFGS

# Facilitate vscode intellisense
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import optax

from common import vis_oct_field, normalize, unroll_identity_block

import open3d as o3d
import argparse
import os

import flow_lines

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
def rotvec_n_to_z(n):
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(normalize(n), z)
    angle = jnp.arctan2(jnp.linalg.norm(axis) + 1e-8, normalize(n)[2])
    return angle * normalize(axis)


@jit
def skew_symmetric(rotvec):
    return jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                      [-rotvec[1], rotvec[0], 0]])


@jit
def rotvec_to_R3(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec) + 1e-8
    A = skew_symmetric(rotvec / rotvec_norm)
    return jnp.eye(
        3) + jnp.sin(rotvec_norm) * A + (1 - jnp.cos(rotvec_norm)) * A @ A


# Note the phi, theta have different convention as in rendering
@jit
def cartesian_to_spherical(v):
    return jnp.arccos(v[2]), jnp.arctan2(v[1], v[0])


@jit
def rotvec_to_R9(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec) + 1e-8
    phi, theta = cartesian_to_spherical(rotvec / rotvec_norm)
    R_zv = R_x90.T @ R_z(-phi) @ R_x90 @ R_z(-theta)
    return R_zv.T @ R_z(rotvec_norm) @ R_zv


@jit
def rotvec_to_sh4(rotvec):
    return rotvec_to_R9(rotvec) @ sh4_canonical


@jit
def rotvec_to_R3_expm(rotvec):
    return jax.scipy.linalg.expm(skew_symmetric(rotvec))


@jit
def rotvec_to_R9_expm(rotvec):
    A = rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_sh4_expm(rotvec):
    return rotvec_to_R9_expm(rotvec) @ sh4_canonical


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
    @value_and_grad
    def loss_func(params):
        return jnp.power(rotvec_to_sh4(params['rotvec']) - sh4_target, 2).mean()

    @jit
    def condition_func(state):
        return (state["loss"] > min_loss) & (state["iter"] < max_iter)

    @jit
    def body_func(state):
        loss, grads = loss_func(state["params"])
        updates, state["opt_state"] = optimizer.update(grads,
                                                       state["opt_state"])
        state["params"] = optax.apply_updates(state["params"], updates)

        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)
    return state["params"]["rotvec"]


@jit
def R3_to_repvec(R, vn):
    idx = jnp.argmin(jnp.abs(jnp.einsum('ji,j->i', R, vn)))
    return R[:, idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('--out_path',
                        type=str,
                        default='results',
                        help='Path to output folder.')
    args = parser.parse_args()

    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_prac.obj")

    V, T, _ = igl.read_off(model_path)
    F = igl.boundary_facets(T)
    # boundary_facets gives opposite orientation for some reason
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)
    boundary_vid = np.unique(F)

    NV = len(V)
    NB = len(boundary_vid)

    VN = igl.per_vertex_normals(V, F)

    # For vertex belongs to sharp edges, current implementation simply pick a random adjacent face normal
    # FIXME: Duplicate vertices to handle sharp edge
    # NOTE: On second thought, I probably should keep them as they are, cause it is inevitable even for face based parallel transport
    # Fid = np.repeat(np.arange(len(F), dtype=np.int64)[:, None], 3, -1)
    # V2F = np.zeros(NV, dtype=np.int64)
    # V2F[F.reshape(-1,)] = Fid.reshape(-1,)

    # SE, _, _, _, _, _ = igl.sharp_edges(V, F, 45 / 180 * np.pi)
    # sharp_vid = np.unique(SE)
    # FN = igl.per_face_normals(V, F, np.array([0., 1., 0.]))
    # VN[sharp_vid] = FN[V2F[sharp_vid]]

    # Cotangent weights
    L = igl.cotmatrix(V, T)
    R9_zn = vmap(rotvec_to_R9_expm)(vmap(rotvec_n_to_z)(VN[boundary_vid]))

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
    # Assume x is like [9, 9, 9, 9, 9 ... 2, 2, 2]
    A_tl = unroll_identity_block(L, 9)
    A_tr = scipy.sparse.csr_matrix((NV * 9, NB * 2))
    boundary_weight = 100.
    A_bl = scipy.sparse.coo_array(
        (boundary_weight * np.ones(9 * NB),
         (np.arange(9 * NB), ((9 * boundary_vid)[..., None] +
                              np.arange(9)[None, ...]).reshape(-1))),
        shape=(9 * NB, 9 * NV)).tocsc()
    A_br = scipy.sparse.block_diag(boundary_weight *
                                   np.stack([sh0_n, sh8_n], -1))
    A = scipy.sparse.vstack(
        [scipy.sparse.hstack([A_tl, A_tr]),
         scipy.sparse.hstack([A_bl, A_br])])
    b = np.concatenate(
        [np.zeros((NV * 9,)), boundary_weight * sh4_n.reshape(-1,)])

    # A @ x = b
    # => (A.T @ A) @ x = A.T @ b
    x, _ = scipy.sparse.linalg.cg(A.T @ A, A.T @ b)
    sh4_opt = x[:NV * 9].reshape(NV, 9)

    # Project to acquire initialize
    rotvecs = vmap(proj_sh4_to_rotvec)(sh4_opt)

    # Optimize field via non-linear objective function
    sh4_n_pad = jnp.zeros((NV, 9))
    sh4_n_pad = sh4_n_pad.at[boundary_vid].set(sh4_n)

    boundary_mask = np.zeros(NV)
    boundary_mask[boundary_vid] = 1

    V_cot_adj_coo = scipy.sparse.coo_array(L)

    L_jax = sparse.BCOO((V_cot_adj_coo.data,
                         jnp.stack([V_cot_adj_coo.row, V_cot_adj_coo.col], -1)),
                        shape=(NV, NV))

    V_vis, F_vis = vis_oct_field(vmap(rotvec_to_R3)(rotvecs), V, T)

    @jit
    def loss_func(rotvec, align_weight=100):
        a = vmap(rotvec_to_sh4)(rotvec)
        loss_smooth = jnp.trace(a.T @ -L_jax @ a)
        loss_align = jnp.where(boundary_mask,
                               (7 / 12 - jnp.einsum('ni,ni->n', a, sh4_n_pad)),
                               0).sum()

        return loss_smooth + align_weight * loss_align

    lbfgs = LBFGS(loss_func)
    rotvecs_opt = lbfgs.run(rotvecs).params

    Rs = vmap(rotvec_to_R3)(rotvecs_opt)
    Q = vmap(R3_to_repvec)(Rs, VN)

    V_vis, F_vis, VC_vis = flow_lines.trace(V,
                                            F,
                                            VN,
                                            Q,
                                            4000,
                                            length_factor=5,
                                            interval_factor=10,
                                            width_factor=0.075)

    ps.init()
    tet = ps.register_volume_mesh("tet", V, T)
    tet.add_vector_quantity("VN", VN)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(model_out_path, stroke_mesh)
