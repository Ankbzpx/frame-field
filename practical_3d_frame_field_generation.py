import igl
import numpy as np
import jax
from jax import vmap, jit, numpy as jnp
from jax.experimental import sparse
from jaxopt import LBFGS

# Facilitate vscode intellisense
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
from common import unroll_identity_block, normalize_aabb, vis_oct_field
from sh_representation import proj_sh4_to_rotvec, R3_to_repvec, rotvec_to_sh4_expm, rotvec_n_to_z, rotvec_to_R3, rotvec_to_R9, sh4_z

import open3d as o3d
import argparse
import os

import flow_lines

import polyscope as ps
from icecream import ic

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
    V = normalize_aabb(V)
    F = igl.boundary_facets(T)
    # boundary_facets gives opposite orientation for some reason
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)
    boundary_vid = np.unique(F)

    NV = len(V)
    NB = len(boundary_vid)

    VN = igl.per_vertex_normals(V, F)

    # TODO: Handle non-smooth vertices (paper precomputes and fixes those SH coefficients)
    # SE, _, _, _, _, _ = igl.sharp_edges(V, F, 45 / 180 * np.pi)
    # sharp_vid = np.unique(SE)

    # Cotangent weights
    L = igl.cotmatrix(V, T)
    rotvec_zn = vmap(rotvec_n_to_z)(VN[boundary_vid])
    R9_zn = vmap(rotvec_to_R9)(rotvec_zn)

    sh4_0 = jnp.array([jnp.sqrt(5 / 12), 0, 0, 0, 0, 0, 0, 0, 0])
    sh4_4 = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, 0])
    sh4_8 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, jnp.sqrt(5 / 12)])

    # R9_zn @ sh4 = sh4_4 + c0 sh4_0 + c1 sh4_8
    # => sh4 = R9_zn.T @ sh4_4 + c0 R9_zn.T @ sh4_0 + c1 R9_zn.T @ sh4_8
    sh4_0_n = jnp.einsum('bji,j->bi', R9_zn, sh4_0)
    sh4_4_n = jnp.einsum('bji,j->bi', R9_zn, sh4_4)
    sh4_8_n = jnp.einsum('bji,j->bi', R9_zn, sh4_8)

    # Build system
    # NV x 9 + NB x 2, have to unroll...
    # Assume x is like [9, 9, 9, 9, 9 ... 2, 2, 2]
    A_tl = unroll_identity_block(L, 9)
    A_tr = scipy.sparse.csr_matrix((NV * 9, NB * 2))

    # Large alignment weight to ensure boundary align
    boundary_weight = 100.
    A_bl = scipy.sparse.coo_array(
        (boundary_weight * np.ones(9 * NB),
         (np.arange(9 * NB), ((9 * boundary_vid)[..., None] +
                              np.arange(9)[None, ...]).reshape(-1))),
        shape=(9 * NB, 9 * NV)).tocsc()
    A_br = scipy.sparse.block_diag(boundary_weight *
                                   np.stack([sh4_0_n, sh4_8_n], -1))
    A = scipy.sparse.vstack(
        [scipy.sparse.hstack([A_tl, A_tr]),
         scipy.sparse.hstack([A_bl, A_br])])
    b = np.concatenate(
        [np.zeros((NV * 9,)), boundary_weight * sh4_4_n.reshape(-1,)])

    # A @ x = b
    # => (A.T @ A) @ x = A.T @ b
    factor = cholesky((A.T @ A).tocsc())
    x = factor(A.T @ b)
    sh4_opt = x[:NV * 9].reshape(NV, 9)

    # Project to acquire initialize
    rotvecs = proj_sh4_to_rotvec(sh4_opt)

    # Optimize field via non-linear objective function
    R9_zn_pad = jnp.repeat(jnp.eye(9)[None, ...], NV, axis=0)
    R9_zn_pad = R9_zn_pad.at[boundary_vid].set(R9_zn)

    L_jax = sparse.BCOO.from_scipy_sparse(-L)

    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(key, (len(boundary_vid),))
    params = {'rotvec': rotvecs, 'theta': theta}

    @jit
    def loss_func(params):
        rotvec = params['rotvec']
        theta = params['theta']

        # LBFGS is second-order optimization method, has to use expm implementation here
        sh4 = vmap(rotvec_to_sh4_expm)(rotvec)

        # sh4_n = R9_zn.T @ sh4_z
        sh4_n = jnp.einsum('bji,bj->bi', R9_zn, vmap(sh4_z)(theta))
        # Use theta parameterization for boundary
        sh4 = sh4.at[boundary_vid].set(sh4_n)

        # Integrated quantity
        loss_smooth = jnp.trace(sh4.T @ L_jax @ sh4)
        return loss_smooth

    lbfgs = LBFGS(loss_func)
    params = lbfgs.run(params).params

    rotvecs_opt = params['rotvec']
    Rs = vmap(rotvec_to_R3)(rotvecs_opt)

    # Recovery boundary rotation
    theta = params['theta']
    rotvec_z = theta[..., None] * jnp.array([0, 0, 1])[None, ...]
    Rz = vmap(rotvec_to_R3)(rotvec_z)
    R3_zn = vmap(rotvec_to_R3)(rotvec_zn)
    # R_n = R3_zn.T @ Rz
    Rs = Rs.at[boundary_vid].set(jnp.einsum('bjk,bji->bki', R3_zn, Rz))

    V_vis_cube, F_vis_cube = vis_oct_field(Rs, V, T)

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
    ps.register_surface_mesh("cube", V_vis_cube, F_vis_cube, enabled=False)
    ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(model_out_path, stroke_mesh)
