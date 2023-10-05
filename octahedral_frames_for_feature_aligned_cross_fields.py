import igl
import numpy as np
from jax import vmap, jit, numpy as jnp
from jaxopt import LBFGS
from common import unroll_identity_block, normalize_aabb, normalize, unpack_stiffness
from sh_representation import R3_to_repvec, rotvec_n_to_z, rotvec_to_R9, proj_sh4_to_R3

import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
import open3d as o3d
import argparse
import os

import flow_lines

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    # enable 64 bit precision
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('-p', type=str, default='2', help='Lp norm.')
    parser.add_argument('--out_path',
                        type=str,
                        default='results',
                        help='Path to output folder.')
    args = parser.parse_args()

    norm = args.p
    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_{norm}_oct.obj")

    V, F = igl.read_triangle_mesh(model_path)
    V = normalize_aabb(V)
    NV = len(V)
    VN = igl.per_vertex_normals(V, F)

    L = igl.cotmatrix(V, F)

    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(VN))

    W = scipy.sparse.coo_array((np.ones(7), (np.arange(7), 1 + np.arange(7))),
                               shape=(7, 9)).todense()

    # R9_zn @ sh4 = sh4_4 + c0 sh4_0 + c1 sh4_8
    # => W @ R9_zn @ sh4 = sh4_4
    As = np.einsum('bj,nji->nbi', W, R9_zn)
    b = np.array([0, 0, 0, np.sqrt(7 / 12), 0, 0, 0])

    if norm != "2":
        p = 1
        if norm == 'inf':
            # FIXME: Don't know how to implement inf norm. Pick max will stop gradient flow for other pairs
            p = 8
        else:
            try:
                p = int(norm)
                p = 1 if p == 0 else p
            except:
                pass

        # Need to adjust weight for different norm
        boundary_weight = 0.5
        np.random.seed(0)
        x = np.random.randn(NV, 9)
        x = vmap(normalize)(x)
        E_i, E_j, E_weight = unpack_stiffness(-L)

        @jit
        def loss_func(x):
            edge_energy = jnp.linalg.norm(E_weight[:, None] * (x[E_i] - x[E_j]),
                                          ord=2,
                                          axis=1)
            loss_smooth = jnp.linalg.norm(edge_energy,
                                          ord=p) / len(edge_energy)**(1 / p)
            loss_align = jnp.linalg.norm(jnp.einsum('bji,bi->bj', As, x) -
                                         b[None, :],
                                         ord=2,
                                         axis=1).mean()
            return loss_smooth + boundary_weight * loss_align

        lbfgs = LBFGS(loss_func)
        x = lbfgs.run(x).params
    else:
        # May need to adjust weight for different geometry
        boundary_weight = 0.1
        A = scipy.sparse.block_diag(As).tocsc()
        b = np.tile(b, NV)
        L_unroll = unroll_identity_block(-L, 9)

        # Linear system
        Q = scipy.sparse.vstack([L_unroll, boundary_weight * A])
        c = np.concatenate([np.zeros(9 * NV), boundary_weight * b])
        factor = cholesky((Q.T @ Q).tocsc())
        x = factor(Q.T @ c).reshape(NV, 9)

    Rs = proj_sh4_to_R3(x)
    Q = vmap(R3_to_repvec)(Rs, VN)

    V_vis, F_vis, VC_vis = flow_lines.trace(V, F, VN, Q, 4000)

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    mesh.add_vector_quantity("VN", VN)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(model_out_path, stroke_mesh)
