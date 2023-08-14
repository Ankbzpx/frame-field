import igl
import numpy as np
from jax import vmap, numpy as jnp, jit
from common import vis_oct_field, unroll_identity_block, normalize
from practical_3d_frame_field_generation import proj_sh4_to_rotvec, rotvec_to_R3, rotvec_to_R9, rotvec_n_to_z, R3_to_repvec

import scipy.sparse
import scipy.sparse.linalg
import osqp
import open3d as o3d
import argparse
import os

import flow_lines

import polyscope as ps
from icecream import ic

qz = np.array([0, 0, 0, 0, np.sqrt(7 / 12), 0, 0, 0, 0])
Bz = np.sqrt(5 / 12) * np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])


def project_z(q):
    return qz + Bz.T @ normalize(Bz @ q)


def project_n(q, R_zn):
    return R_zn.T @ project_z(R_zn @ q)


if __name__ == '__main__':
    # enable 64 bit precision
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('--out_path',
                        type=str,
                        default='results',
                        help='Path to output folder.')
    args = parser.parse_args()

    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_oct.obj")

    V, F = igl.read_triangle_mesh(model_path)
    NV = len(V)
    VN = igl.per_vertex_normals(V, F)

    L = igl.cotmatrix(V, F)

    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(VN))

    W = scipy.sparse.coo_array((np.ones(7), (np.arange(7), 1 + np.arange(7))),
                               shape=(7, 9)).todense()

    # R9_zn a = sh4 + c0 sh0 + c1 sh8
    # => W R9_zn a = sh4
    As = np.einsum('bj,nji->nbi', W, R9_zn)
    b = np.array([0, 0, 0, np.sqrt(7 / 12), 0, 0, 0])

    A = scipy.sparse.block_diag(As).tocsc()
    b = np.tile(b, NV)
    L_unroll = unroll_identity_block(-L, 9)

    # Least square
    Q = scipy.sparse.vstack([L_unroll, A])
    c = np.concatenate([np.zeros(9 * NV), b])
    x, _ = scipy.sparse.linalg.cg(Q.T @ Q, Q.T @ c)

    # Quadratic
    prob = osqp.OSQP()
    prob.setup(P=L_unroll, A=A, l=b, u=b)
    prob.warm_start(x=x)
    res = prob.solve()
    assert res.info.status == 'solved'
    x = res.x

    x = x.reshape(NV, 9)
    # x = vmap(project_n)(x.reshape(NV, 9), R9_zn)

    rotvecs = vmap(proj_sh4_to_rotvec)(x)
    Rs = vmap(rotvec_to_R3)(rotvecs)
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
