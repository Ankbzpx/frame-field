import igl
import numpy as np
from jax import vmap
from common import vis_oct_field, unroll_identity_block, normalize
from practical_3d_frame_field_generation import proj_sh4_to_rotvec, rotvec_to_R3, rotvec_to_R9, rotvec_to_z

import scipy.sparse
import scipy.sparse.linalg
import osqp

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

    V, F = igl.read_triangle_mesh("data/mesh/fandisk_sur.obj")
    NV = len(V)
    VN = igl.per_vertex_normals(V, F)

    L = igl.cotmatrix(V, F)

    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_to_z)(VN))

    W = scipy.sparse.coo_array((np.ones(7), (np.arange(7), 1 + np.arange(7))),
                               shape=(7, 9)).todense()

    As = np.einsum('bj,nji->nbi', W, R9_zn)
    b = np.array([0, 0, 0, np.sqrt(7 / 12), 0, 0, 0])

    A = scipy.sparse.block_diag(As).tocsc()
    b = np.tile(b, NV)
    L_unroll = unroll_identity_block(-L, 9)

    # Least square
    Q = scipy.sparse.vstack([L_unroll, A])
    c = np.concatenate([np.zeros(9 * NV), b])
    x, _ = scipy.sparse.linalg.cg(Q.T @ Q, Q.T @ c)

    x = x.reshape(NV, 9)
    vmap(project_n)(x.reshape(NV, 9), R9_zn)
    x = x.reshape(-1,)

    # Quadratic
    prob = osqp.OSQP()
    prob.setup(P=L_unroll, A=A, l=b, u=b)
    prob.warm_start(x=x)
    res = prob.solve()
    assert res.info.status == 'solved'
    x = res.x

    x = x.reshape(NV, 9)

    rotvecs = vmap(proj_sh4_to_rotvec)(x)
    V_vis, F_vis = vis_oct_field(vmap(rotvec_to_R3)(rotvecs), V, F)

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    ps.register_surface_mesh("Oct_opt", V_vis, F_vis)
    mesh.add_vector_quantity("VN", VN)
    ps.show()
