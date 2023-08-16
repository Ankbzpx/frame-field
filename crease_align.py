import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
from jaxopt import LBFGS
from jax.experimental import sparse

from icecream import ic
import polyscope as ps

from common import vis_oct_field, unroll_identity_block
from practical_3d_frame_field_generation import rotvec_to_R3, rotvec_n_to_z, rotvec_to_R9, sh4_z


def quad_patch(width, height, cfg):
    cfg = cfg % 4
    if cfg < 3:
        V = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1], [1, 0, 1], [1, 0, 0],
                      [1, 0, -1]],
                     dtype=np.float64)
        if cfg == 0:
            F = np.array([[1, 3, 0], [1, 4, 3], [1, 5, 4], [1, 2, 5]],
                         dtype=np.int64)
        elif cfg == 1:
            F = np.array([[0, 4, 3], [0, 1, 4], [1, 5, 4], [1, 2, 5]],
                         dtype=np.int64)
        elif cfg == 2:
            F = np.array([[1, 3, 0], [1, 4, 3], [1, 2, 4], [2, 5, 4]],
                         dtype=np.int64)
    else:
        V = np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1], [1, 0, 1], [1, 0, 0.5],
                      [1, 0, -0.5], [1, 0, -1]],
                     dtype=np.float64)
        F = np.array([[0, 4, 3], [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5]],
                     dtype=np.int64)
    # (x, y) -> (-x, 0, z)
    return V * np.array([-width, 0, height])[None, :], F


def quad_crease(cfg0, cfg1, angle):
    V0, F0 = quad_patch(**cfg0)
    VN0 = igl.per_vertex_normals(V0, F0)
    V1, F1 = quad_patch(**cfg1)

    theta = np.deg2rad(angle)
    R = rotvec_to_R3(0.5 * theta * np.array([0, 0, -1]))
    VN0[:3] = np.einsum('ni,ji->nj', VN0[:3], R)

    length = np.copy(V1[:, 0])
    V1[:, 0] = -length * np.cos(theta)
    V1[:, 1] = length * np.sin(theta)
    F1 = F1[:, ::-1]
    VN1 = igl.per_vertex_normals(V1, F1)

    offset = F0.max()
    F1[F1 == 0] = 2 + 0 - offset
    F1[F1 == 1] = 2 + 1 - offset
    F1[F1 == 2] = 2 + 2 - offset
    F1 += (offset - 2)

    V = np.vstack([V0, V1[3:, :]])
    F = np.vstack([F0, F1])
    VN = np.vstack([VN0, VN1[3:, :]])

    return V, F, VN


@jit
def sh9_n_align(R9_zn, theta):
    sh9_z = sh4_z(theta)
    # R9_zn.T @ sh9_z
    return R9_zn[0, :] * sh9_z[0] + R9_zn[4, :] * sh9_z[4] + R9_zn[
        -1, :] * sh9_z[-1]


if __name__ == '__main__':
    # For this config, oct field starts to align crease at around 10 degree
    cfg0 = {'width': 0.5, 'height': 2.0, 'cfg': 3}
    cfg1 = {'width': 2.0, 'height': 2.0, 'cfg': 0}
    V, F, VN = quad_crease(cfg0, cfg1, 10)
    NV = len(V)
    rotvec_zn = vmap(rotvec_n_to_z)(VN)
    R3_zn = vmap(rotvec_to_R3)(rotvec_zn)
    R9_zn = vmap(rotvec_to_R9)(rotvec_zn)

    # TODO: Face based FEM
    L = igl.cotmatrix(V, F)
    A = sparse.BCOO.from_scipy_sparse(unroll_identity_block(-L, 9))

    key = jax.random.PRNGKey(0)
    thetas = jax.random.normal(key, (NV,))

    @jit
    def loss_func(thetas):
        sh9_unroll = vmap(sh9_n_align)(R9_zn, thetas).reshape(-1,)
        return sh9_unroll.T @ A @ sh9_unroll

    lbfgs = LBFGS(loss_func)
    thetas_opt = lbfgs.run(thetas).params

    # Cube based visualization
    def vis_theta(thetas):
        rotvecs = thetas[:, None] * jnp.repeat(
            jnp.array([0, 0, 1])[None, :], NV, axis=0)
        Rs = vmap(rotvec_to_R3)(rotvecs)
        return vis_oct_field(jnp.einsum('bji,bjk->bik', R3_zn, Rs), V, F)

    V_vis, F_vis = vis_theta(thetas)
    V_vis_opt, F_vis_opt = vis_theta(thetas_opt)

    ps.init()
    patch_mesh = ps.register_surface_mesh("patch", V, F)
    patch_mesh.add_vector_quantity("VN", VN, enabled=False)
    ps.register_surface_mesh("Cubes", V_vis, F_vis, enabled=False)
    ps.register_surface_mesh("Cubes_opt", V_vis_opt, F_vis_opt)
    ps.show()
