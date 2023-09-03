import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
from jaxopt import LBFGS
from jax.experimental import sparse

from icecream import ic
import polyscope as ps

from common import vis_oct_field, unroll_identity_block
from sh_representation import sh4_z, rotvec_n_to_z, rotvec_to_R3, rotvec_to_R9


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


def quad_crease_gap(cfg0, cfg1, angle, gap):
    V0, F0 = quad_patch(**cfg0)
    V1, F1 = quad_patch(**cfg1)

    theta = np.deg2rad(angle)
    V1[:3, 0] = (gap) * np.cos(theta)
    V1[:3, 1] = -(gap) * np.sin(theta)
    length = np.copy(V1[3:, 0])
    V1[3:, 0] = -(length - gap) * np.cos(theta)
    V1[3:, 1] = (length - gap) * np.sin(theta)
    F1 = F1[:, ::-1]

    V0[:, 0] -= gap

    offset = F0.max() + 1
    V = np.vstack([V0, V1])
    F = np.vstack([F0, F1 + offset])
    VN = igl.per_vertex_normals(V, F)

    return V, F, VN


def quad_sample_even(width, height, grid_size, remove_edge=False, **kwargs):
    n_sample_width = int(grid_size * width)
    n_sample_height = int(2 * grid_size * height)

    xx, zz = np.meshgrid(np.linspace(-width, 0, n_sample_width),
                         np.linspace(-height, height, n_sample_height))
    yy = np.zeros((n_sample_height, n_sample_width))

    samples = np.stack([xx, yy, zz], -1)

    if remove_edge:
        samples = samples[:, :-1, :]

    return samples.reshape(-1, 3)


def quad_crease_eval(cfg0, cfg1, angle, grid_size):
    samples_0 = quad_sample_even(**cfg0, grid_size=grid_size)
    samples_1 = quad_sample_even(**cfg1, grid_size=grid_size, remove_edge=True)

    theta = np.deg2rad(angle)
    length = np.copy(samples_1[:, 0])
    samples_1[:, 0] = -length * np.cos(theta)
    samples_1[:, 1] = length * np.sin(theta)

    return np.vstack([samples_0, samples_1])


@jit
def sh4_n_align(R9_zn, theta):
    sh4_z_align = sh4_z(theta)
    # R9_zn.T @ sh4_z_align
    return R9_zn[0, :] * sh4_z_align[0] + R9_zn[4, :] * sh4_z_align[4] + R9_zn[
        -1, :] * sh4_z_align[-1]


if __name__ == '__main__':
    # For this config, oct field starts to align crease at around 10 degree
    cfg0 = {'width': 0.5, 'height': 2.0, 'cfg': 3}
    cfg1 = {'width': 2.0, 'height': 2.0, 'cfg': 0}
    V, F, VN = quad_crease(cfg0, cfg1, 30)

    # Generate toy eval samples
    # for angle in [10, 30, 90, 120, 150]:
    #     samples = quad_crease_eval(cfg0, cfg1, angle, 10)
    #     np.save(f"data/toy_eval/crease_{angle}.npy", samples)

    # Gap ones no connectivity, used to gen toy data
    # for gap in [0.05, 0.1, 0.2, 0.3, 0.4]:
    #     for angle in [10, 30, 90, 120, 150]:
    #         V, F, VN = quad_crease_gap(cfg0, cfg1, angle, gap)
    #         igl.write_triangle_mesh(f"data/toy/crease_{gap}_{angle}.obj", V, F)
    # exit()

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
        sh4_unroll = vmap(sh4_n_align)(R9_zn, thetas).reshape(-1,)
        return sh4_unroll.T @ A @ sh4_unroll

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
