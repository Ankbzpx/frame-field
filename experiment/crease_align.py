import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
from jaxopt import LBFGS
from jax.experimental import sparse

from icecream import ic
import polyscope as ps

from common import vis_oct_field, unroll_identity_block, normalize_aabb
from sh_representation import sh4_z, rotvec_n_to_z, rotvec_to_R3, rotvec_to_R9
import copy


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
    # Crease normal
    R = rotvec_to_R3(-0.5 * theta * np.array([0, 0, 1]))
    VN0[:3] = np.einsum('ni,ji->nj', VN0[:3], R)

    R = rotvec_to_R3((np.pi - theta) * np.array([0, 0, 1]))
    V1 = V1 @ R.T
    F1 = F1[:, ::-1]
    VN1 = igl.per_vertex_normals(np.float64(V1), F1)

    offset = F0.max()
    F1[F1 == 0] = 2 + 0 - offset
    F1[F1 == 1] = 2 + 1 - offset
    F1[F1 == 2] = 2 + 2 - offset
    F1 += (offset - 2)

    V = np.vstack([V0, V1[3:, :]])
    F = np.vstack([F0, F1])
    VN = np.vstack([VN0, VN1[3:, :]])

    return V, F, VN


def quad_sample_even(width,
                     height,
                     n_grid_width,
                     n_grid_height,
                     rm_front_edge=False,
                     rm_back_edge=False,
                     **kwargs):
    n_sample_width = max(1, int(n_grid_width * width))
    n_sample_height = int(2 * n_grid_height * height)

    xx, zz = np.meshgrid(-np.linspace(0, width, n_sample_width)[::-1],
                         np.linspace(-height, height, n_sample_height))
    yy = np.zeros((n_sample_height, n_sample_width))

    samples = np.stack([xx, yy, zz], -1)

    if n_sample_width > 1 and rm_front_edge:
        samples = samples[:, 1:, :]

    if rm_back_edge:
        samples = samples[:, :-1, :]

    return samples.reshape(-1, 3)


def quad_crease_gap(cfg0, cfg1, angle, gap, grid_size=10):
    V0, F0 = quad_patch(**cfg0)
    V1, F1 = quad_patch(**cfg1)

    theta = np.deg2rad(angle)
    R = rotvec_to_R3((np.pi - theta) * np.array([0, 0, 1]))
    t = np.array([gap, 0, 0])

    V0 = V0 - t[None, :]
    V1 = (V1 - t[None, :]) @ R.T
    F1 = F1[:, ::-1]

    offset = F0.max() + 1
    V = np.vstack([V0, V1])
    F = np.vstack([F0, F1 + offset])
    VN = igl.per_vertex_normals(V, F)

    samples_0 = quad_sample_even(**cfg0,
                                 n_grid_width=grid_size,
                                 n_grid_height=grid_size)
    samples_1 = quad_sample_even(**cfg1,
                                 n_grid_width=grid_size,
                                 n_grid_height=grid_size)

    samples_0 = samples_0 - t[None, :]
    samples_1 = (samples_1 - t[None, :]) @ R.T
    samples = np.vstack([samples_0, samples_1])

    cfg0 = copy.deepcopy(cfg0)
    cfg0['width'] = gap
    samples_0 = quad_sample_even(**cfg0,
                                 n_grid_width=2 * grid_size,
                                 n_grid_height=grid_size,
                                 rm_front_edge=True)
    cfg1 = copy.deepcopy(cfg1)
    cfg1['width'] = gap
    samples_1 = quad_sample_even(**cfg1,
                                 n_grid_width=2 * grid_size,
                                 n_grid_height=grid_size,
                                 rm_front_edge=True,
                                 rm_back_edge=True)
    samples_1 = samples_1 @ R.T
    samples_gap = np.vstack([samples_0, samples_1])

    return V, F, VN, samples, samples_gap


@jit
def sh4_n_align(R9_zn, theta):
    sh4_z_align = sh4_z(theta)
    # R9_zn.T @ sh4_z_align
    return R9_zn[0, :] * sh4_z_align[0] + R9_zn[4, :] * sh4_z_align[4] + R9_zn[
        -1, :] * sh4_z_align[-1]


if __name__ == '__main__':
    # Normalize aabb
    scale = 0.45
    # For this config, oct field starts to align crease at around 10 degree
    cfg0 = {'width': scale * 0.5, 'height': scale * 2.0, 'cfg': 3}
    cfg1 = {'width': scale * 2.0, 'height': scale * 2.0, 'cfg': 0}
    V, F, VN = quad_crease(cfg0, cfg1, 30)

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
        return vis_oct_field(jnp.einsum('bji,bjk->bik', R3_zn, Rs), V,
                             0.1 * igl.avg_edge_length(V, F))

    V_vis, F_vis = vis_theta(thetas)
    V_vis_opt, F_vis_opt = vis_theta(thetas_opt)

    ps.init()
    patch_mesh = ps.register_surface_mesh("patch", V, F)
    patch_mesh.add_vector_quantity("VN", VN, enabled=False)
    ps.register_surface_mesh("Cubes", V_vis, F_vis, enabled=False)
    ps.register_surface_mesh("Cubes_opt", V_vis_opt, F_vis_opt)
    ps.show()
