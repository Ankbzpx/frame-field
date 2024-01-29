import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
from sh_representation import rotvec_to_R3, R3_to_sh4_zonal, R3_to_rotvec, rotvec_n_to_z, rotvec_to_R9, rotvec_to_sh4
import matplotlib.pyplot as plt
import matplotlib as mpl

import polyscope as ps
from icecream import ic


@jit
def sh4_tangential_twist(theta, xy_scale):
    return jnp.array([
        jnp.sqrt(5 / 12 * xy_scale) * jnp.cos(4 * theta), 0, 0, 0,
        jnp.sqrt(7 / 12), 0, 0, 0,
        jnp.sqrt(5 / 12 * xy_scale) * jnp.sin(4 * theta)
    ])


# Both `sh_param` are equivalent in terms of loss manifold, but `rotvec_to_R9` introduces `normal` depend initial tangential twist
#   plot wise
# @jit
# def sh_param(theta, normal, xy_scale):
#     R9_zn = rotvec_to_R9(rotvec_n_to_z(normal))
#     return R9_zn.T @ sh4_tangential_twist(theta, xy_scale)


@jit
def sh_param(theta, normal, xy_scale):
    tan_vec = jnp.array([0, 0, 1])

    x_scale = xy_scale
    y_scale = xy_scale
    z_scale = 1

    tan = rotvec_to_R3(theta * normal) @ tan_vec
    cotan = jnp.cross(normal, tan)
    R = jnp.stack([z_scale * normal, x_scale * tan, y_scale * cotan], -1)
    return R3_to_sh4_zonal(R)


@jit
def eval_loss(angle, xy_scale, pi_multi=0.5, dim=10):
    theta = jnp.deg2rad(angle)
    tan_vec = jnp.array([0, 0, 1])
    R = rotvec_to_R3(theta * tan_vec)

    n0 = jnp.array([0, 1, 0])
    n1 = R @ n0

    @jit
    def loss_func(angles):
        phi = angles[0]
        theta = angles[1]

        sh0 = sh_param(phi, n0, xy_scale)
        sh1 = sh_param(theta, n1, xy_scale)

        return jnp.linalg.norm(sh0 - sh1)

    axis = jnp.linspace(0, pi_multi * jnp.pi, dim**2)
    query = jnp.stack(jnp.meshgrid(axis, axis), -1)
    loss = vmap(loss_func)(query.reshape(-1, 2))
    return loss


if __name__ == '__main__':

    pi_multi = 0.5
    dim = 10

    cmap = mpl.colors.LinearSegmentedColormap.from_list('octa',
                                                        [(0, '#461959'),
                                                         (0.25, '#7A316F'),
                                                         (0.75, '#CD6688'),
                                                         (1, '#AED8CC')],
                                                        N=256)

    for xy_scale in [0.5, 0.75, 1.0, 1.5, 2.0]:
        for angle in [10, 30, 60, 90, 120, 150, 170]:

            loss = eval_loss(angle, xy_scale)

            if angle == 90:
                loss_max = loss.max()
                loss_min = loss.min()
                jax.debug.print(
                    'Z scale: {xy_scale}, Loss min: {loss_min}, Loss max: {loss_max}, Loss gap: {loss_gap}',
                    xy_scale=xy_scale,
                    loss_min=loss_min,
                    loss_max=loss_max,
                    loss_gap=loss_max - loss_min)

            X, Y = np.meshgrid(np.linspace(0, pi_multi, dim**2),
                               np.linspace(0, pi_multi, dim**2))

            fig = plt.figure(figsize=(dim, dim))
            plt.contourf(X,
                         Y,
                         loss.reshape(dim**2, dim**2),
                         levels=21,
                         cmap=cmap)
            tag = f"{angle}".zfill(3)
            xy_scale_tag = f'{xy_scale}'.replace('.', '_')
            plt.axis('off')
            plt.savefig(f'plots/loss_l2_{xy_scale_tag}_{tag}.png',
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
