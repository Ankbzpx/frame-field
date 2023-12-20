import numpy as np
from jax import vmap, numpy as jnp, jit
from sh_representation import rotvec_to_R3, R3_to_sh4_zonal

import matplotlib.pyplot as plt

import polyscope as ps
from icecream import ic


@jit
def sh4_tangential_twist(theta):
    return jnp.array([
        jnp.sqrt(5 / 12) * jnp.cos(4 * theta), 0, 0, 0,
        jnp.sqrt(7 / 12), 0, 0, 0,
        jnp.sqrt(5 / 12) * jnp.sin(4 * theta)
    ])


if __name__ == '__main__':

    z_scale = 10

    for angle in [0, 10, 30, 60, 90, 120, 150, 170, 180]:

        theta = np.deg2rad(angle)
        tan_vec = np.array([0, 0, 1])
        R = rotvec_to_R3(theta * tan_vec)

        n0 = np.array([0, 1, 0])
        n1 = R @ n0

        @jit
        def sh_param(theta, normal):
            tan = rotvec_to_R3(theta * normal) @ tan_vec
            cotan = jnp.cross(normal, tan)
            R = jnp.stack([z_scale * normal, tan, cotan], -1)
            return R3_to_sh4_zonal(R)

        @jit
        def loss_func(angles):
            phi = angles[0]
            theta = angles[1]

            sh0 = sh_param(phi, n0)
            sh1 = sh_param(theta, n1)

            return jnp.linalg.norm(sh0 - sh1)

        multi = 0.5
        axis = np.linspace(0, multi * np.pi, 100)
        query = np.stack(np.meshgrid(axis, axis), -1)
        loss = vmap(loss_func)(query.reshape(-1, 2))

        X, Y = np.meshgrid(np.linspace(0, multi, 100),
                           np.linspace(0, multi, 100))

        fig = plt.figure(figsize=(10, 10))
        plt.contourf(X, Y, loss.reshape(100, 100), levels=21)
        tag = f"{angle}".zfill(3)
        z_scale_tag = f'{z_scale}'.replace('.', '_')
        plt.savefig(f'plots/odec_l2_{z_scale_tag}_{tag}.png')
        plt.close()
