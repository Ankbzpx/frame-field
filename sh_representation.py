import numpy as np
from common import normalize
import jax
from jax import numpy as jnp, vmap, grad, jit

from practical_3d_frame_field_generation import rotvec_to_R3, rotvec_to_sh4

import polyscope as ps


# SH basis **pre-multiplied** with r^4
# Reference https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
@jit
def r_2(x, y, z):
    return x**2 + y**2 + z**2


@jit
def r_4(x, y, z):
    return r_2(x, y, z)**2


@jit
def y_00(x, y, z):
    return (1 / 2) * (1 / jnp.pi)**(1 / 2) * r_4(x, y, z)


@jit
def y_4_4(x, y, z):
    return (3 / 4) * (35 / jnp.pi)**(1 / 2) * x * y * (x**2 - y**2)


@jit
def y_4_3(x, y, z):
    return (3 / 4) * (35 / (2 * jnp.pi))**(1 / 2) * y * (3 * x**2 - y**2) * z


@jit
def y_4_2(x, y, z):
    return (3 / 4) * (5 / jnp.pi)**(1 / 2) * x * y * (7 * z**2 - r_2(x, y, z))


@jit
def y_4_1(x, y, z):
    return (3 / 4) * (5 / (2 * jnp.pi))**(1 / 2) * y * (7 * z**3 -
                                                        3 * z * r_2(x, y, z))


@jit
def y_40(x, y, z):
    return (3 / 16) * (1 / jnp.pi)**(1 / 2) * (
        35 * z**4 - 30 * z**2 * r_2(x, y, z) + 3 * r_4(x, y, z))


@jit
def y_41(x, y, z):
    return (3 / 4) * (5 / (2 * jnp.pi))**(1 / 2) * x * (7 * z**3 -
                                                        3 * z * r_2(x, y, z))


@jit
def y_42(x, y, z):
    return (3 / 8) * (5 / jnp.pi)**(1 / 2) * (x**2 - y**2) * (7 * z**2 -
                                                              r_2(x, y, z))


@jit
def y_43(x, y, z):
    return (3 / 4) * (35 / (2 * jnp.pi))**(1 / 2) * x * (x**2 - 3 * y**2) * z


@jit
def y_44(x, y, z):
    return (3 / 16) * (35 / jnp.pi)**(1 / 2) * (x**2 *
                                                (x**2 - 3 * y**2) - y**2 *
                                                (3 * x**2 - y**2))


sh_basis_funcs = [
    y_00, y_4_4, y_4_3, y_4_2, y_4_1, y_40, y_41, y_42, y_43, y_44
]


@jit
def rep_polynomial(v, q):
    x = v[0]
    y = v[1]
    z = v[2]
    q = jnp.concatenate([np.array([3 * 21**(1 / 2) / 2]), q])
    basis = jnp.array(jax.tree_map(lambda f: f(x, y, z), sh_basis_funcs))
    return jnp.dot(q, basis)


if __name__ == '__main__':
    np.random.seed(0)
    rotvec = np.random.randn(3)
    q = rotvec_to_sh4(rotvec)
    R3 = rotvec_to_R3(rotvec)

    v = vmap(normalize)(np.random.randn(1000, 3))

    ps.init()
    ps.register_point_cloud('pc', v, radius=2e-3)

    R3 = rotvec_to_R3(rotvec)
    V_cube = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
                       [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]])

    F_cube = np.array([[7, 6, 2], [2, 3, 7], [0, 4, 5], [5, 1, 0], [0, 2, 6],
                       [6, 4, 0], [7, 3, 1], [1, 5, 7], [3, 2, 0], [0, 1, 3],
                       [4, 6, 7], [7, 5, 4]])
    ps.register_surface_mesh('cube', (V_cube / np.sqrt(3)) @ R3.T, F_cube)

    # Repeat evaluating normalized spherical polynomial gradient converges to one of orthogonal basis
    # Reference: https://github.com/dpa1mer/arff/blob/master/src/io/Octa2Frames.m
    #
    # Intuition:
    # For base polynomial f = x^4 + y^4 + z^4, it has directional derivative df = [4 x^3, 4 y^3, 4 z^3]
    # Let v = (x_0, y_0, z_0), x_0 > y_0 > z_0, |x_0^2 + y_0^2 + z_0^2| = 1
    # Repeat evaluating normalize(df(x_0, y_0, z_0)) will increase the ratio of x component, which will eventually converge to (1, 0, 0)
    # For more general case, the polynomial is rotated on the sphere, so the process will converge to one rotated orthogonal basis
    for _ in range(1000):
        v = vmap(normalize)(vmap(grad(rep_polynomial), in_axes=[0, None])(v, q))

    ps.register_point_cloud('pc_converge', v)
    ps.show()
