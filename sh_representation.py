import numpy as np
import optax

from common import normalize
import jax
from jax import numpy as jnp, vmap, grad, jit, value_and_grad, jacfwd

import polyscope as ps
from icecream import ic

# yapf: disable

# Supplementary of https://dl.acm.org/doi/abs/10.1145/3366786
# Angular momentum matrix (cross product matrix in R^9)
# For small theta, R_x(theta) \approx I + theta * Lx
Lx = jnp.array([[0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(2), 0],
                [0, 0, 0, 0, 0, 0, -jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, -3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, -jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, jnp.sqrt(10), 0, 0, 0, 0, 0],
                [0, 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0]])


# For small theta, R_y(theta) \approx I + theta * Ly
Ly = jnp.array([[0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                [-jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, -jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [0, 0, -3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -jnp.sqrt(10), 0, 0, 0],
                [0, 0, 0, 0, jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, 0, 0, 3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, 0, 0, jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, 0, 0, jnp.sqrt(2), 0]])


# For small theta, R_z(theta) \approx I + theta * Lz
Lz = jnp.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 4.],
    [0, 0, 0, 0, 0, 0, 0, 3., 0],
    [0, 0, 0, 0, 0, 0, 2., 0, 0],
    [0, 0, 0, 0, 0, 1., 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1., 0, 0, 0, 0, 0],
    [0, 0, -2., 0, 0, 0, 0, 0, 0],
    [0, -3., 0, 0, 0, 0, 0, 0, 0],
    [-4., 0, 0, 0, 0, 0, 0, 0, 0],
])

# jax.scipy.linalg.expm(jnp.pi / 2 * Lx)
R_x90 = jnp.array(
    [[0, 0, 0, 0, 0, jnp.sqrt(7 / 2) / 2, 0, -1 / (2 * jnp.sqrt(2)), 0],
     [0, -3 / 4, 0, jnp.sqrt(7) / 4, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1 / (2 * jnp.sqrt(2)), 0, jnp.sqrt(7 / 2) / 2, 0],
     [0, jnp.sqrt(7) / 4, 0, 3 / 4, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 3 / 8, 0, jnp.sqrt(5) / 4, 0, jnp.sqrt(35) / 8],
     [-jnp.sqrt(7 / 2) / 2, 0, -1 / (2 * jnp.sqrt(2)), 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, jnp.sqrt(5) / 4, 0, 1 / 2, 0, -jnp.sqrt(7) / 4],
     [1 / (2 * jnp.sqrt(2)), 0, -jnp.sqrt(7 / 2) / 2, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, jnp.sqrt(35) / 8, 0, -jnp.sqrt(7) / 4, 0, 1 / 8]])

sh4_canonical = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, jnp.sqrt(5 / 12)])

# jax.scipy.linalg.expm(theta * Lz)
@jit
def R_z(theta):
    return jnp.array([
        [jnp.cos(4 * theta), 0, 0, 0, 0, 0, 0, 0, jnp.sin(4 * theta)],
        [0, jnp.cos(3 * theta), 0, 0, 0, 0, 0, jnp.sin(3 * theta), 0],
        [0, 0, jnp.cos(2 * theta), 0, 0, 0, jnp.sin(2 * theta), 0, 0],
        [0, 0, 0, jnp.cos(theta), 0, jnp.sin(theta), 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, -jnp.sin(theta), 0, jnp.cos(theta), 0, 0, 0],
        [0, 0, -jnp.sin(2 * theta), 0, 0, 0, jnp.cos(2 * theta), 0, 0],
        [0, -jnp.sin(3 * theta), 0, 0, 0, 0, 0, jnp.cos(3 * theta), 0],
        [-jnp.sin(4 * theta), 0, 0, 0, 0, 0, 0, 0, jnp.cos(4 * theta)],
    ])
# yapf: enable


# jax.scipy.linalg.expm(theta * Ly)
@jit
def R_y(theta):
    return R_x90 @ R_z(theta) @ R_x90.T


# jax.scipy.linalg.expm(theta * Lx)
@jit
def R_x(theta):
    return R_y(jnp.pi / 2).T @ R_z(theta) @ R_y(jnp.pi / 2)


# R_z(theta) @ sh4_canonical
def sh4_z(theta):
    return jnp.array([0, 0, 0, 0, jnp.sqrt(
        7 / 12), 0, 0, 0, 0]) + jnp.sqrt(5 / 12) * jnp.array(
            [jnp.sin(4 * theta), 0, 0, 0, 0, 0, 0, 0,
             jnp.cos(4 * theta)])


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
@jit
def rotvec_n_to_z(n):
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(normalize(n), z)
    angle = jnp.arctan2(jnp.linalg.norm(axis) + 1e-8, normalize(n)[2])
    return angle * normalize(axis)


@jit
def skew_symmetric(rotvec):
    return jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                      [-rotvec[1], rotvec[0], 0]])


@jit
def R3_to_repvec(R, vn):
    idx = jnp.argmin(jnp.abs(jnp.einsum('ji,j->i', R, vn)))
    return R[:, idx]


# JAX has no logm implementation yet
# May not be correct for pi, but I don't think it matters?
# https://math.stackexchange.com/questions/1972695/converting-from-rotation-matrix-to-axis-angle-with-no-ambiguity
@jit
def R3_to_rotvec(R):
    u = jnp.linalg.eigh(R)[1][:, 2]
    theta = jnp.arccos(0.5 * (jnp.trace(R) - 1))
    u_test = jnp.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    dp = jnp.dot(u, u_test)
    return jnp.where(dp > 0, theta * u, -theta * u)


# Note the phi, theta have different convention as in rendering
@jit
def cartesian_to_spherical(v):
    return jnp.arctan2(jnp.sqrt((v[1]**2 + v[0]**2)),
                       v[2]), jnp.arctan2(v[1], v[0])


@jit
def rotvec_to_R9(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec) + 1e-8
    phi, theta = cartesian_to_spherical(rotvec / rotvec_norm)
    R_zv = R_x90.T @ R_z(-phi) @ R_x90 @ R_z(-theta)
    return R_zv.T @ R_z(rotvec_norm) @ R_zv


# WARNING: Has singularity at (0, 0, 0), (0, 0, z), not suitable for autograd
@jit
def rotvec_to_sh4(rotvec):
    R9 = rotvec_to_R9(rotvec)
    return jnp.sqrt(7 / 12) * R9[:, 4] + jnp.sqrt(5 / 12) * R9[:, -1]


@jit
def rotvec_to_R3(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec) + 1e-8
    A = skew_symmetric(rotvec / rotvec_norm)
    return jnp.eye(
        3) + jnp.sin(rotvec_norm) * A + (1 - jnp.cos(rotvec_norm)) * A @ A


# rotvec_to_R9(rotvec) @ sh4_canonical
@jit
def rotvec_to_R3_expm(rotvec):
    return jax.scipy.linalg.expm(skew_symmetric(rotvec))


@jit
def rotvec_to_R9_expm(rotvec):
    A = rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_sh4_expm(rotvec):
    return rotvec_to_R9_expm(rotvec) @ sh4_canonical


# https://en.wikipedia.org/wiki/Rotation_matrix
@jit
def euler_to_R3(a, b, c):
    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(a), -jnp.sin(a)],
                    [0, jnp.sin(a), jnp.cos(a)]])

    Ry = jnp.array([[jnp.cos(b), 0, jnp.sin(b)], [0, 1, 0],
                    [-jnp.sin(b), 0, jnp.cos(b)]])

    Rz = jnp.array([[jnp.cos(c), -jnp.sin(c), 0], [jnp.sin(c),
                                                   jnp.cos(c), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# TODO: Adjust the threshold since the input may not be valid SH4 coefficients
# @jit
def proj_sh4_to_rotvec(sh4s_target, lr=1e-2, min_loss_diff=1e-5, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    @jit
    def initialize(sh4):
        init_rotvecs = jnp.array([[0, 0, 0], [jnp.pi / 4, 0, 0],
                                  [0, jnp.pi / 4, 0], [0, 0, jnp.pi / 4],
                                  jnp.pi / 4 * normalize(jnp.array([1, 1, 0]))])
        init_sh4s = vmap(rotvec_to_sh4)(init_rotvecs)
        init_idx = jnp.argmax(jnp.einsum('ni,i->n', init_sh4s, sh4))
        return init_rotvecs[init_idx]

    rotvec = vmap(initialize)(sh4s_target)

    # Original implementation uses first order approximation of rotation
    # For small angle a, b, c
    # R(a, b, c) \approx I + a * Lx + b * Ly + c * Lz
    #
    # We want to find R(a, b, c) that rotate sh4 (R @ sh4) to sh4s_target
    # Define objective function (minimize l2 norm -> maximize dot product)
    #   L = sh4s_target.T @ R @ sh4
    #     = sh4s_target.T @ sh4  + a * sh4s_target.T @ Lx @ sh4 + b * sh4s_target.T @ Ly @ sh4 + c * sh4s_target.T @ Lz @ sh4
    #
    # The gradient w.r.t. (a, b, c) is then given by
    #   (sh4s_target.T @ Lx @ sh4, sh4s_target.T @ Ly @ sh4, sh4s_target.T @ Lz @ sh4)
    #
    # Here we leverage autograd to directly optimize over se3

    optimizer = optax.adam(lr)
    params = {'rotvec': rotvec}
    opt_state = optimizer.init(params)

    state = {
        "loss": 100.,
        "loss_diff": 100.,
        "iter": 0,
        "opt_state": opt_state,
        "params": params
    }

    @jit
    @value_and_grad
    def loss_func(params):
        return jnp.power(
            vmap(rotvec_to_sh4_expm)(params['rotvec']) - sh4s_target,
            2).sum(axis=1).mean()

    @jit
    def condition_func(state):
        return (state["loss_diff"] > min_loss_diff) & (state["iter"] < max_iter)

    @jit
    def body_func(state):
        loss, grads = loss_func(state["params"])
        updates, state["opt_state"] = optimizer.update(grads,
                                                       state["opt_state"])
        state["params"] = optax.apply_updates(state["params"], updates)

        state["loss_diff"] = jnp.abs(loss - state["loss"])
        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)
    return state["params"]["rotvec"]


# Section 5.1 of https://dl.acm.org/doi/abs/10.1145/3366786
sh4_z = np.array([0, 0, 0, 0, np.sqrt(7 / 12), 0, 0, 0, 0])
Bz = np.sqrt(5 / 12) * np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])


# exp(t L_z) @ sh4_canonical = sh4_z + Bz.T [cos(4t), sin(4t)].T
# normalize(Bz @ sh4) = normalize([sh4[0], sh4[8]]) is analogous to [cos(4t), sin(4t)]
def project_z(sh4):
    return sh4_z + Bz.T @ normalize(Bz @ sh4)


def project_n(sh4, R_zn):
    return R_zn.T @ project_z(R_zn @ sh4)


# Implement "On the Continuity of Rotation Representations in Neural Networks" by Zhou et al.
@jit
def rot6d_to_R3(rot6d):
    a0 = rot6d[:3]
    a1 = rot6d[3:]
    b0 = normalize(a0)
    b1 = normalize(a1 - jnp.dot(b0, a1) * b0)
    b2 = jnp.cross(b0, b1)
    return jnp.array([b0, b1, b2]).T


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
    return (1 / 2) * jnp.sqrt(1 / jnp.pi) * r_4(x, y, z)


@jit
def y_4_4(x, y, z):
    return (3 / 4) * jnp.sqrt(35 / jnp.pi) * x * y * (x**2 - y**2)


@jit
def y_4_3(x, y, z):
    return (3 / 4) * jnp.sqrt(35 / (2 * jnp.pi)) * y * (3 * x**2 - y**2) * z


@jit
def y_4_2(x, y, z):
    return (3 / 4) * jnp.sqrt(5 / jnp.pi) * x * y * (7 * z**2 - r_2(x, y, z))


@jit
def y_4_1(x, y, z):
    return (3 / 4) * jnp.sqrt(
        5 / (2 * jnp.pi)) * y * (7 * z**3 - 3 * z * r_2(x, y, z))


@jit
def y_40(x, y, z):
    return (3 / 16) * jnp.sqrt(
        1 / jnp.pi) * (35 * z**4 - 30 * z**2 * r_2(x, y, z) + 3 * r_4(x, y, z))


@jit
def y_41(x, y, z):
    return (3 / 4) * jnp.sqrt(
        5 / (2 * jnp.pi)) * x * (7 * z**3 - 3 * z * r_2(x, y, z))


@jit
def y_42(x, y, z):
    return (3 / 8) * jnp.sqrt(
        5 / jnp.pi) * (x**2 - y**2) * (7 * z**2 - r_2(x, y, z))


@jit
def y_43(x, y, z):
    return (3 / 4) * jnp.sqrt(35 / (2 * jnp.pi)) * x * (x**2 - 3 * y**2) * z


@jit
def y_44(x, y, z):
    return (3 / 16) * jnp.sqrt(35 / jnp.pi) * (x**2 * (x**2 - 3 * y**2) - y**2 *
                                               (3 * x**2 - y**2))


sh_basis_funcs = [
    y_00, y_4_4, y_4_3, y_4_2, y_4_1, y_40, y_41, y_42, y_43, y_44
]


@jit
def rep_polynomial(v, sh4):
    x = v[0]
    y = v[1]
    z = v[2]
    sh4 = jnp.concatenate([jnp.array([3 * jnp.sqrt(21) / 2]), normalize(sh4)])
    basis = jnp.array(jax.tree_map(lambda f: f(x, y, z), sh_basis_funcs))
    return jnp.dot(sh4, basis)


# Repeat evaluating normalized spherical polynomial gradient converges to one of orthogonal basis
# Reference: https://github.com/dpa1mer/arff/blob/master/src/io/Octa2Frames.m
#
# Intuition:
# For base polynomial f = x^4 + y^4 + z^4, it has directional derivative df = [4 x^3, 4 y^3, 4 z^3]
# Let v = (x_0, y_0, z_0), x_0 > y_0 > z_0, |x_0^2 + y_0^2 + z_0^2| = 1
# Repeat evaluating normalize(df(x_0, y_0, z_0)) will increase the ratio of x component, which will eventually converge to (1, 0, 0)
# For more general case, the polynomial is rotated on the sphere, so the process will converge to one rotated orthogonal basis
@jit
def proj_sh4_to_R3(sh4s_target, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    n_elem = len(sh4s_target)
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))

    v1 = jax.random.normal(key1, (n_elem, 3))
    v2 = jax.random.normal(key2, (n_elem, 3))
    state = {"loss": 100., "iter": 0, "v1": v1, "v2": v2}

    # sqrt(n_elem * eps**2)
    min_loss = jnp.sqrt(n_elem) * 1e-8

    @jit
    def condition_func(state):
        return (state["loss"] > min_loss) & (state["iter"] < max_iter)

    @jit
    def project_orth(a, b):
        return b - jnp.dot(b, a) * a

    @jit
    def body_func(state):
        v1 = vmap(grad(rep_polynomial))(state['v1'], sh4s_target)
        v1 = vmap(normalize)(v1)
        v2 = vmap(grad(rep_polynomial))(state['v2'], sh4s_target)
        v2 = vmap(project_orth)(v1, v2)
        v2 = vmap(normalize)(v2)

        loss = jnp.linalg.norm(v1 - state['v1'], 'f')

        state["v1"] = v1
        state["v2"] = v2
        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)

    v1 = state['v1']
    v2 = state['v2']
    v3 = jnp.cross(v1, v2, axis=-1)

    return jnp.stack([v1, v2, v3], -1)


if __name__ == '__main__':
    np.random.seed(0)
    rotvec = np.random.randn(3)
    sh4 = rotvec_to_sh4(rotvec)
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

    for _ in range(1000):
        v = vmap(normalize)(vmap(grad(rep_polynomial), in_axes=[0, None])(v,
                                                                          sh4))

    ps.register_point_cloud('pc_converge', v)
    ps.show()
