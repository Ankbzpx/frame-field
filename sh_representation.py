import numpy as np
import optax

from common import normalize, Timer
import jax
from jax import numpy as jnp, vmap, grad, jit, value_and_grad

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
# Also see: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
@jit
def rotvec_n_to_z(n):
    n = normalize(n)
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(n, z)
    axis_norm = jnp.linalg.norm(axis) + 1e-8
    # sin(theta) = |n x z|, cos(theta) = n . z
    angle = jnp.arctan2(axis_norm, n[2])
    return angle * (axis / axis_norm)


@jit
def skew_symmetric3(rotvec):
    return jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                      [-rotvec[1], rotvec[0], 0]])


@jit
def R3_to_repvec(R, vn):
    idx = jnp.argmin(jnp.abs(jnp.einsum('ji,j->i', R, vn)))
    return R[:, idx]


# JAX has no logm implementation yet
# **IMPORTANT** Do NOT use it for interpolation (i.e. Karcher mean)
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


# First order approximation
@jit
def rotvec_to_R9_approx(rotvec):
    return jnp.eye(9) + rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz


@jit
def rotvec_to_R9_spherical(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec)
    phi, theta = cartesian_to_spherical(rotvec / rotvec_norm)
    R_zv = R_x90.T @ R_z(-phi) @ R_x90 @ R_z(-theta)
    return R_zv.T @ R_z(rotvec_norm) @ R_zv


@jit
def rotvec_to_R9_close_z(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec)
    # Jacobian matches expm, but why?
    R_zv = rotvec_to_R9_approx(
        jnp.array([rotvec[1] / rotvec_norm, -rotvec[0] / rotvec_norm, 0]))
    R9_z = R_z(jnp.sign(rotvec[2]) * rotvec_norm)
    return R_zv.T @ R9_z @ R_zv


@jit
def rotvec_to_R9_exact(rotvec):
    # Handle singularities at (0, 0, z)
    close_z = jnp.abs(jnp.dot(normalize(rotvec), jnp.array([0., 0., 1.
                                                           ]))) > 0.999
    return jax.lax.cond(close_z, rotvec_to_R9_close_z, rotvec_to_R9_spherical,
                        rotvec)


# Once differentiable, should be enough for gradient descent
@jit
def rotvec_to_R9(rotvec):
    # https://github.com/google/jax/discussions/10306
    # jnp.where always concretely evaluate both branches, that NaN could occur in one of them
    # Thus, use lax.cond instead (one is lazy evaluation)

    # Handle singularities at (0, 0, 0)
    close_zero = jnp.linalg.norm(rotvec) < 1e-8
    return jax.lax.cond(close_zero, rotvec_to_R9_approx, rotvec_to_R9_exact,
                        rotvec)


@jit
def rotvec_to_sh4(rotvec):
    R9 = rotvec_to_R9(rotvec)
    return jnp.sqrt(7 / 12) * R9[:, 4] + jnp.sqrt(5 / 12) * R9[:, -1]


@jit
def rotvec_to_R3_Rodrigues(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec)
    A = skew_symmetric3(rotvec / rotvec_norm)
    return jnp.eye(
        3) + jnp.sin(rotvec_norm) * A + (1 - jnp.cos(rotvec_norm)) * A @ A


# First order approximation
@jit
def rotvec_to_R3_approx(rotvec):
    return jnp.eye(3) + skew_symmetric3(rotvec)


@jit
def rotvec_to_R3(rotvec):
    return jax.lax.cond(
        jnp.linalg.norm(rotvec) < 1e-8, rotvec_to_R3_approx,
        rotvec_to_R3_Rodrigues, rotvec)


# rotvec_to_R9(rotvec) @ sh4_canonical
@jit
def rotvec_to_R3_expm(rotvec):
    return jax.scipy.linalg.expm(skew_symmetric3(rotvec))


@jit
def rotvec_to_R9_expm(rotvec):
    A = rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_sh4_expm(rotvec):
    return rotvec_to_R9_expm(rotvec) @ sh4_canonical


# https://en.wikipedia.org/wiki/Rotation_matrix
@jit
def eulerXYZ_to_R3(a, b, c):
    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(a), -jnp.sin(a)],
                    [0, jnp.sin(a), jnp.cos(a)]])

    Ry = jnp.array([[jnp.cos(b), 0, jnp.sin(b)], [0, 1, 0],
                    [-jnp.sin(b), 0, jnp.cos(b)]])

    Rz = jnp.array([[jnp.cos(c), -jnp.sin(c), 0], [jnp.sin(c),
                                                   jnp.cos(c), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# TODO: Adjust the threshold since the input may not be valid SH4 coefficients
@jit
def proj_sh4_to_rotvec(sh4s_target, lr=1e-2, min_loss_diff=1e-5, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    # This is still necessary as SO(9) induced by SO(3) cannot cover the full rotation space
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
    # Here we leverage autograd to directly optimize over so3

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
            vmap(rotvec_to_sh4)(params['rotvec']) - sh4s_target,
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


# Adapted from Section 5.1 of https://dl.acm.org/doi/abs/10.1145/3366786
sh4_z_4 = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, 0])

Bz = jnp.sqrt(5 / 12) * jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1]])


# exp(t L_z) @ sh4_canonical = sh4_z_4 + Bz.T [cos(4t), sin(4t)].T
# normalize(Bz @ sh4) = normalize([sh4[0], sh4[8]]) is analogous to [cos(4t), sin(4t)]
@jit
def project_z(sh4, xy_scale):
    # Scaling according to  "Boundary Element Octahedral Fields in Volumes" by Solomon et al.
    return sh4_z_4 + Bz.T @ (xy_scale * normalize(Bz @ sh4))


@jit
def project_n(sh4, R9_zn, xy_scale):
    return R9_zn.T @ project_z(R9_zn @ sh4, xy_scale)


# Implement "On the Continuity of Rotation Representations in Neural Networks" by Zhou et al.
@jit
def rot6d_to_R3(rot6d):
    a0 = rot6d[:3]
    a1 = rot6d[3:]
    b0 = normalize(a0)
    b1 = normalize(a1 - jnp.dot(b0, a1) * b0)
    b2 = jnp.cross(b0, b1)
    return jnp.array([b0, b1, b2]).T


# SH basis **pre-multiplied** with r^4, as x^4 + y*4 + z^4 = y_00 * r^4 + \sum_i y_4i * r^4
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
def y_2_2(x, y, z):
    return (1 / 2) * jnp.sqrt(15 / jnp.pi) * x * y * r_2(x, y, z)


@jit
def y_2_1(x, y, z):
    return (1 / 2) * jnp.sqrt(15 / jnp.pi) * y * z * r_2(x, y, z)


@jit
def y_20(x, y, z):
    return (1 / 4) * jnp.sqrt(
        5 / jnp.pi) * (3 * z**2 * r_2(x, y, z) - r_4(x, y, z))


@jit
def y_21(x, y, z):
    return (1 / 2) * jnp.sqrt(15 / jnp.pi) * x * z * r_2(x, y, z)


@jit
def y_22(x, y, z):
    return (1 / 4) * jnp.sqrt(15 / jnp.pi) * (x**2 - y**2) * r_2(x, y, z)


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


sh0_basis = [y_00]
sh2_basis = [y_2_2, y_2_1, y_20, y_21, y_22]
sh4_basis = [y_4_4, y_4_3, y_4_2, y_4_1, y_40, y_41, y_42, y_43, y_44]

# oct_poly_scale * (x^4 + y^4 + z^4) = oct_00 * y_00 + sqrt(7 / 12) * y_40 + sqrt(5 / 12) * y_44
oct_00 = 3 * jnp.sqrt(21) / 4
# r^4 is NOT in denominator because it has been **pre-multiplied** to basis
oct_poly_scale = 5 * jnp.sqrt(21 / np.pi) / 8


@jit
def eval_sh0_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array(jax.tree_map(lambda f: f(x, y, z), sh0_basis))


@jit
def eval_sh2_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array(jax.tree_map(lambda f: f(x, y, z), sh2_basis))


@jit
def eval_sh4_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array(jax.tree_map(lambda f: f(x, y, z), sh4_basis))


@jit
def eval_oct_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array(jax.tree_map(lambda f: f(x, y, z), sh0_basis + sh4_basis))


@jit
def eval_non_orth_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array(
        jax.tree_map(lambda f: f(x, y, z), sh0_basis + sh2_basis + sh4_basis))


# x^4 + y^4 + z^4
@jit
def oct_polynomial_sh4(v, sh4):
    sh = jnp.hstack([oct_00, sh4])
    return jnp.dot(sh, eval_oct_basis(v) / oct_poly_scale)


# (x^4 + y^4 + z^4) / r^4
# Note that divide by r^4 won't affect polynomial value (if v is unit vector),
#   but it forces gradient to lies on the tangent plane of unit sphere
# It is equivalent to oct_polynomial_sh4(normalize(v), sh4)
# IMPORTANT: sh4 needs to be normalized, otherwise it is not the fourth order tensor
@jit
def oct_polynomial_sh4_unit_norm(v, sh4):
    sh = jnp.hstack([oct_00, sh4])
    return jnp.dot(sh,
                   eval_oct_basis(v) / oct_poly_scale) / r_4(v[0], v[1], v[2])


@jit
def oct_polynomial(v, R3):
    v = R3.T @ v
    x = v[0]
    y = v[1]
    z = v[2]
    return x**4 + y**4 + z**4


# The goal here is to find R \in SO(3) that induces sh4
#
# Note our polynomial f(R(v)) = (e_x^T @ v)^4 + (e_y^T @ v)^4 + (e_z^T @ v)^4
#   is the homogeneous polynomial of symmetric tensor T, whose eigenvectors are (e_x, e_y, e_z)
#
# From section 2 of "Orthogonal Decomposition of Symmetric Tensors" by Elina Robeva
#   "The eigenvector of f are precisely the fixed points of \nabla f"
#
# Thus, we can recover R by applying power iteration to \nabla f
#
# Note: Unlike 'proj_sh4_to_rotvec', it doesn't attempt to find closest sh4 induced from SO(3), so the input may well be invalid
# However, "Representing three-dimensional cross fields using 4th order tensors" by Chemin et al. shows that 4-th order tensor forms a linear space \mathbb{R}^9,
#   such that its eigenvectors correspond to three largest eigenvalues are good enough approximation to its projection on SO(3) / O
#
# Empirically it reaches similar minimal as 'proj_sh4_to_rotvec', while being significantly faster
#
@jit
def proj_sh4_to_R3(sh4s_target, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    # Needs to be normalized
    sh4s_target = vmap(normalize)(sh4s_target)

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
        # Power iteration
        v1 = vmap(grad(oct_polynomial_sh4))(state['v1'], sh4s_target)
        v1 = vmap(normalize)(v1)
        v2 = vmap(grad(oct_polynomial_sh4))(state['v2'], sh4s_target)
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


# Zonal Harmonics
# Reference: http://www.ppsloan.org/publications/StupidSH36.pdf

# z-axis symmetry
# zonal_z_poly_scale * z**4 = zonal_z_00 * y_00 + zonal_z_20 * y_20 + zonal_z_40 * y_40
zonal_z_poly_scale = (3 * 35) / (16 * jnp.sqrt(jnp.pi))
zonal_z_00 = 21 / 8
zonal_z_20 = (3 * jnp.sqrt(5) / 2)
zonal_z_40 = 1

zonal_to_octa_scale = oct_poly_scale / zonal_z_poly_scale


@jit
def zonal_z_polynomial(z):
    return z**4


@jit
def zonal_z_polynomial_sh(z):
    sum = zonal_z_00 * y_00(0, 0, z) + zonal_z_20 * y_20(
        0, 0, z) + zonal_z_40 * y_40(0, 0, z)
    return sum / zonal_z_poly_scale


@jit
def zonal_band_coeff(l):
    return jnp.sqrt(4 * jnp.pi / (2 * l + 1))


@jit
def zonal_sh0_coeffs(u):
    return jnp.hstack([zonal_band_coeff(0) * zonal_z_00 * eval_sh0_basis(u)])


@jit
def zonal_sh2_coeffs(u):
    return jnp.hstack([zonal_band_coeff(2) * zonal_z_20 * eval_sh2_basis(u)])


@jit
def zonal_sh4_coeffs(u):
    return jnp.hstack([zonal_band_coeff(4) * zonal_z_40 * eval_sh4_basis(u)])


@jit
# Rotate to direction u
def zonal_oct_coeffs(u):
    return jnp.hstack([zonal_sh0_coeffs(u), zonal_sh4_coeffs(u)])


@jit
# Rotate to direction u
def zonal_non_orth_coeffs(u):
    return jnp.hstack(
        [zonal_sh0_coeffs(u),
         zonal_sh2_coeffs(u),
         zonal_sh4_coeffs(u)])


@jit
def R3_to_sh4_zonal(R3):
    return zonal_to_octa_scale * vmap(zonal_sh4_coeffs)(R3.T).sum(0)


@jit
def rotvec_to_sh4_zonal(rotvec):
    R3 = rotvec_to_R3(rotvec)
    return R3_to_sh4_zonal(R3)


@jit
def rot6d_to_sh4_zonal(rot6d):
    R3 = rot6d_to_R3(rot6d)
    return R3_to_sh4_zonal(R3)


# x^4 + y^4 + z^4
@jit
def oct_polynomial_zonal(v, R3):
    return vmap(zonal_oct_coeffs)(
        R3.T).sum(0) @ eval_oct_basis(v) / zonal_z_poly_scale


# (x^4 + y^4 + z^4) / r^4
# It is equivalent to oct_polynomial_zonal(normalize(v), sh4)
@jit
def oct_polynomial_zonal_unit_norm(v, R3):
    return vmap(zonal_oct_coeffs)(
        R3.T).sum(0) @ eval_oct_basis(v) / zonal_z_poly_scale / r_4(
            v[0], v[1], v[2])


@jit
def non_orth_polynomial_zonal(v, basis):
    return vmap(zonal_non_orth_coeffs)(
        basis.T).sum(0) @ eval_non_orth_basis(v) / zonal_z_poly_scale


# symmetric & column unit norm
@jit
def vec3_to_symmetric3(vec3):
    a = vec3[0]
    b = vec3[1]
    c = vec3[2]

    # From WolframAlpha
    # |a| < 1
    a = jnp.tanh(a)
    # sqrt(1 - a^2) / sqrt(2) < |b| < sqrt(1 - a^2)
    b = jnp.sign(b) * (jax.nn.sigmoid(b) * (1 - jnp.sqrt(2) / 2) +
                       (jnp.sqrt(2) / 2)) * jnp.sqrt(1 - a**2)
    # |c| < sqrt(1 - b^2)
    c = jnp.tanh(c) * jnp.sqrt(1 - b**2)
    d = jnp.sqrt(1 - a**2 - b**2)
    e = jnp.sqrt(1 - b**2 - c**2)
    f = jnp.sqrt(1 - d**2 - e**2)

    return jnp.array([[a, b, d], [b, c, e], [d, e, f]])


@jit
def vec9_to_A3(vec9):
    # Polar decomposition form
    return rot6d_to_R3(vec9[:6]) @ vec3_to_symmetric3(vec9[6:])


@jit
def A3_to_non_orth_zonal(A3):
    # Ignore constant coefficient
    return zonal_to_octa_scale * vmap(zonal_non_orth_coeffs)(A3.T).sum(0)[1:]


@jit
def vec9_to_non_orth_zonal(vec9):
    A3 = vec9_to_A3(vec9)
    return A3_to_non_orth_zonal(A3)


# Reference: Section 4.2 of "Algebraic Representations for Volumetric Frame Fields" by PALMER et al.
def proj_sh4_sdp(sh4s_target):
    import frame_field_utils
    _sdp_helper = frame_field_utils.SH4SDPProjectHelper()

    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]
    return _sdp_helper.project(sh4s_target)


# "Revisiting the Continuity of Rotation Representations in Neural Networks" by Xiang et al.
@jit
def distance_SO3(R1, R2):
    cos = 0.5 * (jnp.trace(R2 @ jnp.linalg.inv(R1)) - 1)
    cos = jnp.clip(cos, -1, 1)
    return jnp.arccos(cos)


if __name__ == '__main__':
    np.random.seed(0)
    rotvec = np.random.randn(3)
    sh4 = rotvec_to_sh4(rotvec)
    sh4_zonal = rotvec_to_sh4_zonal(rotvec)
    R3 = rotvec_to_R3(rotvec)

    s = normalize(np.random.randn(3))
    print("L2 sh4 ≈ zonal: ", jnp.allclose(sh4, sh4_zonal, atol=1e-6))
    print("L2 poly ≈ zonal oct: ",
          jnp.allclose(oct_polynomial(s, R3), oct_polynomial_zonal(s, R3)))
    print("L2 sh4 oct ≈ zonal oct: ",
          jnp.allclose(oct_polynomial(s, R3), oct_polynomial_sh4(s, sh4)))
    print(
        "Rotational invariance: ",
        jnp.allclose(
            rotvec_to_R9_expm(rotvec) @ sh4_canonical @ eval_sh4_basis(s),
            sh4_canonical @ eval_sh4_basis(R3.T @ s)))

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
        v = vmap(normalize)(vmap(grad(oct_polynomial_sh4),
                                 in_axes=(0, None))(v, sh4))

    dps = vmap(oct_polynomial_sh4, in_axes=(0, None))(v, sh4)
    print(f"Dot product ≈ 1: {jnp.allclose(dps, 1)}")

    ps.register_point_cloud('pc_converge', v)

    for _ in range(1000):
        v = vmap(normalize)(vmap(grad(oct_polynomial), in_axes=(0, None))(v,
                                                                          R3))

    ps.register_point_cloud('pc_converge_origin', v)

    ps.show()
