import igl
import numpy as np
import jax
from jax import Array, vmap, jit, numpy as jnp, value_and_grad
from functools import partial
import scipy
import scipy.linalg
from scipy.spatial.transform import Rotation as R
from extrinsically_smooth_direction_field import normalize
import optax

import polyscope as ps
from icecream import ic

# Supplementary of https://dl.acm.org/doi/abs/10.1145/3366786
# yapf: disable
Lx = jnp.array([[0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(2), 0],
                [0, 0, 0, 0, 0, 0, -jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, -3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, -jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, jnp.sqrt(10), 0, 0, 0, 0, 0],
                [0, 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0]])

Ly = jnp.array([[0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                [-jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
                [0, -jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
                [0, 0, -3 / jnp.sqrt(2), 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -jnp.sqrt(10), 0, 0, 0],
                [0, 0, 0, 0, jnp.sqrt(10), 0, -3 / jnp.sqrt(2), 0, 0],
                [0, 0, 0, 0, 0, 3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
                [0, 0, 0, 0, 0, 0, jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
                [0, 0, 0, 0, 0, 0, 0, jnp.sqrt(2), 0]])

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

# jax.scipy.linalg.expm(jnp.pi / 2 *Lx)
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

f_0 = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, jnp.sqrt(5 / 12)])

# jax.scipy.linalg.expm(theta *Lz)
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


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
@jit
def rotvec_to_z(n):
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(normalize(n), z)
    angle = jnp.arctan2(jnp.linalg.norm(axis), normalize(n)[2])
    return angle * normalize(axis)


@jit
def rotvec_to_R3(rotvec):
    A = jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                   [-rotvec[1], rotvec[0], 0]])

    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_R9(rotvec):
    A = rotvec[0] * Lx + rotvec[1] * Ly + rotvec[2] * Lz
    return jax.scipy.linalg.expm(A)


@jit
def proj_sh4_to_rotvec(sh4_target, lr=1e-2, min_loss=1e-4, max_iter=1000):
    # Should I use different key for each function call?
    key = jax.random.PRNGKey(0)
    rotvec = jax.random.normal(key, (3,))

    optimizer = optax.adam(lr)
    params = {'rotvec': rotvec}
    opt_state = optimizer.init(params)

    state = {"loss": 100., "iter": 0, "opt_state": opt_state, "params": params}

    @jit
    def loss_func(params, f):
        return jnp.power(rotvec_to_R9(params['rotvec']) @ f_0 - f, 2).mean()

    @jit
    def condition_func(state):
        return (state["loss"] > min_loss) & (state["iter"] < max_iter)

    @jit
    def body_func(state):
        loss, grads = value_and_grad(loss_func)(state["params"], sh4_target)
        updates, state["opt_state"] = optimizer.update(grads,
                                                       state["opt_state"])
        state["params"] = optax.apply_updates(state["params"], updates)

        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)
    return state["params"]["rotvec"]


if __name__ == '__main__':
    V, T, _ = igl.read_off('data/tet/bunny.off')
    F = igl.boundary_facets(T)
    # boundary_facets gives opposite orientation for some reason
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)
    VN = igl.per_vertex_normals(V, F)

    np.random.seed(0)
    rotvec_target = np.random.randn(10, 3)
    f_target = jnp.einsum('nij,j->ni', vmap(rotvec_to_R9)(rotvec_target), f_0)

    rotvec = vmap(proj_sh4_to_rotvec)(f_target)

    vis_idx = 7
    vec_vis = np.array([0, 1, 0])
    vec_vis_opt = R.from_rotvec(rotvec[vis_idx]).as_matrix() @ vec_vis
    vec_vis_target = R.from_rotvec(rotvec_target[vis_idx]).as_matrix() @ vec_vis

    ps.init()
    pc = ps.register_point_cloud("o", np.array([0, 0, 0])[None, ...])
    pc.add_vector_quantity("src_opt",
                           vec_vis_opt[None, ...],
                           enabled=True,
                           length=0.1)
    pc.add_vector_quantity("tar",
                           vec_vis_target[None, ...],
                           enabled=True,
                           length=0.1)
    ps.show()
