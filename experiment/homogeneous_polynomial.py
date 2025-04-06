from common import normalize, super_fibonacci
from loss import (
    cosine_similarity,
)
from sh_representation import (
    oct_polynomial_sh4,
    oct_polynomial_sh4_unit_norm,
    project_n,
    quaternion_to_rotvec,
    rotvec_n_to_z,
    rotvec_to_R9,
    sh4_canonical,
)

import jax
from jax import grad, numpy as jnp, vmap
import numpy as np
import pcax

from icecream import ic
import polyscope as ps


def sample_sh4_uniform(N):
    qs = super_fibonacci(N)
    rotvecs = vmap(quaternion_to_rotvec)(qs)
    R9 = vmap(rotvec_to_R9)(rotvecs)
    return R9 @ sh4_canonical


def gradient_map(n, q):
    return grad(oct_polynomial_sh4)(n, q)


def loss_func(n, q):
    grad_n = gradient_map(normalize(n), normalize(q))
    return 20 - jnp.dot(4 * normalize(n), grad_n)


def loss_func2(n, q):
    R9_zn = rotvec_to_R9(rotvec_n_to_z(n))
    q_n = project_n(q, R9_zn, 1)
    return 1 - cosine_similarity(q, q_n)
    # return jnp.linalg.norm(q - q_n, ord=1)


if __name__ == "__main__":
    # Experiment scaling
    s = 5
    n = s * np.array([0.0, 0.0, 1.0])

    ic(
        grad(oct_polynomial_sh4)(n, sh4_canonical),
        4 * jnp.pow(jnp.linalg.norm(s), 3) * normalize(n),
    )
    ic(grad(oct_polynomial_sh4)(normalize(n), sh4_canonical), 4 * normalize(n))

    ic(
        grad(oct_polynomial_sh4, argnums=1)(n, sh4_canonical)
        / jnp.pow(jnp.linalg.norm(s), 4)
    )
    ic(grad(oct_polynomial_sh4_unit_norm, argnums=1)(n, sh4_canonical))
    exit()

    key = jax.random.PRNGKey(0)

    N = 1000000
    n = jnp.array([0.0, 0.0, 1.0])

    q = jax.random.normal(key, (N, 9))
    # q = vmap(normalize)(q)

    loss1 = vmap(loss_func, in_axes=(None, 0))(n, q)
    loss2 = vmap(loss_func2, in_axes=(None, 0))(n, q)

    # Fit a PCA to visualize
    sh4_uniform = sample_sh4_uniform(N)
    state = pcax.fit(sh4_uniform, 3)
    sh4_uniform_viz = pcax.transform(state, sh4_uniform)

    q_viz = pcax.transform(state, q)
    # g1_viz = pcax.transform(state, g1)
    # g2_viz = pcax.transform(state, g2)

    ps.init()
    ps.register_point_cloud("sh4_uniform_viz", sh4_uniform_viz)
    q_ps = ps.register_point_cloud("q_viz", q_viz)
    q_ps.add_scalar_quantity("loss1", loss1, enabled=True)
    q_ps.add_scalar_quantity("loss2", loss2)
    ps.show()

    exit()

    critical_points = jnp.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    loss_critical = vmap(loss_func, in_axes=(0, None))(critical_points, sh4_canonical)
    loss_critical_grad = -vmap(grad(loss_func), in_axes=(0, None))(
        critical_points, sh4_canonical
    )

    ic(loss_critical)
    ic(loss_critical_grad)

    vs = jax.random.normal(key, (N, 3))
    loss = vmap(loss_func, in_axes=(0, None))(vs, sh4_canonical)
    loss_grad = -vmap(grad(loss_func), in_axes=(0, None))(vs, sh4_canonical)
    loss_grad = vmap(normalize)(loss_grad)

    ps.init()
    ps_viz = ps.register_point_cloud("vs", vs, point_render_mode="quad")
    ps_viz.add_scalar_quantity("loss", loss, enabled=True)
    ps_viz.add_vector_quantity("loss_grad", loss_grad, enabled=True)
    ps_viz = ps.register_point_cloud("critical_points", critical_points)
    ps_viz.add_scalar_quantity("loss_critical", loss_critical, enabled=True)
    ps_viz.add_vector_quantity("loss_critical_grad", loss_critical_grad, enabled=True)
    ps.show()
