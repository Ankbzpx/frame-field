from common import normalize, super_fibonacci
from loss import (
    align_sh4_explicit_cosine,
    align_sh4_explicit_l2,
    align_sh4_functional_grad,
    cosine_similarity,
)
from sh_representation import (
    oct_polynomial_sh4,
    proj_sh4_sdp,
    proj_sh4_to_R3,
    quaternion_to_rotvec,
    rotvec_to_R9,
    sh4_canonical,
)

import jax
from jax import grad, jit, numpy as jnp, vmap
import numpy as np
import pcax

from icecream import ic
import polyscope as ps


def sample_sh4_uniform(N):
    qs = super_fibonacci(N)
    rotvecs = vmap(quaternion_to_rotvec)(qs)
    R9 = vmap(rotvec_to_R9)(rotvecs)
    return R9 @ sh4_canonical


def align_loss(n, q):
    return align_sh4_functional_grad(q, n)


def align_loss2(n, q):
    return align_sh4_explicit_cosine(q, n)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    N = 100000
    n = jnp.array([0.0, 0.0, 1.0])[None, :]
    n = jnp.repeat(n, N, axis=0)
    q = jax.random.normal(key, (N, 9))
    # q = vmap(normalize)(q)

    loss1 = align_loss(n, q)
    loss2 = align_loss2(n, q)

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

    sh4s = sample_sh4_uniform(N)
    R3s = proj_sh4_to_R3(sh4s)

    v0 = R3s[..., 0]
    grad_v0 = vmap(grad(oct_polynomial_sh4))(R3s[..., 0], sh4s)

    ic(jnp.linalg.norm(grad_v0, axis=1).mean())
    ic(5 * jnp.sqrt(21 / np.pi) / 8)
    ic((3 * 35) / (16 * jnp.sqrt(jnp.pi)))

    ic(jnp.isclose(v0, vmap(normalize)(grad_v0)).sum(), len(sh4s) * 3)
