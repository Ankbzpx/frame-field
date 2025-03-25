import os

import jax.scipy.spatial
import jax.scipy.spatial.transform
import numpy as np
import pcax


# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from common import normalize
from loss import cosine_similarity
from sh_representation import (
    grad,
    project_n,
    rotvec_n_to_z,
    rotvec_n_to_z_raw,
    rotvec_to_R9,
    sh4_canonical,
    sh4_z,
)

import jax
from jax import hessian, jacfwd, jit, numpy as jnp, vmap

from icecream import ic
import polyscope as ps


def super_fibonacci(n):
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041

    Q = np.empty(shape=(n, 4), dtype=float)

    for i in range(n):
        s = i + 0.5
        r = np.sqrt(s / n)
        R = np.sqrt(1.0 - s / n)
        alpha = 2.0 * np.pi * s / phi
        beta = 2.0 * np.pi * s / psi
        Q[i, 0] = r * np.sin(alpha)
        Q[i, 1] = r * np.cos(alpha)
        Q[i, 2] = R * np.sin(beta)
        Q[i, 3] = R * np.cos(beta)

    return Q


@jit
def quaternion_to_rotvec(q):
    rot = jax.scipy.spatial.transform.Rotation(q)
    return rot.as_rotvec()


if __name__ == "__main__":
    np.random.seed(0)

    # n = np.random.randn(3,)
    n = np.array([0.0, 0.0, 1.0])
    R9_zn = rotvec_to_R9(rotvec_n_to_z(n))
    sh4_test = R9_zn @ sh4_canonical

    N = 100000
    qs = super_fibonacci(N)
    rotvecs = vmap(quaternion_to_rotvec)(qs)

    # N = 1000
    # thetas = jnp.linspace(-jnp.pi, jnp.pi, N)
    # z = jnp.array([0., 0., 1.])
    # rotvecs = z[None, :] * thetas[:, None]

    R9 = vmap(rotvec_to_R9)(rotvecs)
    sh4 = R9 @ sh4_canonical

    state = pcax.fit(sh4, 3)
    sh4_viz = pcax.transform(state, sh4)

    ps.init()
    ps.register_point_cloud("sh4_viz", sh4_viz)
    # for s in np.linspace(0.1, 1.2, 2):
    #     sh4_tmp = sh4 * s
    #     sh4_tmp_n = vmap(project_n, in_axes=(0, None, None))(sh4_tmp, R9_zn, 1)
    #     loss_tmp = 1 - vmap(cosine_similarity)(sh4_tmp, sh4_tmp_n)
    #     ps.register_point_cloud(f"sh4_viz_{s}", pcax.transform(state, sh4 * s)).add_scalar_quantity("loss_tmp", loss_tmp, enabled=True)

    NS = 10000
    key = jax.random.PRNGKey(0)
    r9 = jax.random.normal(key, (NS, 9))
    r9 = vmap(normalize)(r9)
    r9_n = vmap(project_n, in_axes=(0, None, None))(r9, R9_zn, 1)
    loss_cosine_r9 = 1 - vmap(cosine_similarity)(r9, r9_n)
    loss_l1_r9 = jnp.linalg.norm(r9 - r9_n, 1, axis=1)
    loss_l2_r9 = jnp.linalg.norm(r9 - r9_n, 2, axis=1)

    r9_viz = pcax.transform(state, r9)
    r9_n_viz = pcax.transform(state, r9_n)
    curve_indices = np.arange(2 * NS).reshape(2, -1).T
    ps.register_curve_network(
        "projection", jnp.vstack([r9_viz, r9_n_viz]), curve_indices, radius=2e-3
    )

    # embedding = umap.UMAP().fit_transform(sh4)
    # ic(embedding)

    # ps.init()
    # ps_viz = ps.register_point_cloud("sh4_viz", sh4_viz)
    # ps_viz.add_scalar_quantity("loss_cosine", loss_cosine)
    # ps_viz.add_scalar_quantity("loss_l1", loss_l1)
    # ps_viz.add_scalar_quantity("loss_l2", loss_l2)

    ps.register_point_cloud("r9_n_viz", r9_n_viz)

    ps_viz_r9 = ps.register_point_cloud("r9_viz", r9_viz)
    ps_viz_r9.add_scalar_quantity("loss_cosine_r9", loss_cosine_r9)
    ps_viz_r9.add_scalar_quantity("loss_l1_r9", loss_l1_r9)
    ps_viz_r9.add_scalar_quantity("loss_l2_r9", loss_l2_r9)
    ps.show()

    exit()

    # ps.init()
    # ps_viz = ps.register_point_cloud("rotvecs", rotvecs)

    # ps.show()

    exit()

    q = np.random.randn(
        9,
    )

    R9_zn = rotvec_to_R9(rotvec_n_to_z(n))

    ic(jacfwd(project_n)(q, R9_zn, 1))

    # R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal))
    # sh4_n = vmap(project_n, in_axes=(0, 0, None))(sh4, R9_zn, 1)

    # ic(jnp.sqrt(5 / 12))
    # ic(jnp.linalg.norm(((Bz.T @ Bz) @ q) / jnp.linalg.norm(Bz @ q)))
