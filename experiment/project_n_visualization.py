import os

import jax.scipy.spatial
import jax.scipy.spatial.transform
import numpy as np
import pcax


# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from common import normalize, super_fibonacci
from loss import cosine_similarity
from sh_representation import (
    project_n,
    project_n_scaled,
    quaternion_to_rotvec,
    rotvec_n_to_z,
    rotvec_to_R9,
    sh4_canonical,
)

import jax
from jax import numpy as jnp, vmap

import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    n = np.array([0.0, 0.0, 1.0])
    # n = np.random.randn(3, )
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
    # for s in np.linspace(0.1, 2.0, 20):
    #     sh4_tmp = sh4 * s
    #     # sh4_tmp_n = vmap(project_n, in_axes=(0, None, None))(sh4_tmp, R9_zn, 1)
    #     # loss_tmp = 1 - vmap(cosine_similarity)(sh4_tmp, sh4_tmp_n)
    #     ps.register_point_cloud(f"sh4_viz_{s}", pcax.transform(state, sh4_tmp))
    #     # .add_scalar_quantity("loss_tmp", loss_tmp, enabled=True)

    # ps.show()
    # exit()

    NS = 1000
    key = jax.random.PRNGKey(0)
    r9 = jax.random.normal(key, (NS, 9))
    r9 = vmap(normalize)(r9)
    r9_n = vmap(project_n, in_axes=(0, None, None))(r9, R9_zn, 1)
    loss_cosine_r9 = 1 - vmap(cosine_similarity)(r9, r9_n)
    loss_l1_r9 = jnp.linalg.norm(r9 - r9_n, 1, axis=1)
    loss_l2_r9 = jnp.linalg.norm(r9 - r9_n, 2, axis=1)

    r9_n_scaled = vmap(project_n_scaled, in_axes=(0, None))(r9, R9_zn)
    loss_cosine_r9_scaled = 1 - vmap(cosine_similarity)(r9, r9_n_scaled)
    loss_l1_r9_scaled = jnp.linalg.norm(r9 - r9_n_scaled, 1, axis=1)
    loss_l2_r9_scaled = jnp.linalg.norm(r9 - r9_n_scaled, 2, axis=1)

    r9_viz = pcax.transform(state, r9)
    r9_n_viz = pcax.transform(state, r9_n)
    r9_n_scaled_viz = pcax.transform(state, r9_n_scaled)
    curve_indices = np.arange(2 * NS).reshape(2, -1).T
    ps.register_curve_network(
        "projection", jnp.vstack([r9_viz, r9_n_viz]), curve_indices, radius=2e-3
    )
    ps.register_curve_network(
        "projection_scaled",
        jnp.vstack([r9_viz, r9_n_scaled_viz]),
        curve_indices,
        radius=2e-3,
    )

    ps.register_point_cloud("r9_n_viz", r9_n_viz)
    ps.register_point_cloud("r9_n_scaled_viz", r9_n_scaled_viz)

    ps_viz_r9 = ps.register_point_cloud("r9_viz", r9_viz)
    ps_viz_r9.add_scalar_quantity("loss_cosine_r9", loss_cosine_r9)
    ps_viz_r9.add_scalar_quantity("loss_l1_r9", loss_l1_r9)
    ps_viz_r9.add_scalar_quantity("loss_l2_r9", loss_l2_r9)

    ps_viz_r9.add_scalar_quantity("loss_cosine_r9_scaled", loss_cosine_r9_scaled)
    ps_viz_r9.add_scalar_quantity("loss_l1_r9_scaled", loss_l1_r9_scaled)
    ps_viz_r9.add_scalar_quantity("loss_l2_r9_scaled", loss_l2_r9_scaled)
    ps.show()
