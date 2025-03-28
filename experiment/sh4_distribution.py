from common import normalize, super_fibonacci
from model_jax import LipMLP
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
import matplotlib.pyplot as plt
import pcax

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    N = 100000
    qs = super_fibonacci(N)
    rotvecs = vmap(quaternion_to_rotvec)(qs)
    R9 = vmap(rotvec_to_R9)(rotvecs)
    sh4 = R9 @ sh4_canonical

    state = pcax.fit(sh4, 3)

    sh4_viz = pcax.transform(state, sh4)

    key_data, key_model = jax.random.split(jax.random.PRNGKey(0))

    mlp = LipMLP(3, 256, 4, 9, key_model, activation="sin")

    x = jax.random.uniform(key_data, (N, 3), minval=-1, maxval=1)
    z = jnp.zeros((N, 0))
    q = mlp(x, z)

    q_viz = pcax.transform(state, q)

    ps.init()
    ps.register_point_cloud("sh4_viz", sh4_viz)
    ps.register_point_cloud("q_viz", q_viz)
    ps.show()

    exit()

    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 20))

    for i, ax in enumerate(axes):
        ax.hist(sh4[:, i], bins=30, alpha=0.7, color=f"C{i}")

    plt.tight_layout()
    plt.show()
