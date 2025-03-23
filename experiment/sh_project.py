from common import Timer
from experiment.tet_parameterization import make_compatible
from sh_representation import (
    distance_SO3,
    proj_sh4_sdp,
    proj_sh4_to_R3,
    proj_sh4_to_rotvec,
    rotvec_to_R3,
)

from jax import jit, numpy as jnp, vmap
import numpy as np

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)
    sh4 = np.random.randn(100000, 9)

    timer = Timer()

    R1 = vmap(rotvec_to_R3)(proj_sh4_to_rotvec(sh4))

    timer.log("grad")

    R2 = proj_sh4_to_R3(proj_sh4_sdp(sh4))

    timer.log("sdp")

    dists = vmap(distance_SO3)(R1, vmap(make_compatible)(R1, R2))

    degree_max = jnp.rad2deg(dists.max())
    degree_min = jnp.rad2deg(dists.mean())

    ic(degree_max, degree_min)
