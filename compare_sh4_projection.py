import numpy as np
from jax import numpy as jnp, vmap, jit

from common import Timer
from sh_representation import rotvec_to_R3, proj_sh4_to_rotvec, proj_sh4_sdp, proj_sh4_to_R3, distance_SO3
from tet_parameterization import make_compatible

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    np.random.seed(0)
    sh4 = np.random.randn(100000, 9)

    timer = Timer()

    R1 = vmap(rotvec_to_R3)(proj_sh4_to_rotvec(sh4))

    timer.log('grad')

    R2 = proj_sh4_to_R3(proj_sh4_sdp(sh4))

    timer.log('sdp')

    dists = vmap(distance_SO3)(R1, vmap(make_compatible)(R1, R2))

    radians_max = jnp.rad2deg(dists.max())
    radians_min = jnp.rad2deg(dists.mean())

    ic(radians_max, radians_min)
