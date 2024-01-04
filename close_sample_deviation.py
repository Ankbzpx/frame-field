import numpy as np
from jax import numpy as jnp, vmap, jit

from common import normalize
from sh_representation import rotvec_to_sh4, proj_sh4_sdp, proj_sh4_to_R3, rotvec_to_R3, distance_SO3
from tet_parameterization import make_compatible

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    np.random.seed(110)
    rotvec = np.random.randn(3)
    R1 = rotvec_to_R3(rotvec)
    sh4 = rotvec_to_sh4(rotvec)

    # noise
    sigma = 1e-1
    sh4_noisy = sh4[None, :] + sigma * np.random.randn(100000, 9)

    # Normalize because we do the same for evaluating the polynomial
    sh4_proj = proj_sh4_sdp(normalize(sh4_noisy))

    norm_dists = vmap(jnp.linalg.norm)(sh4[None, :] - sh4_proj)

    ic(norm_dists.max(), norm_dists.mean())

    R2 = vmap(make_compatible, in_axes=(None, 0))(R1, proj_sh4_to_R3(sh4_proj))
    angle_dist = vmap(distance_SO3, in_axes=(None, 0))(R1, R2)

    ic(jnp.rad2deg(angle_dist.max()), jnp.rad2deg(angle_dist.mean()))
