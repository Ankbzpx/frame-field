import numpy as np
from jax import numpy as jnp, vmap, jit

from common import normalize
from sh_representation import rotvec_to_sh4, proj_sh4_sdp, proj_sh4_to_R3, rotvec_to_R3, distance_SO3
from tet_parameterization import make_compatible

import seaborn as sns

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    sample_size = 1000000
    np.random.seed(0)

    rotvec = np.random.randn(sample_size, 3)
    R1 = vmap(rotvec_to_R3)(rotvec)
    sh4 = vmap(rotvec_to_sh4)(rotvec)

    # noise
    sigma = 1e-1
    sh4_noisy = sh4 + sigma * np.random.randn(sample_size, 9)

    # Normalize because we do the same for evaluating the polynomial
    sh4_proj = proj_sh4_sdp(normalize(sh4_noisy))

    norm_dists = vmap(jnp.linalg.norm)(sh4 - sh4_proj)

    R2 = vmap(make_compatible)(R1, proj_sh4_to_R3(sh4_proj))
    angle_dist = vmap(distance_SO3)(R1, R2)
    angle_dist_degree = vmap(jnp.rad2deg)(angle_dist)

    ic(angle_dist_degree.max(), angle_dist_degree.min())

    plt = sns.kdeplot(angle_dist_degree, bw_method=0.5, cut=0)
    plt.figure.savefig('plots/density_plot.png')
