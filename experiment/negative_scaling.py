import os

import numpy as np


# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from common import super_fibonacci, vis_oct_field
from sh_representation import (
    proj_sh4_to_R3,
    quaternion_to_rotvec,
    rotvec_to_R9,
    sh4_canonical,
)

import jax
from jax import vmap

import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)
    N = 100
    qs = super_fibonacci(N)
    rotvecs = vmap(quaternion_to_rotvec)(qs)

    R9 = vmap(rotvec_to_R9)(rotvecs)
    sh4 = R9 @ sh4_canonical

    R3 = proj_sh4_to_R3(sh4)
    V_octa, F_octa = vis_oct_field(R3, rotvecs, 0.1)

    R3_2 = proj_sh4_to_R3(-sh4)
    V_octa_2, F_octa_2 = vis_oct_field(R3_2, rotvecs, 0.1)

    ps.init()
    ps.register_surface_mesh("octa", V_octa, F_octa)
    ps.register_surface_mesh("octa 2", V_octa_2, F_octa_2)
    ps.show()
