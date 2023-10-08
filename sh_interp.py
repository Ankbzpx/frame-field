import numpy as np
import jax
from jax import vmap, numpy as jnp, jit, grad
from common import vis_oct_field, normalize
from sh_representation import rotvec_to_R3, rotvec_to_sh4, proj_sh4_to_R3, proj_sh4_to_rotvec

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    np.random.seed(0)

    n_step = 5

    rotvec_0 = np.random.randn(3,)
    rotvec_1 = np.random.randn(3,)

    rotvecs_interp = np.linspace(1, 0, n_step)[:, None] * rotvec_0[
        None, :] + np.linspace(0, 1, n_step)[:, None] * rotvec_1[None, :]

    V_0 = np.stack(
        [np.zeros(n_step),
         np.linspace(-1, 1, n_step),
         np.zeros(n_step)], -1)

    V_0_vis, F_0_vis = vis_oct_field(
        vmap(rotvec_to_R3)(rotvecs_interp), V_0, 0.1)

    sh4_0 = rotvec_to_sh4(rotvec_0)
    sh4_1 = rotvec_to_sh4(rotvec_1)

    sh4_interp = np.linspace(1, 0, n_step)[:, None] * sh4_0[
        None, :] + np.linspace(0, 1, n_step)[:, None] * sh4_1[None, :]

    V_1 = np.stack(
        [np.ones(n_step),
         np.linspace(-1, 1, n_step),
         np.zeros(n_step)], -1)

    V_1_vis, F_1_vis = vis_oct_field(proj_sh4_to_R3(sh4_interp), V_1, 0.1)

    V_2 = np.stack(
        [2 * np.ones(n_step),
         np.linspace(-1, 1, n_step),
         np.zeros(n_step)], -1)

    # It gives valid sh4 induced from SO(3)
    sh4_interp_valid = vmap(rotvec_to_sh4)(proj_sh4_to_rotvec(sh4_interp))
    V_2_vis, F_2_vis = vis_oct_field(proj_sh4_to_R3(sh4_interp_valid), V_2, 0.1)

    V_3 = np.stack(
        [3 * np.ones(n_step),
         np.linspace(-1, 1, n_step),
         np.zeros(n_step)], -1)

    sh4_interp_normalized = vmap(normalize)(sh4_interp)
    V_3_vis, F_3_vis = vis_oct_field(proj_sh4_to_R3(sh4_interp_normalized), V_3,
                                     0.1)

    ps.init()
    ps.register_surface_mesh('rotvec interp', V_0_vis, F_0_vis)
    ps.register_surface_mesh('sh4 interp', V_1_vis, F_1_vis)
    ps.register_surface_mesh('sh4 interp (valid)', V_2_vis, F_2_vis)
    ps.register_surface_mesh('sh4 interp (normalize)', V_3_vis, F_3_vis)
    ps.show()
