import numpy as np
from jax import vmap, jit, numpy as jnp

from sh_representation import (R3_to_sh4_zonal, zonal_sh4_coeffs, rotvec_to_R3,
                               proj_sh4_sdp, zonal_to_octa_scale, project_z)

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    np.random.seed(0)
    theta = np.random.randn()
    R = rotvec_to_R3(theta * jnp.array([0, 0, 1]))

    sh4_4_z = zonal_to_octa_scale * zonal_sh4_coeffs(np.array([0, 0, 1]))[4]
    sh4_4_xy = zonal_to_octa_scale * (zonal_sh4_coeffs(np.array([1, 0, 0])) +
                                      zonal_sh4_coeffs(np.array([0, 1, 0])))[4]
    assert np.isclose(sh4_4_z + sh4_4_xy, np.sqrt(7 / 12))

    sh4 = R3_to_sh4_zonal(R)
    ic(sh4)

    z_scale = 2
    R_z_scale = np.copy(R)
    R_z_scale[:, 2] *= z_scale
    sh4_z_scale = R3_to_sh4_zonal(R_z_scale)
    err = np.abs(sh4 @ proj_sh4_sdp(sh4_z_scale).reshape(-1,) - 1)
    # Scale z does not change sh4_0**2 + sh4_8**2
    assert np.isclose(sh4_z_scale[0]**2 + sh4_z_scale[8]**2, 5 / 12)
    # sh4_4 should remain the same despite the twist
    #   and it should also be scaled by z_scale^4, as (z_scale * z)^4 = z_scale^4 * z^4,
    #   if ignore the contribution of xy
    assert np.isclose(sh4_z_scale[4],
                      z_scale**4 * (np.sqrt(7 / 12) - sh4_4_xy) + sh4_4_xy)
    ic(sh4_z_scale, err)

    xy_scale = 0.5
    R_xy_scale = np.copy(R)
    R_xy_scale[:, :2] *= xy_scale
    sh4_xy_scale = R3_to_sh4_zonal(R_xy_scale)
    err = np.abs(sh4 @ proj_sh4_sdp(sh4_xy_scale).reshape(-1,) - 1)
    # Scale z does not change the portion of sh4_4 contributed by z
    sh4_4_xy_scale = zonal_to_octa_scale * (
        zonal_sh4_coeffs(np.array([xy_scale, 0, 0])) +
        zonal_sh4_coeffs(np.array([0, xy_scale, 0])))[4]
    assert np.isclose(sh4_xy_scale[4] - sh4_4_xy_scale, sh4_4_z)
    # sqrt(sh4_0**2 + sh4_8**2) should remain the same despite the twist
    #   and it should also be scaled by xy_scale^4, as (xy_scale * v)^4 = xy_scale^4 * v^4
    assert np.isclose(np.sqrt(sh4_xy_scale[0]**2 + sh4_xy_scale[8]**2),
                      xy_scale**4 * np.sqrt(5 / 12))
    # sh4_4 should be scaled similar, if ignore the contribution of z
    assert np.isclose(sh4_xy_scale[4], sh4_4_z + xy_scale**4 * sh4_4_xy)
    # it is equivalent to "Boundary Element Octahedral Fields in Volumes", except for sh4_4
    np.isclose(np.delete(project_z(sh4, xy_scale**4), 4),
               np.delete(sh4_xy_scale, 4))
    ic(sh4_xy_scale, err)
