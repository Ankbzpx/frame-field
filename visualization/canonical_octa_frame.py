import numpy as np
from jax import numpy as jnp, vmap, jit

import math
import pymeshlab
import igl
from tqdm import tqdm

from common import write_triangle_mesh_VC, color_map, triangulation, fibonacci_sphere
from sh_representation import oct_polynomial_zonal_unit_norm

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    for xz_scale in tqdm([0.5, 0.75, 0.85, 0.95, 1.0, 1.5, 2.0]):
        save_name = f"octa_{xz_scale}"
        save_name = save_name.replace('.', '_')
        save_path = f'tmp/{save_name}.obj'

        # Note it is slightly different from c_0^2 + c_8^2 = const, but it doesn't affect xz relations
        R = np.diag([xz_scale, 1.0, xz_scale])
        V = fibonacci_sphere(100000)
        poly_val = vmap(oct_polynomial_zonal_unit_norm, in_axes=(0, None))(V, R)

        V = V * (1 + poly_val)[:, None]
        V, F = triangulation(V)
        poly_val = vmap(oct_polynomial_zonal_unit_norm, in_axes=(0, None))(V, R)
        color_scale = poly_val / poly_val.max() - 1e-2
        VC = color_map(color_scale)

        # Normalize w.r.t. y
        V_aabb_max = V.max(0, keepdims=True)
        V_aabb_min = V.min(0, keepdims=True)
        V_center = 0.5 * (V_aabb_max + V_aabb_min)
        V -= V_center
        scale = (V_aabb_max[0, 1] - V_center[0, 1]) / 0.95
        V /= scale

        write_triangle_mesh_VC(save_path, V, F, VC)
