import numpy as np
from jax import numpy as jnp, vmap, jit

import math
import pymeshlab
import igl
from tqdm import tqdm

from common import write_triangle_mesh_VC
from sh_representation import oct_polynomial_zonal_unit_norm

import polyscope as ps
from icecream import ic


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))    # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)    # radius at y

        theta = phi * i    # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    return xyz


if __name__ == '__main__':

    for xz_scale in tqdm([0.5, 0.75, 0.85, 0.95, 1.0, 1.5, 2.0]):
        save_name = f"octa_{xz_scale}"
        save_name = save_name.replace('.', '_')
        save_path = f'tmp/{save_name}.obj'

        # Note it is slightly different from c_0^2 + c_8^2 = const, but it doesn't affect xz relations
        R = np.diag([xz_scale, 1.0, xz_scale])
        V = fibonacci_sphere(100000)
        poly_val = vmap(oct_polynomial_zonal_unit_norm, in_axes=(0, None))(V, R)

        colors = jnp.array([[255, 255, 255], [174, 216, 204], [205, 102, 136],
                            [122, 49, 111], [70, 25, 89]]) / 255.0

        @jit
        def color_interp(val):
            idx = jnp.int32(val // 0.25)
            t = val % 0.25 / 0.25

            c0 = colors[idx]
            c1 = colors[idx + 1]

            return (1 - t) * c0 + t * c1

        V = V * (1 + poly_val)[:, None]

        m = pymeshlab.Mesh(V)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "mesh")
        ms.generate_surface_reconstruction_ball_pivoting()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=5000)
        ms.save_current_mesh(save_path)
        V, F = igl.read_triangle_mesh(save_path)

        poly_val = vmap(oct_polynomial_zonal_unit_norm, in_axes=(0, None))(V, R)
        color_scale = poly_val / poly_val.max() - 1e-2
        VC = vmap(color_interp)(color_scale)

        # Normalize w.r.t. y
        V_aabb_max = V.max(0, keepdims=True)
        V_aabb_min = V.min(0, keepdims=True)
        V_center = 0.5 * (V_aabb_max + V_aabb_min)
        V -= V_center
        scale = (V_aabb_max[0, 1] - V_center[0, 1]) / 0.95
        V /= scale

        write_triangle_mesh_VC(save_path, V, F, VC)
