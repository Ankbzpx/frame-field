import numpy as np
from jax import numpy as jnp, vmap, jit

from sh_representation import oct_polynomial
import math
import pymeshlab
import igl
import open3d as o3d

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

    save_path = 'tmp/octa.obj'

    R = np.eye(3)
    V = fibonacci_sphere(100000)
    poly_val = vmap(oct_polynomial, in_axes=(0, None))(V, R)

    colors = jnp.array([[255, 255, 255], [174, 216, 204], [205, 102, 136],
                        [122, 49, 111], [70, 25, 89]]) / 255.0

    @jit
    def color_interp(val):
        idx = jnp.int32(val // 0.25)
        t = val % 0.25 / 0.25

        c0 = colors[idx]
        c1 = colors[idx + 1]

        return (1 - t) * c0 + t * c1

    VC = vmap(color_interp)(poly_val)
    V = V * (1 + poly_val)[:, None]

    m = pymeshlab.Mesh(V)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.save_current_mesh(save_path)
    V, F = igl.read_triangle_mesh(save_path)

    octa_mesh = o3d.geometry.TriangleMesh()
    octa_mesh.vertices = o3d.utility.Vector3dVector(V)
    octa_mesh.triangles = o3d.utility.Vector3iVector(F)
    octa_mesh.vertex_colors = o3d.utility.Vector3dVector(VC)
    o3d.io.write_triangle_mesh(save_path, octa_mesh)
