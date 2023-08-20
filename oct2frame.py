import jax
from jax import numpy as jnp, vmap, jit
import numpy as np

import igl

from icecream import ic
import polyscope as ps

from common import vis_oct_field
import flow_lines

from practical_3d_frame_field_generation import R3_to_repvec


@jit
def oct_grad(q, x, y, z):
    dx = jnp.array([
        2. * jnp.pi**(-1 / 2) * x * (x**2 + y**2 + z**2),
        (-3 / 4) * (35. * jnp.pi**(-1))**(1 / 2) * y * ((-3) * x**2 + y**2),
        (9 / 2) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * x * y * z, (-3 / 4) *
        (5. * jnp.pi**(-1))**(1 / 2) * y * (3. * x**2 + y**2 + (-6) * z**2),
        (-9 / 2) * ((5 / 2) * jnp.pi**(-1))**(1 / 2) * x * y * z,
        (9 / 4) * jnp.pi**(-1 / 2) * x * (x**2 + y**2 + (-4) * z**2), (3 / 4) *
        ((5 / 2) * jnp.pi**(-1))**(1 / 2) * z * ((-9) * x**2 +
                                                 (-3) * y**2 + 4. * z**2),
        (-3 / 2) * (5. * jnp.pi**(-1))**(1 / 2) * x * (x**2 + (-3) * z**2),
        (9 / 4) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * (x**2 + (-1) * y**2) * z,
        (3 / 4) * (35. * jnp.pi**(-1))**(1 / 2) * (x**3 + (-3) * x * y**2)
    ])

    dy = jnp.array([
        2. * jnp.pi**(-1 / 2) * y * (x**2 + y**2 + z**2),
        (3 / 4) * (35. * jnp.pi**(-1))**(1 / 2) * x * (x**2 + (-3) * y**2),
        (9 / 4) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * (x**2 + (-1) * y**2) * z,
        (-3 / 4) *
        (5. * jnp.pi**(-1))**(1 / 2) * x * (x**2 + 3. * y**2 + (-6) * z**2),
        (3 / 4) * ((5 / 2) * jnp.pi**(-1))**(1 / 2) * z *
        ((-3) * x**2 + (-9) * y**2 + 4. * z**2),
        (9 / 4) * jnp.pi**(-1 / 2) * y * (x**2 + y**2 + (-4) * z**2),
        (-9 / 2) * ((5 / 2) * jnp.pi**(-1))**(1 / 2) * x * y * z,
        (3 / 2) * (5. * jnp.pi**(-1))**(1 / 2) * y * (y**2 + (-3) * z**2),
        (-9 / 2) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * x * y * z, (-3 / 4) *
        (35. * jnp.pi**(-1))**(1 / 2) * (3. * x**2. * y + (-1) * y**3)
    ])

    dz = jnp.array([
        2. * jnp.pi**(-1 / 2) * z * (x**2 + y**2 + z**2), 
        0,
        (-3 / 4) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * y *
        ((-3) * x**2 + y**2), 9. * (5. * jnp.pi**(-1))**(1 / 2) * x * y * z,
        (-9 / 4) * ((5 / 2) * jnp.pi**(-1))**(1 / 2) * y * (x**2 + y**2 +
                                                            (-4) * z**2),
        jnp.pi**(-1 / 2) * ((-9) * x**2. * z + (-9) * y**2. * z + 6. * z**3),
        (-9 / 4) *
        ((5 / 2) * jnp.pi**(-1))**(1 / 2) * x * (x**2 + y**2 + (-4) * z**2),
        (9 / 2) * (5. * jnp.pi**(-1))**(1 / 2) * (x**2 + (-1) * y**2) * z,
        (3 / 4) * ((35 / 2) * jnp.pi**(-1))**(1 / 2) * x * (x**2 + (-3) * y**2),
        0
    ])

    return jnp.array([dx @ q, dy @ q, dz @ q])


sh9 = np.load('sh9.npy')
sh9 = np.hstack([(np.sqrt(189) / 4. * np.ones(len(sh9)))[:, None], sh9])

np.random.seed(0)
v1 = np.random.randn(len(sh9), 3)
v2 = np.random.randn(len(sh9), 3)
delta = 1

V, F = igl.read_triangle_mesh('data/mesh/rocker_arm.ply')

thr = 1e-6**0.9 * jnp.sqrt(len(sh9))

while delta > thr:
    w1 = vmap(oct_grad)(sh9, v1[:, 0], v1[:, 1], v1[:, 2])
    w2 = vmap(oct_grad)(sh9, v2[:, 0], v2[:, 1], v2[:, 2])

    norm = jnp.linalg.norm(w1, ord=2, axis=1, keepdims=True)
    v1_new = w1 / norm
    delta = jnp.linalg.norm(v1 - v1_new)
    v1 = v1_new
    v2 = w2 - jnp.einsum('ni,ni->n', w2, v1)[:, None] * v1
    v2 = v2 / jnp.linalg.norm(v2, ord=2, axis=1, keepdims=True)

v3 = jnp.cross(v1, v2, axis=1)

Rs = jnp.stack([v1, v2, v3], -1)
VN = igl.per_vertex_normals(V, F)
Q = vmap(R3_to_repvec)(Rs, VN)

V_vis, F_vis, VC_vis = flow_lines.trace(V,
                                        F,
                                        VN,
                                        Q,
                                        4000,
                                        length_factor=5,
                                        interval_factor=10,
                                        width_factor=0.075)

ps.init()
ps.register_surface_mesh('rocket arm', V, F)
flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
ps.show()