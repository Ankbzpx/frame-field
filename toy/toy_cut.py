import os
import igl

import numpy as np

import polyscope as ps
from icecream import ic


# V, F = igl.read_triangle_mesh("output/octa_hessian/polyhedral_50000.obj")
V, F = igl.read_triangle_mesh("data/mesh/polyhedral.obj")

roi = {
    "105": np.array([0.49, 0, 0.52]),
    "75": np.array([-0.86, 0, 0.99]),
    "90": np.array([-0.86, 0, -0.5]),
    "30": np.array([0, 0, -1.88]),
    "150": np.array([0.5, 0, -0.49]),
}

scale = 0.05
octa_res = 16
axis = np.linspace(-1, 1, octa_res)
xyz = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
xyz = np.stack([xyz[:, 0], np.zeros_like(xyz[:, 0]), xyz[:, 1]], axis=-1)

octa_samples = []
for offset in roi.values():
    ic(offset)
    octa_samples.append(scale * xyz + offset[None, :])
octa_samples = np.vstack(octa_samples)

ps.init()
# ps.register_point_cloud("octa_samples", octa_samples)
# ps.register_point_cloud("roi", np.vstack(list(roi.values())))

for tag, offset in roi.items():
    ps.register_point_cloud(tag, scale * xyz + offset[None, :])
ps.register_surface_mesh("recon", V, F)
ps.show()

np.save("octa_samples.npy", octa_samples)
