from render_p2s import render_mesh
import numpy as np
import os

from icecream import ic

pos = (1.10421, 0.114716, 1.02771)
rot = (93.4667, 1.34732, -52.0689)
scale = (0.017552, 0.017552, 0.017552)

folder_path = os.path.expandvars(
    '$HOME/frame-field/output/octa_hessian/debug_iters/')

suffix = 10 * (np.arange(1000) + 1)

for tag in suffix:
    out_tag = f"{tag}".zfill(5)
    mesh_path = os.path.join(folder_path, f"fandisk_scan_{tag}.obj")
    mesh_save_path = os.path.join(
        os.path.expandvars('$HOME/dataset/teaser_sequence/'),
        f"fandisk_{out_tag}.png")
    render_mesh(mesh_save_path, mesh_path, pos, rot, scale, face_normals=True)
