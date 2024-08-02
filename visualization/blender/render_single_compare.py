import blendertoolbox as bt
import bpy
import numpy as np
import os
import json
from glob import glob
from icecream import ic

from render_p2s import render_mesh_numpy, load_template

suffix = 500 * (1 + np.arange(20))
methods = ['hessian', 'digs']

root = os.path.expandvars('$HOME/dataset/octa_results/debug/')
save_path = os.path.expandvars('$HOME/dataset/debug_render')
model_name = '00010218_4769314c71814669ba5d3512'

with open(f'abc_poses.json') as f:
    poses = json.load(f)

pose = poses[model_name]
location = pose['location']
rotation = list(map(lambda x: np.rad2deg(x), pose['rotation_euler']))
scale = pose['scale']

for method in methods:
    save_folder = os.path.join(save_path, method)
    os.makedirs(save_folder, exist_ok=True)

    for tag in suffix:
        model_path = glob(os.path.join(root, method,
                                       f'{model_name}_{tag}.*'))[0]
        model_save_path = os.path.join(save_folder, f'{model_name}_{tag}.png')
        if not os.path.exists(model_save_path):
            render_mesh_numpy(model_save_path,
                              model_path,
                              location,
                              rotation,
                              scale,
                              face_normals=True)
