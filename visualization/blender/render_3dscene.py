from glob import glob
import json
import os

import blendertoolbox as bt
import bpy
import numpy as np
from render_p2s import render_mesh, render_pointcloud

from icecream import ic


if __name__ == "__main__":
    methods = ["CapUDF", "ours", "S2DF"]

    models = [
        "burghers_1000",
        "copyroom_1000",
        "lounge_1000",
        "stonewall_1000",
        "totempole_1000",
    ]
    root_folder = os.path.expandvars("$HOME/dataset/3d_scene")
    render_folder = os.path.join(root_folder, "render")

    with open("3dscene_poses.json") as f:
        poses = json.load(f)

    for model in models:
        save_folder = os.path.join(render_folder, model)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        pose = poses[model]
        location = pose["location"]
        rotation = list(map(lambda x: np.rad2deg(x), pose["rotation_euler"]))
        scale = pose["scale"]

        input_path = os.path.join(root_folder, "scans", f"{model}.ply")
        input_save_path = os.path.join(save_folder, "input.png")
        if not os.path.exists(input_save_path):
            render_pointcloud(input_save_path, input_path, location, rotation, scale)

        for method in methods:
            model_path = glob(os.path.join(root_folder, method, f"{model}.*"))[0]
            model_save_path = os.path.join(save_folder, f"{method}.png")
            if not os.path.exists(model_save_path):
                render_mesh(
                    model_save_path,
                    model_path,
                    location,
                    rotation,
                    scale,
                    face_normals=True,
                )
