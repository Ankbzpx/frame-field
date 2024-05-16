import bpy
import os
from glob import glob
import json
from icecream import ic

root_folder = os.path.expandvars('$HOME/dataset')
for dataset in ['abc', 'thingi10k']:
    poses = {}
    model_folder = os.path.join(root_folder, 'p2s', dataset, 'gt')
    model_path_list = glob(os.path.join(model_folder, '*.ply'))

    for model_path in model_path_list:
        model_name = model_path.split('/')[-1].split('.')[0]
        print(model_path)
        save_path = '/'.join(model_path.split('/')[:-2])
        save_path = save_path.replace('p2s', 'p2s_render_template')
        save_file_path = os.path.join(save_path, f"{model_name}.blend")

        bpy.ops.wm.open_mainfile(filepath=save_file_path)
        mesh = bpy.data.objects[model_name]

        pose = {
            'location': list(mesh.location),
            'rotation_euler': list(mesh.rotation_euler),
            'scale': list(mesh.scale)
        }
        poses[model_name] = pose

    with open(f'{dataset}_poses.json', 'w') as f:
        json.dump(poses, f)
