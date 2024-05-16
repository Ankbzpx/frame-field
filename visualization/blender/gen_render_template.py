import blendertoolbox as bt
import bpy
from glob import glob
import os
from tqdm import tqdm

from icecream import ic

root_folder = os.path.expandvars('$HOME/dataset')
for dataset in ['abc', 'thingi10k']:
    model_folder = os.path.join(root_folder, 'p2s', dataset, 'gt')
    model_path_list = glob(os.path.join(model_folder, '*.ply'))

    for model_path in tqdm(model_path_list):
        model_name = model_path.split('/')[-1].split('.')[0]
        print(model_path)
        save_path = '/'.join(model_path.split('/')[:-2])
        save_path = save_path.replace('p2s', 'p2s_render_template')
        save_file_path = os.path.join(save_path, f"{model_name}.blend")

        if os.path.exists(save_file_path):
            continue

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        bpy.ops.wm.open_mainfile(filepath="template.blend")

        bpy.data.objects['Camera'].select_set(False)
        bpy.data.objects['spot'].select_set(True)
        bpy.ops.object.delete()

        location = (1.12, -0.14, 0)
        rotation = (90, 0, 227)
        scale = (1.25, 1.25, 1.25)

        bt.readMesh(model_path, location, rotation, scale)
        bpy.data.objects[model_name].active_material = bpy.data.materials[
            'MeshMaterial']

        bpy.ops.wm.save_mainfile(filepath=save_file_path)
