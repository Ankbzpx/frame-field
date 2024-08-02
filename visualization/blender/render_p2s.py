import blendertoolbox as bt
import bpy
import os
import json

from glob import glob
import numpy as np
import open3d as o3d

from icecream import ic


def load_template():
    bpy.ops.wm.open_mainfile(filepath="template.blend")
    bpy.data.objects['Camera'].select_set(False)
    bpy.data.objects['spot'].select_set(True)
    bpy.ops.object.delete()


def render_optix(save_file_path):
    bpy.data.scenes['Scene'].render.filepath = save_file_path
    bpy.ops.wm.save_mainfile(filepath='tmp.blend')

    cmd = 'blender -b tmp.blend -f 0 -- --cycles-device OPTIX'
    os.system(cmd)
    os.system(f'mv {save_file_path}0000.png {save_file_path}')
    os.system('rm tmp.blend')


def render_pointcloud(save_path,
                      pc_path,
                      location,
                      rotation,
                      scale,
                      radius=3e-3):
    load_template()
    pc_o3d = o3d.io.read_point_cloud(pc_path)
    V = np.array(pc_o3d.points)
    mesh = bt.readNumpyPoints(V, location, rotation, scale)
    mat = bpy.data.materials['MeshMaterial']

    # turn a mesh into point cloud using geometry node
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    bpy.ops.object.modifier_add(type='NODES')
    bpy.ops.node.new_geometry_nodes_modifier()
    tree = mesh.modifiers[-1].node_group

    IN = tree.nodes['Group Input']
    OUT = tree.nodes['Group Output']
    MESH2POINT = tree.nodes.new('GeometryNodeMeshToPoints')
    MESH2POINT.location.x -= 100
    MESH2POINT.inputs['Radius'].default_value = radius
    MATERIAL = tree.nodes.new('GeometryNodeSetMaterial')

    tree.links.new(IN.outputs['Geometry'], MESH2POINT.inputs['Mesh'])
    tree.links.new(MESH2POINT.outputs['Points'], MATERIAL.inputs['Geometry'])
    tree.links.new(MATERIAL.outputs['Geometry'], OUT.inputs['Geometry'])

    # assign the material to point cloud
    MATERIAL.inputs[2].default_value = mat

    render_optix(save_path)


def render_mesh(save_path,
                mesh_path,
                location,
                rotation,
                scale,
                face_normals=False):
    load_template()
    mesh = bt.readMesh(mesh_path, location, rotation, scale)

    if face_normals:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.set_normals_from_faces()
            bpy.ops.mesh.faces_shade_flat()
            bpy.ops.object.mode_set(mode="OBJECT")
        mesh.select_set(False)
        bpy.context.view_layer.objects.active = None

    mesh.active_material = bpy.data.materials['MeshMaterial']
    render_optix(save_path)


# Set face_normals to true for DiGS / NSH to avoid unlit faces
def render_mesh_numpy(save_path,
                      mesh_path,
                      location,
                      rotation,
                      scale,
                      face_normals=False):
    load_template()
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    V = np.asarray(mesh_o3d.vertices)
    F = np.asarray(mesh_o3d.triangles)
    mesh = bt.readNumpyMesh(V, F, location, rotation, scale)

    if face_normals:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.set_normals_from_faces()
            bpy.ops.mesh.faces_shade_flat()
            bpy.ops.object.mode_set(mode="OBJECT")
        mesh.select_set(False)
        bpy.context.view_layer.objects.active = None

    mesh.active_material = bpy.data.materials['MeshMaterial']
    render_optix(save_path)


if __name__ == '__main__':
    methods = [
        'APSS', 'digs', 'EAR_viz', 'line_processing_viz',
        'neural_singular_hessian', 'nksr', 'nksr_ks', 'ours_digs_5',
        'ours_digs_10', 'ours_hessian_5', 'ours_hessian_10',
        'graph_laplacian_viz', 'siren', 'SPR'
    ]

    root_folder = os.path.expandvars('$HOME/dataset')

    # re_render_model_list = [
    #     '00010218_4769314c71814669ba5d3512', '00990546_db31ddca9d3585c330dcce3a'
    # ]
    re_render_model_list = []
    # re_render_method_list = ['line_processing_viz', 'graph_laplacian_viz']
    re_render_method_list = []

    failure_case_list = []

    for dataset in ['abc', 'thingi10k']:
        pick_path = f'{dataset}_pick.txt'
        with open(pick_path, 'r') as f:
            model_list = f.read().splitlines()

        with open(f'{dataset}_poses.json') as f:
            poses = json.load(f)

        for noise_level in ['1e-2', '2e-3']:
            tag = f"{dataset}_{noise_level}"

            for model_name in model_list:
                render_model = (model_name in re_render_model_list)

                save_folder = os.path.join(root_folder, 'p2s_render', tag,
                                           model_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                pose = poses[model_name]
                location = pose['location']
                rotation = list(
                    map(lambda x: np.rad2deg(x), pose['rotation_euler']))
                scale = pose['scale']

                # GT
                gt_path = os.path.join(root_folder, 'p2s', dataset, 'gt',
                                       f"{model_name}.ply")
                gt_save_path = os.path.join(save_folder, 'gt.png')

                if render_model or not os.path.exists(gt_save_path):
                    try:
                        render_mesh_numpy(gt_save_path,
                                          gt_path,
                                          location,
                                          rotation,
                                          scale,
                                          face_normals=False)
                    except:
                        failure_case_list.append(f'{model_name}_gt')

                # input
                input_path = os.path.join(root_folder, 'p2s', dataset,
                                          noise_level, f"{model_name}.ply")
                input_save_path = os.path.join(save_folder, 'input.png')
                if render_model or not os.path.exists(input_save_path):
                    try:
                        render_pointcloud(input_save_path, input_path, location,
                                          rotation, scale)
                    except:
                        failure_case_list.append(f'{model_name}_{tag}_input')

                for method in methods:
                    render_method = render_model or (method
                                                     in re_render_method_list)

                    model_path = glob(
                        os.path.join(root_folder, 'octa_results', method, tag,
                                     f'{model_name}.*'))[0]
                    model_save_path = os.path.join(save_folder, f'{method}.png')
                    if render_method or not os.path.exists(model_save_path):
                        try:
                            render_mesh_numpy(model_save_path,
                                              model_path,
                                              location,
                                              rotation,
                                              scale,
                                              face_normals=True)
                        except:
                            failure_case_list.append(
                                f'{model_name}_{tag}_{method}')

    print(failure_case_list)
