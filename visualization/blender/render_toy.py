import blendertoolbox as bt
import bpy
import numpy as np
import os
from icecream import ic

from render_p2s import load_template

input_path = os.path.expandvars('$HOME/frame-field/data/toy/')
viz_path = os.path.expandvars('$HOME/frame-field/output/toy/')

location = (0, 0, 0)
rotation = (0, 0, 0)
scale = (1, 1, 1)


def eulerXYZ_to_R3(a, b, c):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                   [0, np.sin(a), np.cos(a)]])

    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0],
                   [-np.sin(b), 0, np.cos(b)]])

    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c),
                                                np.cos(c), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# Compensate small rotation
R = eulerXYZ_to_R3(np.pi / 6, np.pi / 3, np.pi / 4)

for angle in [30, 90, 120]:
    data = np.load(os.path.join(input_path, f'crease_4_{angle}.npz'))
    V = data['samples_sup']
    V = np.float64(V @ R)

    load_template()
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
    MESH2POINT.inputs['Radius'].default_value = 0.020
    MATERIAL = tree.nodes.new('GeometryNodeSetMaterial')

    tree.links.new(IN.outputs['Geometry'], MESH2POINT.inputs['Mesh'])
    tree.links.new(MESH2POINT.outputs['Points'], MATERIAL.inputs['Geometry'])
    tree.links.new(MATERIAL.outputs['Geometry'], OUT.inputs['Geometry'])

    MATERIAL.inputs[2].default_value = mat

    bpy.ops.wm.save_mainfile(filepath=f"{angle}.blend")
