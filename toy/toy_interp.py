import bpy

from common import vis_oct_field
from sh_representation import proj_sh4_sdp, proj_sh4_to_R3, R3_to_sh4_zonal

import blendertoolbox as bt

import os
import igl
import jax
from jax import jit, numpy as jnp, vmap
import numpy as np

from icecream import ic
import polyscope as ps


render_folder = os.path.expandvars("$HOME/dataset/rendering/toy/")

data = np.load("toy_data.npz")
V = data["V"]
R_end = data["R"]

R_begin = np.repeat(np.eye(3)[None, ...], len(R_end), axis=0)
q_begin = vmap(R3_to_sh4_zonal)(R_begin)
q_end = vmap(R3_to_sh4_zonal)(R_end)
V_gt, F_gt = igl.read_triangle_mesh("data/mesh/polyhedral.obj")

cam_pos = np.array(
    [
        [0.001797, 1.8808, 0.13566],
        [0.498648, 0.487685, 0.13566],
        [0.48795, -0.517896, 0.13566],
        [-0.859957, 0.498383, 0.13566],
        [-0.859957, -0.988594, 0.13566],
    ]
)


def render_optix(save_file_path):
    bpy.data.scenes["Scene"].render.filepath = save_file_path
    bpy.ops.wm.save_mainfile(filepath="tmp.blend")

    cmd = "blender -b tmp.blend -f 0 -- --cycles-device OPTIX"
    os.system(cmd)
    os.system(f"mv {save_file_path}0000.png {save_file_path}")
    os.system("rm tmp.blend")


interval = 1 / 60.0

counter = 0
for frac in np.arange(0, 1 + interval, interval):
    bpy.ops.wm.open_mainfile(filepath="octa.blend")
    bpy.data.objects["Camera"].select_set(False)
    bpy.data.objects["octa"].select_set(True)
    bpy.ops.object.delete()

    q = frac * q_end + (1.0 - frac) * q_begin
    q = proj_sh4_sdp(q)
    R = proj_sh4_to_R3(q)

    V_vis_octa, F_vis_octa = vis_oct_field(R, V, 0.15 / 84)

    mesh = bt.readNumpyMesh(V_vis_octa, F_vis_octa, [0, 0, 0], [90, 0, 0], [1, 1, 1])
    mesh.active_material = bpy.data.materials["octa"]

    gp = bpy.data.grease_pencils.new("LineArtGP")
    gp_obj = bpy.data.objects.new("LineArtObject", gp)
    bpy.context.collection.objects.link(gp_obj)

    gp_obj.select_set(True)
    bpy.context.view_layer.objects.active = gp_obj

    layer = gp.layers.new(name="LineArtLayer", set_active=True)
    mod = gp_obj.grease_pencil_modifiers.new(name="LineArt", type="GP_LINEART")
    mod.source_type = "OBJECT"
    mod.source_object = bpy.data.objects["numpy mesh object"]

    mod.thickness = 1
    mod.target_layer = "LineArtLayer"

    mat = bpy.data.materials.new(name="line")
    bpy.data.materials.create_gpencil_data(mat)
    mat.grease_pencil.color = (0.619371, 0.006397, 1.0, 1)
    gp_obj.data.materials.append(mat)

    mod.target_material = mat

    ## Circle
    gp = bpy.data.grease_pencils.new("LineArtGP")
    gp_obj = bpy.data.objects.new("LineArtObject", gp)
    bpy.context.collection.objects.link(gp_obj)

    gp_obj.select_set(True)
    bpy.context.view_layer.objects.active = gp_obj

    layer = gp.layers.new(name="LineArtLayer", set_active=True)
    mod = gp_obj.grease_pencil_modifiers.new(name="LineArt", type="GP_LINEART")
    mod.source_type = "OBJECT"
    mod.source_object = bpy.data.objects["Circle"]

    mod.thickness = 1
    mod.target_layer = "LineArtLayer"

    mat = bpy.data.materials.new(name="line")
    bpy.data.materials.create_gpencil_data(mat)
    mat.grease_pencil.color = (0, 0, 0, 1)
    gp_obj.data.materials.append(mat)

    mod.target_material = mat

    tag = str(counter).zfill(3)
    for i in range(5):
        cam = bpy.data.objects["Camera"]
        cam.location = cam_pos[i]

        save_path = os.path.join(render_folder, f"pose_{i}_{tag}.png")
        render_optix(save_path)

    # ps.init()
    # ps.register_surface_mesh("gt", V_gt, F_gt)
    # ps.register_surface_mesh("octa", V_vis_octa, F_vis_octa)
    # ps.show()

    counter += 1
