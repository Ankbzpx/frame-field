import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
from skimage.measure import marching_cubes
import igl
import argparse
import json
import os

import model_jax
from config import Config
from common import normalize, vis_oct_field, rm_unref_vertices
from sh_representation import proj_sh4_to_R3, R3_to_repvec, rotvec_n_to_z, rotvec_to_R3, \
    rotvec_to_R9, project_n
import flow_lines
import open3d as o3d
import pymeshlab

import polyscope as ps
from icecream import ic


def eval(cfg: Config,
         out_dir,
         vis_mc=False,
         project_vn=False,
         vis_cube=False,
         vis_flowline=False):
    model_key = jax.random.PRNGKey(0)
    if len(cfg.mlp_types) == 1:
        model: model_jax.MLP = getattr(model_jax,
                                       cfg.mlp_types[0])(**cfg.mlp_cfgs[0],
                                                         key=model_key)
    else:
        model: model_jax.MLP = model_jax.MLPComposer(model_key, cfg.mlp_types,
                                                     cfg.mlp_cfgs)

    model = eqx.tree_deserialise_leaves(f"checkpoints/{cfg.name}.eqx", model)

    grid_res = 512
    grid_min = -1.0
    grid_max = 1.0
    group_size = 1200000

    @jit
    def infer(x):
        return model(x)[:, 0]

    @jit
    def infer_grad(x):
        return model.call_grad(x)

    indices = np.linspace(grid_min, grid_max, grid_res)
    grid = np.stack(np.meshgrid(indices, indices, indices), -1).reshape(-1, 3)

    sdfs_list = []
    for x in np.array_split(grid, len(grid) // group_size, axis=0):
        sdf = infer(x)
        sdfs_list.append(np.array(sdf))
    sdfs = np.concatenate(sdfs_list).reshape(grid_res, grid_res, grid_res)
    sdfs = np.swapaxes(sdfs, 0, 1)

    spacing = 1. / (grid_res - 1)
    V, F, _, _ = marching_cubes(sdfs, 0., spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)

    A = igl.adjacency_matrix(F)
    (n_c, C, K) = igl.connected_components(A)

    if n_c > 1:
        VF, NI = igl.vertex_triangle_adjacency(F, F.max() + 1)

        V_filter = np.argwhere(C != np.argmax(K)).reshape(-1,)
        FV = np.split(VF, NI[1:-1])
        F_filter = np.unique(np.concatenate([FV[vid] for vid in V_filter]))
        F = np.delete(F, F_filter, axis=0)
        V, F = rm_unref_vertices(V, F)

    m = pymeshlab.Mesh(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=10000)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(2.0))

    # I believe there is no other option to pass meshlab mesh back to python
    ms.save_current_mesh(f'tmp/{cfg.name}.obj')
    V, F = igl.read_triangle_mesh(f'tmp/{cfg.name}.obj')
    os.system(f'rm tmp/{cfg.name}.obj')

    # Project on isosurface
    (sdf, _), VN = infer_grad(V)
    VN = vmap(normalize)(VN)
    V = V - sdf[:, None] * VN
    V = np.array(V)

    if vis_mc:
        ps.init()
        mesh_vis = ps.register_surface_mesh(f"{cfg.name}", V, F)
        mesh_vis.add_vector_quantity("VN", VN, enabled=True)
        ps.show()
        exit()

    (_, aux), VN = infer_grad(V)

    def R_from_sh4():
        sh4 = aux[:, :9]
        if project_vn:
            R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(VN))
            sh4 = vmap(project_n)(sh4.reshape(len(V), 9), R9_zn)

        return proj_sh4_to_R3(sh4)

    if cfg.loss_cfg.rot:
        Rs = vmap(rotvec_to_R3)(aux[:, 9:])
    else:
        Rs = R_from_sh4()

    if vis_cube:
        V_vis, F_vis = vis_oct_field(Rs, V, F)

        ps.init()
        mesh = ps.register_surface_mesh("mesh", V, F)
        mesh.add_vector_quantity("VN", VN)
        ps.register_surface_mesh("Oct frames", V_vis, F_vis)
        if cfg.loss_cfg.rot:
            Rs2 = R_from_sh4()
            V_vis2, F_vis2 = vis_oct_field(Rs2, V, F)
            ps.register_surface_mesh("Oct frames sh4", V_vis2, F_vis2)
        ps.show()
        exit()

    Q = vmap(R3_to_repvec)(Rs, VN)
    V_vis, F_vis, VC_vis = flow_lines.trace(V,
                                            F,
                                            VN,
                                            Q,
                                            4000,
                                            length_factor=5,
                                            interval_factor=10,
                                            width_factor=0.075)

    if vis_flowline:
        ps.init()
        mesh = ps.register_surface_mesh("mesh", V, F)
        mesh.add_vector_quantity("VN", VN)
        flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
        flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
        ps.show()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_mc.obj", V, F)

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(f"{out_dir}/{cfg.name}_stroke.obj", stroke_mesh)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--vis_mc',
                        action='store_true',
                        help='Visualize MC mesh only')
    parser.add_argument('--project_vn',
                        action='store_true',
                        help='Project sh4 to vertex normal')
    parser.add_argument('--vis_cube',
                        action='store_true',
                        help='Visualize cube')
    parser.add_argument('--vis_flowline',
                        action='store_true',
                        help='Visualize flowline')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    eval(cfg, "output", args.vis_mc, args.project_vn, args.vis_cube,
         args.vis_flowline)
