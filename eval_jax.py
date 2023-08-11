import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
from skimage.measure import marching_cubes
import igl
import argparse
import json

import model_jax
from config import Config
from common import normalize, vis_oct_field, rm_unref_vertices
from practical_3d_frame_field_generation import proj_sh4_to_rotvec, rotvec_to_R3, R3_to_repvec
import flow_lines
import pyfqmr
import open3d as o3d

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--mc',
                        action='store_true',
                        help='Visualize MC mesh only')
    parser.add_argument('--fl', action='store_true', help='Visualize flowline')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    model = getattr(model_jax, cfg.mlp_type)(**cfg.mlp_cfg,
                                             key=jax.random.PRNGKey(0))
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

    if args.mc:
        ps.init()
        ps.register_surface_mesh(f"{cfg.name}", V, F)
        ps.show()
        exit()

    # Simplify mesh
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(V, F)
    mesh_simplifier.simplify_mesh(target_count=10000,
                                  aggressiveness=7,
                                  preserve_border=True)
    V, F, _ = mesh_simplifier.getMesh()

    # Project on isosurface
    (sdf, _), VN = infer_grad(V)
    V = V - sdf[:, None] * vmap(normalize)(VN)
    V = np.array(V)

    (_, sh9), VN = infer_grad(V)
    rotvecs = vmap(proj_sh4_to_rotvec)(sh9)
    Rs = vmap(rotvec_to_R3)(rotvecs)

    Q = vmap(R3_to_repvec)(Rs, VN)

    V_vis, F_vis, VC_vis = flow_lines.trace(V,
                                            F,
                                            VN,
                                            Q,
                                            4000,
                                            length_factor=5,
                                            interval_factor=10,
                                            width_factor=0.075)

    if args.fl:
        ps.init()
        mesh = ps.register_surface_mesh("mesh", V, F)
        mesh.add_vector_quantity("VN", VN)
        flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
        flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
        ps.show()

    igl.write_triangle_mesh(f"output/{cfg.name}_mc.obj", V, F)

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(f"output/{cfg.name}_stroke.obj", stroke_mesh)
