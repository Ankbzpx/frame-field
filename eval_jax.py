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
from config_utils import config_latent, config_model
from common import normalize, vis_oct_field, filter_components
from sh_representation import proj_sh4_to_R3, R3_to_repvec, rotvec_n_to_z, rotvec_to_R3, \
    rotvec_to_R9, project_n, rot6d_to_R3, R3_to_sh4_zonal
import flow_lines
import open3d as o3d
import pymeshlab

import polyscope as ps
from icecream import ic

import time


# infer: R^3 -> R (sdf)
def extract_surface(infer, grid_res=512, grid_min=-1.0, grid_max=1.0):
    # Smaller batch is somehow faster
    group_size = 4 * grid_res**2
    iter_size = grid_res**3 // group_size

    @jit
    def infer_sdf():
        indices = jnp.linspace(grid_min, grid_max, grid_res)
        grid = jnp.stack(jnp.meshgrid(indices, indices, indices),
                         -1).reshape(iter_size, group_size, 3)

        query_data = {"grid": grid, "sdf": jnp.zeros((iter_size, group_size))}

        @jit
        def body_func(i, query_data):
            sdf = infer(query_data["grid"][i])
            query_data["sdf"] = query_data["sdf"].at[i].set(sdf)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["sdf"].reshape(grid_res, grid_res, grid_res)

    sdf = infer_sdf()
    # This step is surprising slow, memory copy?
    sdf_np = np.swapaxes(np.array(sdf), 0, 1)

    spacing = 1. / (grid_res - 1)
    # It outputs inverse VN, even with gradient_direction set to ascent
    V, F, VN_inv, _ = marching_cubes(sdf_np,
                                     0.,
                                     spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)

    return V, F, -VN_inv


# Reduce face count to speed up visualization
# TODO: Use edge collapsing like one in Instant meshes
def meshlab_remesh(V, F):
    m = pymeshlab.Mesh(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=len(F) // 10)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(1.0))

    # I believe there is no other option to pass meshlab mesh back to python
    ms.save_current_mesh(f'tmp/{cfg.name}.obj')
    V, F = igl.read_triangle_mesh(f'tmp/{cfg.name}.obj')
    os.system(f'rm tmp/{cfg.name}.obj')
    return V, F


def eval(cfg: Config, interp, out_dir, vis_mc=False, vis_flowline=False):

    latents, latent_dim = config_latent(cfg)
    tokens = interp.split('_')
    # Interpolate latent
    i = int(tokens[0])
    j = int(tokens[1])
    t = float(tokens[2])
    latent = (1 - t) * latents[i] + t * latents[j]

    model_key = jax.random.PRNGKey(0)
    model = config_model(cfg, model_key, latent_dim)
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{cfg.name}.eqx", model)

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    @jit
    def infer_grad(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model.call_grad(x, z)

    start_time = time.time()
    V, F, VN = extract_surface(infer)
    print("Extract surface", time.time() - start_time)

    start_time = time.time()
    V, F = filter_components(V, F, VN)
    print("Filter VN", time.time() - start_time)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    interp_tag = "" if len(cfg.sdf_paths) == 1 else f"{interp}_"
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_{interp_tag}mc.obj", V, F)

    if vis_mc:
        # TODO support latent
        sdf_data = dict(np.load(cfg.sdf_paths[0]))
        sur_sample = sdf_data['samples_on_sur']
        sur_normal = sdf_data['normals_on_sur']
        (_, aux), _ = infer_grad(sur_sample)

        if cfg.loss_cfg.rot6d:
            Rs = vmap(rot6d_to_R3)(aux[:, :6])
        else:
            sh4 = aux[:, :9]
            Rs = proj_sh4_to_R3(sh4)

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, sur_sample, 0.005)

        _, VN = infer_grad(V)

        ps.init()
        mesh = ps.register_surface_mesh(f"{cfg.name}", V, F)
        mesh.add_vector_quantity('VN', VN)
        ps.register_surface_mesh('Oct frames supervise', V_vis_sup, F_vis_sup)
        pc = ps.register_point_cloud('sur_sample', sur_sample, radius=1e-4)
        pc.add_vector_quantity('sur_normal', sur_normal, enabled=True)
        ps.show()
        exit()

    start_time = time.time()
    V, F = meshlab_remesh(V, F)
    print("Meshlab Remesh", time.time() - start_time)

    start_time = time.time()
    # Project on isosurface
    (sdf, _), VN = infer_grad(V)
    VN = vmap(normalize)(VN)
    V = V - sdf[:, None] * VN
    V = np.array(V)

    print("Project SDF", time.time() - start_time)
    start_time = time.time()

    (_, aux), VN = infer_grad(V)

    if cfg.loss_cfg.rot6d:
        Rs = vmap(rot6d_to_R3)(aux[:, :6])
        sh4 = vmap(R3_to_sh4_zonal)(Rs)
    else:
        sh4 = aux[:, :9]
        Rs = proj_sh4_to_R3(sh4)

    print("Project SO(3)", time.time() - start_time)

    print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")

    sh4 = vmap(normalize)(sh4)
    L = igl.cotmatrix(V, F)
    smoothness = np.trace(sh4.T @ -L @ sh4)
    print(f"Smoothness {smoothness}")

    start_time = time.time()

    Q = vmap(R3_to_repvec)(Rs, VN)
    V_vis, F_vis, VC_vis = flow_lines.trace(V, F, VN, Q, 4000)
    print("Trace flowlines", time.time() - start_time)

    if vis_flowline:
        ps.init()
        mesh = ps.register_surface_mesh("mesh", V, F)
        mesh.add_vector_quantity("VN", VN)
        flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
        flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
        ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(f"{out_dir}/{cfg.name}_{interp_tag}stroke.obj",
                               stroke_mesh)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--interp',
                        type=str,
                        default='0_1_0',
                        help='Interpolation progress')
    parser.add_argument('--vis_mc',
                        action='store_true',
                        help='Visualize MC mesh only')
    parser.add_argument('--vis_flowline',
                        action='store_true',
                        help='Visualize flowline')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    eval(cfg, args.interp, "output", args.vis_mc, args.vis_flowline)
