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
from common import normalize, vis_oct_field, filter_components
from sh_representation import proj_sh4_to_R3, R3_to_repvec, rotvec_n_to_z, rotvec_to_R3, \
    rotvec_to_R9, project_n
import flow_lines
import open3d as o3d
import pymeshlab

import polyscope as ps
from icecream import ic

import time


def eval(cfg: Config,
         interp,
         out_dir,
         vis_mc=False,
         project_vn=False,
         vis_cube=False,
         vis_flowline=False):
    model_key = jax.random.PRNGKey(0)

    n_models = len(cfg.sdf_paths)
    # preload data in memory to speedup training
    assert n_models > 0

    # Latent
    # Compute (n_models - 1) dim simplex vertices as latent
    # Reference: https://mathoverflow.net/a/184585
    q, _ = jnp.linalg.qr(jnp.ones((n_models, 1)), mode="complete")
    latents = q[:, 1:]
    latent_dim = (n_models - 1)

    tokens = interp.split('_')
    # Interpolate latent
    i = int(tokens[0])
    j = int(tokens[1])
    t = float(tokens[2])
    latent = (1 - t) * latents[i] + t * latents[j]

    if len(cfg.mlp_types) == 1:
        cfg.mlps[0].in_features += latent_dim
        model: model_jax.MLP = getattr(model_jax,
                                       cfg.mlp_types[0])(**cfg.mlp_cfgs[0],
                                                         key=model_key)
    else:
        if cfg.conditioning:
            # Do NOT modify config inside eqx.Module, because the __init__ will be called twice
            cfg.mlps[1].in_features += cfg.mlps[0].in_features
            MultiMLP = model_jax.MLPComposerCondition
        else:
            MultiMLP = model_jax.MLPComposer

        for mlp_cfg in cfg.mlps:
            mlp_cfg.in_features += latent_dim

        model: model_jax.MLP = MultiMLP(
            model_key,
            cfg.mlp_types,
            cfg.mlp_cfgs,
        )

    model = eqx.tree_deserialise_leaves(f"checkpoints/{cfg.name}.eqx", model)

    grid_res = 512
    grid_min = -1.0
    grid_max = 1.0
    # Smaller batch is somehow faster
    group_size = 4 * grid_res**2
    iter_size = grid_res**3 // group_size

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    @jit
    def infer_grad(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model.call_grad(x, z)

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

    start_time = time.time()
    sdf = infer_sdf()
    print("Inference SDF", time.time() - start_time)
    start_time = time.time()

    # This step is unfortunately slow
    sdf_np = np.swapaxes(np.array(sdf), 0, 1)

    spacing = 1. / (grid_res - 1)
    V, F, _, _ = marching_cubes(sdf_np, 0., spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)
    V, F = filter_components(V, F)

    if vis_mc:
        ps.init()
        ps.register_surface_mesh(f"{cfg.name}", V, F)
        ps.show()
        exit()

    m = pymeshlab.Mesh(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=len(F) // 10)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(1.0))

    # I believe there is no other option to pass meshlab mesh back to python
    ms.save_current_mesh(f'tmp/{cfg.name}.obj')
    V, F = igl.read_triangle_mesh(f'tmp/{cfg.name}.obj')
    os.system(f'rm tmp/{cfg.name}.obj')

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
    sh4 = aux[:, :9]

    print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")

    sh4 = vmap(normalize)(sh4)
    L = igl.cotmatrix(V, F)
    smoothness = np.trace(sh4.T @ -L @ sh4)
    print(f"Smoothness {smoothness}")

    def R_from_sh4(sh4):
        if project_vn:
            R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(VN))
            sh4 = vmap(project_n)(sh4.reshape(len(V), 9), R9_zn)

        return proj_sh4_to_R3(sh4)

    if cfg.loss_cfg.rot:
        Rs = vmap(rotvec_to_R3)(aux[:, 9:])
    else:
        Rs = R_from_sh4(sh4)

    print("Project SO(3)", time.time() - start_time)

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

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    interp_tag = "" if n_models == 0 else interp
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_{interp_tag}_mc.obj", V, F)

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(f"{out_dir}/{cfg.name}_{interp_tag}_stroke.obj",
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

    eval(cfg, args.interp, "output", args.vis_mc, args.project_vn,
         args.vis_cube, args.vis_flowline)
