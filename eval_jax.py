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
from common import normalize, vis_oct_field, filter_components, Timer
from sh_representation import (proj_sh4_to_R3, proj_sh4_to_rotvec, R3_to_repvec,
                               rotvec_n_to_z, rotvec_to_R3, rotvec_to_R9,
                               project_n, rot6d_to_R3, R3_to_sh4_zonal,
                               rot6d_to_sh4_zonal)
import flow_lines
import open3d as o3d
import pymeshlab

import polyscope as ps
from icecream import ic


# infer: R^3 -> R
def voxel_infer(infer,
                grid_res=512,
                grid_min=-1.0,
                grid_max=1.0,
                group_size_mul=2):
    # Smaller batch is somehow faster
    group_size = group_size_mul * grid_res**2
    iter_size = grid_res**3 // group_size

    # Cannot pass jitted function as argument to another jitted function
    @jit
    def infer_scalar():
        indices = jnp.linspace(grid_min, grid_max, grid_res)
        grid = jnp.stack(jnp.meshgrid(indices, indices, indices), -1)

        query_data = {
            "grid": grid.reshape(iter_size, group_size, 3),
            "val": jnp.zeros((iter_size, group_size))
        }

        @jit
        def body_func(i, query_data):
            val = infer(query_data["grid"][i])
            query_data["val"] = query_data["val"].at[i].set(val)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["val"].reshape(grid_res, grid_res, grid_res), grid

    return infer_scalar()


# infer: R^3 -> R (sdf)
def extract_surface(infer, grid_res=512, grid_min=-1.0, grid_max=1.0):

    sdf, _ = voxel_infer(infer, grid_res, grid_min, grid_max, 4)
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
def meshlab_remesh(cfg, V, F):
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


def eval(cfg: Config,
         out_dir,
         model,
         latent,
         vis_mc=False,
         vis_smooth=False,
         vis_flowline=False,
         geo_only=False,
         interp_tag=''):

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    @jit
    def infer_grad(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model.call_grad(x, z)

    if cfg.loss_cfg.rot6d:
        param_func = rot6d_to_sh4_zonal
    else:
        param_func = lambda x: x

    @jit
    def infer_smoothness(x):
        z = latent[None, ...].repeat(len(x), 0)
        jac, _ = model.call_jac_param(x, z, param_func)
        return vmap(jnp.linalg.norm, in_axes=[0, None])(jac, 'f')

    if vis_smooth:
        smoothness, grid_samples = voxel_infer(infer_smoothness)

        ps.init()
        pc = ps.register_point_cloud('grid_samples',
                                     grid_samples.reshape(-1, 3),
                                     point_render_mode='quad')
        pc.add_scalar_quantity('smoothness',
                               smoothness.reshape(-1),
                               enabled=True)
        ps.show()
        exit()

    timer = Timer()

    V, F, VN = extract_surface(infer)

    timer.log('Extract surface')

    V, F = filter_components(V, F, VN)

    timer.log('Filter components')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_{interp_tag}mc.obj", V, F)

    if geo_only:
        return

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
            # rotvec = proj_sh4_to_rotvec(sh4)
            # Rs = vmap(rotvec_to_R3)(rotvec)
            Rs = proj_sh4_to_R3(sh4)

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, sur_sample, 0.005)

        ps.init()
        mesh = ps.register_surface_mesh(f"{cfg.name}", V, F)
        ps.register_surface_mesh('Oct frames supervise', V_vis_sup, F_vis_sup)
        pc = ps.register_point_cloud('sur_sample', sur_sample, radius=1e-4)
        pc.add_vector_quantity('sur_normal', sur_normal, enabled=True)

        if cfg.loss_cfg.tangent:

            @jit
            def infer_extra(x):
                z = latent[None, ...].repeat(len(x), 0)
                (_, aux), _, vec_potential = vmap(model._single_call_grad)(x, z)
                return aux[:, :3], aux[:, 3:], vec_potential

            _, TAN_sup, vec_potential_sup = infer_extra(sur_sample)
            potential_interp = vmap(jnp.linalg.norm)(vec_potential_sup)

            pc.add_vector_quantity('TAN_sup', TAN_sup, enabled=True)
            pc.add_scalar_quantity('potential_interp',
                                   potential_interp,
                                   enabled=True)

        ps.show()
        exit()

    timer.reset()

    V, F = meshlab_remesh(cfg, V, F)

    timer.log('Meshlab Remesh')

    # Project on isosurface
    (sdf, _), VN = infer_grad(V)
    VN = vmap(normalize)(VN)
    V = V - sdf[:, None] * VN
    V = np.array(V)

    timer.log('Project SDF')

    (_, aux), VN = infer_grad(V)

    if cfg.loss_cfg.rot6d:
        Rs = vmap(rot6d_to_R3)(aux[:, :6])
        sh4 = vmap(R3_to_sh4_zonal)(Rs)
    else:
        sh4 = aux[:, :9]
        # rotvec = proj_sh4_to_rotvec(sh4)
        # Rs = vmap(rotvec_to_R3)(rotvec)
        Rs = proj_sh4_to_R3(sh4)

    timer.log('Project SO(3)')

    print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")

    sh4 = vmap(normalize)(sh4)
    L = igl.cotmatrix(V, F)
    smoothness = np.trace(sh4.T @ -L @ sh4)
    print(f"Smoothness {smoothness}")

    timer.reset()

    Q = vmap(R3_to_repvec)(Rs, VN)
    V_vis, F_vis, VC_vis = flow_lines.trace(V, F, VN, Q, 4000)

    timer.log('Trace flowlines')

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
    parser.add_argument('--vis_smooth',
                        action='store_true',
                        help='Visualize smoothness')
    parser.add_argument('--vis_flowline',
                        action='store_true',
                        help='Visualize flowline')
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='Output folder')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    latents, latent_dim = config_latent(cfg)
    tokens = args.interp.split('_')
    # Interpolate latent
    i = int(tokens[0])
    j = int(tokens[1])
    t = float(tokens[2])
    latent = (1 - t) * latents[i] + t * latents[j]

    model_key = jax.random.PRNGKey(0)
    model = config_model(cfg, model_key, latent_dim)
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{cfg.name}.eqx", model)

    eval(cfg, args.output, model, latent, args.vis_mc, args.vis_smooth,
         args.vis_flowline)
