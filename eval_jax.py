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
from config_utils import config_latent, config_model, load_sdf
from common import (normalize, vis_oct_field, filter_components, Timer,
                    voxel_tet_from_grid_scale, ps_register_curve_network,
                    rm_unref_vertices, write_triangle_mesh_VC, aabb_compute)
from sh_representation import (proj_sh4_to_R3, proj_sh4_to_rotvec, R3_to_repvec,
                               rotvec_n_to_z, rotvec_to_R3, rotvec_to_R9,
                               project_n, rot6d_to_R3, R3_to_sh4_zonal,
                               rotvec_to_sh4, rot6d_to_sh4_zonal, proj_sh4_sdp)
import pymeshlab

import polyscope as ps
from icecream import ic


# infer: R^3 -> R
def voxel_infer(infer,
                grid_res=512,
                grid_bl=np.array([-1.0, -1.0, -1.0]),
                grid_tr=np.array([1.0, 1.0, 1.0]),
                group_size_mul=1,
                out_dim=1):
    # Smaller batch is somehow faster
    group_size = group_size_mul * grid_res**2
    iter_size = grid_res**3 // group_size

    # Cannot pass jitted function as argument to another jitted function
    @jit
    def infer_scalar():
        # For consistency with partition, we ignore the endpoint
        idx_x = jnp.linspace(grid_bl[0], grid_tr[0], grid_res, endpoint=False)
        idx_y = jnp.linspace(grid_bl[1], grid_tr[1], grid_res, endpoint=False)
        idx_z = jnp.linspace(grid_bl[2], grid_tr[2], grid_res, endpoint=False)
        grid = jnp.stack(jnp.meshgrid(idx_x, idx_y, idx_z), -1)

        query_data = {
            "grid": grid.reshape(iter_size, group_size, 3),
            "val": jnp.zeros((iter_size, group_size, out_dim))
        }

        @jit
        def body_func(i, query_data):
            val = infer(query_data["grid"][i]).reshape(-1, out_dim)
            query_data["val"] = query_data["val"].at[i].set(val)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["val"].reshape(grid_res, grid_res, grid_res,
                                         out_dim), grid

    return infer_scalar()


# infer: R^3 -> R (sdf)
def extract_surface(infer, grid_res=512, grid_min=-1.0, grid_max=1.0):

    grid_max_res = 512

    if grid_res > grid_max_res:
        # Have to partition
        div = int(np.ceil(grid_res / grid_max_res))
        interval = (grid_max - grid_min) / div

        part_idx = np.arange(div)
        part_offsets = np.stack(np.meshgrid(part_idx, part_idx, part_idx),
                                -1).reshape(-1, 3)

        part_bl = grid_min + part_offsets * interval
        part_tr = grid_min + (part_offsets + 1) * interval

        block_list = []
        for i in range(div**3):
            sdf, _ = voxel_infer(infer,
                                 grid_max_res,
                                 grid_bl=part_bl[i],
                                 grid_tr=part_tr[i])
            block_list.append(np.array(sdf[..., 0]))

        sdf_np = np.stack(block_list).reshape(div, div, div, grid_max_res,
                                              grid_max_res, grid_max_res)
        sdf_np = np.transpose(sdf_np, (0, 3, 1, 4, 2, 5)).reshape(
            grid_res, grid_res, grid_res)
    else:
        sdf, _ = voxel_infer(infer,
                             grid_res,
                             grid_bl=np.array([grid_min, grid_min, grid_min]),
                             grid_tr=np.array([grid_max, grid_max, grid_max]))
        # This step is surprising slow, gpu to cpu memory copy?
        sdf_np = np.array(sdf[..., 0])

    sdf_np = np.swapaxes(sdf_np, 0, 1)
    spacing = 1. / grid_res
    # It outputs inverse VN, even with gradient_direction set to ascent
    V, F, VN_inv, _ = marching_cubes(sdf_np,
                                     0.,
                                     spacing=(spacing, spacing, spacing))
    dim = grid_max - grid_min
    V = dim * (V - np.abs(grid_min) / dim)
    return V, F, -VN_inv


# Reduce face count to speed up visualization
# TODO: Use edge collapsing like one in Instant meshes
def meshlab_edge_collapse(save_path, V, F, num_faces):
    m = pymeshlab.Mesh(V, F)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_faces)

    # I believe there is no other option to pass meshlab mesh back to python
    ms.save_current_mesh(save_path)
    V, F = igl.read_triangle_mesh(save_path)
    return V, F


def batch_call(func,
               input,
               num_out_args=1,
               out_map_func=lambda x: [x],
               group_size=256**2):

    n_iters = len(input) // group_size

    if n_iters == 0:
        output = func(input)
        output = out_map_func(output)
    else:
        output = {}
        for i in range(num_out_args):
            output[i] = None

        input_splits = jnp.array_split(input, n_iters)
        for input_batch in input_splits:
            output_ = func(input_batch)
            output_ = out_map_func(output_)

            for i in range(num_out_args):
                output[
                    i] = output_[i] if output[i] is None else jnp.concatenate(
                        [output[i], output_[i]])

        output = list(output.values())

    if num_out_args == 1:
        output = output[0]

    return output


def eval(cfg: Config,
         model: model_jax.MLP,
         latent,
         grid_res=512,
         vis_singularity=False,
         vis_mc=False,
         vis_smooth=False,
         vis_flowline=False,
         save_octa=False,
         single_object=True,
         edge_collapse=False,
         trace_flowline=False,
         miq=False,
         interp_tag=''):

    # Map network output to sh4 parameterization
    if cfg.loss_cfg.rot6d:
        param_func = rot6d_to_sh4_zonal
        proj_func = vmap(rot6d_to_R3)
    elif cfg.loss_cfg.rotvec:
        param_func = rotvec_to_sh4
        proj_func = vmap(rotvec_to_R3)
    else:
        param_func = lambda x: x
        proj_func = proj_sh4_to_R3

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)

    @jit
    def infer_grad(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model.call_grad(x, z)

    timer = Timer()

    if vis_smooth:

        @jit
        def infer_smoothness(x):
            z = latent[None, ...].repeat(len(x), 0)
            jac, _ = model.call_jac_param(x, z, param_func)
            return vmap(jnp.linalg.norm, in_axes=(0, None))(jac, 'f')

        smoothness, grid_samples = voxel_infer(infer_smoothness)

        timer.log('Infer gradient F-norm')

        ps.init()
        pc = ps.register_point_cloud('grid_samples',
                                     grid_samples.reshape(-1, 3),
                                     point_render_mode='quad')
        pc.add_scalar_quantity('smoothness',
                               smoothness.reshape(-1),
                               enabled=True)
        ps.show()
        exit()

    infer_sdf = lambda x: infer(x)[:, 0]
    V, F, VN = extract_surface(infer_sdf, grid_res=grid_res)

    timer.log('Extract surface')

    # if single_object:
    #     # If the output has artifacts, it have large amount of flipped components / ghost geometries
    #     V, F, no_artifacts = filter_components(V, F, VN)

    #     timer.log('Filter components')
    # else:
    #     no_artifacts = False

    if vis_singularity:
        import frame_field_utils

        V_tet, T = voxel_tet_from_grid_scale(16, 1)

        def out_map_func(out):
            (sdf_, aux_), VN_ = out
            return sdf_, aux_, VN_

        sdf, aux, VN = batch_call(infer_grad, V_tet, 3, out_map_func)

        # V_tet, T, V_id = frame_field_utils.tet_reduce(V_tet, VN, sdf < 0, T)
        # aux = aux[V_id]
        # VN = VN[V_id]
        sh4 = vmap(param_func)(aux)

        timer.log('Extract parameterization')

        sh4 = proj_sh4_sdp(sh4)
        sh4_bary = sh4[T].mean(axis=1)
        Rs_bary = proj_sh4_to_R3(sh4_bary)

        timer.log('Project and interpolate SH4')

        TT, TTi = igl.tet_tet_adjacency(T)
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE, E2T = frame_field_utils.tet_edge_one_ring(
            T, TT)
        uE_singularity_mask = frame_field_utils.tet_frame_singularity(
            uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum,
            Rs_bary)
        uE_singular = uE[uE_singularity_mask]

        timer.log('Compute singularity')

        F_b = igl.boundary_facets(T)
        F_b = np.stack([F_b[:, 2], F_b[:, 1], F_b[:, 0]], -1)

        V_b, F_b = rm_unref_vertices(V_tet, F_b)
        # igl.write_triangle_mesh(f"{cfg.out_dir}/{cfg.name}_tet_bound.obj",
        #                         np.float64(V_b), F_b)

        # V_uE, uE_singular = rm_unref_vertices(V_tet, uE_singular)
        # data = {
        #     'V': V_uE.reshape(-1,).tolist(),
        #     'uE': uE_singular.reshape(-1,).tolist()
        # }
        # with open(f'{cfg.out_dir}/{cfg.name}.json', 'w') as f:
        #     json.dump(data, f)

        # exit()

        ps.init()
        ps.register_surface_mesh('tet boundary', V_b, F_b, enabled=False)
        ps.register_surface_mesh('mc', V, F)
        if uE_singularity_mask.sum() > 0:
            ps_register_curve_network('singularity', V_tet, uE_singular)
        ps.show()

        param_path = os.path.join(f"{cfg.out_dir}/{cfg.name}.npz")
        np.savez(param_path, V=V_tet, T=T, sh4=sh4, sdf=sdf)

        exit()

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)

    # Recovery input scale
    # TODO: support latent
    sdf_data = load_sdf(cfg.sdf_paths[0])
    sur_sample = sdf_data['samples_on_sur']
    pc_center, pc_scale, _ = aabb_compute(sur_sample)

    save_name = f"{cfg.name}_{interp_tag}" if interp_tag != '' else f"{cfg.name}"

    # Octahedral field
    if save_octa and len(cfg.mlp_cfgs) > 1:
        aux = batch_call(infer, V)[:, 1:]
        sh4 = param_func(aux)

        if cfg.loss_cfg.xy_scale != 1:
            sh4 = proj_sh4_sdp(sh4)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")
        Rs = proj_func(sh4)

        timer.log('Infer octahedral frames')

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, V, 0.64 / grid_res)
        V_vis_sup = V_vis_sup * pc_scale + pc_center
        igl.write_triangle_mesh(
            os.path.join(cfg.out_dir, f"{save_name}_octa.obj"), V_vis_sup,
            F_vis_sup)

    V = V * pc_scale + pc_center
    igl.write_triangle_mesh(os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F)

    # Quadratic edge collapsing reduces vertex count while preserves original appeal
    if edge_collapse:
        # Reduced mesh is preferable for visualization / downstream task
        V, F = meshlab_edge_collapse(mc_save_path, V, F,
                                     20000 if no_artifacts else 60000)

        timer.log('Meshlab edge collapsing')

        qc_save_path = f"{cfg.out_dir}/{cfg.name}_{interp_tag}mc_qc.obj"
        igl.write_triangle_mesh(qc_save_path, V, F)

    if miq:
        import frame_field_utils

        aux = infer(V)[:, 1:]
        sh4 = param_func(aux)
        sh4 = proj_sh4_sdp(sh4)

        FN = igl.per_face_normals(V, F, np.float64([0, 1, 0]))

        sh4 = sh4[F].mean(1)
        Rs = proj_sh4_to_R3(sh4)

        Q = vmap(R3_to_repvec)(Rs, FN)

        UV, FUV = frame_field_utils.miq(np.float64(V),
                                        F,
                                        np.float64(Q),
                                        gradient_size=75)

        timer.log('MIQ')

        from mesh_helper import OBJMesh, write_obj

        mesh = OBJMesh(V, F)
        mesh.uvs = UV
        mesh.face_uvs_idx = FUV

        write_obj(f'{cfg.out_dir}/{cfg.name}_param.obj', mesh)

        exit()

    if vis_mc:

        ps.init()
        mesh = ps.register_surface_mesh(f"{cfg.name}", V, F)
        if len(cfg.mlp_cfgs) > 1:
            ps.register_surface_mesh('Oct frames supervise', V_vis_sup,
                                     F_vis_sup)

        # pc = ps.register_point_cloud('sur_sample', sur_sample, radius=1e-4)
        # pc.add_vector_quantity('sur_normal', sur_normal, enabled=True)
        ps.show()
        exit()

    timer.reset()

    if trace_flowline:
        import frame_field_utils

        # Project on isosurface
        (sdf, _), VN = infer_grad(V)
        VN = vmap(normalize)(VN)
        V = V - sdf[:, None] * VN
        V = np.array(V)

        timer.log('Project SDF')

        (_, aux), VN = infer_grad(V)
        sh4 = vmap(param_func)(aux)

        if cfg.loss_cfg.xy_scale != 1:
            sh4 = proj_sh4_sdp(sh4)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")

        L = igl.cotmatrix(V, F)
        smoothness = np.trace(sh4.T @ -L @ sh4)
        print(f"Smoothness {smoothness}")

        Rs = proj_func(aux)
        timer.log('Project SO(3)')

        timer.reset()

        Q = vmap(R3_to_repvec)(Rs, VN)

        timer.log('Project to representation vectors')

        V_vis, F_vis, VC_vis = frame_field_utils.trace(V, F, VN, Q, 4000)

        timer.log('Trace flowlines')

        if vis_flowline:
            ps.init()
            mesh = ps.register_surface_mesh("mesh", V, F)
            mesh.add_vector_quantity("VN", VN)
            flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
            flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
            ps.show()

        write_triangle_mesh_VC(
            f"{cfg.out_dir}/{cfg.name}_{interp_tag}stroke.obj", V_vis, F_vis,
            VC_vis)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--interp',
                        type=str,
                        default='0_1_0',
                        help='Interpolation progress')
    parser.add_argument('--vis_singularity',
                        action='store_true',
                        help='Visualize octahedron singularity')
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
    cfg.out_dir = args.output

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
        os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)

    eval(cfg, model, latent, args.vis_singularity, args.vis_mc, args.vis_smooth,
         args.vis_flowline)
