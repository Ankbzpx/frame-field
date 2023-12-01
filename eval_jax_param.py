import equinox as eqx
import numpy as np
import jax
from jax import vmap, jit, numpy as jnp
from skimage.measure import marching_cubes

import model_jax
from config import Config
from config_utils import config_latent, config_model, eval_data_scale
from sh_representation import (rotvec_to_sh4, rot6d_to_sh4_zonal, rot6d_to_R3,
                               rotvec_to_R3, proj_sh4_to_R3, proj_sh4_sdp)
from common import voxel_tet_from_grid_scale, ps_register_basis, filter_components
from eval_jax import extract_surface
from train_jax_param import ParamMLP
import igl

import json
import argparse

import polyscope as ps
from icecream import ic

from mesh_helper import OBJMesh, write_obj

import frame_field_utils


def extract_dc(save_path,
               f,
               f_grad,
               f_map,
               grid_res,
               grid_min=-1.0,
               grid_max=1.0):
    min_corner = np.float64([grid_min, grid_min, grid_min])
    max_corner = np.float64([grid_max, grid_max, grid_max])

    V_mc_param, F_mc_param = frame_field_utils.dual_contouring_serial(
        f, f_grad, min_corner, max_corner, grid_res, grid_res, grid_res)

    F_mc_param_tri = np.hstack([
        np.stack([F_mc_param[:, 0], F_mc_param[:, 1], F_mc_param[:, 2]], -1),
        np.stack([F_mc_param[:, 0], F_mc_param[:, 2], F_mc_param[:, 3]], -1)
    ]).reshape(-1, 3)

    # ps.init()
    # ps.register_surface_mesh('mesh', V_mc_param, F_mc_param_tri)
    # ps.show()

    VN = igl.per_vertex_normals(V_mc_param, F_mc_param_tri)

    # filter_components should not break quad pairing
    V_mc_param, F_mc_param_tri, _ = filter_components(V_mc_param,
                                                      F_mc_param_tri, VN)
    V_mc_param = f_map(V_mc_param)

    F_mc_param = F_mc_param_tri.reshape(-1, 6)
    F_mc_param = np.stack([
        F_mc_param[:, 0], F_mc_param[:, 1], F_mc_param[:, 2], F_mc_param[:, -1]
    ], -1)
    mesh = OBJMesh(vertices=V_mc_param,
                   faces=F_mc_param_tri,
                   faces_quad=F_mc_param)
    write_obj(save_path, mesh)


def eval(cfg: Config,
         out_dir,
         model: model_jax.MLP,
         model_octa: model_jax.MLP,
         inverse: bool,
         dc: bool = False,
         vis_octa=False):

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

    if vis_octa or not inverse:

        @jit
        def infer_mapping_jac(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model.call_jac(x, z)

        @jit
        def infer_octa(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model_octa.call_aux(x, z)

        grid_scale = 1.25 * eval_data_scale(cfg)
        # low res because SDP is expensive
        V, T = voxel_tet_from_grid_scale(16, grid_scale)

        J, V_param = infer_mapping_jac(V)
        J = J if inverse else jnp.transpose(J, (0, 2, 1))
        dets = vmap(jnp.linalg.det)(J)
        print(f'Num of flips: {(dets < 0).sum()}')

        V_cart = V_param if inverse else V
        aux = infer_octa(V_cart)
        sh4 = vmap(param_func)(aux)
        # SDP is necessary for visualizing accurate octa field
        sh4_octa = proj_sh4_sdp(sh4)
        Rs = proj_func(sh4_octa)

        ps.init()
        ps_register_basis('GT', Rs, V_cart)
        ps_register_basis('Param', J, V_cart)
        ps.register_volume_mesh('tet param', V_param, T)
        ps.show()

    if inverse:

        # There is no guarantee of the aabb of inverse parameterization
        grid_res = 64
        grid_max = 1.5
        grid_min = -grid_max

        @jit
        def infer_mapping(x):
            z = latent[None, ...].repeat(len(x), 0)
            return model(x, z)

        @jit
        def infer_sdf_param(x):
            z = latent
            x_cart = model.single_call(x, z)
            sdf = model_octa.single_call(x_cart, z)[0]
            return sdf

        @jit
        def infer_sdf_grad_param(x):
            return eqx.filter_grad(infer_sdf_param)(x)

        @jit
        def infer_sdf(x):
            z = latent
            return model_octa.single_call(x, z)[0]

        @jit
        def infer_sdf_grad(x):
            return eqx.filter_grad(infer_sdf)(x)

        if dc:
            extract_dc(f'{out_dir}/{cfg.name}_param_dc.obj', infer_sdf_param,
                       infer_sdf_grad_param, infer_mapping, grid_res, grid_min,
                       grid_max)

            extract_dc(f'{out_dir}/{cfg.name}_dc.obj', infer_sdf,
                       infer_sdf_grad, lambda x: x, grid_res, grid_min,
                       grid_max)

        else:
            indices = jnp.linspace(grid_min, grid_max, grid_res)
            grid = jnp.stack(jnp.meshgrid(indices, indices, indices), -1)
            grid = grid.reshape(-1, 3)

            V_mc_param, F_mc_param, VN = extract_surface(vmap(infer_sdf_param),
                                                         grid_res=grid_res,
                                                         grid_min=grid_min,
                                                         grid_max=grid_max)
            V_mc_param, F_mc_param, _ = filter_components(
                V_mc_param, F_mc_param, VN)
            V_mc_param = infer_mapping(V_mc_param)

            V_mc, F_mc, VN = extract_surface(vmap(infer_sdf),
                                             grid_res=grid_res,
                                             grid_min=grid_min,
                                             grid_max=grid_max)
            V_mc, F_mc, _ = filter_components(V_mc, F_mc, VN)

            igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_param_mc.obj",
                                    np.float64(V_mc_param), F_mc_param)
            igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_mc.obj",
                                    np.float64(V_mc), F_mc)

            ps.init()
            ps.register_surface_mesh('MC param', V_mc_param, F_mc_param)
            ps.register_surface_mesh('MC', V_mc, F_mc)
            ps.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--inverse',
                        action='store_true',
                        help='Inverse parameterization')
    parser.add_argument('--dc', action='store_true', help='Dual contouring')
    args = parser.parse_args()

    name = args.config.split('/')[-1].split('.')[0]
    suffix = "param_inv" if args.inverse else "param"

    cfg = Config(**json.load(open(args.config)))
    cfg.name = f'{name}_{suffix}'

    model_key = jax.random.PRNGKey(0)

    latents, latent_dim = config_latent(cfg)
    latent = latents[0]
    model_octa = config_model(cfg, model_key, latent_dim)
    model_octa: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{name}.eqx", model_octa)

    mlp_cfg = cfg.mlp_cfgs[0]
    mlp_type = cfg.mlp_types[0]

    # Single vector potential
    mlp_cfg['out_features'] = 3
    model = ParamMLP(**mlp_cfg, key=model_key)

    # 3 scalar fields
    # model = ParamMLP(mlp_types=[mlp_type] * 3,
    #                  mlp_cfgs=[mlp_cfg] * 3,
    #                  key=model_key)
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{cfg.name}.eqx", model)

    eval(cfg, 'output', model, model_octa, args.inverse, args.dc)
