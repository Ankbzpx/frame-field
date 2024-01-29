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

    V, F = frame_field_utils.dual_contouring_serial(f, f_grad, min_corner,
                                                    max_corner, grid_res,
                                                    grid_res, grid_res)

    F_tri = np.hstack([
        np.stack([F[:, 0], F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 0], F[:, 2], F[:, 3]], -1)
    ]).reshape(-1, 3)

    # ps.init()
    # ps.register_surface_mesh('mesh', V, F_tri)
    # ps.show()
    # exit()

    VN = igl.per_vertex_normals(V, F_tri)

    # filter_components should not break quad pairing
    V, F_tri, _ = filter_components(V, F_tri, VN)
    V = f_map(V)

    F = F_tri.reshape(-1, 6)
    F = np.stack([F[:, 0], F[:, 1], F[:, 2], F[:, -1]], -1)
    mesh = OBJMesh(vertices=V, faces=F_tri, faces_quad=F)
    write_obj(save_path, mesh)

    return V, F_tri


def extract_mc(save_path, f, f_map, grid_res, grid_min=-1.0, grid_max=1.0):
    V, F, VN = extract_surface(vmap(f),
                               grid_res=grid_res,
                               grid_min=grid_min,
                               grid_max=grid_max)
    V, F, _ = filter_components(V, F, VN)
    V = f_map(V)

    igl.write_triangle_mesh(save_path, np.float64(V), F)

    return V, F


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
            V_param, F_param = extract_dc(f'{out_dir}/{cfg.name}_param_dc.obj',
                                          infer_sdf_param, infer_sdf_grad_param,
                                          infer_mapping, grid_res, grid_min,
                                          grid_max)

            V, F = extract_dc(f'{out_dir}/{cfg.name}_dc.obj', infer_sdf,
                              infer_sdf_grad, lambda x: x, grid_res, grid_min,
                              grid_max)

        else:
            V_param, F_param = extract_mc(f'{out_dir}/{cfg.name}_param_mc.obj',
                                          infer_sdf_param, infer_mapping,
                                          grid_res, grid_min, grid_max)

            V, F = extract_mc(f'{out_dir}/{cfg.name}_mc.obj', infer_sdf,
                              lambda x: x, grid_res, grid_min, grid_max)

        ps.init()
        ps.register_surface_mesh('Param', V_param, F_param)
        ps.register_surface_mesh('Euclidean', V, F)
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

    # 1 general vector field
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{cfg.name}.eqx", model)

    eval(cfg, 'output', model, model_octa, args.inverse, args.dc)
