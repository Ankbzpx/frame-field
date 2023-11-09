import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
import igl
import json
import argparse

import model_jax
from config import Config
from config_utils import config_model, config_latent, config_toy_training_data
from train_jax import train
from common import vis_oct_field, Timer
from eval_jax import extract_surface, meshlab_edge_collapse
from sh_representation import (proj_sh4_to_R3, rot6d_to_R3, euler_to_R3,
                               rotvec_to_R3, proj_sh4_to_rotvec)

import polyscope as ps
from icecream import ic


def eval(cfg: Config,
         model: model_jax.MLP,
         latent,
         samples_sup,
         samples_interp,
         out_dir,
         vis=False):

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 0]

    timer = Timer()

    V, F, _ = extract_surface(infer)

    timer.log('Extract surface')

    @jit
    def infer_aux(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 1:]

    def vis_oct(samples):
        aux = infer_aux(samples)
        if cfg.loss_cfg.rot6d:
            Rs = vmap(rot6d_to_R3)(aux[:, :6])
        elif cfg.loss_cfg.rotvec:
            Rs = vmap(rotvec_to_R3)(aux[:, :3])
        else:
            sh4 = aux[:, :9]
            # rotvec = proj_sh4_to_rotvec(sh4)
            # Rs = vmap(rotvec_to_R3)(rotvec)
            Rs = proj_sh4_to_R3(sh4)

        return vis_oct_field(Rs, samples, 0.01)

    V_vis_sup, F_vis_sup = vis_oct(samples_sup)
    V_vis_interp, F_vis_interp = vis_oct(samples_interp)

    # Compensate small rotation
    R = euler_to_R3(np.pi / 6, np.pi / 3, np.pi / 4)
    V = np.float64(V @ R)
    V_vis_sup = np.float64(V_vis_sup @ R)
    V_vis_interp = np.float64(V_vis_interp @ R)

    if vis:
        ps.init()
        ps.register_surface_mesh("mesh", V, F)
        ps.register_surface_mesh("Oct frames supervise", V_vis_sup, F_vis_sup)
        ps.register_surface_mesh("Oct frames interpolation", V_vis_interp,
                                 F_vis_interp)
        if cfg.loss_cfg.tangent:

            @jit
            def infer_extra(x):
                z = latent[None, ...].repeat(len(x), 0)
                (_, aux), _, vec_potential = vmap(model._single_call_grad)(x, z)
                return aux[:, :3], aux[:, 3:], vec_potential

            VN_interp, TAN_interp, vec_potential_interp = infer_extra(
                samples_interp)

            potential_interp = vmap(jnp.linalg.norm)(vec_potential_interp)

            pc_interp = ps.register_point_cloud('interp',
                                                samples_interp @ R,
                                                radius=1e-4)
            pc_interp.add_vector_quantity('VN_interp',
                                          VN_interp @ R,
                                          enabled=True)
            pc_interp.add_vector_quantity('TAN_interp',
                                          TAN_interp @ R,
                                          enabled=True)
            pc_interp.add_scalar_quantity('potential_interp',
                                          potential_interp,
                                          enabled=True)

        ps.show()

    # V_cube = np.vstack([V_vis_sup, V_vis_interp])
    # F_cube = np.vstack([F_vis_sup, F_vis_interp + F_vis_sup.max() + 1])

    mc_path = f"{out_dir}/{cfg.name}_mc.obj"
    meshlab_edge_collapse(mc_path, V, F)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_sup.obj", V_vis_sup,
                            F_vis_sup)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_interp.obj", V_vis_interp,
                            F_vis_interp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/toy.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    args = parser.parse_args()

    # 1, 2, 3, 4
    # 150, 135, 120, 90, 60, 45, 30
    for gap in [4]:
        for theta in [150, 120, 90, 60, 30]:
            name = f"crease_{gap}_{theta}"

            config = json.load(open(args.config))
            # Placeholder
            config['sdf_paths'] = ['/']
            cfg = Config(**config)
            cfg.name = name

            toy_sample = np.load(f"data/toy/crease_{gap}_{theta}.npz")
            samples_sup = toy_sample['samples_sup']
            samples_vn_sup = toy_sample['samples_vn_sup']
            samples_interp = toy_sample['samples_interp']

            model_key, data_key = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), 2)

            latents, latent_dim = config_latent(cfg)
            model = config_model(cfg, model_key, latent_dim)

            if args.eval:
                model: model_jax.MLP = eqx.tree_deserialise_leaves(
                    f"checkpoints/{cfg.name}.eqx", model)
            else:
                data = config_toy_training_data(cfg, data_key, samples_sup,
                                                samples_vn_sup, latents, gap)
                model = train(cfg, model, data, 'checkpoints')

            eval(cfg, model, jnp.zeros((0,)), samples_sup, samples_interp,
                 "output/toy", args.vis)

            # exit()
