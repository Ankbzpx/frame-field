import argparse
import json
import os
import igl
from glob import glob

import torch
import torch.nn.functional as F

from pathlib import Path

from config import Config
from config_utils import load_sdf
from train_pytorch import OctaGuidedSDF
from model_pytorch import Siren, LipschitzMLP
from common import Timer, vis_oct_field, aabb_compute
from sh_representation import proj_sh4_to_R3

import jax
from jax import numpy as jnp, vmap, jit
from torch2jax import t2j, connect, j2t

connect(torch.sin, jnp.sin)

from eval_jax import extract_surface, batch_call

import polyscope as ps
from icecream import ic


def eval(cfg: Config,
         model: OctaGuidedSDF,
         grid_res=512,
         vis_mc=False,
         save_octa=False):
    sdf_mlp_jax = t2j(model.sdf_mlp)
    sdf_mlp_params = {k: t2j(v) for k, v in model.sdf_mlp.named_parameters()}

    @jit
    def infer_sdf(x):
        return sdf_mlp_jax(x, state_dict=sdf_mlp_params)

    # FIXME: I should probably t2j the module, but got struck at torch.sum and _div
    @torch.no_grad()
    def infer_octa(x):
        return t2j(model.octa_mlp(j2t(x)))

    timer = Timer()
    save_name = cfg.name

    V, F, VN = extract_surface(infer_sdf, grid_res=grid_res)

    timer.log('Extract surface')

    sdf_data = load_sdf(cfg.sdf_paths[0])
    sur_sample = sdf_data['samples_on_sur']
    pc_center, pc_scale, _ = aabb_compute(sur_sample)

    if save_octa:
        sh4 = batch_call(infer_octa, V)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")
        Rs = proj_sh4_to_R3(sh4)

        timer.log('Infer octahedral frames')

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, V, 0.64 / grid_res)
        V_vis_sup = V_vis_sup * pc_scale + pc_center
        igl.write_triangle_mesh(
            os.path.join(cfg.out_dir, f"{save_name}_octa.obj"), V_vis_sup,
            F_vis_sup)

    V = V * pc_scale + pc_center
    igl.write_triangle_mesh(os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F)

    if vis_mc:
        ps.init()
        ps.register_surface_mesh(f"{cfg.name}", V, F)
        # ps.register_surface_mesh("Octa", V_vis_sup, F_vis_sup)
        ps.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--vis_mc',
                        action='store_true',
                        help='Visualize MC mesh only')
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='Output folder')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]
    cfg.out_dir = args.output

    model_name = cfg.sdf_paths[0].split('/')[-1].split('.')[0]

    model = OctaGuidedSDF(cfg)
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{cfg.name}.ckpt")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    eval(cfg, model, vis_mc=args.vis_mc)
