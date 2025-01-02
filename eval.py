import argparse
import json
import os
from glob import glob

import torch
import torch.nn.functional as F

from pathlib import Path

from config import Config
from train import OctaGuidedSDF
from model import Siren, LipschitzMLP
from common import Timer, vis_oct_field
from sh_representation import proj_sh4_to_R3

import jax
from jax import numpy as jnp, vmap, jit
from torch2jax import t2j, connect, j2t

connect(torch.sin, jnp.sin)

from eval_jax import extract_surface, batch_call

import polyscope as ps
from icecream import ic

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

    mlp_cfgs = cfg.mlp_cfgs
    sdf_mlp = Siren(**mlp_cfgs[0])
    octa_mlp = LipschitzMLP(**mlp_cfgs[1])
    model = OctaGuidedSDF(sdf_mlp, octa_mlp, cfg)

    # TODO: choose version with args
    checkpoint_root = "lightning_logs"
    # https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
    versions = sorted(Path(checkpoint_root).iterdir(), key=os.path.getmtime)

    checkpoint_path = glob(os.path.join(versions[-1], "checkpoints",
                                        "*.ckpt"))[0]
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.sdf_mlp.eval()
    model.octa_mlp.eval()

    sdf_mlp_jax = t2j(model.sdf_mlp)
    sdf_mlp_params = {k: t2j(v) for k, v in model.sdf_mlp.named_parameters()}

    @jit
    def infer_sdf(x):
        return sdf_mlp_jax(x, state_dict=sdf_mlp_params)

    timer = Timer()

    grid_res = 512
    V, F, VN = extract_surface(infer_sdf, grid_res=grid_res)

    timer.log('Extract surface')

    # FIXME: I should probably t2j the module, but got struck at torch.sum and _div
    octa_mlp = model.octa_mlp.cuda()

    @torch.no_grad()
    def infer_octa(x):
        return t2j(octa_mlp(j2t(x)))

    sh4 = batch_call(infer_octa, V)
    Rs = proj_sh4_to_R3(sh4)

    timer.log('Infer octahedral frames')

    V_vis_sup, F_vis_sup = vis_oct_field(Rs, V, 0.64 / grid_res)

    ps.init()
    ps.register_surface_mesh("Surface", V, F)
    ps.register_surface_mesh("Octa", V_vis_sup, F_vis_sup)
    ps.show()
