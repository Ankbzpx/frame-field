import os
import sys

import torch.utils.dlpack


sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
from glob import glob
import json
from pathlib import Path

from common import aabb_compute, Timer, vis_oct_field
from config import Config
from config_utils import load_sdf
from eval_jax import batch_call, extract_surface
from sh_representation import proj_sh4_to_R3

import igl
import jax
from jax import jit, numpy as jnp, vmap
from model_pytorch import LipschitzMLP, Siren
import torch
import torch.nn.functional as F
from train_pytorch import OctaGuidedSDF

from icecream import ic
import polyscope as ps


def j2t(x_jax):
    x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
    return x_torch


def t2j(x_torch):
    x_torch = x_torch.contiguous()
    x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
    return x_jax


def eval(
    cfg: Config, model: OctaGuidedSDF, grid_res=512, vis_mc=False, save_octa=False
):
    @torch.no_grad()
    def infer_sdf(x):
        return t2j(model.sdf_mlp(j2t(x)))

    @torch.no_grad()
    def infer_octa(x):
        return t2j(model.octa_mlp(j2t(x)))

    timer = Timer()
    save_name = cfg.name

    with jax.disable_jit():
        V, F, VN = extract_surface(infer_sdf, grid_res=grid_res)

    timer.log("Extract surface")

    sdf_data = load_sdf(cfg.sdf_paths[0])
    sur_sample = sdf_data["samples_on_sur"]
    pc_center, pc_scale, _ = aabb_compute(sur_sample)

    if save_octa:
        sh4 = batch_call(infer_octa, V)

        print(f"SH4 norm {vmap(jnp.linalg.norm)(sh4).mean()}")
        Rs = proj_sh4_to_R3(sh4)

        timer.log("Infer octahedral frames")

        V_vis_sup, F_vis_sup = vis_oct_field(Rs, V, 0.64 / grid_res)
        V_vis_sup = V_vis_sup * pc_scale + pc_center
        igl.write_triangle_mesh(
            os.path.join(cfg.out_dir, f"{save_name}_octa.obj"), V_vis_sup, F_vis_sup
        )

    V = V * pc_scale + pc_center
    igl.write_triangle_mesh(os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F)

    if vis_mc:
        ps.init()
        ps.register_surface_mesh(f"{cfg.name}", V, F)
        # ps.register_surface_mesh("Octa", V_vis_sup, F_vis_sup)
        ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--vis_mc", action="store_true", help="Visualize MC mesh only")
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split("/")[-1].split(".")[0]
    cfg.out_dir = args.output
    cfg.sdf_paths = [os.path.join("..", path) for path in cfg.sdf_paths]

    model_name = cfg.sdf_paths[0].split("/")[-1].split(".")[0]

    model = OctaGuidedSDF(cfg)
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{cfg.name}.ckpt")
    checkpoint_path = "lightning_logs/version_34/checkpoints/epoch=0-step=10000.ckpt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()

    eval(cfg, model, vis_mc=args.vis_mc)
