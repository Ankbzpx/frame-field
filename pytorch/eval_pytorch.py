import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))

from config import Config

from common_pytorch import aabb_compute
from dataset_pytorch import load_sdf
import igl
from skimage.measure import marching_cubes
import torch
from tqdm import tqdm
from train_pytorch import OctaGuidedSDF

from icecream import ic
import polyscope as ps


def eval(cfg: Config, model: OctaGuidedSDF, grid_res=512, vis_mc=False):
    save_name = cfg.name

    axis = torch.linspace(-1, 1, grid_res)
    grid_pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(
        -1, 3
    )
    group_size = grid_res**2

    sdfs = []
    for pt_group in tqdm(torch.split(grid_pts, group_size)):
        sdfs.append(model.sdf_mlp(pt_group.float().cuda()).detach().cpu()[:, 0])
    sdfs = torch.concat(sdfs)
    sdfs = sdfs.reshape(grid_res, grid_res, grid_res).numpy()

    spacing = 1.0 / grid_res
    V, F, _, _ = marching_cubes(sdfs, 0, spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)

    sdf_data = load_sdf(cfg.sdf_paths[0])
    sur_sample = sdf_data["samples_on_sur"]
    pc_center, pc_scale, _ = aabb_compute(sur_sample)

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)

    V = V * pc_scale + pc_center
    igl.write_triangle_mesh(os.path.join(cfg.out_dir, f"{save_name}.obj"), V, F)

    if vis_mc:
        ps.init()
        ps.register_surface_mesh(f"{cfg.name}", V, F)
        ps.show()
