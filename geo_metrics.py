import numpy as np

import os
import open3d as o3d

import igl
import torch

import pytorch3d
from pytorch3d import ops

import polyscope as ps
from icecream import ic


# Match how we generate data samples
def compute_normalize_aabb(V, scale=0.95):
    V = np.copy(V)

    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale

    return V_center, scale, (V_aabb_max - V_aabb_min)


model_list = ['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas']
dataset_folder = '$HOME/dataset/deep_geometric_prior_data'
dataset_folder = os.path.expandvars(dataset_folder)

gt_path = os.path.join(dataset_folder, 'ground_truth')
ref_path = os.path.join(dataset_folder, 'our_reconstructions')
scan_path = os.path.join(dataset_folder, 'scans')
ours_path = 'output/deep_geometric_prior'

np.random.seed(0)
sample_size = 1000000

for model in model_list:
    print(model)

    pc_gt_o3d = o3d.io.read_point_cloud(os.path.join(gt_path, f'{model}.xyz'))
    pc_gt = np.asarray(pc_gt_o3d.points)

    pc_ref_o3d = o3d.io.read_point_cloud(
        os.path.join(ref_path, f'{model}.npy.xyz.ply'))
    pc_ref = np.asarray(pc_ref_o3d.points)

    # Recover scale from scan
    pc_scan_o3d = o3d.io.read_point_cloud(
        os.path.join(scan_path, f'{model}.ply'))
    pc_scan = np.asarray(pc_scan_o3d.points)
    pc_center, pc_scale, pc_aabb = compute_normalize_aabb(pc_scan)

    chamfer_scale = 1 / (0.1 * pc_aabb.max())
    f_score_threshold = 0.02 * pc_aabb.max()

    pc_gt_sample = np.random.permutation(pc_gt)[:sample_size]
    pc_ref_sample = np.random.permutation(pc_ref)[:sample_size]

    pc_gt_sample_cuda = torch.from_numpy(pc_gt_sample).float().cuda()
    pc_ref_sample_cuda = torch.from_numpy(pc_ref_sample).float().cuda()

    def eval_chamfer_f_score(pc_cuda):
        out = pytorch3d.ops.knn_points(pc_gt_sample_cuda[None, ...],
                                       pc_cuda[None, ...])
        dists_l1_1 = torch.linalg.norm(pc_gt_sample_cuda -
                                       pc_cuda[out.idx.reshape(-1,)],
                                       dim=1,
                                       ord=1)

        out = pytorch3d.ops.knn_points(pc_cuda[None, ...],
                                       pc_gt_sample_cuda[None, ...])
        dists_l1_2 = torch.linalg.norm(pc_cuda -
                                       pc_gt_sample_cuda[out.idx.reshape(-1,)],
                                       dim=1,
                                       ord=1)

        chamfer_dists_l1 = chamfer_scale * (torch.mean(dists_l1_1) +
                                            torch.mean(dists_l1_2))

        precision = torch.sum(dists_l1_1 < f_score_threshold) / sample_size
        recall = torch.sum(dists_l1_2 < f_score_threshold) / sample_size
        f_1_score = 2 * precision * recall / (precision + recall)

        return chamfer_dists_l1.cpu().numpy().item(), f_1_score.cpu().numpy(
        ).item()

    print(f'Ref {model}_full {eval_chamfer_f_score(pc_ref_sample_cuda)}')

    for input_size in ['2500', '5000', '10000', 'full']:
        for reg_cfg in ['reg', 'no_reg']:

            V, F = igl.read_triangle_mesh(
                os.path.join(ours_path,
                             f'{model}_{input_size}_{reg_cfg}_mc.obj'))
            V = V * pc_scale + pc_center

            pc_ours = V
            pc_ours_sample = np.random.permutation(pc_ours)[:sample_size]
            pc_ours_sample_cuda = torch.from_numpy(
                pc_ours_sample).float().cuda()

            print(
                f'Ours {model}_{input_size}_{reg_cfg} {eval_chamfer_f_score(pc_ours_sample_cuda)}'
            )
