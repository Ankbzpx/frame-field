import numpy as np
import trimesh
import os
from glob import glob
from scipy.spatial import cKDTree
import pandas as pd
from tqdm import tqdm

import igl

import polyscope as ps
from icecream import ic


# Reference: https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py
def compute_metrics(recon_points, gt_points, f1_thr, n_worker=8):
    recon_kd_tree = cKDTree(recon_points)
    gt_kd_tree = cKDTree(gt_points)
    re2gt_distances, _ = recon_kd_tree.query(gt_points, workers=n_worker)
    gt2re_distances, _ = gt_kd_tree.query(recon_points, workers=n_worker)
    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5 * (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))

    precision = np.sum(re2gt_distances < f1_thr) / len(gt_points)
    recall = np.sum(gt2re_distances < f1_thr) / len(recon_points)
    f_1_score = 2 * precision * recall / (precision + recall)

    return chamfer_dist, hausdorff_distance, f_1_score


if __name__ == '__main__':

    gt_root = os.path.expandvars('$HOME/dataset/p2s')
    result_root = os.path.expandvars('$HOME/dataset/octa_results')

    dataset_list = ['abc', 'thingi10k']
    noise_level_list = ['1e-2', '2e-3']
    method_list = [
        'digs', 'EAR', 'APSS', 'point_laplacian', 'SPR', 'nksr',
        'line_processing', 'siren', 'ours_hessian_5', 'ours_hessian_10',
        'ours_digs_5', 'ours_digs_10', 'neural_singular_hessian', 'SALD',
        'IterativePFN', 'RFEPS', 'NeurCAD'
    ]

    seed = 0
    sample_size = 1000000
    f1_percent = 5e-3
    filter_thr = 0.075

    np.random.seed(seed)

    collection = {}
    metrics_column = ['item', 'chamfer', 'hausdorff', 'f1']

    for dataset in dataset_list:
        print(dataset)
        gt_folder = os.path.join(gt_root, dataset, 'gt')
        model_list = sorted(os.listdir(gt_folder))
        for model in tqdm(model_list):
            gt_mesh: trimesh.Trimesh = trimesh.load(
                os.path.join(gt_folder, model))
            gt_samples, _ = trimesh.sample.sample_surface(gt_mesh,
                                                          sample_size,
                                                          seed=seed)
            aabb = gt_mesh.bounding_box.bounds
            max_bound = np.max(aabb[1] - aabb[0])
            f1_thr = f1_percent * max_bound

            def append_collection(tag, result_samples):
                chamfer_dist, hausdorff_distance, f_1_score = compute_metrics(
                    result_samples, gt_samples, f1_thr)

                metrics_frame = pd.DataFrame(
                    [[model_name, chamfer_dist, hausdorff_distance, f_1_score]],
                    columns=metrics_column)

                if tag not in collection:
                    collection[tag] = metrics_frame
                else:
                    collection[tag] = pd.concat(
                        [collection[tag], metrics_frame])

            # Sub kd tree for connected component test
            gt_sub_kd_tree = cKDTree(gt_samples[:int(0.1 * sample_size)])

            model_name = model.split('.')[0]
            for noise_level in noise_level_list:
                for method in method_list:
                    tag = f"{method}_{dataset}_{noise_level}"
                    result_path = glob(
                        os.path.join(result_root, method,
                                     f"{dataset}_{noise_level}",
                                     f'{model_name}.*'))[0]
                    result_mesh: trimesh.Trimesh = trimesh.load(result_path)
                    if hasattr(result_mesh, 'faces'):
                        result_samples, _ = trimesh.sample.sample_surface(
                            result_mesh, sample_size, seed=seed)
                        append_collection(tag, result_samples)
                    else:
                        idx_permute = np.random.permutation(
                            len(result_mesh.vertices))
                        idx = idx_permute[:sample_size]
                        result_samples = result_mesh.vertices[idx]
                        append_collection(tag, result_samples)

    for tag, tag_frames in collection.items():
        tag_frames.to_csv(os.path.join('output', 'metrics', f"{tag}.csv"))
