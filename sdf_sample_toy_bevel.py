import pickle

import jax
from jax import jit, numpy as jnp, vmap
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import trimesh

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    mesh: trimesh.Trimesh = trimesh.load("data/toy/bevel.obj")
    V = mesh.vertices
    F = mesh.faces
    FN = mesh.face_normals

    sample_size = 10000
    V_sample, I_sample = trimesh.sample.sample_surface_even(mesh, sample_size)
    VN_sample = FN[I_sample]

    radius = np.linalg.norm(V[4] - V[51])
    mask = np.linalg.norm(V_sample[:, :2] - V[4][:2][None, :], axis=-1) > radius

    noise_mask = np.logical_not(mask)
    V_noise = V_sample[noise_mask] + 0.5 * radius * np.random.normal(
        0, 1, (noise_mask.sum(), 3)
    )

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(V_noise)
    o3d.io.write_point_cloud("data/sdf/bevel_sample.ply", pc_o3d)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(V_sample)
    pc_o3d.normals = o3d.utility.Vector3dVector(VN_sample)
    o3d.io.write_point_cloud("data/sdf/bevel.ply", pc_o3d)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(V_sample[mask])
    pc_o3d.normals = o3d.utility.Vector3dVector(VN_sample[mask])
    o3d.io.write_point_cloud("data/sdf/bevel_mask.ply", pc_o3d)
