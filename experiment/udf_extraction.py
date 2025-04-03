import os


# Set off, cause conditional flow can evaluate NaN branch
# from jax.config import config
# config.update("jax_debug_nans", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from functools import partial

# https://bguillard.github.io/meshudf/
from custom_mc._marching_cubes_lewiner import udf_mc_lewiner
import jax
from jax import grad, jit, numpy as jnp, value_and_grad, vmap
import numpy as np
import trimesh

from icecream import ic
import polyscope as ps


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def get_udf_normals_grid(infer_val_and_grad, N=56, max_batch=int(256 * 256)):
    """
    Fills a dense N*N*N regular grid by querying the decoder network
    Inputs:
        infer_val_and_grad: coordinate network to evaluate
        N: grid size
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
    Returns:
        dfs: (N,N,N) tensor representing distance field values on the grid
        df_grads: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
    """
    ################
    # 1: setting up the empty grid
    ################
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    overall_index = np.arange(0, N**3, 1)

    zz = (overall_index % N * voxel_size) + voxel_origin[0]
    yy = (np.floor_divide(overall_index, N) % N * voxel_size) + voxel_origin[1]
    xx = (
        np.floor_divide(np.floor_divide(overall_index, N), N) % N * voxel_size
    ) + voxel_origin[2]
    xyz = jnp.stack([xx, yy, zz], axis=-1)
    num_samples = N**3

    ################
    # 2: Run forward pass to fill the grid
    ################
    dfs = []
    df_grads = []

    head = 0
    ## FIRST: fill distance field grid without gradients
    while head < num_samples:
        # xyz coords
        sample_subset = xyz[head : min(head + max_batch, num_samples)]
        df, df_grad = infer_val_and_grad(sample_subset)

        dfs.append(df)
        df_grads.append(df_grad)
        head += max_batch

    dfs = jnp.concat(dfs)
    df_grads = jnp.vstack(df_grads)
    df_grads = -vmap(normalize)(df_grads)

    dfs = dfs.reshape(N, N, N)
    df_grads = df_grads.reshape(N, N, N, 3)
    return np.asarray(dfs).copy(), np.asarray(df_grads).copy()


def get_mesh_udf(infer_val_and_grad, N_MC=128):
    """
    Computes a triangulated mesh from a distance field network conditioned on the latent vector
    Inputs:
        infer_val_and_grad: coordinate network to evaluate
        N_MC: grid size
    Returns:
        verts: vertices of the mesh
        faces: faces of the mesh
    """
    ### 1: sample grid
    df_values, normals = get_udf_normals_grid(infer_val_and_grad, N_MC)
    df_values[df_values < 0] = 0

    ### 2: run our custom MC on it
    N = df_values.shape[0]
    voxel_size = 2.0 / (N - 1)
    verts, faces, _, _ = udf_mc_lewiner(
        df_values,
        normals,
        spacing=[voxel_size] * 3,
    )

    verts = verts - 1  # since voxel_origin = [-1, -1, -1]
    ### 3: evaluate vertices DF, and remove the ones that are too far
    pred_df_verts = infer_val_and_grad(verts)[0]
    pred_df_verts = np.asarray(pred_df_verts).copy()

    # Remove faces that have vertices far from the surface
    filtered_faces = faces[np.max(pred_df_verts[faces], axis=1) < voxel_size / 6]
    filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
    ### 4: clean the mesh a bit
    # Remove NaNs, flat triangles, duplicate faces
    filtered_mesh = filtered_mesh.process(
        validate=False
    )  # DO NOT try to consistently align winding directions: too slow and poor results
    filtered_mesh.update_faces(filtered_mesh.unique_faces())
    filtered_mesh.update_faces(filtered_mesh.nondegenerate_faces())
    # Fill single triangle holes
    filtered_mesh.fill_holes()
    filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)

    # Re-process the mesh until it is stable:
    n_verts, n_faces, n_iter = 0, 0, 0
    while (n_verts, n_faces) != (
        len(filtered_mesh_2.vertices),
        len(filtered_mesh_2.faces),
    ) and n_iter < 10:
        filtered_mesh_2 = filtered_mesh_2.process(validate=False)
        filtered_mesh_2.update_faces(filtered_mesh_2.unique_faces())
        filtered_mesh_2.update_faces(filtered_mesh_2.nondegenerate_faces())
        (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
        n_iter += 1
        filtered_mesh_2 = trimesh.Trimesh(
            filtered_mesh_2.vertices, filtered_mesh_2.faces
        )

    filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)
    return filtered_mesh_2.vertices, filtered_mesh_2.faces


def sphere_sdf(p, r=0.5):
    return jnp.linalg.norm(p) - r


def sphere_udf(p, r=0.5):
    return jnp.abs(jnp.linalg.norm(p) - r)


if __name__ == "__main__":
    dim = 256
    ax = jnp.linspace(-1, 1, dim)
    pts = jnp.stack(jnp.meshgrid(ax, ax, ax), -1).reshape(-1, 3)

    sdf = vmap(sphere_sdf)(pts)
    udf = vmap(sphere_udf)(pts)

    sdf_grad = vmap(grad(sphere_sdf))(pts)
    udf_grad = vmap(grad(sphere_udf))(pts)

    udf = udf.reshape(dim, dim, dim)
    udf_grad = udf_grad.reshape(dim, dim, dim, 3)

    udf = np.array(udf)
    udf_grad = np.array(udf_grad)

    V, F = get_mesh_udf(vmap(value_and_grad(sphere_udf)), dim)

    ps.init()
    ps.register_surface_mesh("Sphere", V, F)

    pc_viz = ps.register_point_cloud("pts", pts)
    pc_viz.add_scalar_quantity("sdf", sdf)
    pc_viz.add_scalar_quantity("udf", udf)

    pc_viz.add_vector_quantity("sdf_grad", sdf_grad)
    pc_viz.add_vector_quantity("udf_grad", udf_grad)
    ps.show()
