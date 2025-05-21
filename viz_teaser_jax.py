import argparse
import json
import os

from common import aabb_compute, vis_oct_field
from config import Config
from config_utils import config_latent, config_model, load_sdf
from eval_jax import extract_surface
import model_jax
from sh_representation import (
    eulerXYZ_to_R3,
    proj_sh4_sdp,
    proj_sh4_to_R3,
)

import equinox as eqx
import igl
import jax
from jax import jit, numpy as jnp
import matplotlib
import numpy as np

from icecream import ic
import polyscope as ps


jax.config.update("jax_default_matmul_precision", "tensorfloat32")

matplotlib.use("Agg")

tags = ["00994034_9299b4c10539bb6b50b162d7", "75658", "117"]
obj_poses = [
    {
        "t": [1.27352, -0.128447, 0.985097],
        "R": [10.8916, 18.7741, 97.8997],
        "s": [1.47165, 1.47165, 1.47165],
    },
    {
        "t": [1.06809, 0.018084, 0.869257],
        "R": [6.97875, -2.99218, 88.3063],
        "s": [1.58881, 1.58881, 1.58881],
    },
    {
        "t": [1.12769, 0.096429, 0.745711],
        "R": [5.08369, 20.2637, -116.484],
        "s": [1.71952, 1.71952, 1.71952],
    },
]
plane_poses = [
    {
        "t": [1.54814, 0.212535, 1.07617],
        "R": [99.4114, 2.59422, -305.441],
        "s": [0.389667, 0.389667, 0.389667],
    },
    {
        "t": [1.08668, -0.415991, 0.301808],
        "R": [1.41436, 0.742931, -101.795],
        "s": [0.366439, 0.366439, 0.366439],
    },
    {
        "t": [1.34736, 0.194946, 0.728997],
        "R": [-6.90613, 28.5447, -133.83],
        "s": [0.215312, 0.215312, 0.215312],
    },
]
octa_scales = [1.0, 0.8, 0.8]


def sample_plane(dim, skip_sides=False):
    axis = np.linspace(-1, 1, dim)
    if skip_sides:
        axis = axis[1:-1]
    xy = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    z = np.zeros(len(xy))
    xyz = np.hstack([xy, z[:, None]])
    return xyz


def apply_T(T, x):
    A = T[:3, :3]
    t = T[:3, 3][None, :]
    return x @ A.T + t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/udf.json", help="Path to config file."
    )
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    latents, latent_dim = config_latent(cfg)
    model_key, data_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 2)
    model = config_model(cfg, model_key, latent_dim)

    for tag, obj_pose, plane_pose, octa_scale in zip(
        tags, obj_poses, plane_poses, octa_scales
    ):
        sdf_path = f"checkpoints/teaser/input/{tag}.ply"
        sdf_data = load_sdf(sdf_path)
        sur_sample = sdf_data["samples_on_sur"]
        pc_center, pc_scale, _ = aabb_compute(sur_sample)

        T_normalize = np.eye(4)
        T_normalize[:3, :3] *= pc_scale
        T_normalize[:3, 3] = pc_center

        model: model_jax.MLP = eqx.tree_deserialise_leaves(
            os.path.join(cfg.checkpoints_dir, f"teaser/{tag}.eqx"), model
        )

        @jit
        def infer_sdf(x):
            z = jnp.empty((0,))[None, ...].repeat(len(x), 0)
            return model.mlps[0](x, z)

        @jit
        def infer_udf(x):
            z = jnp.empty((0,))[None, ...].repeat(len(x), 0)
            df = model.mlps[0](x, z)
            df = jnp.clip(df / 1000, min=1e-10)
            return jnp.sqrt(df)

        @jit
        def infer_octa(x):
            z = jnp.empty((0,))[None, ...].repeat(len(x), 0)
            return model.mlps[1](x, z)

        if tag == "117":
            V, F, _ = extract_surface(infer_udf, grid_res=256, iso=6e-3)
        else:
            V, F, _ = extract_surface(infer_sdf, grid_res=256)
        V = apply_T(T_normalize, V)

        R_plane = eulerXYZ_to_R3(
            np.deg2rad(plane_pose["R"][0]),
            np.deg2rad(plane_pose["R"][1]),
            np.deg2rad(plane_pose["R"][2]),
        )
        s_plane = np.array(plane_pose["s"])
        t_plane = np.array(plane_pose["t"])
        T_plane = np.eye(4)
        T_plane[:3, :3] = np.diag(s_plane) @ R_plane
        T_plane[:3, 3] = t_plane

        R_obj = eulerXYZ_to_R3(
            np.deg2rad(obj_pose["R"][0]),
            np.deg2rad(obj_pose["R"][1]),
            np.deg2rad(obj_pose["R"][2]),
        )
        s_obj = np.array(obj_pose["s"])
        t_obj = np.array(obj_pose["t"])
        T_obj = np.eye(4)
        T_obj[:3, :3] = np.diag(s_obj) @ R_obj
        T_obj[:3, 3] = t_obj

        T_relative = np.linalg.inv(T_obj) @ T_plane

        V_plane = np.array(
            [
                [-1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [1, -1, 0],
            ]
        )
        # V_plane = apply_T(T_relative, V_plane)
        F_plane = np.array([[0, 1, 2], [0, 2, 3]])

        image_res = 512
        samples_image = sample_plane(image_res)
        samples_image = apply_T(T_relative, samples_image)

        if tag == "117":
            sdf = infer_udf(
                apply_T(jnp.linalg.inv(T_normalize), samples_image)
            ).reshape(
                -1,
            )
        else:
            sdf = infer_sdf(
                apply_T(jnp.linalg.inv(T_normalize), samples_image)
            ).reshape(
                -1,
            )

        octa_scale = octa_scale * 0.4 * plane_pose["s"][0]

        octa_res = 16
        samples_octa = sample_plane(octa_res, True)
        samples_octa = apply_T(T_relative, samples_octa)
        octa = infer_octa(apply_T(jnp.linalg.inv(T_normalize), samples_octa))
        octa = proj_sh4_sdp(octa)
        Rs = proj_sh4_to_R3(octa)
        V_octa, F_octa = vis_oct_field(Rs, samples_octa, octa_scale / octa_res)

        V = apply_T(T_obj, V)
        # V_plane = apply_T(T_obj, V_plane)
        samples_image = apply_T(T_obj, samples_image)
        V_octa = apply_T(T_obj, V_octa)

        np.save(f"output/{tag}.npy", sdf)
        igl.write_triangle_mesh(
            f"output/{tag}_octa.obj", np.float64(V_octa), np.int64(F_octa)
        )
        V_octa_2 = apply_T(np.linalg.inv(T_obj @ T_relative), V_octa)
        igl.write_triangle_mesh(
            f"output/{tag}_octa2.obj", np.float64(V_octa_2), np.int64(F_octa)
        )

        ps.init()
        ps.register_surface_mesh("extract", V, F)

        ps.register_point_cloud("samples_image", samples_image).add_scalar_quantity(
            "sdf", sdf
        )
        ps.register_surface_mesh("octa", V_octa, F_octa)
        ps.register_surface_mesh("octa 2", V_octa_2, F_octa)
        ps.register_surface_mesh("pl", V_plane, F_plane)
        ps.show()
