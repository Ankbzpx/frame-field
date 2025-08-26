from functools import partial
import os

from common import normalize, normalize_aabb

import cv2
import igl
import jax
from jax import grad, jit, numpy as jnp, vmap
import numpy as np
from pyrr import Matrix44
from skimage.measure import marching_cubes

from icecream import ic
import polyscope as ps


def unproj_depth_map(depth, intr, mask=None):
    height, width = np.shape(depth)
    N = height * width
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - intr[0, 2]) * depth / intr[0, 0]
    Y = (yy - intr[1, 2]) * depth / intr[1, 1]
    pts_view = np.stack((X, Y, depth), -1).reshape(-1, 3)
    if mask is not None:
        mask = np.reshape(mask, (N))
        pts_view = pts_view[mask > 0]

    return pts_view


# Reference: https://en.wikipedia.org/wiki/Slab_method
@jit
def ray_aabb_intersect(origin, dir, low=-1, high=1):
    tl = (low - origin) / dir
    th = (high - origin) / dir

    t_near = jnp.max(jnp.minimum(tl, th))
    t_far = jnp.min(jnp.maximum(tl, th))
    return t_near, t_far


def sample_stratified(key, t_near, t_far, sample_size):
    rands = jax.random.uniform(key, (sample_size,)) + jnp.arange(sample_size)
    dists = rands / sample_size * (t_far - t_near) + t_near
    return dists


def sample_pdf(node, weight, sample_size):
    pdf = weight / (weight.sum() + 1e-5)
    # Pad 0 to get node idx
    cdf = jnp.concat([jnp.zeros((1,)), jnp.cumsum(pdf)])
    unif_samples = jnp.linspace(0, 1, sample_size + 1)
    unif_samples = 0.5 * (unif_samples[:-1] + unif_samples[1:])

    # Right most insertion, in case of duplicates
    #   interval / right node idx
    idx = jnp.searchsorted(cdf, unif_samples, side="right")

    t = (unif_samples - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1] + 1e-5)
    return node[idx - 1] + t * (node[idx] - node[idx - 1])


@jit
def sphere_sdf(x, radius=0.5):
    return jnp.linalg.norm(x) - radius


@jit
def eval_sdf(dir, dist):
    samples = origin[None, :] + dist[:, None] * dir[None, :]
    sdf = vmap(sphere_sdf)(samples)
    return sdf


def debug_samples(dirs, dists):
    sdfs = vmap(eval_sdf)(dirs, dists)
    debug_mask = (sdfs < 0).sum(1) > 0

    img = jnp.zeros((res, res)).reshape(
        -1,
    )
    img = img.at[valid_mask].set(debug_mask)
    img = np.uint(255 * img).reshape(res, res)

    cv2.imwrite("debug.png", img)


if __name__ == "__main__":
    # model_name = "00010218_4769314c71814669ba5d3512"
    # gt_folder = os.path.expandvars("$HOME/dataset/p2s/abc/gt")

    # model_path = os.path.join(gt_folder, f"{model_name}.ply")
    # V, F = igl.read_triangle_mesh(model_path)
    # # Ray is in normalized space
    # V = normalize_aabb(V)

    res = 128
    fov_deg = 55

    # Test geometry
    row = np.linspace(-1, 1, res)
    grid_pts = np.stack(np.meshgrid(row, row, row), -1).reshape(-1, 3)
    sdfs = vmap(sphere_sdf)(grid_pts)
    spacing = 1.0 / res
    V, F, _, _ = marching_cubes(
        np.array(sdfs).reshape(res, res, res),
        0.0,
        spacing=(spacing, spacing, spacing),
    )
    V = 2 * (V - 0.5)

    origin = np.array([-1.3933814, -0.03670767, 0.35455924])
    target = np.array([-0.42886664, 0.22563264, 0.32479309])
    z = np.array([0.0, 1.0, 0.0])

    T = Matrix44.look_at(origin, target, z)
    T = np.array(T).T

    # Generate rays
    fx = 0.5 * res / np.tan(np.deg2rad(fov_deg) / 2)
    K = np.array([[fx, 0.0, (res - 1) / 2], [0.0, fx, (res - 1) / 2], [0.0, 0.0, 1.0]])

    R = T[:3, :3]
    t = T[:3, 3]
    pts_view = unproj_depth_map(-np.ones((res, res)), K)
    pts = (pts_view - t[None, :]) @ R

    # Intersection w.r.t. [-1, 1] bbox
    dirs = vmap(normalize)(pts - origin[None, :])
    t_near, t_far = vmap(ray_aabb_intersect, in_axes=(None, 0))(origin, dirs)
    valid_mask = jnp.logical_and(t_near < t_far, t_far > 0)

    coarse_sample_size = 64
    fine_sample_size = 16

    dirs = dirs[valid_mask]
    t_near = t_near[valid_mask]
    t_far = t_far[valid_mask]

    sample_ray_coarse = jit(partial(sample_stratified, sample_size=coarse_sample_size))
    sample_ray_fine = jit(partial(sample_pdf, sample_size=fine_sample_size))

    data_key = jax.random.PRNGKey(0)
    keys = jax.random.split(data_key, valid_mask.sum())
    dists = vmap(sample_ray_coarse)(keys, t_near, t_far)

    @jit
    def sample_dist_fine(dir, dist, num_levels=4):
        @jit
        def sample_dist(level, dist):
            inv_s = 32 * 2**level
            samples = origin[None, :] + dist[:, None] * dir[None, :]
            sdf = vmap(sphere_sdf)(samples)

            prev_sdf, next_sdf = sdf[:-1], sdf[1:]
            mid_sdf = 0.5 * (prev_sdf + next_sdf)
            prev_dist, next_dist = dist[:-1], dist[1:]

            cos_val = (next_sdf - prev_sdf) / (next_dist - prev_dist + 1e-5)
            prev_cos_val = jnp.concat([jnp.zeros_like(cos_val[:1]), cos_val[:-1]])
            cos_val = jnp.minimum(prev_cos_val, cos_val)

            dist_intv = next_dist - prev_dist
            est_prev_sdf = mid_sdf - cos_val * dist_intv * 0.5
            est_next_sdf = mid_sdf + cos_val * dist_intv * 0.5

            prev_cdf = jax.nn.sigmoid(est_prev_sdf * inv_s)
            next_cdf = jax.nn.sigmoid(est_next_sdf * inv_s)
            alpha = jnp.clip((prev_cdf - next_cdf) / (prev_cdf + 1e-5), 0.0, 1.0)
            weight = alpha * jnp.cumprod(1 - alpha)
            return jnp.concat([dist, sample_ray_fine(dist, weight)])

        for level in range(num_levels):
            dist = sample_dist(level, dist)
        return jnp.sort(dist)

    dists = vmap(sample_dist_fine)(dirs, dists)

    @jit
    # Choose the finest inv_s since we don't hav
    # e network
    def eval_opacity(dir, dist, inv_s=32 * 2**4):
        dist = jnp.concat([dist, jnp.array([1e10])])
        samples = origin[None, :] + dist[:, None] * dir[None, :]
        sdf = vmap(sphere_sdf)(samples)

        prev_sdf, next_sdf = sdf[:-1], sdf[1:]
        mid_sdf = 0.5 * (prev_sdf + next_sdf)
        prev_dist, next_dist = dist[:-1], dist[1:]

        cos_val = (next_sdf - prev_sdf) / (next_dist - prev_dist + 1e-5)
        prev_cos_val = jnp.concat([jnp.zeros_like(cos_val[:1]), cos_val[:-1]])
        cos_val = jnp.minimum(prev_cos_val, cos_val)

        dist_intv = next_dist - prev_dist
        est_prev_sdf = mid_sdf - cos_val * dist_intv * 0.5
        est_next_sdf = mid_sdf + cos_val * dist_intv * 0.5

        prev_cdf = jax.nn.sigmoid(est_prev_sdf * inv_s)
        next_cdf = jax.nn.sigmoid(est_next_sdf * inv_s)
        alpha = jnp.clip((prev_cdf - next_cdf) / (prev_cdf + 1e-5), 0.0, 1.0)
        weight = alpha * jnp.cumprod(1 - alpha)
        return weight

    opacity = vmap(eval_opacity)(dirs, dists).sum(-1)
    img = jnp.zeros((res, res)).reshape(
        -1,
    )
    img = img.at[valid_mask].set(opacity)
    img = np.uint(255 * img).reshape(res, res)

    cv2.imwrite("test.png", img)

    exit()

    ps.init()
    ps.register_point_cloud("pts", pts)

    ps.register_point_cloud("samples", samples)
    ps.register_surface_mesh("ms", V, F)
    ps.show()

    exit()

    ps.init()
    ps.register_point_cloud("pts", pts)

    pt_near = origin[None, :] + t_near[:, None] * dirs
    pt_far = origin[None, :] + t_far[:, None] * dirs
    ps.register_point_cloud("pt_near", pt_near[valid_mask])
    ps.register_point_cloud("pt_far", pt_far[valid_mask])

    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=fov_deg, aspect=1)
    extrinsics = ps.CameraExtrinsics(mat=T)
    params = ps.CameraParameters(intrinsics, extrinsics)
    cam = ps.register_camera_view("cam", params)

    ps.register_point_cloud("p0", target[None, :])
    ps.register_point_cloud("p1", origin[None, :])
    ps.register_surface_mesh("ms", V, F)
    ps.show()
