import os
import pickle


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

import sys

import blendertoolbox as bt
import bpy
import igl
import jax
from jax import grad, jit, numpy as jnp, vmap
import jax.scipy.spatial
import jax.scipy.spatial.transform
import numpy as np
import point_cloud_utils as pcu
import trimesh

from icecream import ic


# Copied from common.py
def vis_oct_field(R3s, V, size):
    V_cube = np.array(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
        ]
    )

    F_cube = np.array(
        [
            [7, 6, 2],
            [2, 3, 7],
            [0, 4, 5],
            [5, 1, 0],
            [0, 2, 6],
            [6, 4, 0],
            [7, 3, 1],
            [1, 5, 7],
            [3, 2, 0],
            [0, 1, 3],
            [4, 6, 7],
            [7, 5, 4],
        ]
    )

    NV = len(V)
    F_vis = (
        np.repeat(F_cube[None, ...], NV, 0)
        + (len(V_cube) * np.arange(NV))[:, None, None]
    ).reshape(-1, 3)

    V_vis = (V[:, None, :] + np.einsum("nij,bj->nbi", R3s, size * V_cube)).reshape(
        -1, 3
    )

    return V_vis, F_vis


def vis_drop_field(R3s, V, size):
    V_drop, F_drop = igl.read_triangle_mesh("drop.obj")

    NV = len(V)
    F_vis = (
        np.repeat(F_drop[None, ...], NV, 0)
        + (len(V_drop) * np.arange(NV))[:, None, None]
    ).reshape(-1, 3)

    V_vis = (V[:, None, :] + np.einsum("nji,bj->nbi", R3s, size * V_drop)).reshape(
        -1, 3
    )

    return V_vis, F_vis


class StepInterpolator:
    def __init__(self, viz_path):
        with open(viz_path, "rb") as f:
            data = pickle.load(f)

        self.V = data["V"]
        self.max_steps = len(data.keys()) - 1

        qs = []
        vns = []
        for i in np.arange(self.max_steps):
            q = data[i]["q"]
            vn = data[i]["vn"]

            qs.append(q)
            vns.append(vn)

        self.qs = np.stack(qs)
        self.vns = np.stack(vns)

    def eval(self, t):
        t = np.clip(t, 0, self.max_steps - 1)

        left_idx = int(np.floor(t))
        right_idx = int(np.ceil(t))

        if left_idx == right_idx:
            q_interp = self.qs[left_idx]
            vn_interp = self.vns[left_idx]
        else:
            q_interp = (t - left_idx) * self.qs[right_idx] + (right_idx - t) * self.qs[
                left_idx
            ]
            vn_interp = (t - left_idx) * self.vns[right_idx] + (
                right_idx - t
            ) * self.vns[left_idx]

        return q_interp, vn_interp


@jit
def rotvec_to_R3(rotvec):
    rot = jax.scipy.spatial.transform.Rotation.from_rotvec(rotvec)
    return rot.as_matrix()


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


@jit
def rotvec_y_to_n(n):
    n = normalize(n)
    y = jnp.array([0, 1, 0])
    axis = jnp.cross(n, y)
    axis_norm = jnp.linalg.norm(axis) + 1e-8
    # sin(theta) = |n x y|, cos(theta) = n . y
    angle = jnp.arctan2(axis_norm, n[1])
    return angle * (axis / axis_norm)


@jit
def grad_oct_polynomial_sh4(v, sh4):
    x = v[0]
    y = v[1]
    z = v[2]

    x2 = x * x
    y2 = y * y
    z2 = z * z

    x3 = x * x * x
    y3 = y * y * y
    z3 = z * z * z

    coeffs = jnp.array(
        [
            1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(35.0 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(35.0 / 2 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(5.0 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(5.0 / 2.0 / jnp.pi),
            3.0 / 16.0 * jnp.sqrt(1.0 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(5.0 / 2.0 / jnp.pi),
            3.0 / 8.0 * jnp.sqrt(5.0 / jnp.pi),
            3.0 / 4.0 * jnp.sqrt(35.0 / 2 / jnp.pi),
            3.0 / 16.0 * jnp.sqrt(35.0 / jnp.pi),
        ]
    )

    dx = jnp.array(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * x,
            3.0 * x2 * y - y3,
            6.0 * x * y * z,
            6.0 * y * z2 - 3.0 * x2 * y - y3,
            -6.0 * x * y * z,
            -60.0 * x * z2 + 6.0 * (x2 + y2 + z2) * 2.0 * x,
            4.0 * z3 - 9.0 * x2 * z - 3.0 * y2 * z,
            12.0 * x * z2 - 4.0 * x3,
            3.0 * x2 * z - 3.0 * y2 * z,
            4.0 * x3 - 12.0 * x * y2,
        ]
    )

    dy = jnp.array(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * y,
            x3 - 3.0 * x * y2,
            3.0 * x2 * z - 3.0 * y2 * z,
            6.0 * x * z2 - x3 - 3.0 * x * y2,
            4.0 * z3 - 3.0 * x2 * z - 9.0 * y2 * z,
            -60.0 * y * z2 + 6.0 * (x2 + y2 + z2) * 2.0 * y,
            -6.0 * x * y * z,
            -12.0 * y * z2 + 4.0 * y3,
            -6.0 * x * y * z,
            -12.0 * x2 * y + 4.0 * y3,
        ]
    )

    dz = jnp.array(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * z,
            0.0,
            3.0 * x2 * y - y3,
            12.0 * x * y * z,
            12.0 * y * z2 - 3.0 * x2 * y - 3.0 * y3,
            20.0 * z3 - 60.0 * x2 * z - 60.0 * y2 * z + 6.0 * (x2 + y2 + z2) * 2.0 * z,
            12.0 * x * z2 - 3.0 * x3 - 3.0 * x * y2,
            12.0 * x2 * z - 12.0 * y2 * z,
            x3 - 3.0 * x * y2,
            0.0,
        ]
    )

    sh = jnp.hstack([3 * jnp.sqrt(21) / 4, sh4])
    return jnp.stack(
        [
            (coeffs * dx * sh).sum(),
            (coeffs * dy * sh).sum(),
            (coeffs * dz * sh).sum(),
        ]
    ) / (5 * jnp.sqrt(21 / np.pi) / 8)


def proj_sh4_sdp(sh4s_target):
    import frame_field_utils

    _sdp_helper = frame_field_utils.SH4SDPProjectHelper()

    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]
    return _sdp_helper.project(sh4s_target)


@jit
def proj_sh4_to_R3(sh4s_target, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    # Needs to be normalized
    sh4s_target = vmap(normalize)(sh4s_target)

    n_elem = len(sh4s_target)
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))

    v1 = jax.random.normal(key1, (n_elem, 3))
    v2 = jax.random.normal(key2, (n_elem, 3))
    state = {"loss": 100.0, "iter": 0, "v1": v1, "v2": v2}

    # sqrt(n_elem * eps**2)
    min_loss = jnp.sqrt(n_elem) * 1e-8

    @jit
    def condition_func(state):
        return (state["loss"] > min_loss) & (state["iter"] < max_iter)

    @jit
    def project_orth(a, b):
        return b - jnp.dot(b, a) * a

    @jit
    def body_func(state):
        # Power iteration
        v1 = vmap(grad_oct_polynomial_sh4)(state["v1"], sh4s_target)
        v1 = vmap(normalize)(v1)
        v2 = vmap(grad_oct_polynomial_sh4)(state["v2"], sh4s_target)
        v2 = vmap(project_orth)(v1, v2)
        v2 = vmap(normalize)(v2)

        loss = jnp.linalg.norm(v1 - state["v1"], "f")

        state["v1"] = v1
        state["v2"] = v2
        state["loss"] = loss
        state["iter"] += 1
        return state

    state = jax.lax.while_loop(condition_func, body_func, state)

    v1 = state["v1"]
    v2 = state["v2"]
    v3 = jnp.cross(v1, v2, axis=-1)

    return jnp.stack([v1, v2, v3], -1)


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]
    opt = argv[0]

    options = ["octa", "dir", "ncolor"]

    if opt not in options:
        print("Invalid options")
        exit()

    save_folder = f"render_{opt}"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    pc_path = os.path.expandvars("$HOME/dataset/p2s/thingi10k/1e-2/54725.ply")
    pc = trimesh.load(pc_path)
    pc_center, pc_scale, _ = aabb_compute(pc.vertices)

    spacing = 1e-2
    sample_idx = pcu.downsample_point_cloud_poisson_disk(pc.vertices, spacing)
    V = pc.vertices

    gt_path = os.path.expandvars("$HOME/dataset/p2s/thingi10k/gt/54725.ply")
    V_gt, F_gt = igl.read_triangle_mesh(gt_path)

    num_frames = 10000
    count = 0
    fids = [i for i in range(0, num_frames, 10)] + [num_frames - 1]
    for fid in fids:
        tag = str(fid).zfill(6)
        frame_data = np.load(os.path.join("../../tmp", f"{tag}.npz"))
        sdf = frame_data["sdf"] * pc_scale
        q = frame_data["q"]
        vn = frame_data["vn"]
        V_proj = V - sdf[:, None] * vn

        res = 2048
        bt.blenderInit(res, res, 100, 1.5)

        location = (1.1543, 0.014629, 0.794958)
        rotation = (0, 0, 60)
        scale = (1.377, 1.377, 1.377)

        mesh_gt = bt.readNumpyMesh(
            np.float32(V_gt), np.int32(F_gt), location, rotation, scale
        )
        color_gt = bt.colorObj(bt.derekBlue, 0.5, 1.5, 1.0, 0.0, 0.0)
        alpha = 0.2
        transmission = 0.5
        bt.setMat_transparent(mesh_gt, color_gt, alpha, transmission)
        bt.invisibleGround(shadowBrightness=0.9)

        # ===== Octa frame
        if opt == "octa":
            q = proj_sh4_sdp(q[sample_idx])
            Rs = proj_sh4_to_R3(q)
            V_octa, F_octa = vis_oct_field(Rs, V_proj[sample_idx], spacing)
            mesh_octa = bt.readNumpyMesh(
                np.float32(V_octa), np.int32(F_octa), location, rotation, scale
            )
            color_octa = bt.colorObj(
                (0.243245, 0.869022, 0.92549, 1), 0.5, 1.0, 1.0, 0.0, 2.0
            )
            bt.setMat_singleColor(mesh_octa, color_octa, 1.0)

        # ===== VN dir
        if opt == "dir":
            Rs = vmap(rotvec_to_R3)(vmap(rotvec_y_to_n)(vn[sample_idx]))
            V_drop, F_drop = vis_drop_field(Rs, V_proj[sample_idx], 0.2 * spacing)
            mesh_drop = bt.readNumpyMesh(
                np.float32(V_drop), np.int32(F_drop), location, rotation, scale
            )
            color_drop = bt.colorObj(
                (0.243245, 0.869022, 0.92549, 1), 0.5, 1.0, 1.0, 0.0, 2.0
            )
            bt.setMat_singleColor(mesh_drop, color_drop, 1.0)

        # ===== VN color
        if opt == "ncolor":
            mesh_pc = bt.readNumpyPoints(V_proj, location, rotation, scale)
            mesh_pc = bt.setPointColors(mesh_pc, vn)
            # set material ptColor = (vertex_RGBA, H, S, V_proj, Bright, Contrast)
            ptColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
            ptSize = 0.01
            bt.setMat_pointCloudColored(mesh_pc, ptColor, ptSize)

        camLocation = (3, 0, 1.73566)
        lookAtLocation = (0, 0, 0.2)
        focalLength = 40  # (UI: click camera > Object Data > Focal Length)
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        lightAngle = (55, -50, -240)
        strength = 2
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

        bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

        # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + "/test.blend")
        # exit()

        tag = f"{count}".zfill(6)
        bt.renderImage(f"{save_folder}/{tag}.png", cam)
        count += 1
