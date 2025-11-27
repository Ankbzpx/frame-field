import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse

from common import (
    normalize_aabb,
    Timer,
    unroll_identity_block,
    write_triangle_mesh_VC,
)
from sh_representation import (
    oct_00,
    oct_poly_scale,
    proj_sh4_to_R3,
    R3_to_repvec,
)

import frame_field_utils
import igl
from jax import jit, numpy as jnp, vmap
import numpy as np
import potpourri3d as pp3d
import scipy.sparse
import scipy.sparse.linalg

import polyscope as ps


# oct_00 * grad(y_00)(n)
def J0(v):
    x = v[0]
    y = v[1]
    z = v[2]
    c = 1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi)

    x2 = x * x
    y2 = y * y
    z2 = z * z

    return (
        oct_00
        * c
        * jnp.array(
            [
                2.0 * (x2 + y2 + z2) * 2.0 * x,
                2.0 * (x2 + y2 + z2) * 2.0 * y,
                2.0 * (x2 + y2 + z2) * 2.0 * z,
            ]
        )
    )


# \frac{\nabla F(q, v)}{\nabla q}
def J4(v):
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
    return coeffs[None, :] * jnp.stack([dx, dy, dz])


@jit
def eval_A(v):
    # return J4(v).T @ J4(v)
    return J4(v).T


@jit
def eval_b(v):
    # return 4 * oct_poly_scale * v @ J4(v) - J0(v) @ J4(v)
    return 4 * oct_poly_scale * v - J0(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input file.")
    parser.add_argument(
        "--out_path", type=str, default="../results", help="Path to output folder."
    )
    args = parser.parse_args()

    # Large alignment weight to ensure boundary align
    model_path = args.input
    model_name = model_path.split("/")[-1].split(".")[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_prac.obj")

    timer = Timer()

    V, F = igl.read_triangle_mesh(model_path)
    V = normalize_aabb(V)
    VN = igl.per_vertex_normals(V, F)

    timer.log("Load and preprocess mesh")

    As = vmap(eval_A)(VN)
    bs = vmap(eval_b)(VN)

    A = scipy.sparse.block_diag(As).T.tocsc()
    b = bs.reshape(
        -1,
    )

    # libigl gives zero matrix, likely caused by incompatible version
    # L = igl.cotmatrix(V, F)
    L = pp3d.cotan_laplacian(V, F)
    L_unroll = unroll_identity_block(-L, 9)

    M = scipy.sparse.vstack(
        [
            scipy.sparse.hstack([L_unroll, A.T]),
            scipy.sparse.hstack([A, scipy.sparse.csc_matrix((A.shape[0], A.shape[0]))]),
        ]
    ).tocsc()
    n = np.concatenate([np.zeros(9 * len(VN)), b])

    timer.log("Build sparse system")

    # Pad epsilon
    M += scipy.sparse.diags(1e-10 * np.ones(len(n)))
    x = scipy.sparse.linalg.cg(M, n)[0][: len(VN) * 9]

    timer.log("Solve (Linear)")

    sh4 = x.reshape(len(VN), 9)
    R3 = proj_sh4_to_R3(sh4)

    timer.log("Project SO(3)")

    Q = vmap(R3_to_repvec)(R3, VN)

    timer.log("Project to representation vectors")

    V_vis, F_vis, VC_vis = frame_field_utils.trace(V, F, VN, Q, 4000)

    timer.log("Trace flowlines")

    ps.init()
    mesh_viz = ps.register_surface_mesh("Mesh", V, F)
    mesh_viz.add_vector_quantity("VN", VN, enabled=True)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.show()

    save_folder = os.path.dirname(model_out_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    write_triangle_mesh_VC(model_out_path, V_vis, F_vis, VC_vis)
