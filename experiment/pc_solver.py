import os
import pdb
import pickle

from common import (
    block_diag,
    normalize,
    normalize_aabb,
    Timer,
    unroll_identity_block,
    vis_oct_field,
)
from sh_representation import (
    eval_sh4_basis,
    grad_oct_polynomial_sh4,
    oct_00,
    oct_poly_scale,
    oct_polynomial_sh4,
    oct_polynomial_sh4_unit_norm,
    proj_sh4_sdp,
    proj_sh4_to_R3,
    R3_to_repvec,
    R3_to_rotvec,
    sh4_canonical,
    y_00,
)

# https://github.com/3DFin/delaunay_geogram?tab=readme-ov-file
from delaunay_geogram import ParallelDelaunay3D
import frame_field_utils_bind
import igl
import jax
from jax import grad, jacfwd, jit, numpy as jnp, vmap
import numpy as np
import point_cloud_utils as pcu
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm

from icecream import ic
import polyscope as ps


# oct_00 * grad(y_00)(n)
@jit
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
@jit
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
    return J4(v).T


@jit
def eval_b(v):
    return 4 * oct_poly_scale * v - J0(v)


def sample_V_close(V):
    kd_tree = scipy.spatial.KDTree(V)
    dists, _ = kd_tree.query(V, k=51, workers=-1)
    sigmas = dists[:, -1:]
    V_close = V + sigmas * np.random.randn(len(V), 3)
    return V_close


def solve_saddle_direct(L, A, b):
    M = scipy.sparse.vstack(
        [
            scipy.sparse.hstack([L, A.T]),
            scipy.sparse.hstack([A, scipy.sparse.csc_matrix((A.shape[0], A.shape[0]))]),
        ]
    ).tocsc()
    x = scipy.sparse.linalg.spsolve(M, b)[: A.shape[1]]
    return x


def solve_saddle_iterative(L, A, b, eps=1e-10):
    M = scipy.sparse.vstack(
        [
            scipy.sparse.hstack([L, A.T]),
            scipy.sparse.hstack([A, scipy.sparse.csc_matrix((A.shape[0], A.shape[0]))]),
        ]
    ).tocsc()
    M += scipy.sparse.diags(eps * np.ones(len(b)))
    x, _ = scipy.sparse.linalg.cg(M, b)
    return x[: A.shape[1]]


# Lx = 0, s.t. Ax = b
def solve_laplace_dirichlet(L, val, idx):
    NV = L.shape[0]
    NB = len(val)

    row = np.arange(NB)
    data = np.ones(NB)
    A = scipy.sparse.coo_matrix((data, (row, idx)), shape=(NB, NV))
    b = np.concatenate([np.zeros(A.shape[1]), val])
    return solve_saddle_direct(L, A, b)


# (M + tL) x_t = M x_0
# Natural boundary condition
def diffuse_natural(L, M, x_0, t):
    return frame_field_utils_bind.solve_iterative_cuda(M + t * L, M @ x_0)


def update_q(L_unroll, M_unroll, q, n, w_align, delta_t):
    As = vmap(eval_A)(n)
    bs = vmap(eval_b)(n)

    A = block_diag(As)
    b = bs.reshape(
        -1,
    )

    LHS_q = scipy.sparse.identity(A.shape[0]) + delta_t * (
        L_unroll + w_align * (A @ M_unroll @ A.T)
    )
    rhs_q = delta_t * w_align * (A @ M_unroll @ b) + q.reshape(
        -1,
    )
    q = frame_field_utils_bind.solve_iterative_cuda(LHS_q, rhs_q)
    q = q.reshape(-1, 9)
    q = vmap(normalize)(q)
    return q


def update_n(M_unroll, M, N, q, r, w_reg, delta_t):
    LHS_r = (
        scipy.sparse.identity(M_unroll.shape[0]) + delta_t * (16 * w_reg + 1) * M_unroll
    )
    rhs_r = (
        delta_t * M @ (4 * w_reg * vmap(grad_oct_polynomial_sh4)(r, q) + N) + r
    ).reshape(
        -1,
    )
    r = frame_field_utils_bind.solve_iterative_cuda(LHS_r, rhs_r)
    r = r.reshape(-1, 3)
    r = vmap(normalize)(r)
    return r


def solve_n(M_unroll, M, N, q, r, w_reg):
    LHS_r = (16 * w_reg + 1) * M_unroll
    rhs_r = (M @ (4 * w_reg * vmap(grad_oct_polynomial_sh4)(r, q) + N)).reshape(
        -1,
    )
    r = frame_field_utils_bind.solve_iterative_cuda(LHS_r, rhs_r)
    r = r.reshape(-1, 3)
    r = vmap(normalize)(r)
    return r


def eval_VN(T, TV, G, u):
    TVN_int = (G @ u).reshape(3, -1).T
    TVN_int = vmap(normalize)(TVN_int)

    VN = jnp.zeros((len(u), 3))
    VN = VN.at[T].add(np.repeat((TV[:, None] * TVN_int)[:, None, :], 4, axis=1))
    VN = vmap(normalize)(VN)
    return VN


def integrate_poisson(T, L, M, G, n):
    n = M @ n
    tn = n[T].mean(1)
    div_n = G.T @ tn.T.reshape(
        -1,
    )
    u = frame_field_utils_bind.solve_iterative_cuda(-L, div_n)
    return u


def pc_to_mesh(V_on, VN_on, iso_idx=None):
    # Sample points
    NV_on = len(V_on)

    # Assume no need to sample
    if iso_idx is not None:
        V = V_on
        VN = VN_on
    else:
        iso_idx = np.arange(NV_on)
        V_close = sample_V_close(V_on)
        off_scale = 1.2
        aabb_min = off_scale * V_on.min(0)
        aabb_max = off_scale * V_on.max(0)
        V_off = np.random.uniform(aabb_min, aabb_max, (len(V_on), 3))

        V = np.vstack([V_on, V_close, V_off])
        VN = np.zeros_like(V)
        VN[:NV_on] = VN_on

    # Trianglulation
    dt = ParallelDelaunay3D()
    dt.set_vertices(V)
    T = dt.cell_to_vertices()

    L = igl.cotmatrix(V, T)
    G = igl.grad(V, T)
    M = igl.massmatrix(V, T)

    TV = igl.volume(V, T)

    u = integrate_poisson(T, L, M, G, VN)
    u -= u[iso_idx].mean()
    VN = eval_VN(T, TV, G, u)
    V_recon, F_recon, _, _ = igl.marching_tets(V, T, u, 0)
    return V_recon, F_recon


if __name__ == "__main__":
    np.random.seed(0)
    timer = Timer()

    data = {}

    import open3d as o3d

    test_pc_path = os.path.expandvars("$HOME/dataset/p2s/thingi10k/1e-2/54725.ply")
    pc_o3d = o3d.io.read_point_cloud(test_pc_path)
    V_raw = np.asarray(pc_o3d.points)
    VN_raw = np.asarray(pc_o3d.normals)

    timer.log("Load input")

    # Integrate initially to fix orientation
    V, F = pc_to_mesh(V_raw, VN_raw)
    VN = igl.per_vertex_normals(V, F)

    ps.init()
    ps.register_surface_mesh("Init", V, F)

    sample_idx = pcu.downsample_point_cloud_poisson_disk(V, 5e-3)
    V = V[sample_idx]
    VN = VN[sample_idx]

    data["V"] = V

    timer.log("Init integration")

    NV = len(V)
    V_off = sample_V_close(V)
    # V_off = np.random.uniform(-1, 1, (len(V), 3))

    # Estimate VN_close
    kd_tree = scipy.spatial.KDTree(V)
    dists, indices = kd_tree.query(V_off, k=15, workers=-1)
    weights = dists / np.sum(dists, axis=-1, keepdims=True)
    VN_close = (VN[indices] * weights[..., None]).sum(1)
    VN_close = vmap(normalize)(VN_close)

    timer.log("Sample neighbor")

    V = np.vstack([V, V_off])
    VN = np.vstack([VN, VN_close])

    # **IMPORTANT** PSD laplacian here
    L, M = robust_laplacian.point_cloud_laplacian(V)
    L_unroll = unroll_identity_block(L, 9)
    M_unroll = unroll_identity_block(M, 3)

    timer.log("Build Laplacian")

    q = np.random.randn(len(V), 9)
    q = vmap(normalize)(q)
    Rs = proj_sh4_to_R3(q[:NV])
    V_vis_0, F_vis_0 = vis_oct_field(Rs, V[:NV], 0.01)

    ps.register_surface_mesh("Octa 0", V_vis_0, F_vis_0, enabled=False)
    pc_viz = ps.register_point_cloud("V", V[:NV])
    pc_viz.add_color_quantity("VN 0", VN[:NV])
    # pc_viz.add_vector_quantity("VN 0", VN[:NV])

    data[0] = {"q": q[:NV], "vn": VN[:NV]}

    w_align = 100.0
    w_reg = 100.0
    delta_t = 2.5

    r = VN
    n_iters = 10
    for iter in tqdm(range(n_iters)):
        q = update_q(
            L_unroll, M_unroll, q, r, 2 * w_align if iter == 0 else w_align, delta_t
        )
        r = update_n(M_unroll, M, VN, q, r, w_reg, delta_t)

        Rs = proj_sh4_to_R3(q[:NV])

        if iter == n_iters - 1:
            # pc_viz.add_vector_quantity("VN 1", r[:NV])
            pc_viz.add_color_quantity("VN 1", r[:NV])
            V_vis, F_vis = vis_oct_field(Rs, V[:NV], 0.01)
            ps.register_surface_mesh("Octa 1", V_vis, F_vis)

        data[iter + 1] = {"q": q[:NV], "vn": r[:NV]}

    timer.log("Optimization")

    V_refined, F_refined = pc_to_mesh(V[:NV], r[:NV])
    ps.register_surface_mesh("Final", V_refined, F_refined)

    timer.log("Final integration")
    ps.show()

    # with open("tmp/step_viz.bin", "wb") as f:
    #     pickle.dump(data, f)
