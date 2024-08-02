import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import igl
import numpy as np
from jax import vmap, jit, numpy as jnp
from jaxopt import LBFGS
from common import (unroll_identity_block, normalize_aabb, normalize,
                    unpack_stiffness, Timer, write_triangle_mesh_VC)
from sh_representation import R3_to_repvec, rotvec_n_to_z, rotvec_to_R9, proj_sh4_to_R3, proj_sh4_sdp

import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize

import argparse

import frame_field_utils

import polyscope as ps
from icecream import ic


def solve_least_square(A, b, C, d, soft_weight=0.):

    if isinstance(soft_weight, float) or isinstance(soft_weight, int):
        soft_weight = soft_weight * np.ones(C.shape[0])

    if soft_weight.sum() > 0:
        W = scipy.sparse.diags(soft_weight)
        M = scipy.sparse.vstack([A, W @ C])
        n = np.concatenate([b, W @ d])

        x, info = scipy.sparse.linalg.cg(M.T @ M, M.T @ n)
        assert info == 0

        return x
    else:
        M = scipy.sparse.vstack([
            scipy.sparse.hstack([A, C.T]),
            scipy.sparse.hstack(
                [C, scipy.sparse.csc_matrix((C.shape[0], C.shape[0]))])
        ]).tocsc()
        n = np.concatenate([b, d])

        x = scipy.sparse.linalg.spsolve(M, n)[:len(b)]

        return x


if __name__ == '__main__':
    # enable 64 bit precision
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('-p', type=str, default='2', help='Lp norm.')
    parser.add_argument(
        '-w',
        type=float,
        default=0.,
        help='Boundary weight. If bigger than 0 then treat soft constraints')
    parser.add_argument('--out_path',
                        type=str,
                        default='../results',
                        help='Path to output folder.')
    args = parser.parse_args()

    norm = args.p
    # May need to adjust for different geometry / norm (i.e. 1e-2 for cylinder)
    boundary_weight = args.w
    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_{norm}_oct.obj")

    timer = Timer()

    V, F = igl.read_triangle_mesh(model_path)
    V = normalize_aabb(V)
    NV = len(V)
    VN = igl.per_vertex_normals(V, F)

    timer.log('Load and preprocess mesh')

    L = igl.cotmatrix(V, F)

    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(VN))

    W = scipy.sparse.coo_array((np.ones(7), (np.arange(7), 1 + np.arange(7))),
                               shape=(7, 9)).todense()

    # R9_zn @ sh4 = sh4_4 + c0 sh4_0 + c1 sh4_8
    # => W @ R9_zn @ sh4 = sh4_4
    As = np.einsum('bj,nji->nbi', W, R9_zn)
    b = np.array([0, 0, 0, np.sqrt(7 / 12), 0, 0, 0])

    timer.log('Build stiffness and RHS')
    np.random.seed(0)

    if norm != "2":
        p = 1
        if norm == 'inf':
            # TODO: Implement inf norm.
            # The objective function is min\|{w_e * \|f_e_i - f_e_j\|_2}, e \in \mathcal{E}\|_\infin
            #
            # According to https://math.stackexchange.com/questions/2589887/how-can-the-infinity-norm-minimization-problem-be-rewritten-as-a-linear-program,
            #   the problem can be re-written as
            #       min     z
            #       s.t. z\bm{1} >= w_e * \|f_e_i - f_e_j\|_2, z\bm{1} >= -w_e * \|f_e_i - f_e_j\|_2, for any e \in \mathcal{E}|
            #   which is a linear program with quadratic constraints

            raise NotImplementedError('L infinity norm not implemented')
        else:
            try:
                p = int(norm)
                p = 1 if p == 0 else p
            except:
                raise ValueError('Invalid p norm.')

        x = np.random.randn(NV, 9)
        x = vmap(normalize)(x)
        # We are interested in pair (non diagonal) weight, so no need to negate
        E_i, E_j, E_weight = unpack_stiffness(L)

        @jit
        def loss_func(x):
            # (\sum_e (w * (|sh4_i - sh4_j|_2)^p))^(1 / p)
            edge_energy = jnp.linalg.norm(x[E_i] - x[E_j], ord=2, axis=1)
            loss_smooth = jnp.linalg.norm(E_weight * edge_energy,
                                          ord=p) / len(edge_energy)**(1 / p)
            # Inherently soft constraints...
            loss_align = jnp.linalg.norm(jnp.einsum('bji,bi->bj', As, x) -
                                         b[None, :],
                                         ord=2,
                                         axis=1).mean()
            return loss_smooth + boundary_weight * loss_align

        timer.log('Solve (L-BFGS')

        # TODO: Add rerun for degenerated case
        lbfgs = LBFGS(loss_func)
        x = lbfgs.run(x).params
    else:
        A = scipy.sparse.block_diag(As).tocsc()
        b = np.tile(b, NV)
        L_unroll = unroll_identity_block(-L, 9)

        timer.log('Build sparse system')

        # Cache for further concatenation
        B = A
        bc = b

        while True:
            x = solve_least_square(L_unroll, np.zeros(9 * NV), B, bc,
                                   boundary_weight)
            x = x.reshape(NV, 9)

            # Project back to octahedral variety.
            # Not exactly sure whether the paper refers to SDP or the closest normal aligned projection. Use SDP for now.
            x_proj = proj_sh4_sdp(x)
            d_norm = vmap(jnp.linalg.norm)(x - x_proj)

            # Follow Sec 5.5 Non-Triviality Constraint
            rerun_mask = d_norm > 0.665
            N_rerun = rerun_mask.sum()

            if N_rerun == 0:
                x = x_proj
                break

            print(f'Find {N_rerun} degenerated frame. Re-run...')

            rerun_vid = jnp.where(rerun_mask)[0]

            R = scipy.sparse.hstack([
                scipy.sparse.coo_array(
                    (np.ones(9 * N_rerun),
                     (np.arange(9 * N_rerun),
                      ((9 * rerun_vid)[..., None] +
                       np.arange(9)[None, ...]).reshape(-1))),
                    shape=(9 * N_rerun, 9 * NV)).tocsc(),
            ])
            r = x_proj[rerun_mask].reshape(9 * N_rerun)

            B = scipy.sparse.vstack([A, R])
            bc = np.concatenate([b, r])

        timer.log('Solve (Linear)')

    Rs = proj_sh4_to_R3(x)

    timer.log('Project SO(3)')

    Q = vmap(R3_to_repvec)(Rs, VN)

    timer.log('Project to representation vectors')

    V_vis, F_vis, VC_vis = frame_field_utils.trace(V, F, VN, Q, 4000)

    timer.log('Trace flowlines')

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    mesh.add_vector_quantity("VN", VN)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.show()

    write_triangle_mesh_VC(model_out_path, V_vis, F_vis, VC_vis)
