import igl
import numpy as np
import jax
from jax import vmap, jit, numpy as jnp
from jax.experimental import sparse
from jaxopt import LBFGS

# Facilitate vscode intellisense
import scipy.sparse
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
from common import unroll_identity_block, normalize_aabb, vis_oct_field, ps_register_curve_network
from sh_representation import (proj_sh4_to_rotvec, R3_to_repvec,
                               rotvec_to_sh4_expm, rotvec_n_to_z, rotvec_to_R3,
                               rotvec_to_R9, sh4_z, R3_to_sh4_zonal)

import open3d as o3d
import argparse
import os

import flow_lines

import polyscope as ps
from icecream import ic


def handle_sharp_vertices(V, F, sharp_angle=45):
    SE, _, _, _, _, _ = igl.sharp_edges(V, F, sharp_angle / 180 * np.pi)

    # Use directed edge for traversal
    SDE = np.vstack([SE, SE[:, ::-1]])
    E_sort_idx = np.argsort(SDE[:, 0])
    uV, uV_count = np.unique(SDE[:, 0][E_sort_idx], return_counts=True)
    Vid_joint = uV[np.where(uV_count > 2)[0]]
    V2V = dict(
        zip(uV, np.split(SDE[:, 1][E_sort_idx],
                         np.cumsum(uV_count)[:-1])))

    # For directed edges, force them to consistent along the path
    Vid_start = uV[np.where(uV_count != 2)[0]]
    V_mask = np.zeros(len(V), dtype=bool)
    V2V_ordered = {}

    def traverse(v):
        V_mask[v] = True
        for vi in V2V[v]:
            if not V_mask[vi]:

                # For whatever reason, it could be double?!
                v = np.int64(v)
                vi = np.int64(vi)

                if v not in V2V_ordered:
                    V2V_ordered[v] = [vi]
                else:
                    V2V_ordered[v].append(vi)

                traverse(vi)

    # Ignore joint
    V_mask[Vid_joint] = True
    for vi in Vid_start:
        traverse(vi)

    # For cycles
    Vid_cycle_start = []
    for vi in uV[np.logical_not(V_mask[uV])]:
        if not V_mask[vi]:
            traverse(vi)
            Vid_cycle_start.append(vi)

    # print(f"Find {len(Vid_cycle_start)} cycles")

    Vid_start = np.concatenate([Vid_cycle_start, Vid_start])

    # Trace edges to get path
    path_list = []

    for vi in Vid_start:
        if vi in V2V_ordered:
            for vj in V2V_ordered[vi]:

                # For whatever reason, it could be double?!
                vi = np.int64(vi)
                vj = np.int64(vj)

                path = [vi]
                end = False
                while not end:
                    path.append(vj)

                    if vj in V2V_ordered:
                        vj = V2V_ordered[vj][0]
                    else:
                        end = True

                path_list.append(path)

    # Form a cycle
    for i in range(len(Vid_cycle_start)):
        v0 = np.int64(path_list[i][-1])
        v1 = np.int64(Vid_cycle_start[i])
        path_list[i].append(v1)
        V2V_ordered[v0] = [v1]

    # ps.init()
    # for i in range(len(path_list)):
    #     path = np.array(path_list[i])
    #     edges = np.stack([path[:-1], path[1:]], -1)
    #     ps_register_curve_network(f'{i}', V, edges)
    # ps.show()
    # return

    # Find adjacent faces
    V2T = {}
    V2Ti = {}
    V2T_joint = {}
    V2Ti_joint = {}

    TT, TTi = igl.triangle_triangle_adjacency(F)
    FN = igl.per_face_normals(V, F, np.array([0., 0., 0.])[None, :])

    for i in range(len(F)):
        for j in range(3):
            v0 = F[i][j]
            v1 = F[i][(j + 1) % 3]

            # If the directed edge lies on this triangle
            # Edge direction matters because we want consistent normal
            if v0 in V2V_ordered and v1 in V2V_ordered[v0]:

                # **IMPORTANT** Match the vertex order of V2V_ordered
                idx = np.argwhere(V2V_ordered[v0] == v1)[0][0]

                # Index of current triangle
                if v0 not in V2T:
                    V2T[v0] = [0] * len(V2V_ordered[v0])
                V2T[v0][idx] = i

                # Index of adjacent triangle sharing the edge
                if v0 not in V2Ti:
                    V2Ti[v0] = [0] * len(V2V_ordered[v0])
                V2Ti[v0][idx] = TT[i][j]

            # For joint, we ignore edge direction and think it as a heat source
            # So it's always from joint to neighbor (order doesn't matter here)
            if v0 in Vid_joint:
                if v0 in V2V and v1 in V2V[v0]:

                    # Index of current triangle
                    if v0 not in V2T_joint:
                        V2T_joint[v0] = [i]
                    else:
                        V2T_joint[v0].append(i)

                    # Index of adjacent triangle sharing the edge
                    if v0 not in V2Ti_joint:
                        V2Ti_joint[v0] = [TT[i][j]]
                    else:
                        V2Ti_joint[v0].append(TT[i][j])

    # Note the adjacent of triangles of joint sharp edges may NOT form one-ring
    # I don't think there is a better approach, unless I can trace a path that has at least one consistent normal?
    def compatible_oct_joint(v0):
        fns = np.vstack([FN[V2T_joint[v0]], FN[V2Ti_joint[v0]]])
        dps = np.abs(np.einsum('ni,mi->nm', fns, fns))

        xx, yy = np.where(dps == np.min(dps))

        n0 = fns[xx[0]]
        n2 = np.cross(n0, fns[yy[0]])
        return np.stack([n0, np.cross(n0, n2), n2], -1)

    if len(Vid_joint) > 0:
        # First fix joint (> 2 adjacent sharp edges)
        Rs_sharp = np.stack([compatible_oct_joint(v) for v in Vid_joint])
        Vid_sharp = Vid_joint
    else:
        Vid_sharp = np.zeros((0,), dtype=np.int64)
        Rs_sharp = np.zeros((0, 3, 3), dtype=np.float64)

    # Directed, just pick left triangle and edge dir (I don't think there is much else I can do)
    def compatible_oct_edge(e):
        v0 = e[0]
        v1 = e[1]
        fid = V2T[v0][np.where(V2V_ordered[v0] == v1)[0][0]]

        n0 = V[v1] - V[v0]
        n0 /= (np.linalg.norm(n0) + 1e-8)
        # Face normal is always orthogonal to one of its edge
        n1 = FN[fid]
        n1 /= (np.linalg.norm(n1) + 1e-8)

        return np.stack([n0, n1, np.cross(n0, n1)], -1)

    for i in range(len(path_list)):
        path = np.array(path_list[i])
        edges = np.stack([path[:-1], path[1:]], -1)

        Vid_path = edges[:, 1]
        Rs_path = np.stack([compatible_oct_edge(e) for e in edges])

        v0 = edges[0, 0]
        if v0 not in Vid_joint and v0 not in Vid_cycle_start:
            Vid_path = np.concatenate([[v0], Vid_path])
            Rs_path = np.vstack([[Rs_path[0]], Rs_path])

        Vid_sharp = np.concatenate([Vid_sharp, Vid_path])
        Rs_sharp = np.vstack([Rs_sharp, Rs_path])

    assert len(Vid_sharp) == len(Rs_sharp)

    # FIXME: Figure out why this check fails
    # assert np.alltrue(np.sort(Vid_sharp) == uV)

    return Vid_sharp, Rs_sharp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('-w', type=float, default=100, help='Boundary weight')
    parser.add_argument('--out_path',
                        type=str,
                        default='results',
                        help='Path to output folder.')
    parser.add_argument('--sharp',
                        action='store_true',
                        help='Handle sharp edge')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save for seamless parameterization')
    args = parser.parse_args()

    # Large alignment weight to ensure boundary align
    boundary_weight = args.w
    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_prac.obj")

    print("Load and preprocess tetrahedral mesh")

    V, T, _ = igl.read_off(model_path)
    V = normalize_aabb(V)
    F = igl.boundary_facets(T)
    # boundary_facets gives opposite orientation for some reason
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)
    boundary_vid = np.unique(F)

    NV = len(V)

    VN = igl.per_vertex_normals(V, F)

    if args.sharp:
        # Handle non-smooth vertices (paper precomputes and fixes those SH coefficients)
        sharp_vid, Rs_sharp = handle_sharp_vertices(V, F)
        sh4_sharp = vmap(R3_to_sh4_zonal)(Rs_sharp)
        NS = len(sharp_vid)
        boundary_vid = boundary_vid[np.logical_not(
            np.in1d(boundary_vid, sharp_vid))]

    NB = len(boundary_vid)

    print("Build stiffness and RHS")

    # Cotangent weights
    L = igl.cotmatrix(V, T)
    rotvec_zn = vmap(rotvec_n_to_z)(VN[boundary_vid])
    R9_zn = vmap(rotvec_to_R9)(rotvec_zn)

    sh4_0 = jnp.array([jnp.sqrt(5 / 12), 0, 0, 0, 0, 0, 0, 0, 0])
    sh4_4 = jnp.array([0, 0, 0, 0, jnp.sqrt(7 / 12), 0, 0, 0, 0])
    sh4_8 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, jnp.sqrt(5 / 12)])

    # R9_zn @ sh4 = sh4_4 + c0 sh4_0 + c1 sh4_8
    # => sh4 = R9_zn.T @ sh4_4 + c0 R9_zn.T @ sh4_0 + c1 R9_zn.T @ sh4_8
    sh4_0_n = jnp.einsum('bji,j->bi', R9_zn, sh4_0)
    sh4_4_n = jnp.einsum('bji,j->bi', R9_zn, sh4_4)
    sh4_8_n = jnp.einsum('bji,j->bi', R9_zn, sh4_8)

    print("Build sparse system")

    # Build system
    # NV x 9 + NB x 2, have to unroll...
    # Assume x is like [9, 9, 9, 9, 9 ... 2, 2, 2]
    A_tl = unroll_identity_block(L, 9)
    A_tr = scipy.sparse.csr_matrix((NV * 9, NB * 2))
    # Pick boundary vertex
    A_bl = scipy.sparse.coo_array(
        (boundary_weight * np.ones(9 * NB),
         (np.arange(9 * NB), ((9 * boundary_vid)[..., None] +
                              np.arange(9)[None, ...]).reshape(-1))),
        shape=(9 * NB, 9 * NV)).tocsc()
    A_br = scipy.sparse.block_diag(boundary_weight *
                                   np.stack([sh4_0_n, sh4_8_n], -1))
    A = scipy.sparse.vstack(
        [scipy.sparse.hstack([A_tl, A_tr]),
         scipy.sparse.hstack([A_bl, A_br])])
    b = np.concatenate(
        [np.zeros((NV * 9,)), boundary_weight * sh4_4_n.reshape(-1,)])

    if args.sharp:
        sharp_weight = boundary_weight
        # Pick sharp vertex
        S_l = scipy.sparse.hstack([
            scipy.sparse.coo_array(
                (sharp_weight * np.ones(9 * NS),
                 (np.arange(9 * NS), ((9 * sharp_vid)[..., None] +
                                      np.arange(9)[None, ...]).reshape(-1))),
                shape=(9 * NS, 9 * NV)).tocsc(),
        ])
        S_r = scipy.sparse.csr_matrix((NS * 9, NB * 2))

        A = scipy.sparse.vstack([A, scipy.sparse.hstack([S_l, S_r])])
        b = np.concatenate([b, sharp_weight * sh4_sharp.reshape(-1)])

    print("Solve alignment (Linear)")

    # A @ x = b
    # => (A.T @ A) @ x = A.T @ b
    factor = cholesky((A.T @ A).tocsc())
    x = factor(A.T @ b)
    sh4_opt = x[:NV * 9].reshape(NV, 9)

    print("Project SO(3)")

    # Project to acquire initialize
    rotvecs = proj_sh4_to_rotvec(sh4_opt)

    # Optimize field via non-linear objective function
    R9_zn_pad = jnp.repeat(jnp.eye(9)[None, ...], NV, axis=0)
    R9_zn_pad = R9_zn_pad.at[boundary_vid].set(R9_zn)

    L_jax = sparse.BCOO.from_scipy_sparse(-L)

    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(key, (len(boundary_vid),))
    params = {'rotvec': rotvecs, 'theta': theta}

    @jit
    def loss_func(params):
        rotvec = params['rotvec']
        theta = params['theta']

        # LBFGS is second-order optimization method, has to use expm implementation here
        sh4 = vmap(rotvec_to_sh4_expm)(rotvec)

        # sh4_n = R9_zn.T @ sh4_z
        sh4_n = jnp.einsum('bji,bj->bi', R9_zn, vmap(sh4_z)(theta))
        # Use theta parameterization for boundary
        sh4 = sh4.at[boundary_vid].set(sh4_n)

        if args.sharp:
            # Preserve sharp
            sh4 = sh4.at[sharp_vid].set(sh4_sharp)

        # Integrated quantity
        loss_smooth = jnp.trace(sh4.T @ L_jax @ sh4)
        return loss_smooth

    print("Solve smoothness (L-BFGS)")

    lbfgs = LBFGS(loss_func)
    params = lbfgs.run(params).params

    rotvecs_opt = params['rotvec']
    Rs = vmap(rotvec_to_R3)(rotvecs_opt)

    # Recovery boundary rotation
    theta = params['theta']
    rotvec_z = theta[..., None] * jnp.array([0, 0, 1])[None, ...]
    Rz = vmap(rotvec_to_R3)(rotvec_z)
    R3_zn = vmap(rotvec_to_R3)(rotvec_zn)
    # R_n = R3_zn.T @ Rz
    Rs = Rs.at[boundary_vid].set(jnp.einsum('bjk,bji->bki', R3_zn, Rz))

    if args.save:
        print("Save for parameterization")

        # Save SH parametrization
        sh4 = vmap(R3_to_sh4_zonal)(Rs)

        param_path = os.path.join(args.out_path, f"{model_name}_prac.npz")
        np.savez(param_path, V=V, T=T, Rs=Rs, sh4=sh4)

    V_vis_cube, F_vis_cube = vis_oct_field(Rs, V,
                                           0.1 * igl.avg_edge_length(V, T))

    print("Trace flowlines")

    Q = vmap(R3_to_repvec)(Rs, VN)

    V_vis, F_vis, VC_vis = flow_lines.trace(V, F, VN, Q, 4000)

    ps.init()
    tet = ps.register_volume_mesh("tet", V, T)
    tet.add_vector_quantity("VN", VN)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.register_surface_mesh("cube", V_vis_cube, F_vis_cube, enabled=False)
    ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(model_out_path, stroke_mesh)
