import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
import pickle

from common import ps_register_curve_network, Timer
from sh_representation import (proj_sh4_to_R3, proj_sh4_sdp)

import frame_field_utils

import polyscope as ps
from icecream import ic


# FIXME: Make it more efficient
def build_traversal_graph(T):
    NT = len(T)

    # Build traversal graph
    # NT x 4 x 3
    E = np.stack([
        np.stack([T[:, 0], T[:, 1]], -1),
        np.stack([T[:, 1], T[:, 2]], -1),
        np.stack([T[:, 2], T[:, 0]], -1),
        np.stack([T[:, 1], T[:, 3]], -1),
        np.stack([T[:, 3], T[:, 2]], -1),
        np.stack([T[:, 2], T[:, 1]], -1),
        np.stack([T[:, 0], T[:, 2]], -1),
        np.stack([T[:, 2], T[:, 3]], -1),
        np.stack([T[:, 3], T[:, 0]], -1),
        np.stack([T[:, 0], T[:, 3]], -1),
        np.stack([T[:, 3], T[:, 1]], -1),
        np.stack([T[:, 1], T[:, 0]], -1)
    ], 1).reshape(-1, 2)

    # F_id // 4 gives T_id
    F_id = np.arange(T.size)

    # Directed face
    # F (0, 1, 2) (1, 3, 2) (0, 2, 3) (0, 3, 1)
    F = np.stack([
        np.stack([T[:, 0], T[:, 1], T[:, 2]], -1),
        np.stack([T[:, 1], T[:, 3], T[:, 2]], -1),
        np.stack([T[:, 0], T[:, 2], T[:, 3]], -1),
        np.stack([T[:, 0], T[:, 3], T[:, 1]], -1)
    ], 1).reshape(-1, 3)

    # Undirected edge, because I don't think traverse tet clockwise / counter-clockwise matters?
    ue, ue_idx, ue_idx_inv = np.unique(np.sort(E, axis=1),
                                       axis=0,
                                       return_index=True,
                                       return_inverse=True)

    E_id = np.arange(len(ue))
    E_map = E_id[np.argsort(ue_idx)]
    E_map_inv = np.zeros((np.max(E_map) + 1,), dtype=np.int64)
    E_map_inv[E_map] = E_id

    # We sort edge by its first occurrence in tet face
    E = ue[E_map]
    F2E = E_map_inv[ue_idx_inv].reshape(NT * 4, 3)

    TF2E = F2E.reshape(NT, 4, 3)

    face_table = np.array([[3, 1, 2], [3, 2, 0], [0, 1, 3], [2, 1, 0]])

    # next face id sharing the same edge in tet
    def next_face_id(e_id, f_id):
        t_id = f_id // 4
        t_f_id_t = f_id % 4

        F2E = TF2E[t_id]
        t_f_e_id = np.argwhere(F2E[t_f_id_t] == e_id)[0][0]
        return t_id * 4 + face_table[t_f_id_t][t_f_e_id]

    # Build F2F
    # Again, not sure if there is a better way than for loop / list comprehension
    # Maybe using F2E?
    F2Fid = dict([(f"{f[0]}_{f[1]}_{f[2]}", f_id) for f, f_id in zip(F, F_id)])

    def opposite_face_id(f_id):
        v0, v1, v2 = F[f_id]

        key_0 = f"{v0}_{v2}_{v1}"
        key_1 = f"{v2}_{v1}_{v0}"
        key_2 = f"{v1}_{v0}_{v2}"

        oppo_f_id = -1

        for key in {key_0, key_1, key_2}:
            if key in F2Fid:
                oppo_f_id = F2Fid[key]

        return oppo_f_id

    F2F = -np.ones(T.size, dtype=np.int64)
    F2F[F_id] = np.array([opposite_face_id(f_id) for f_id in F_id])

    _, uf_inv, uf_count = np.unique(np.sort(F, axis=1),
                                    axis=0,
                                    return_counts=True,
                                    return_inverse=True)

    # TODO: Filter non-manifold
    T_boundary_id = np.unique(F_id[(uf_count == 1)[uf_inv]] // 4)
    E_boundary = np.unique(F2E[(uf_count == 1)[uf_inv]])

    E2F_list = np.vstack([
        np.stack([F2E[:, 0], F_id], -1),
        np.stack([F2E[:, 1], F_id], -1),
        np.stack([F2E[:, 2], F_id], -1)
    ])
    E2F_sort_idx = np.lexsort(E2F_list.T[::-1])
    E2F_list_sorted = E2F_list[E2F_sort_idx]

    _, f_count = np.unique(E2F_list_sorted[:, 0], return_counts=True)

    split_indices = np.cumsum(f_count)[:-1]
    E2F_list = np.split(E2F_list_sorted[:, 1], split_indices)

    # To compute singularity, we need to sort based on one-ring traversal
    def build_one_ring(e_id, f_ids):
        f_id = f_ids[0]
        f_ids_sort = [f_id]

        if e_id in E_boundary:
            reach_boundary = False
            while not reach_boundary:
                prev_f_id = F2F[f_id]
                if prev_f_id == -1:
                    reach_boundary = True
                else:
                    f_id = next_face_id(e_id, prev_f_id)
                    f_ids_sort.append(f_id)

            f_ids_sort = f_ids_sort[::-1]
            f_id = f_ids_sort[-1]

            reach_boundary = False
            while not reach_boundary:
                next_f_id = F2F[next_face_id(e_id, f_id)]
                if next_f_id == -1:
                    reach_boundary = True
                else:
                    f_id = next_f_id
                    f_ids_sort.append(next_f_id)

            if len(f_ids_sort) == 1:
                return np.zeros((0, 2))
            else:
                return np.array([[f_ids_sort[i], f_ids_sort[i + 1]]
                                 for i in range(len(f_ids_sort) - 1)])

        else:
            anchor_id = f_id
            reach_boundary = False
            while not reach_boundary:
                next_f_id = F2F[next_face_id(e_id, f_id)]
                if next_f_id == anchor_id:
                    reach_boundary = True
                else:
                    f_id = next_f_id
                    f_ids_sort.append(next_f_id)

            return np.array(
                [[f_ids_sort[i], f_ids_sort[i + 1]] if i != len(f_ids_sort) - 1
                 else [f_ids_sort[i], f_ids_sort[0]]
                 for i in range(len(f_ids_sort))])

    E2DE = [
        build_one_ring(e_id, el)
        for (e_id, el) in zip(np.arange(len(E2F_list)), E2F_list)
    ]

    return E, E2DE, E_boundary


def edge_one_ring(V, T):
    # Faces 0:012 1:013 2:123 3:203
    TT, TTi = igl.tet_tet_adjacency(T)

    # Edges (undirected) 0:01 1:02 2:03 3:12 4:13 5:23
    TE_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    # E2F 0:01 1:03 2:13 3:02 4:12 5:23
    EF_list = [[0, 1], [0, 3], [1, 3], [0, 2], [1, 2], [2, 3]]

    NT = len(T)

    # Hash
    def get_uEid(e):
        return f"{e[0]}_{e[1]}"

    def get_uE(i0, i1):
        if i0 < i1:
            e = np.array([i0, i1], dtype=np.int64)
        else:
            e = np.array([i1, i0], dtype=np.int64)
        return e, get_uEid(e)

    uE2uEid = {}
    # Use list to preserve order
    uE = []
    uE2T = []
    uE_boundary_mask = []
    # If the adjacent T is boundary w.r.t. ue
    uE2T_boundary = []
    # Maps each edge of tet to uE_id
    E2uE = -np.ones((NT, 6), dtype=np.int64)
    # Maps each edge of tet to id of adjacent tets w.r.t. the edge
    E2T = -np.ones((NT, 6, 2), dtype=np.int64)
    uE_count = 0
    for i in range(NT):
        for j in range(6):
            i0, i1 = TE_list[j]
            E2T[i, j] = TT[i][EF_list[j]]
            # If it is on boundary face
            # In other words, if the edge is not boundary, it should be adjacent to a loop of tets
            is_boundary = (E2T[i, j] == -1).sum() > 0
            ue, ue_tag = get_uE(T[i, i0], T[i, i1])

            if ue_tag not in uE2uEid:
                uE2uEid[ue_tag] = uE_count
                uE.append(ue)
                uE2T.append([i])
                uE_boundary_mask.append(is_boundary)
                uE2T_boundary.append([is_boundary])
                E2uE[i, j] = uE_count
                uE_count += 1
            else:
                ue_id = uE2uEid[ue_tag]
                E2uE[i, j] = ue_id
                uE2T[ue_id].append(i)
                uE2T_boundary[ue_id].append(is_boundary)

                if is_boundary:
                    uE_boundary_mask[ue_id] = True

    uE = np.array(uE)
    uE_boundary_mask = np.array(uE_boundary_mask)

    # Sort one-ring
    uE_mark = np.zeros(uE_count, dtype=bool)

    def next_t_id(t_id, e_id, last_t_id):
        for t_id_ in E2T[t_id, e_id]:
            if t_id_ != last_t_id:
                return t_id, t_id_

    for i in range(NT):
        for j in range(6):
            ue_id = E2uE[i, j]

            if not uE_mark[ue_id]:
                uE_mark[ue_id] = True
                T_adj = uE2T[ue_id]

                # Since the edge is undirected, the traverse order (clockwise / counterclockwise) doesn't matter
                if len(T_adj) < 3:
                    continue

                if uE_boundary_mask[ue_id]:
                    t_id_boundary = np.array(T_adj)[uE2T_boundary[ue_id]]
                    # assert len(t_id_boundary) == 2
                    t_id = t_id_boundary[0]
                    t_id_end = t_id_boundary[1]
                else:
                    t_id = i
                    t_id_end = i

                t_id_last = -1
                T_adj_sorted = []

                non_manifold = False
                finished = False
                while not finished:
                    e_id = np.argwhere(E2uE[t_id] == ue_id)[0][0]
                    t_id_last, t_id = next_t_id(t_id, e_id, t_id_last)
                    # assert t_id in T_adj

                    # Non-manifold one-ring
                    if t_id == -1:
                        non_manifold = True
                        break

                    finished = (t_id == t_id_end)
                    T_adj_sorted.append(t_id)

                if non_manifold:
                    uE2T[ue_id] = T_adj
                else:
                    if uE_boundary_mask[ue_id]:
                        T_adj_sorted.insert(0, t_id_boundary[0])

                    # assert len(T_adj_sorted) == len(T_adj)
                    uE2T[ue_id] = np.array(T_adj_sorted, dtype=np.int64)

    return uE, uE_boundary_mask, uE2T


# Let basis R = [x|y|z], where x, y, z are basis vectors in world coordinate
# For a vector v, we have
#   x * v[0] + y * v[1] + z * v[2] = R @ v
# v is the local coordinate of R, R @ v transforms v to world coordinate
#
# If v_i, v_j represents the same vector in world coordinate, we have
#   v = R_j @ v_j = R_i @ v_i
# Thus
#   v_j \approx \PI_{ji} @ v_i, where \PI_{ji} \approx R_j.T @ R_i
@jit
def transition_matrix(R_i, R_j):

    @jit
    def match_basis(e, R):
        dp = jnp.einsum('b,ba->a', e, R)
        max_idx = jnp.argmax(jnp.abs(dp))
        return max_idx, jnp.sign(dp)[max_idx]

    idx, sign = vmap(match_basis, in_axes=(1, None))(R_i, R_j)
    return jnp.eye(3)[:, idx] * sign[None, :]


@jit
def is_singular(T_adj, Rs):
    R_adj = Rs[T_adj]
    transitions = vmap(transition_matrix)(R_adj, jnp.roll(R_adj, -1, axis=0))

    @jit
    def body_func(i, m):
        return transitions[i] @ m

    m = jax.lax.fori_loop(0, len(transitions), body_func, jnp.eye(3))
    return jnp.linalg.norm(m - jnp.eye(3)) > 1e-7


def edge_singularity(uE, uE_boundary_mask, uE2T):
    uE_interior_mask = np.logical_not(uE_boundary_mask)
    uE_singularity_mask = np.zeros(len(uE), dtype=bool)
    uE_singularity_mask[uE_interior_mask] = np.array([
        is_singular(uE2T[ue_id], Rs_bary)
        for ue_id in np.arange(len(uE))[uE_interior_mask]
    ])

    return uE_singularity_mask


# https://stackoverflow.com/questions/2933470/how-do-i-call-setattr-on-the-current-module
def set_var_by_name(name, val):
    globals()[name] = val


def save_tmp(vars: dict, name='tmp', folder='tmp'):
    with open(f'{folder}/{name}.pkl', 'wb') as f:
        pickle.dump(vars, f)


def load_tmp(name='tmp', folder='tmp'):
    with open(f'{folder}/{name}.pkl', 'rb') as f:
        data = pickle.load(f)

    for key in data.keys():
        set_var_by_name(key, data[key])

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()

    timer = Timer()

    if args.config is not None:
        name = args.config.split('/')[-1].split('.')[0]
        data = np.load(f'output/{name}.npz')
        V_mc, F_mc = igl.read_triangle_mesh(f'output/{name}_mc.obj')
    else:
        data = np.load('results/prism_prac.npz')
        # data = np.load('results/join_prac.npz')

    V = np.float64(data['V'])
    T = np.int64(data['T'])
    sh4: np.array = data['sh4']

    timer.log('Load data')

    T = frame_field_utils.tet_fix_index_order(V, T)

    timer.log('Fix index order')

    # L = igl.cotmatrix(V, T)
    # M = igl.massmatrix(V, T)
    # M_inv = scipy.sparse.diags(1 / M.diagonal())

    # # point-wise energy
    # V_energy = vmap(jnp.linalg.norm)(M_inv @ (-L) @ sh4)

    # ps.init()
    # pc = ps.register_volume_mesh('pc', V, T)
    # if V_mc is not None:
    #     ps.register_surface_mesh('mc', V_mc, F_mc)
    # pc.add_scalar_quantity('Energy', V_energy, enabled=True)
    # ps.show()

    # Use tet based representation, so the singularities are defined on edges, allowing us to cut directly along faces
    # Follow "Boundary Aligned Smooth 3D Cross-Frame Field" by Jin Huang et al., we transform from vertex based to tet via simple averaging
    V_bary = V[T].mean(axis=1)
    sh4_bary = sh4[T].mean(axis=1)
    Rs_bary = proj_sh4_to_R3(sh4_bary)

    # Technically, we could directly infer sh4 at each tets barycenter, but given NT >> NV, SDP would be extremely heavy..
    sh4_octa = proj_sh4_sdp(sh4)
    # Small linear interpolated sh4 won't deviate much from the variety, hence its recovery would be sufficiently accurate
    sh4_bary_octa = sh4_octa[T].mean(axis=1)
    Rs_bary_octa = proj_sh4_to_R3(sh4_bary_octa)

    timer.log('Project and interpolate SH4')

    TT, TTi = igl.tet_tet_adjacency(T)
    uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE, E2T = frame_field_utils.tet_edge_one_ring(
        T, TT)

    timer.log('Build edge one ring')

    uE_singularity_mask = frame_field_utils.tet_frame_singularity(
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, Rs_bary)
    uE_singularity_mask_octa = frame_field_utils.tet_frame_singularity(
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum,
        Rs_bary_octa)

    timer.log('Compute singularity')

    save_tmp({
        'V': V,
        'T': T,
        'Rs': Rs_bary_octa,
        'sh4s': sh4_bary_octa,
        'uE': uE,
        'uE_boundary_mask': uE_boundary_mask,
        'uE_non_manifold_mask': uE_non_manifold_mask,
        'uE_singularity_mask': uE_singularity_mask_octa,
        'uE2T': uE2T,
        'uE2T_cumsum': uE2T_cumsum,
        'E2uE': E2uE,
        'E2T': E2T
    })

    F = igl.boundary_facets(T)
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)

    ps.init()
    ps.register_surface_mesh('tet', V, F, enabled=False)
    if V_mc is not None:
        ps.register_surface_mesh('mc', V_mc, F_mc)
    if uE_singularity_mask.sum() > 0:
        ps_register_curve_network('singularity',
                                  V,
                                  uE[uE_singularity_mask],
                                  enabled=False)
        ps_register_curve_network('singularity SDP', V,
                                  uE[uE_singularity_mask_octa])
    ps.show()
