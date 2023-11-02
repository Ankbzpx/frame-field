import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
import pickle
import queue

from common import (ps_register_curve_network, Timer, normalize,
                    unroll_identity_block, ps_register_basis,
                    surface_vertex_topology, rm_unref_vertices)
from sh_representation import (proj_sh4_to_R3, proj_sh4_sdp)

import frame_field_utils
import scipy.sparse

from sksparse.cholmod import cholesky

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


# Transform R_j so that it is compatible with R_i
@jit
def make_compatible(R_i, R_j):

    @jit
    def match_basis(e, R):
        dp = jnp.einsum('b,ba->a', e, R)
        max_idx = jnp.argmax(jnp.abs(dp))
        return max_idx, jnp.sign(dp)[max_idx]

    idx, sign = vmap(match_basis, in_axes=(1, None))(R_i, R_j)

    return R_j[:, idx] * sign[None, :]


def comb_oct_field(T, Rs, sh4s, TT):
    NT = len(T)

    Rs_comb = np.zeros_like(Rs)
    T_mark = np.zeros(NT).astype(bool)

    # Minimal spanning tree
    bfs_queue = queue.PriorityQueue()

    for t_id_start in range(NT):
        if T_mark[t_id_start]:
            continue

        bfs_queue.put((0, t_id_start))
        T_mark[t_id_start] = True
        Rs_comb[t_id_start] = Rs[t_id_start]

        while not bfs_queue.empty():
            _, t_i = bfs_queue.get()
            sh4_i = sh4s[t_i]
            R_i = Rs_comb[t_i]

            for t_j in TT[t_i]:

                if t_j == -1 or T_mark[t_j]:
                    continue

                sh4_j = sh4s[t_j]
                R_j = Rs[t_j]

                Rs_comb[t_j] = make_compatible(R_i, R_j)

                # Smoothness first
                smoothness = np.linalg.norm(sh4_i - sh4_j)
                smooth_weight = smoothness
                bfs_queue.put((smooth_weight, t_j))
                T_mark[t_j] = True

    assert T_mark.sum() == NT

    return Rs_comb


def mark_oct_field_mismatch(T, Rs, TT, TTi):
    NT = len(T)
    TT_mismatch = np.zeros_like(T).astype(bool)

    for t_i in range(NT):
        R_i = Rs[t_i]
        for j in range(4):

            if TT_mismatch[t_i, j]:
                continue

            t_j = TT[t_i, j]

            if t_j == -1:
                continue

            R_j = Rs[t_j]

            T_ji = transition_matrix(R_i, R_j)

            mismatch = np.linalg.norm(T_ji - np.eye(3)) > 1e-7

            if mismatch:
                TT_mismatch[t_i, j] = True
                TT_mismatch[t_j, TTi[t_i, j]] = True

    return TT_mismatch


def tet_uF_count(T, TT, TTi):
    NT = len(T)
    F_mark = np.zeros_like(T).astype(bool)
    uF_count = 0
    for t_i in range(NT):
        for j in range(4):

            if F_mark[t_i, j]:
                continue

            t_j = TT[t_i, j]

            if t_j == -1:
                F_mark[t_i, j] = True
                uF_count += 1
                continue

            F_mark[t_i, j] = True
            F_mark[t_j, TTi[t_i, j]] = True
            uF_count += 1

    return uF_count


# Assume convex tet of genus 0
def is_tet_manifold(T, TT, TTi, uE=None, F2uF=None, **kwargs):
    NV = len(np.unique(T))
    NC = len(T)

    if F2uF is None:
        F2uF = frame_field_utils.tet_uF(T, TT, TTi)

    NF = F2uF.max() + 1

    if uE is None:
        E = np.stack([
            np.stack([T[:, 0], T[:, 1]], -1),
            np.stack([T[:, 0], T[:, 2]], -1),
            np.stack([T[:, 0], T[:, 3]], -1),
            np.stack([T[:, 1], T[:, 2]], -1),
            np.stack([T[:, 1], T[:, 3]], -1),
            np.stack([T[:, 2], T[:, 3]], -1)
        ], 1).reshape(-1, 2)

        NE = len(np.unique(np.sort(E, axis=-1), axis=0))
    else:
        NE = len(uE)

    return NV - NE + NF - NC == 1


def extract_seams(Rs_comb, F, T, TT, TTi, F2uF, uF2uE, **kwargs):
    TT_mismatch = frame_field_utils.tet_frame_mismatch(T, TT, TTi, Rs_comb)
    uF_seam, uF_seam_idx = np.unique(F2uF[TT_mismatch], return_index=True)
    F_seam = F[TT_mismatch][uF_seam_idx]
    uE_seam = np.unique(uF2uE[uF_seam])
    V_seam = np.unique(F_seam)
    return V_seam, uE_seam, uF_seam, F_seam


def dissolve_non_manifold(Rs_comb, uE_seam, uF_seam, uE_singularity_mask, uE2T,
                          uE2T_cumsum, uE2uF, uE2uF_cumsum, uF2uE, F2uF,
                          **kwargs):
    uF_seam_mask = np.zeros(F2uF.max() + 1).astype(bool)
    uF_seam_mask[uF_seam] = True

    uE_F_seams_count = np.array([
        uF_seam_mask[uE2uF[uE2uF_cumsum[uE_id]:uE2uF_cumsum[uE_id + 1]]].sum()
        for uE_id in uE_seam
    ])

    # Filter out irregular edges
    uE_non_manifold_singular = uE_seam[uE_F_seams_count == 3]
    # TODO: Non-manifold interior seams fine? Maybe we should relax it?
    uE_non_manifold_interior = uE_seam[uE_F_seams_count == 4]

    # TODO: Figure out what happen if singular edge forms a node. Catch it for now
    assert np.all(uE_singularity_mask[uE_non_manifold_singular])
    assert np.all(np.logical_not(uE_singularity_mask[uE_non_manifold_interior]))

    # Dissolve non-manifold comb
    # TODO: Don't know how correct this method is, or whether it would introduce more singularities
    uE_ids_non_manifold = np.concatenate(
        [uE_non_manifold_singular, uE_non_manifold_interior])

    if len(uE_ids_non_manifold) == 0:
        return Rs_comb, False

    print(
        f"Dissolve {len(uE_ids_non_manifold)} non-manifold edges. {len(uE_non_manifold_singular)} occurs at singular edge"
    )

    for uE_id in uE_ids_non_manifold:
        T_adj = uE2T[uE2T_cumsum[uE_id]:uE2T_cumsum[uE_id + 1]]
        Rs_one_ring = Rs_comb[T_adj]
        transitions = vmap(transition_matrix)(Rs_one_ring,
                                              np.roll(Rs_one_ring, -1, axis=0))

        start_idx = -1
        for i in range(len(transitions)):
            if np.linalg.norm(transitions[i] - np.eye(3)) < 1e-7:
                start_idx = i

        # TODO: Don't know if it's possible to have one singular edge with full singular neighbor
        #       while also being combed non-manifold
        assert start_idx != -1

        idx_reordered = (np.arange(len(transitions)) +
                         start_idx) % len(transitions)

        # Start with identity transition, dissolve the first two flip
        count = 0
        for i in idx_reordered:
            t_i = transitions[i]
            j = (i + 1) % len(transitions)
            t_j = transitions[j]

            if np.linalg.norm(t_i - t_j) > 1e-7:
                count += 1

                if count > 2:
                    break

                Rs_comb[T_adj[j]] = make_compatible(Rs_comb[T_adj][i],
                                                    Rs_comb[T_adj][j])
    return Rs_comb, True


def dissolve_uncuttable(Rs_comb, uE_seam, uF_seam, uE, uE_singularity_mask,
                        uE_ids_singular, V_singular_mask, uF2uE, F2uF, uF2T,
                        uE2uF, uE2uF_cumsum, **kwargs):
    # Update the mask
    uF_seam_mask = np.zeros(F2uF.max() + 1).astype(bool)
    uF_seam_mask[uF_seam] = True

    # BFS all seam faces
    Q_bfs = queue.Queue()
    uE_mark = np.zeros(len(uE)).astype(bool)
    uF_mark = np.zeros(F2uF.max() + 1).astype(bool)

    uF_seam_uncuttable = []

    # TODO: Use uF based BFS
    for uE_id_start in uE_ids_singular:

        if uE_mark[uE_id_start]:
            continue

        uE_mark[uE_id_start] = True
        Q_bfs.put(uE_id_start)

        while not Q_bfs.empty():
            uE_id = Q_bfs.get()
            uF_ids = uE2uF[uE2uF_cumsum[uE_id]:uE2uF_cumsum[uE_id + 1]]

            for uF_id in uF_ids:

                if not uF_seam_mask[uF_id]:
                    continue

                if uF_mark[uF_id]:
                    continue

                uE_ids_next = uF2uE[uF_id]

                # Verify if the seam can be cut
                cuttable = True

                # Uncuttable if the seam face has two singular edges
                if uE_singularity_mask[uE_ids_next].sum() >= 2:
                    cuttable = False

                # Uncuttable if any tets adjacent to seam face have 4 singular vertices
                if np.any(V_singular_mask[T[uF2T[uF_id]]].sum(1) == 4):
                    cuttable = False

                if not cuttable:
                    uF_seam_uncuttable.append(uF_id)

                for uE_id_next in uE_ids_next:

                    if uE_mark[uE_id_next]:
                        continue

                    uE_mark[uE_id_next] = True
                    Q_bfs.put(uE_id_next)

                uF_mark[uF_id] = True

    # Verify that seams have been fully traversed
    assert uE_mark.sum() == len(uE_seam)
    assert uF_mark.sum() == len(uF_seam)

    if len(uF_seam_uncuttable) == 0:
        return Rs_comb, False

    print(f"Dissolve {len(uF_seam_uncuttable)} uncuttable faces")
    dirty = False

    for uF_seam_invalid in uF_seam_uncuttable:
        # Get its adjacent tets
        t_ids = uF2T[uF_seam_invalid]
        t_ids_cuttable = np.argwhere(V_singular_mask[T[t_ids]].sum(1) != 4)

        if len(t_ids_cuttable) == 0:
            # It means both tets are uncuttable. Needs further dissolve
            t_pick = 0
            print(f"Seam between tets {t_ids} cannot be dissolved")
            dirty = True
        else:
            t_pick = t_ids_cuttable[0, 0]

        t_id = t_ids[t_pick]
        t_id_other = t_ids[0] if t_pick == 1 else t_ids[1]

        Rs_comb[t_id] = make_compatible(Rs_comb[t_id_other], Rs_comb[t_id])

    return Rs_comb, dirty


def tet_cut_coloring(V_seam, uE_seam, uF_seam, F_seam, V, T, TT, uE,
                     V_singular_mask, F2uF, uF2T, uF2uE, uE2uF, uE2uF_cumsum,
                     **kwargs):
    # Update the mask
    uE_seam_mask = np.zeros(len(uE)).astype(bool)
    uE_seam_mask[uE_seam] = True

    uF_seam_mask = np.zeros(F2uF.max() + 1).astype(bool)
    uF_seam_mask[uF_seam] = True

    # Seam vertex mask
    V_seam_mask = np.zeros(len(V)).astype(bool)
    V_seam_mask[V_seam] = True

    # Traverse all seams and fix face orientation
    Q_bfs = queue.Queue()
    uF_mark = np.zeros(F2uF.max() + 1).astype(bool)

    uF = np.zeros((F2uF.max() + 1, 3)).astype(np.int64)
    uF[uF_seam] = F_seam
    uFN = np.zeros((F2uF.max() + 1, 3)).astype(np.float64)
    T_coloring_mask = np.zeros(len(T)).astype(bool)

    V2T_seed = np.zeros(len(V)).astype(np.int64)
    V2T_seed_inv = np.zeros(len(V)).astype(np.int64)

    for uF_id_start in uF_seam:

        if uF_mark[uF_id_start]:
            continue

        Q_bfs.put(uF_id_start)
        uF_mark[uF_id_start] = True

        while not Q_bfs.empty():
            uF_id = Q_bfs.get()
            f = uF[uF_id]

            t_ids_adj = uF2T[uF_id]
            t_barys_adj = V[T[t_ids_adj]].mean(1)
            verts = V[f]
            vn = np.cross(normalize(verts[1] - verts[0]),
                          normalize(verts[2] - verts[0]))

            uFN[uF_id] = vn
            if np.dot(t_barys_adj[0] - t_barys_adj[1], vn) > 0:
                t_id = t_ids_adj[0]
                t_id_inv = t_ids_adj[1]
            else:
                t_id = t_ids_adj[1]
                t_id_inv = t_ids_adj[0]

            # Prevent leak
            V2T_seed[f] = t_id
            V2T_seed_inv[f] = t_id_inv
            T_coloring_mask[t_id] = True

            for uE_id in uF2uE[uF_id]:

                if not uE_seam_mask[uE_id]:
                    continue

                uF_ids = uE2uF[uE2uF_cumsum[uE_id]:uE2uF_cumsum[uE_id + 1]]
                for uF_id_adj in uF_ids:

                    if not uF_seam_mask[uF_id_adj]:
                        continue

                    if uF_mark[uF_id_adj]:
                        continue

                    f_adj = uF[uF_id_adj]

                    # Verify adjacent face orientation
                    match = False
                    for i in range(3):
                        for j in range(3):
                            match = (f[i]
                                     == f_adj[j]) and (f[(i + 1) % 3]
                                                       == f_adj[(j + 1) % 3])

                            if match:
                                break

                        # Break double for loop ...
                        if match:
                            break

                    if match:
                        uF[uF_id_adj] = f_adj[::-1]

                    uF_mark[uF_id_adj] = True
                    Q_bfs.put(uF_id_adj)

    assert uF_mark.sum() == len(uF_seam)

    # Might doable via uE one-ring traversal...
    # But let's cheese it through with simpler vertex_tet_adjacency
    V2T, V2T_cumsum = frame_field_utils.vertex_tet_adjacency(T)
    VT_adj_list = np.split(V2T, V2T_cumsum[1:-1])

    V_interior_mask = np.logical_xor(V_singular_mask, V_seam_mask)
    V_interior = np.arange(len(V))[V_interior_mask]

    for vid in V_interior:
        VT_adj = VT_adj_list[vid]

        # Start from seed
        t_id = V2T_seed[vid]

        assert t_id in VT_adj

        VT_adj_coloring = [t_id]

        T_mark = np.zeros(len(T)).astype(bool)
        T_mark[t_id] = True

        q = queue.Queue()
        q.put(t_id)

        while not q.empty():
            t_id = q.get()

            for i in range(4):
                uF_id = F2uF[t_id, i]

                # Must not leak through seams
                if uF_seam_mask[uF_id]:
                    continue

                t_id_next = TT[t_id, i]

                if T_mark[t_id_next]:
                    continue

                T_mark[t_id_next] = True

                if t_id_next in VT_adj:
                    q.put(t_id_next)
                    T_coloring_mask[t_id_next] = True
                    VT_adj_coloring.append(t_id_next)

        VT_adj_list[vid] = np.array(VT_adj_coloring)

    return T_coloring_mask, uF[uF_seam], uFN[
        uF_seam], VT_adj_list, V2T_seed, V2T_seed_inv


def tet_cut(uE_seam, uF_seam, F_seam, Rs_comb, VT_adj_list, V2T_seed,
            V2T_seed_inv, V, T, uE, uE_ids_singular, V_singular_mask, F2uF,
            uF2uE, uE2uF, uE2uF_cumsum, **kwargs):
    # Update the mask
    uF_seam_mask = np.zeros(F2uF.max() + 1).astype(bool)
    uF_seam_mask[uF_seam] = True

    uF = np.zeros((F2uF.max() + 1, 3)).astype(np.int64)
    uF[uF_seam] = F_seam

    # BFS all seam faces
    Q_bfs = queue.Queue()
    uE_mark = np.zeros(len(uE)).astype(bool)
    uF_mark = np.zeros(F2uF.max() + 1).astype(bool)

    # TODO: If traverse face by face, vertices shared by multiple isolated tets may appear multiple times
    #  Should those be handled different?
    V_mark = np.zeros(len(V)).astype(bool)

    vid_base = len(V)
    V_append = []
    T_new = np.copy(T)

    V_i = []
    V_j = []
    transitions_ji = []

    for uE_id_start in uE_ids_singular:

        if uE_mark[uE_id_start]:
            continue

        uE_mark[uE_id_start] = True
        Q_bfs.put(uE_id_start)

        while not Q_bfs.empty():
            uE_id = Q_bfs.get()
            uF_ids = uE2uF[uE2uF_cumsum[uE_id]:uE2uF_cumsum[uE_id + 1]]

            for uF_id in uF_ids:

                if not uF_seam_mask[uF_id]:
                    continue

                if uF_mark[uF_id]:
                    continue

                for vid in uF[uF_id]:

                    # No need to cut singularity
                    if V_singular_mask[vid]:
                        continue

                    if V_mark[vid]:
                        continue

                    V_append.append(V[vid])

                    T_adj = T_new[VT_adj_list[vid]]
                    T_adj[T_adj == vid] = vid_base
                    T_new[VT_adj_list[vid]] = T_adj

                    # Build constraints
                    R_i = Rs_comb[V2T_seed[vid]]
                    R_j = Rs_comb[V2T_seed_inv[vid]]
                    T_ji = transition_matrix(R_i, R_j)

                    assert np.linalg.norm(T_ji - np.eye(3)) > 1e-7

                    V_i.append(vid)
                    V_j.append(vid_base)
                    transitions_ji.append(T_ji)

                    vid_base += 1
                    V_mark[vid] = True

                uE_ids_next = uF2uE[uF_id]

                for uE_id_next in uE_ids_next:

                    if uE_mark[uE_id_next]:
                        continue

                    uE_mark[uE_id_next] = True
                    Q_bfs.put(uE_id_next)

                uF_mark[uF_id] = True

    V_append = np.stack(V_append)
    V_i = np.stack(V_i)
    V_j = np.stack(V_j)
    transitions_ji = np.stack(transitions_ji)

    # Verify that seams have been fully traversed
    assert uE_mark.sum() == len(uE_seam)
    assert uF_mark.sum() == len(uF_seam)

    TT, TTi = igl.tet_tet_adjacency(T_new)
    TT_mismatch = frame_field_utils.tet_frame_mismatch(T_new, TT, TTi, Rs_comb)

    assert TT_mismatch.sum() == 0

    return np.vstack([V, V_append]), T_new, TT, TTi, V_i, V_j, transitions_ji


# "Mesh Modification Using Deformation Gradients" by Sumner
@jit
def deformation_gradient(V, V_deform):
    V = jnp.stack([V[1] - V[0], V[2] - V[0], V[3] - V[0]], -1)
    V_deform = jnp.stack([
        V_deform[1] - V_deform[0], V_deform[2] - V_deform[0],
        V_deform[3] - V_deform[0]
    ], -1)

    return V_deform @ jnp.linalg.inv(V)


# "Mixed-Integer Quadrangulation" by Bommes et al.
@jit
def local_distortion(J):
    det = jnp.linalg.det(J)
    _, S, _ = jnp.linalg.svd(J)
    return jnp.abs(det * S - 1).sum()


@jit
def uniform_laplacian(weight, TT):
    mask = (TT > 0)
    return (weight[TT] * mask).mean(1)


def tet_solve_param(Rs_comb, V_i, V_j, transitions_ji, V, T):
    # Follow "CubeCover-Parameterization of 3D Volumes" by Nieser et al., we want to minimize
    #   E = \int \|\nabla f - X\|^2 dV
    # with its Euler–Lagrange equation yields
    #   \Delta f = div X

    # For boundary condition
    # 1. Fix f for one vertex
    # 2. For each cuts, PI_ji @ f_i - f_j = 0
    #
    # So we need to unroll...

    NV = len(V)
    NT = len(T)
    NC = len(V_i)

    # Boundary conditions
    idx = np.argwhere(np.abs(transitions_ji) == 1)
    col_idx_i = (V_i * 3).repeat(3) + idx[:, -1]
    # Should be equivalent to np.arange(3 * NC)
    row_idx_i = 3 * idx[:, 0] + idx[:, 1]
    data_i = np.take_along_axis(transitions_ji.reshape(-1, 3),
                                idx[:, -1][:, None],
                                axis=-1)[:, 0]

    # -I
    col_idx_j = (3 * V_j[:, None] + np.arange(3)[None, :]).reshape(-1)
    row_idx_j = np.arange(3 * NC)
    data_j = -np.ones(3 * NC)

    # Just to be safe
    assert np.all(np.isclose(row_idx_i, row_idx_j))

    # Fix the first vertex to (0, 0, 0) (put it at the end)
    # I @ f_i = 0
    A_row = np.concatenate([row_idx_i, row_idx_j, 3 * NC + np.arange(3)])
    A_col = np.concatenate([col_idx_i, col_idx_j, np.arange(3)])
    A_data = np.concatenate([data_i, data_j, np.ones(3)])

    A = scipy.sparse.coo_array((A_data, (A_row, A_col)),
                               shape=(3 * (NC + 1), 3 * NV)).tocsc()

    assert len(A.data) == NC * 3 * 2 + 3

    b = np.zeros(3 * (NC + 1))

    stiffness_weight = np.ones(NT)
    G = igl.grad(V, T)
    vol = igl.volume(V, T)
    M = scipy.sparse.diags(np.hstack([vol, vol, vol]))

    no_flip = False

    while not no_flip:
        S = scipy.sparse.diags(
            np.hstack([stiffness_weight, stiffness_weight, stiffness_weight]))

        # Discrete divergence operator
        div = G.T @ M @ S
        L = unroll_identity_block(div @ G, 3)

        # Note: div = [ div_x | div_y | div_z ]
        #   so we need to transpose before reshape
        div_u = div @ Rs_comb[..., 0].T.reshape(-1)
        div_v = div @ Rs_comb[..., 1].T.reshape(-1)
        div_w = div @ Rs_comb[..., 2].T.reshape(-1)
        # [div_u_0, div_v_0, div_w_0, ..., div_u_n, div_v_n, div_w_n]
        div_X = np.stack([div_u, div_v, div_w], -1).reshape(-1)

        # Assemble system
        Q = scipy.sparse.vstack([L, A])
        c = np.concatenate([div_X, b])

        factor = cholesky((Q.T @ Q).tocsc())
        UVW = factor(Q.T @ c).reshape(NV, 3)

        J = vmap(deformation_gradient)(V[T], UVW[T])

        flip_mask = vmap(jnp.linalg.det)(J) < 0
        num_flips = flip_mask.sum()
        no_flip = num_flips == 0

        distort = vmap(local_distortion)(J)

        # Default hyperparameters in MIQ are (c = 1, d = 5)
        c = 5
        d = 15

        # For some reason, if I use smoothed weight, one flip cannot be resolved after long iterations
        # distort = uniform_laplacian(distort, TT)
        weight = np.clip(c * np.abs(distort), 0, d)

        stiffness_weight += weight

        print(f"Number of flips: {num_flips}")

    return UVW


# Some hacky function so it behaves like jupyter notebook
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

    # Visualize the difference
    # F_b = igl.boundary_facets(T)
    # F_b = np.stack([F_b[:, 2], F_b[:, 1], F_b[:, 0]], -1)

    # ps.init()
    # ps.register_surface_mesh('tet boundary', V, F_b, enabled=False)
    # if V_mc is not None:
    #     ps.register_surface_mesh('mc', V_mc, F_mc)
    # if uE_singularity_mask.sum() > 0:
    #     ps_register_curve_network('singularity',
    #                               V,
    #                               uE[uE_singularity_mask],
    #                               enabled=False)
    #     ps_register_curve_network('singularity SDP', V,
    #                               uE[uE_singularity_mask_octa])
    # ps.show()

    # Use SDP one
    Rs = Rs_bary_octa
    sh4s = sh4_bary_octa
    uE_singularity_mask = uE_singularity_mask_octa
    V_bary = V[T].mean(1)
    F = np.stack([
        np.stack([T[:, 0], T[:, 1], T[:, 2]], -1),
        np.stack([T[:, 0], T[:, 1], T[:, 3]], -1),
        np.stack([T[:, 1], T[:, 2], T[:, 3]], -1),
        np.stack([T[:, 2], T[:, 0], T[:, 3]], -1)
    ], 1)

    uE_ids_singular = np.arange(len(uE))[uE_singularity_mask]
    # Singular vertices will not be split during cut
    V_singular_mask = np.zeros(len(V)).astype(bool)
    V_singular_mask[np.unique(uE[uE_singularity_mask])] = True

    # Share the adjacency map
    TT, TTi = igl.tet_tet_adjacency(T)

    F2uF, uF2T = frame_field_utils.tet_uF_map(T, TT, TTi)
    uE2uF, uE2uF_cumsum, uF2uE = frame_field_utils.tet_uE_uF_map(
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE,
        F2uF)

    # Splitting is surprising heavy, can take like 6s...
    # uE2T_adj_list = np.split(uE2T, uE2T_cumsum[1:-1])
    # uE2uF_adj_list = np.split(uE2uF, uE2uF_cumsum[1:-1])

    timer.log('Build traversal data structure')

    tet_data = {
        'V': V,
        'T': T,
        'F': F,
        'uE': uE,
        'uE_boundary_mask': uE_boundary_mask,
        'uE_non_manifold_mask': uE_non_manifold_mask,
        'uE_singularity_mask': uE_singularity_mask,
        'uE2T': uE2T,
        'uE2T_cumsum': uE2T_cumsum,
        'E2uE': E2uE,
        'E2T': E2T,
        'TT': TT,
        'TTi': TTi,
        'uE_ids_singular': uE_ids_singular,
        'V_singular_mask': V_singular_mask,
        'uE2T': uE2T,
        'uE2T_cumsum': uE2T_cumsum,
        'uE2uF': uE2uF,
        'uE2uF_cumsum': uE2uF_cumsum,
        'F2uF': F2uF,
        'uF2T': uF2T,
        'uF2uE': uF2uE
    }

    # -------------------------------------------------------------------------

    # Verify euler characteristics
    assert is_tet_manifold(**tet_data)

    Rs_comb = frame_field_utils.tet_comb_frame(T, TT, Rs, sh4s)

    timer.log('Comb frame field')

    V_seam, uE_seam, uF_seam, F_seam = extract_seams(Rs_comb=Rs_comb,
                                                     **tet_data)

    timer.log('Extract seams')

    if len(V_seam) > 0:

        # --------------------------------------------------------------------------

        # FIXME: This step only works for very simple cases
        #   I known very little of robust tetrahedron operators...
        print(
            "Attempt to refine seams so they can be cut. Very basic implementation so it's very likely to fail.."
        )

        flag = True
        while flag:
            Rs_comb, flag = dissolve_uncuttable(uE_seam=uE_seam,
                                                uF_seam=uF_seam,
                                                Rs_comb=Rs_comb,
                                                **tet_data)
            V_seam, uE_seam, uF_seam, F_seam = extract_seams(Rs_comb=Rs_comb,
                                                             **tet_data)

            # Manifold is needed for cut coloring
            Rs_comb, flag = dissolve_non_manifold(uE_seam=uE_seam,
                                                  uF_seam=uF_seam,
                                                  Rs_comb=Rs_comb,
                                                  **tet_data)
            V_seam, uE_seam, uF_seam, F_seam = extract_seams(Rs_comb=Rs_comb,
                                                             **tet_data)

        # Verify the F_seam is now manifold
        # TODO: It is still possible that one vertex is shared by two multiple tets.
        #   Should those be dissolved as well?
        assert not surface_vertex_topology(
            *rm_unref_vertices(V, F_seam))[-1].any()

        timer.log('Refine seams')

        # --------------------------------------------------------------------------

        T_coloring_mask, F_seam, uFN, VT_adj_list, V2T_seed, V2T_seed_inv = tet_cut_coloring(
            V_seam=V_seam,
            uE_seam=uE_seam,
            uF_seam=uF_seam,
            F_seam=F_seam,
            **tet_data)

        V, T, TT, TTi, V_i, V_j, transitions_ji = tet_cut(
            uE_seam=uE_seam,
            uF_seam=uF_seam,
            F_seam=F_seam,
            Rs_comb=Rs_comb,
            VT_adj_list=VT_adj_list,
            V2T_seed=V2T_seed,
            V2T_seed_inv=V2T_seed_inv,
            **tet_data)

        timer.log('Cut tetrahedron')

    else:
        V_i = np.zeros((0,))
        V_j = np.zeros((0,))
        transitions_ji = np.zeros((0, 3, 3))

    UVW = tet_solve_param(Rs_comb, V_i, V_j, transitions_ji, V, T)

    timer.log('Solve parameterization')

    # Evaluate the gradient of the potential (should be close to Rs_comb in least square sense)
    NT = len(T)
    G = igl.grad(V, T)
    grad_uvw = G @ UVW
    grad_uvw = np.stack([grad_uvw[:NT], grad_uvw[NT:2 * NT], grad_uvw[2 * NT:]],
                        1)

    ps.init()
    ps.register_volume_mesh('tet_param', UVW, T)
    ps.register_volume_mesh('mesh', V, T)
    ps_register_basis('Comb', grad_uvw, UVW[T].mean(1))
    ps.show()
