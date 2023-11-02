import numpy as np
import igl
import pickle
import queue

from jax import vmap, jit, numpy as jnp
from common import vis_oct_field, ps_register_curve_network, rm_unref_vertices, surface_vertex_topology, normalize, Timer
from tet_parameterization import transition_matrix
import frame_field_utils

import polyscope as ps
from icecream import ic


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


def ps_register_basis(name,
                      Rs,
                      V,
                      length=0.02,
                      radius=0.0015,
                      enabled=True,
                      **wargs):
    pc = ps.register_point_cloud(name, V, radius=1e-4, enabled=enabled, **wargs)

    pc.add_vector_quantity('x',
                           Rs[..., 0],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)
    pc.add_vector_quantity('y',
                           Rs[..., 1],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)
    pc.add_vector_quantity('z',
                           Rs[..., 2],
                           radius=radius,
                           length=length,
                           enabled=True,
                           **wargs)


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


if __name__ == '__main__':

    tet_data = load_tmp()

    V = tet_data['V']
    T = tet_data['T']
    V_bary = V[T].mean(1)
    F = np.stack([
        np.stack([T[:, 0], T[:, 1], T[:, 2]], -1),
        np.stack([T[:, 0], T[:, 1], T[:, 3]], -1),
        np.stack([T[:, 1], T[:, 2], T[:, 3]], -1),
        np.stack([T[:, 2], T[:, 0], T[:, 3]], -1)
    ], 1)
    Rs = tet_data['Rs']
    sh4s = tet_data['sh4s']
    uE = tet_data['uE']
    uE_boundary_mask = tet_data['uE_boundary_mask']
    uE_non_manifold_mask = tet_data['uE_non_manifold_mask']
    uE_singularity_mask = tet_data['uE_singularity_mask']
    uE2T = tet_data['uE2T']
    uE2T_cumsum = tet_data['uE2T_cumsum']
    E2uE = tet_data['E2uE']

    timer = Timer()

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

    tet_data.update({
        'F': F,
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
    })

    # -------------------------------------------------------------------------

    # Verify euler characteristics
    assert is_tet_manifold(**tet_data)

    Rs_comb = frame_field_utils.tet_comb_frame(T, TT, Rs, sh4s)

    timer.log('Comb frame field')

    V_seam, uE_seam, uF_seam, F_seam = extract_seams(Rs_comb=Rs_comb,
                                                     **tet_data)

    timer.log('Extract seams')

    # ps.init()
    # ps.register_surface_mesh('seam', V, F_seam)
    # ps_register_basis('Comb', Rs_comb, V_bary)
    # ps_register_curve_network('singularity',
    #                           V,
    #                           uE[uE_singularity_mask],
    #                           radius=1e-3)
    # ps.show()

    # exit()

    # --------------------------------------------------------------------------

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
    assert not surface_vertex_topology(*rm_unref_vertices(V, F_seam))[-1].any()

    timer.log('Refine seams')

    # --------------------------------------------------------------------------

    T_coloring_mask, F_seam, uFN, VT_adj_list, V2T_seed, V2T_seed_inv = tet_cut_coloring(
        V_seam=V_seam,
        uE_seam=uE_seam,
        uF_seam=uF_seam,
        F_seam=F_seam,
        **tet_data)

    V, T, TT, TTi, V_i, V_j, transitions_ji = tet_cut(uE_seam=uE_seam,
                                                      uF_seam=uF_seam,
                                                      F_seam=F_seam,
                                                      Rs_comb=Rs_comb,
                                                      VT_adj_list=VT_adj_list,
                                                      V2T_seed=V2T_seed,
                                                      V2T_seed_inv=V2T_seed_inv,
                                                      **tet_data)

    ps.init()
    ps.register_surface_mesh('seam', V, F_seam)
    ps.register_volume_mesh('tet 0', V, T[T_coloring_mask])
    ps_register_curve_network('singularity',
                              V,
                              uE[uE_singularity_mask],
                              radius=1e-3)
    ps.show()

    tet_data.update({
        'V': V,
        'T': T,
        'TT': TT,
        'TTi': TTi,
        'V_i': V_i,
        'V_j': V_j,
        'transitions_ji': transitions_ji,
        'Rs_comb': Rs_comb
    })
    # save_tmp(tet_data)

    exit()

    # --------------------------------------------------------------------------

    uE_singular = uE[uE_singularity_mask]
    TT_mask = TT_mismatch.sum(1) > 0

    ps.init()
    ps.register_surface_mesh('seam', V, F_seam)
    # ps_register_basis('Raw', Rs, V_bary, enabled=False)
    ps_register_basis('Comb', Rs_comb[TT_mask], V_bary[TT_mask])
    ps_register_curve_network('singularity', V, uE_singular, radius=1e-3)
    # ps_register_curve_network('singularity valence 3', V, uE_singular3)
    ps.show()
