import igl
import numpy as np
import jax
from jax import vmap, numpy as jnp, jit
import pickle

from common import ps_register_curve_network, vis_oct_field, normalize_aabb, Timer
from sh_representation import proj_sh4_to_R3, R3_to_sh4_zonal

import polyscope as ps
from icecream import ic


def edge_one_ring(T):
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

                finished = False
                while not finished:
                    e_id = np.argwhere(E2uE[t_id] == ue_id)[0][0]
                    t_id_last, t_id = next_t_id(t_id, e_id, t_id_last)
                    # assert t_id in T_adj

                    finished = (t_id == t_id_end)
                    T_adj_sorted.append(t_id)

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


if __name__ == '__main__':
    timer = Timer()

    # data = np.load('results/prism_prac.npz')
    data = np.load('results/cylinder_prac.npz')

    V = np.float64(data['V'])
    V = normalize_aabb(V)
    T = np.int64(data['T'])
    Rs: np.array = data['Rs']
    sh4: np.array = data['sh4']

    timer.log('Load data')

    uE, uE_boundary_mask, uE2T = edge_one_ring(T)

    timer.log('Build edge one ring')

    # Use tet based representation, so the singularities are defined on edges, allowing us to cut directly along faces
    # Follow "Boundary Aligned Smooth 3D Cross-Frame Field" by Jin Huang et al., we transform from vertex based to tet via simple averaging
    V_bary = V[T].mean(axis=1)

    # Do NOT normalize it directly!
    sh4_bary = sh4[T].mean(axis=1)
    Rs_bary = proj_sh4_to_R3(sh4_bary)
    sh4_bary = vmap(R3_to_sh4_zonal)(Rs_bary)

    timer.log('Interpolate value')

    # save_tmp({
    #     'V': V,
    #     'T': T,
    #     'Rs': Rs,
    #     'sh4': sh4,
    #     'uE': uE,
    #     'uE_boundary_mask': uE_boundary_mask,
    #     'uE2T': uE2T,
    #     'V_bary': V_bary,
    #     'Rs_bary': Rs_bary,
    #     'sh4_bary': sh4_bary
    # })

    # exit()

    # load_tmp()

    uE_interior_mask = np.logical_not(uE_boundary_mask)
    uE_singularity_mask = np.zeros(len(uE), dtype=bool)
    uE_singularity_mask[uE_interior_mask] = np.array([
        is_singular(uE2T[ue_id], Rs_bary)
        for ue_id in np.arange(len(uE))[uE_interior_mask]
    ])

    timer.log('Compute singularity')

    F = igl.boundary_facets(T)
    F = np.stack([F[:, 2], F[:, 1], F[:, 0]], -1)

    ps.init()
    ps.register_surface_mesh('tet', V, F)
    ps_register_curve_network('boundary', V, uE[uE_boundary_mask], radius=1e-4)
    ps_register_curve_network('singularity', V, uE[uE_singularity_mask])
    ps.show()
