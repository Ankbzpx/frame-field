import igl
import numpy as np
import jax
from jax import Array, vmap, jit, numpy as jnp
from functools import partial
from typing import Callable
import scipy
import scipy.sparse.linalg

from common import normalize_aabb, normalize, rm_unref_vertices, Timer

import open3d as o3d
import argparse
import os

import frame_field_utils

import polyscope as ps
from icecream import ic


@jit
@partial(vmap)
def per_face_basis(verts):
    e01 = verts[1] - verts[0]
    e02 = verts[2] - verts[0]
    fn = normalize(jnp.cross(e01, e02))
    u = normalize(e01)
    v = normalize(jnp.cross(e01, fn))

    # T_l_w
    T = jnp.stack([u, v])

    return fn, T


# Use half edge graph for traversal
# TODO: Add crease normal
def build_traversal_graph(V, F):
    V, F = rm_unref_vertices(V, F)

    E_id = np.arange(F.size)
    E = np.stack([
        np.stack([F[:, 0], F[:, 1]], -1),
        np.stack([F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 2], F[:, 0]], -1)
    ], 1).reshape(-1, 2)

    # Not sure if there is a more efficient approach
    E2Eid = dict([(f"{e[0]}_{e[1]}", e_id) for e, e_id in zip(E, E_id)])

    def opposite_edge_id(e_id):
        v0, v1 = E[e_id]
        key = f"{v1}_{v0}"
        return E2Eid[key] if key in E2Eid else -1

    E2E = -np.ones(F.size, dtype=np.int64)
    E2E[E_id] = np.array(list(map(opposite_edge_id, E_id)))

    # Use row-wise unique to filter boundary and nonmanifold vertices
    E_row_sorted = np.sort(E, axis=1)
    _, ue_inv, ue_count = np.unique(E_row_sorted,
                                    axis=0,
                                    return_counts=True,
                                    return_inverse=True)

    V_boundary = np.full((len(V)), False)
    V_boundary[list(np.unique(E[(ue_count == 1)[ue_inv]][:, 0]))] = True

    V_nonmanifold = np.full((len(V)), False)
    V_nonmanifold[list(np.unique(E[(ue_count > 2)[ue_inv]][:, 0]))] = True

    # build V2E map
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    E_sort_idx = np.lexsort(E.T[::-1])
    E_sorted = E[E_sort_idx]
    E_id_sorted = E_id[E_sort_idx]

    # Since edges are directed, `e_count` directly match the number of incident faces
    _, e_count = np.unique(E_sorted[:, 0], return_counts=True)
    pad_width = np.max(e_count)

    split_indices = np.cumsum(e_count)[:-1]
    V2E_list = np.split(E_id_sorted, split_indices)
    V2E = np.array([
        np.concatenate([el, -np.ones(pad_width - len(el))]) for el in V2E_list
    ]).astype(np.int64)

    return V, F, E, V2E, E2E, V_boundary, V_nonmanifold


@jit
def prev_edge_id(e_id):
    return jnp.where(e_id % 3 == 0, e_id + 2, e_id - 1)


@jit
def next_edge_id(e_id):
    return jnp.where(e_id % 3 == 2, e_id - 2, e_id + 1)


@jit
@partial(vmap, in_axes=(0, None, None))
def edge_call(e_id, f: Callable, val_default: Array):
    return jnp.where(e_id == -1, val_default, f(e_id))


@jit
@partial(vmap, in_axes=(0, None, None))
def one_ring_traversal(e_ids, func: Callable, val_default: Array):
    return edge_call(e_ids, func, val_default)


# "Computing Vertex Normals from Polygonal Facets" by Grit Thuermer and Charles A. Wuethrich, JGT 1998, Vol 3
@jit
def angle_weighted_face_normal(e_id, V, E, FN):
    f_id = e_id // 3
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]

    d_0 = V[cur_edge[1]] - V[cur_edge[0]]
    d_1 = V[prev_edge[0]] - V[prev_edge[1]]

    angle = jnp.arccos(
        jnp.dot(d_0, d_1) / jnp.linalg.norm(d_0) / jnp.linalg.norm(d_1))

    angle = jnp.where(jnp.isnan(angle), 0, angle)

    return angle * FN[f_id]


@jit
def per_vertex_normal(V, E, V2E, FN):
    vn = one_ring_traversal(
        V2E, jax.tree_util.Partial(angle_weighted_face_normal, V=V, E=E, FN=FN),
        jnp.zeros((3,)))
    vn = vmap(normalize)(jnp.sum(vn, 1))
    return vn


# Assume normal to be tangent plane, x axis is one of its connected edges
@jit
@partial(vmap, in_axes=(0, None, None, None))
def local_vertex_basis(eid, VN, V, E):
    v0, v1 = E[eid[0]]
    u = normalize(V[v1] - V[v0])
    v = jnp.cross(VN[v0], u)
    T = jnp.stack([u, v])
    return T


@jit
def per_vertex_basis(V, E, V2E, FN):
    VN = per_vertex_normal(V, E, V2E, FN)
    T = local_vertex_basis(V2E, VN, V, E)
    return VN, T


# Get rotation matrix from a to b. Reference: https://math.stackexchange.com/a/897677
@jit
def rotate_coplanar(a, b):
    cos = jnp.dot(a, b)
    cross = jnp.cross(a, b)
    sin = jnp.linalg.norm(cross)

    # pure rotation matrix
    G = jnp.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    # transformation of coordinate: a, normal (a x b), binormal
    F = jnp.array([a, normalize(b - cos * a), normalize(cross)])

    return F.T @ G @ F


@jit
@partial(vmap)
def face_curvature_tensor(verts, vert_normals, T_f):
    e0 = verts[2] - verts[1]
    e1 = verts[0] - verts[2]
    e2 = verts[1] - verts[0]

    vn0 = vert_normals[0]
    vn1 = vert_normals[1]
    vn2 = vert_normals[2]

    # direction of basis in local coordinate
    u = T_f[0]
    v = T_f[1]

    a_0u = jnp.dot(e0, u)
    a_0v = jnp.dot(e0, v)

    a_1u = jnp.dot(e1, u)
    a_1v = jnp.dot(e1, v)

    a_2u = jnp.dot(e2, u)
    a_2v = jnp.dot(e2, v)

    A = jnp.array([[a_0u, a_0v, 0], [0, a_0u, a_0v], [a_1u, a_1v, 0],
                   [0, a_1u, a_1v], [a_2u, a_2v, 0], [0, a_2u, a_2v]])

    b = jnp.array([
        jnp.dot(vn2 - vn1, u),
        jnp.dot(vn2 - vn1, v),
        jnp.dot(vn0 - vn2, u),
        jnp.dot(vn0 - vn2, v),
        jnp.dot(vn1 - vn0, u),
        jnp.dot(vn1 - vn0, v)
    ])

    X = jnp.linalg.inv(A.T @ A) @ A.T @ b

    TM = jnp.array([[X[0], X[1]], [X[1], X[2]]])

    return TM


# Curvature tensor is defined on local coordinate system, thus invariant to the rotation of bases
@jit
def transform_face_metric(e_id, T_v, E, VN, FN, T_f, TfM):
    f_id = e_id // 3
    cur_edge = E[e_id]
    v_id = cur_edge[0]
    # rotate from face normal to vertex normal
    R = rotate_coplanar(FN[f_id], VN[v_id])

    T_coplanar = jnp.einsum('ij,bj->bi', R, T_f[f_id])
    uf = T_coplanar[0]
    vf = T_coplanar[1]

    up = T_v[v_id, 0]
    vp = T_v[v_id, 1]

    v0 = jnp.array([jnp.dot(up, uf), jnp.dot(up, vf)])
    v1 = jnp.array([jnp.dot(vp, uf), jnp.dot(vp, vf)])

    TM = TfM[f_id]

    a = v0.T @ TM @ v0
    b = v0.T @ TM @ v1
    c = v1.T @ TM @ v1

    TM = jnp.array([[a, b], [b, c]])
    return TM


# Reference: https://en.wikipedia.org/wiki/Heron%27s_formula
@jit
def face_area(verts):
    v0, v1, v2 = verts

    a = jnp.linalg.norm(v1 - v0)
    b = jnp.linalg.norm(v2 - v0)
    c = jnp.linalg.norm(v2 - v1)

    area_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(area_2) / 4


# Reference: https://www.alecjacobson.com/weblog/?p=1146
@jit
def hybrid_area(a, b, c):
    e0 = a - b
    e1 = a - c
    e2 = b - c

    e0_norm = jnp.linalg.norm(e0)
    e1_norm = jnp.linalg.norm(e1)
    e2_norm = jnp.linalg.norm(e2)

    cos_a = jnp.einsum('i,i', e0, e1) / e0_norm / e1_norm
    cos_b = jnp.einsum('i,i', -e0, e2) / e0_norm / e2_norm
    cos_c = jnp.einsum('i,i', -e1, -e2) / e1_norm / e2_norm
    cos_sum = cos_a + cos_b + cos_c

    area = 0.5 * jnp.linalg.norm(jnp.cross(e0, e1))

    is_obtuse = (cos_a < 0.) | (cos_b < 0.) | (cos_c < 0.)
    is_obtuse_angle = cos_a < 0.

    area = jnp.where(is_obtuse, 0.25 * area,
                     0.5 * area * (cos_c + cos_b) / cos_sum)
    area = jnp.where(is_obtuse_angle, 2 * area, area)

    return area


@jit
def vertex_area(e_id, E, V):
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]
    return hybrid_area(V[cur_edge[0]], V[cur_edge[1]], V[prev_edge[0]])


@jit
def cotangent(ei, ej, E, V, FA):
    A = V[E[ej][0]]
    B = V[E[ej][1]]
    C = V[E[ei][1]]
    area = FA[ei // 3]

    len2 = lambda x: jnp.einsum('i,i', x, x)

    a = len2(B - C)
    b = len2(A - C)
    c = len2(B - A)

    return 0.25 * (b + c - a) / area


# For a boundary vertex of n adjacent vertices, only n-1 half-edges are included in V2E
# Here we retrieve the missing one to support cotangent weight calculation
@jit
def boundary_auxillary_edge(eids, E2E):
    ref_eid = jnp.where(eids == -1, -1, vmap(prev_edge_id)(eids))
    src_eid = jnp.where(eids == -1, -1, E2E[eids])
    in_bool = vmap(jnp.isin, in_axes=(0, None))(ref_eid, src_eid) == 0
    e_aux = ref_eid[jnp.argwhere(in_bool, size=1)[0][0]]
    # Assume boundary has less valence
    e_replace = jnp.argwhere(eids == -1, size=1)[0][0]
    return eids.at[e_replace].set(e_aux)


@jit
def cotangent_edge_weight(e_id, E, E2E, V, FA):
    cot_a = cotangent(e_id, prev_edge_id(e_id), E, V, FA)
    e_id_op = E2E[e_id]
    cot_b = jnp.where(e_id_op == -1, 0.,
                      cotangent(e_id_op, prev_edge_id(e_id_op), E, V, FA))
    return 0.5 * (cot_a + cot_b)


def cotangent_weight(V, E, FA, V2E, E2E, V_boundary):
    V2E_aux = np.copy(V2E)
    V2E_aux[V_boundary] = vmap(boundary_auxillary_edge,
                               in_axes=(0, None))(V2E[V_boundary], E2E)
    return one_ring_traversal(
        V2E_aux,
        jax.tree_util.Partial(cotangent_edge_weight, E=E, E2E=E2E, V=V, FA=FA),
        0.)


# "Estimating Curvatures and Their Derivatives on Triangle Meshes" by Szymon Rusinkiewicz
@jit
def fit_curvature_tensor(V, F, E, V2E, FN, T_f, VN, T_v):
    TfM = face_curvature_tensor(V[F], VN[F], T_f)

    TvM = one_ring_traversal(
        V2E,
        jax.tree_util.Partial(transform_face_metric,
                              T_v=T_v,
                              E=E,
                              VN=VN,
                              FN=FN,
                              T_f=T_f,
                              TfM=TfM), jnp.zeros((2, 2)))

    Ws = one_ring_traversal(V2E, jax.tree_util.Partial(vertex_area, E=E, V=V),
                            0.)

    TvM = jnp.sum(Ws[..., None, None] * TvM, axis=1) / jnp.sum(Ws, 1)[:, None,
                                                                      None]
    return TfM, TvM


@jit
def principal_curvature(T, TM):
    eigvals, eigvecs = vmap(jnp.linalg.eigh)(TM)
    eigvecs = jnp.einsum('bij,bni->bnj', T, eigvecs)
    return eigvals, eigvecs


@jit
@partial(vmap, in_axes=(0, 0, None, None, None, None, None, None))
def smooth(ws, e_ids, E, E2E, FA, alpha, beta, NV):
    f_areas = jnp.where(e_ids == -1, 0, FA[e_ids // 3])
    opp_e_ids = E2E[e_ids]
    opp_f_areas = jnp.where(e_ids == -1, 0, FA[opp_e_ids // 3])
    ms = 0.5 * (f_areas + opp_f_areas)
    m_sum = jnp.sum(f_areas)

    edges = E[e_ids]
    vid = edges[0, 0]
    w_sum = jnp.sum(ws)

    vid_i = edges[:, 0]
    vid_i_next = vid_i + NV
    vid_j = edges[:, 1]
    vid_j_next = vid_j + NV

    alpha_i = alpha[vid_i]
    alpha_j = alpha[vid_j]
    beta_i = beta[vid_i]
    beta_j = beta[vid_j]

    inner_ii = jnp.einsum('bi,bi->b', alpha_i, alpha_j)
    inner_ij = jnp.einsum('bi,bi->b', alpha_i, beta_j)
    inner_ji = jnp.einsum('bi,bi->b', beta_i, alpha_j)
    inner_jj = jnp.einsum('bi,bi->b', beta_i, beta_j)

    idx_i = jnp.concatenate(
        [jnp.array([vid, vid + NV]), vid_i, vid_i, vid_i_next, vid_i_next])

    idx_j = jnp.concatenate(
        [jnp.array([vid, vid + NV]), vid_j, vid_j_next, vid_j, vid_j_next])

    weights = jnp.concatenate([
        jnp.array([w_sum, w_sum]), -inner_ii * ws, -inner_ij * ws,
        -inner_ji * ws, -inner_jj * ws
    ])

    mass = jnp.concatenate([
        jnp.array([m_sum, m_sum]), inner_ii * ms, inner_ij * ms, inner_ji * ms,
        inner_jj * ms
    ])

    return idx_i, idx_j, weights, mass


@jit
def tangent_to_rotation(tangent, normal):
    bitangent = jnp.cross(tangent, normal)
    return jnp.stack([tangent, bitangent, normal], -1)


if __name__ == '__main__':
    # enable 64 bit precision
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file.')
    parser.add_argument('--out_path',
                        type=str,
                        default='results',
                        help='Path to output folder.')
    args = parser.parse_args()

    model_path = args.input
    model_name = model_path.split('/')[-1].split('.')[0]
    model_out_path = os.path.join(args.out_path, f"{model_name}_ext.obj")

    timer = Timer()

    V, F = igl.read_triangle_mesh(model_path)
    V = normalize_aabb(V)

    timer.log('Load and preprocess mesh')

    # My attempt to abuse vmap for graph like data structure
    # Might well just use for loop in the future...
    V, F, E, V2E, E2E, V_boundary, V_nonmanifold = build_traversal_graph(V, F)

    timer.log('Build traversal graph')

    NV = len(V)
    FN, _ = per_face_basis(V[F])
    VN = per_vertex_normal(V, E, V2E, FN)
    FA = vmap(face_area)(V[F])
    Ws = cotangent_weight(V, E, FA, V2E, E2E, V_boundary)

    # Projection matrix to tangent plane (I - n n^T)
    Pv = jnp.repeat(jnp.eye(3)[None, ...], NV, axis=0) - jnp.einsum(
        'bi,bj->bij', VN, VN)

    # Local coordinate with random tangent vector as basis
    key = jax.random.PRNGKey(0)
    alpha = jnp.einsum('bij,bi->bj', Pv, jax.random.normal(key, (NV, 3)))
    alpha = vmap(normalize)(alpha)
    beta = vmap(jnp.cross)(alpha, VN)

    timer.log('Build local coordinate frame')

    idx_i, idx_j, weights, mass = smooth(Ws, V2E, E, E2E, FA, alpha, beta, NV)

    # Filter entries to build sparse system
    idx_i = idx_i.reshape(-1)
    idx_j = idx_j.reshape(-1)
    weights = weights.reshape(-1)
    mass = mass.reshape(-1)

    valid_mask = weights != 0
    idx_i = np.int32(idx_i[valid_mask])
    idx_j = np.int32(idx_j[valid_mask])
    weights = np.float64(weights[valid_mask])
    mass = np.float64(mass[valid_mask])

    timer.log('Build stiffness and mass entries')

    A = scipy.sparse.coo_array((weights, (idx_i, idx_j)),
                               shape=(2 * NV, 2 * NV)).tocsc()

    M = scipy.sparse.coo_array((mass, (idx_i, idx_j)),
                               shape=(2 * NV, 2 * NV)).tocsc()

    timer.log('Build sparse system')

    # Generalized eigenproblem
    # Reference: Algorithm 2 in Globally Optimal Direction Fields by Knöppel et al.
    np.random.seed(0)
    X = np.random.randn(2 * NV, 1)
    solve = scipy.sparse.linalg.factorized(A)

    for _ in range(30):
        X = solve(M @ X)
        X /= np.sqrt(X.T @ M @ X)

    a = X[:NV, 0]
    b = X[NV:, 0]

    timer.log('Solve (Generalized eigenproblem)')

    # representation vector
    Q = vmap(normalize)(a[:, None] * alpha + b[:, None] * beta)
    V_vis, F_vis, VC_vis = frame_field_utils.trace(V, F, VN, Q, 4000)

    timer.log('Trace flowlines')

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    mesh.add_vector_quantity("VN", VN)
    flow_line_vis = ps.register_surface_mesh("flow_line", V_vis, F_vis)
    flow_line_vis.add_color_quantity("VC_vis", VC_vis, enabled=True)
    ps.show()

    stroke_mesh = o3d.geometry.TriangleMesh()
    stroke_mesh.vertices = o3d.utility.Vector3dVector(V_vis)
    stroke_mesh.triangles = o3d.utility.Vector3iVector(F_vis)
    stroke_mesh.vertex_colors = o3d.utility.Vector3dVector(VC_vis)
    o3d.io.write_triangle_mesh(model_out_path, stroke_mesh)
