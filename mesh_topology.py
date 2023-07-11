import igl
import numpy as np
import jax
from jax import Array, vmap, jit, numpy as jnp
from typing import Callable
from functools import partial

# enable 64 bit precision
from jax.config import config

config.update("jax_enable_x64", True)

import polyscope as ps
from icecream import ic


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


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
# TODO: deal with crease normal
def build_traversal_graph(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(F.flatten(),
                                                         return_index=True,
                                                         return_inverse=True)

    V_map = V_unique[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_unique

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_map]

    edge_id = np.arange(F.size)
    E = np.stack([
        np.stack([F[:, 0], F[:, 1]], -1),
        np.stack([F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 2], F[:, 0]], -1)
    ], 1).reshape(-1, 2)

    # Not sure if there is a more efficient approach
    E2Eid = dict([(f"{e[0]}_{e[1]}", e_id) for e, e_id in zip(E, edge_id)])

    def opposite_edge_id(e_id):
        v0, v1 = E[e_id]
        key = f"{v1}_{v0}"
        return E2Eid[key] if key in E2Eid else -1

    E2E = -np.ones(F.size, dtype=np.int64)
    E2E[edge_id] = list(map(opposite_edge_id, edge_id))

    V_boundary = np.full((len(V)), False)
    V_manifold = np.full((len(V)), True)

    # Check boundary, manifold for undirected edge
    ue, ue_count = np.unique(np.sort(E, axis=1), axis=0, return_counts=True)
    V_boundary[ue[ue_count == 1]] = True
    V_manifold[ue[ue_count > 2]] = False

    edge_sort_idx = np.argsort(E[:, 0])
    edge_sorted = E[edge_sort_idx]
    edge_id_sorted = edge_id[edge_sort_idx]

    # Since edges are directed, `e_count` directly match the number of incident faces
    _, e_count = np.unique(edge_sorted[:, 0], return_counts=True)
    pad_width = np.max(e_count)

    # build V2E map
    split_indices = np.cumsum(e_count)[:-1]
    V2E_list = np.split(edge_id_sorted, split_indices)
    V2E = np.array([
        np.concatenate([el, -np.ones(pad_width - len(el))]) for el in V2E_list
    ]).astype(np.int64)

    return V, F, E, V2E, E2E, V_boundary, V_manifold


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
def area(verts):
    v0, v1, v2 = verts

    a = jnp.linalg.norm(v1 - v0)
    b = jnp.linalg.norm(v2 - v0)
    c = jnp.linalg.norm(v2 - v1)

    area_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(area_2) / 4


@jit
def _vertex_area(a, b, c):
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
    return _vertex_area(V[cur_edge[0]], V[cur_edge[1]], V[prev_edge[0]])


@jit
def cotangent(ei, ej, E, V):
    A = V[E[ej][0]]
    B = V[E[ej][1]]
    C = V[E[ei][1]]

    len2 = lambda x: jnp.einsum('i,i', x, x)

    a = len2(B - C)
    b = len2(A - C)
    c = len2(B - A)

    return b + c - a


@jit
def cotangent_weight(e_id, E, E2E, V, FA):
    area_a = FA[e_id // 3]
    cot_a = 0.25 * cotangent(e_id, prev_edge_id(e_id), E, V) / area_a

    e_id_op = E2E[e_id]
    area_b = FA[e_id_op // 3]
    cot_b = 0.25 * cotangent(e_id_op, prev_edge_id(e_id_op), E, V) / area_b
    return 0.5 * (cot_a + cot_b)


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


if __name__ == '__main__':
    V, F = igl.read_triangle_mesh("data/bunny.obj")

    V, F, E, V2E, E2E, V_boundary, V_manifold = build_traversal_graph(V, F)
    FN, T_f = per_face_basis(V[F])
    VN, T_v = per_vertex_basis(V, E, V2E, FN)
    TfM, TvM = fit_curvature_tensor(V, F, E, V2E, FN, T_f, VN, T_v)
    FA = vmap(area)(V[F])

    Ws = one_ring_traversal(
        V2E, jax.tree_util.Partial(cotangent_weight, E=E, E2E=E2E, V=V, FA=FA),
        0.)

    ic(Ws.shape, V2E.shape)
    exit()

    _, pvecs_f = principal_curvature(T_f, TfM)
    _, pvecs_v = principal_curvature(T_v, TvM)

    ps.init()
    sur = ps.register_surface_mesh("sur", V, F)
    sur.add_vector_quantity('pf0', pvecs_f[:, 0], defined_on='faces')
    sur.add_vector_quantity('pf1', pvecs_f[:, 1], defined_on='faces')
    sur.add_vector_quantity('pv0', pvecs_v[:, 0])
    sur.add_vector_quantity('pv1', pvecs_v[:, 1])
    ps.show()
