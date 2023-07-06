import igl
import polyscope as ps
import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jit

# enable 64 bit precision
from jax.config import config

config.update("jax_enable_x64", True)

from icecream import ic


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


@jit
def local_face_basis(f, V):
    v0 = V[f[0]]
    v1 = V[f[1]]
    v2 = V[f[2]]

    e01 = v1 - v0
    e02 = v2 - v0
    fn = normalize(jnp.cross(e01, e02))
    u = normalize(e01)
    v = normalize(jnp.cross(e01, fn))

    # T_l_w
    T = jnp.stack([u, v])

    return fn, v0, T


@jit
def prev_edge_id(e_id):
    return jnp.where(e_id % 3 == 0, e_id + 2, e_id - 1)


@jit
def next_edge_id(e_id):
    return jnp.where(e_id % 3 == 2, e_id - 2, e_id + 1)


# "Computing Vertex Normals from Polygonal Facets" by Grit Thuermer and Charles A. Wuethrich, JGT 1998, Vol 3
@jit
def _angle_weighted_face_normal(e_id, V, FN, E):
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
def angle_weighted_face_normal(e_id, V, FN, E):

    return jnp.where(e_id == -1, jnp.zeros((3,)),
                     _angle_weighted_face_normal(e_id, V, FN, E))


# TODO: handle manifold
@jit
def vertex_normals(e_ids, V, FN, E):
    vn = vmap(angle_weighted_face_normal,
              in_axes=(0, None, None, None))(e_ids, V, FN, E)
    vn = normalize(jnp.sum(vn, 0))
    return vn


# Assume normal to be tangent plane, x axis is one of its connected edges
@jit
def local_vertex_basis(eid, vn, V, E):
    v0, v1 = E[eid[0]]
    u = normalize(V[v1] - V[v0])
    v = jnp.cross(vn, u)
    T = jnp.stack([u, v])
    return T


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


# "Estimating Curvatures and Their Derivatives on Triangle Meshes" by Rusinkiewicz
@jit
def face_curvature_tensor(f, T, V, VN):
    v0 = V[f[0]]
    v1 = V[f[1]]
    v2 = V[f[2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    vn0 = VN[f[0]]
    vn1 = VN[f[1]]
    vn2 = VN[f[2]]

    # direction of basis in local coordinate
    u = T[0]
    v = T[1]

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


@jit
def triangle_area(a, b, c):
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


# Curvature tensor is defined on local coordinate system, thus invariant to the rotation of bases
@jit
def transformed_face_tensor(e_id, T, E, VN, FN, Tf, TfM):
    f_id = e_id // 3
    cur_edge = E[e_id]
    # rotate from face normal to vertex normal
    R = rotate_coplanar(FN[f_id], VN[cur_edge[0]])

    T_coplanar = jnp.einsum('ij,bj->bi', R, Tf[f_id])
    uf = T_coplanar[0]
    vf = T_coplanar[1]

    up = T[0]
    vp = T[1]

    v0 = jnp.array([jnp.dot(up, uf), jnp.dot(up, vf)])
    v1 = jnp.array([jnp.dot(vp, uf), jnp.dot(vp, vf)])

    TM = TfM[f_id]

    a = v0.T @ TM @ v0
    b = v0.T @ TM @ v1
    c = v1.T @ TM @ v1

    TM = jnp.array([[a, b], [b, c]])
    return TM


@jit
def vertex_area(e_id, E, V):
    cur_edge = E[e_id]
    prev_edge = E[prev_edge_id(e_id)]
    return triangle_area(V[cur_edge[0]], V[cur_edge[1]], V[prev_edge[0]])


@jit
def area_weighted_face_tensor(e_id, T, V, E, VN, FN, Tf, TfM):
    tensor = jnp.where(e_id == -1, jnp.zeros((2, 2)),
                       transformed_face_tensor(e_id, T, E, VN, FN, Tf, TfM))
    weight = jnp.where(e_id == -1, 0., vertex_area(e_id, E, V))
    return tensor, weight


@jit
def vertex_curvature_tensor(e_ids, T, V, E, VN, FN, Tf, TfM):
    TMs, Ws = vmap(area_weighted_face_tensor,
                   in_axes=(0, None, None, None, None, None, None,
                            None))(e_ids, T, V, E, VN, FN, Tf, TfM)

    TvM = jnp.sum(Ws[:, None, None] * TMs, axis=0) / jnp.sum(Ws)
    return TvM


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

    return V, F, E, V2E, V_boundary, V_manifold


if __name__ == '__main__':
    V, F = igl.read_triangle_mesh("data/mountain.obj")
    # V, F = igl.read_triangle_mesh("data/fandisk.ply")

    V, F, E, V2E, V_boundary, V_manifold = build_traversal_graph(V, F)

    FN, Os, Tf = vmap(local_face_basis, in_axes=(0, None))(F, V)
    VN = vmap(vertex_normals, in_axes=(0, None, None, None))(V2E, V, FN, E)

    Tv = vmap(local_vertex_basis, in_axes=(0, 0, None, None))(V2E, VN, V, E)

    TfM = vmap(face_curvature_tensor, in_axes=(0, 0, None, None))(F, Tf, V, VN)

    TvM = vmap(vertex_curvature_tensor,
               in_axes=(0, 0, None, None, None, None, None, None))(V2E, Tv, V,
                                                                   E, VN, FN,
                                                                   Tf, TfM)

    _, eigvecs_v = vmap(jnp.linalg.eigh)(TvM)
    eigvecs_vw = jnp.einsum('bji,bnj->bni', Tv, eigvecs_v)

    _, eigvecs_f = vmap(jnp.linalg.eigh)(TfM)
    eigvecs_fw = jnp.einsum('bji,bnj->bni', Tf, eigvecs_f)

    ps.init()
    sur = ps.register_surface_mesh('sur', V, F)
    # sur.add_vector_quantity('eigvec0', eigvecs_vw[:, 0], enabled=True)
    # sur.add_vector_quantity('eigvec1', eigvecs_vw[:, 1], enabled=True)
    sur.add_vector_quantity('eigvec0f',
                            eigvecs_fw[:, 0],
                            defined_on='faces',
                            enabled=True)
    sur.add_vector_quantity('eigvec1f',
                            eigvecs_fw[:, 1],
                            defined_on='faces',
                            enabled=True)
    ps.show()

    ic(Tf.shape, Tv.shape, TfM.shape)

    # pd1, pd2, pv1, pv2 = igl.principal_curvature(V, F)

    # ps.init()
    # mesh = ps.register_surface_mesh('sur', V, F)
    # mesh.add_vector_quantity('eig_max', eigvecs_max, defined_on='faces', enabled=True)
    # mesh.add_vector_quantity('eig_min', eigvecs_min, defined_on='faces', enabled=True)
    # mesh.add_vector_quantity('pd1', pd1, enabled=True)
    # mesh.add_vector_quantity('pd2', pd2, enabled=True)
    # ps.show()

    exit()

    ps.init()
    mesh = ps.register_surface_mesh('mountain', V, F)
    mesh.add_vector_quantity("VN", VN)
    ps.show()

    exit()

    ic(V2E)

    V2E = V2E[0]
    V2E = V2E[V2E >= 0]

    V2E[0] // 3
    V2E[1] // 3

    exit()

    exit()

    sort_idx = np.argsort(E[:, 0])

    edge_id = edge_id[sort_idx]
    E = E[sort_idx]

    ic(sort_idx)
    ic(edge_id)
    ic(E)

    exit()

    E = np.vstack([
        np.stack([F[:, 0], F[:, 1]], -1),
        np.stack([F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 2], F[:, 0]], -1)
    ])
    sort_idx = np.argsort(E[:, 0])
    E = E[sort_idx]

    ic(E)

    exit()

    ic(F.flatten()[np.sort(V_unique_idx)])
    ic(V_unique)
    ic(V_unique_idx_inv[0])
    ic(V_unique[V_unique_idx_inv[0]])
    ic((V_unique[np.argsort(V_unique_idx)])[V_unique_idx_inv[0]])
    ic()

    exit()

    ps.init()
    ps.register_surface_mesh('fandisk', V, F)
    ps.show()

    exit()

    E = np.vstack([
        np.stack([F[:, 0], F[:, 1]], -1),
        np.stack([F[:, 1], F[:, 2]], -1),
        np.stack([F[:, 2], F[:, 0]], -1)
    ])

    sort_idx = np.argsort(E[:, 0])
    E = E[sort_idx]

    _, e_count = np.unique(E[:, 0], return_counts=True)
    pad_width = np.max(e_count)

    split_indices = np.cumsum(e_count)[:-1]

    V2E_list = np.split(E[:, 1], split_indices)

    V2E = np.array([
        np.concatenate([el, -np.ones(pad_width - len(el))]) for el in V2E_list
    ]).astype(np.int64)

    exit()

    @jit
    def boundary_edge(F):
        E = edge_list(F)
        V2E = E[:, 0]
        E2 = E[:, 1]
        edges_sorted = jnp.hstack([
            jnp.where(V2E < E2, V2E, E2)[:, None],
            jnp.where(V2E > E2, V2E, E2)[:, None]
        ])
        edge_unique, edge_count = jnp.unique(edges_sorted,
                                             axis=0,
                                             return_counts=True)
        return edge_unique[edge_count == 1]

    E = edge_list(F)

    ic(E)

    V_unique, V_unique_idx, VE_count = jnp.unique(E[:, 0],
                                                  return_inverse=True,
                                                  return_counts=True)
    jnp.max(VE_count)

    ic(V_unique_idx)

    # ps.init()
    # ps.register_surface_mesh("test_mesh", np.array(V), np.array(F))
    # ps.register_point_cloud("test_pt", np.array(test_pt[None, :]))
    # ps.show()

    # ic(VE_count.min(), VE_count.max())

    # ic(FN.shape)
    # ic(E.shape)

    exit()

    # def vertex_normal(f, fn, V):

    # https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
    def face_curvature_tensor(f, V, VN):
        v0 = V[f[0]]
        v1 = V[f[1]]
        v2 = V[f[2]]

        e01 = v1 - v0
        e02 = v2 - v0

    local_face_basis(F[0], V)

    exit()

    ic(local_origin.shape, local_T.shape)
