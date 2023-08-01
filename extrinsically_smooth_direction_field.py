import igl
import numpy as np
import jax
from jax import vmap, jit, numpy as jnp

from functools import partial
import scipy

from common import vis_oct_field, normalize, build_traversal_graph, per_face_basis, per_vertex_normal, face_area, cotangent_weight

import polyscope as ps
from icecream import ic


@jit
@partial(vmap, in_axes=(0, 0, None, None, None, None, None, None))
def dirichlet(ws, e_ids, E, E2E, FA, alpha, beta, NV):
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
def to_rotation(x, z):
    y = jnp.cross(x, z)
    return jnp.stack([x, y, z], -1)


if __name__ == '__main__':
    # enable 64 bit precision
    from jax.config import config
    config.update("jax_enable_x64", True)

    V, F = igl.read_triangle_mesh("data/mesh/fandisk.ply")

    V, F, E, V2E, E2E, V_boundary, V_nonmanifold = build_traversal_graph(V, F)
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

    idx_i, idx_j, weights, mass = dirichlet(Ws, V2E, E, E2E, FA, alpha, beta,
                                            NV)

    idx_i = idx_i.reshape(-1)
    idx_j = idx_j.reshape(-1)
    weights = weights.reshape(-1)
    mass = mass.reshape(-1)

    valid_mask = weights != 0
    idx_i = np.int32(idx_i[valid_mask])
    idx_j = np.int32(idx_j[valid_mask])
    weights = weights[valid_mask]
    mass = mass[valid_mask]

    A = scipy.sparse.coo_array((weights, (idx_i, idx_j)),
                               shape=(2 * NV, 2 * NV)).tocsc()

    M = scipy.sparse.coo_array((mass, (idx_i, idx_j)),
                               shape=(2 * NV, 2 * NV)).tocsc()

    # Algorithm 2 in Globally Optimal Direction Fields by Knöppel et al.
    np.random.seed(0)
    X = np.random.randn(2 * NV, 1)
    solve = scipy.sparse.linalg.factorized(A)

    for _ in range(30):
        X = solve(M @ X)
        X /= np.sqrt(X.T @ M @ X)

    a = X[:NV, 0]
    b = X[NV:, 0]

    # representation vector
    tangent = vmap(normalize)(a[:, None] * alpha + b[:, None] * beta)

    V_vis, F_vis = vis_oct_field(vmap(to_rotation)(tangent, VN), V, F)

    ps.init()
    mesh = ps.register_surface_mesh("mesh", V, F)
    ps.register_surface_mesh("Oct_opt", V_vis, F_vis)
    mesh.add_vector_quantity("VN", VN)
    ps.show()
