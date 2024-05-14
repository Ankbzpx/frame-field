from jax import numpy as jnp, jit, vmap
from common import normalize
from sh_representation import (oct_polynomial_zonal_unit_norm, rotvec_to_R9,
                               rotvec_n_to_z, project_n, project_z,
                               oct_polynomial_sh4_unit_norm, sh4_canonical)


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)

    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


@jit
def eikonal(x, norm=1):
    return jnp.abs(jnp.linalg.norm(x) - norm)


@jit
def double_well_potential(x):
    return 16 * (x - 0.5)**4 - 8 * (x - 0.5)**2 + 1


@jit
def align_basis_explicit(basis, normal):
    # Normalization matters here because we want the dot product to be either 0 or 1
    dps = jnp.einsum('bij,bi->bj', basis, vmap(normalize)(normal))
    return double_well_potential(jnp.abs(dps)).sum(-1)


@jit
def align_basis_functional(basis, normal):
    poly_val = vmap(oct_polynomial_zonal_unit_norm)(normal, basis)
    return jnp.abs(1 - poly_val)


@jit
def align_sh4_explicit(sh4, normal, xy_scale=1):
    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal))
    sh4_n = vmap(project_n, in_axes=(0, 0, None))(sh4, R9_zn, xy_scale)
    return vmap(jnp.linalg.norm, in_axes=(0, None))(sh4 - sh4_n, 1)


# FIXME: jnp.linalg.norm ord internally has python if that cannot be traced
@jit
def align_sh4_explicit_l2(sh4, normal, xy_scale=1):
    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal))
    sh4_n = vmap(project_n, in_axes=(0, 0, None))(sh4, R9_zn, xy_scale)
    return vmap(jnp.linalg.norm, in_axes=(0, None))(sh4 - sh4_n, 2)


@jit
def align_sh4_explicit_cosine(sh4, normal, xy_scale=1):
    R9_zn = vmap(rotvec_to_R9)(vmap(rotvec_n_to_z)(normal))
    sh4_n = vmap(project_n, in_axes=(0, 0, None))(sh4, R9_zn, xy_scale)
    return (1 - vmap(cosine_similarity)(sh4, sh4_n))


@jit
def align_sh4_functional(sh4, normal):
    poly_val = vmap(oct_polynomial_sh4_unit_norm)(normal, vmap(normalize)(sh4))
    # poly_val shouldn't exceed 1 but just to be safe
    return jnp.abs(1 - poly_val)
