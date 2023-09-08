from jax import numpy as jnp, jit


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)

    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


@jit
def eikonal(x):
    return jnp.abs(jnp.linalg.norm(x) - 1)


@jit
def double_well_potential(x):
    return 16 * (x - 0.5)**4 - 8 * (x - 0.5)**2 + 1
