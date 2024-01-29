from model_jax import curl
from jax import numpy as jnp, vmap, jit, jacfwd

from common import normalize
import polyscope as ps
from icecream import ic

line = jnp.linspace(-1, 1, 10)

grid = jnp.stack(jnp.meshgrid(line, line), -1).reshape(-1, 2)
grid = jnp.hstack([grid, jnp.zeros(len(grid))[:, None]])


@jit
def cal_vec_potential(x):
    return jnp.exp(-(x[0]**2 + x[1]**2) / 0.4) * jnp.array([0, 0, 1])


vec_potential = vmap(cal_vec_potential)(grid)

jac = vmap(jacfwd(cal_vec_potential))(grid)
curls = vmap(curl)(jac)

ps.init()
pc = ps.register_point_cloud('grid', grid)
pc.add_vector_quantity('vec_potential', vec_potential, enabled=True)
pc.add_vector_quantity('curls', curls, enabled=True)
ps.show()
