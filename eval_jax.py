import equinox as eqx

import numpy as np
import jax
from jax import jit
from model_jax import StandardMLP
from skimage.measure import marching_cubes
import igl

import polyscope as ps
from icecream import ic

if __name__ == '__main__':

    mlp_cfg = {
        'in_features': 3,
        'hidden_features': 256,
        'hidden_layers': 8,
        'out_features': 10,
    }
    model = StandardMLP(**mlp_cfg, key=jax.random.PRNGKey(0), activation='elu')
    model = eqx.tree_deserialise_leaves(f"checkpoints/fandisk.eqx", model)

    grid_res = 512
    grid_min = -1.0
    grid_max = 1.0
    group_size = 1200000

    @jit
    def infer(x):
        return model(x)

    indices = np.linspace(grid_min, grid_max, grid_res)
    grid = np.stack(np.meshgrid(indices, indices, indices), -1).reshape(-1, 3)

    sdfs_list = []
    for x in np.array_split(grid, len(grid) // group_size, axis=0):
        sdf, sh9 = infer(x)
        sdfs_list.append(np.array(sdf))
    sdfs = np.concatenate(sdfs_list).reshape(grid_res, grid_res, grid_res)

    spacing = 1. / grid_res
    V, F, _, _ = marching_cubes(sdfs, 0., spacing=(spacing, spacing, spacing))

    igl.write_triangle_mesh("weird.obj", V, F)

    ps.init()
    ps.register_surface_mesh("fandisk", V, F)
    ps.show()
