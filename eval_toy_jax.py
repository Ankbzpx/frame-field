import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
from skimage.measure import marching_cubes
import igl
import json

import model_jax
from config import Config
from common import vis_oct_field
from sh_representation import proj_sh4_to_R3

import polyscope as ps
from icecream import ic

import time


def eval(cfg: Config, eval_samples, out_dir):
    model: model_jax.MLP = model_jax.MLPComposer(
        jax.random.PRNGKey(0),
        cfg.mlp_types,
        cfg.mlp_cfgs,
    )
    model = eqx.tree_deserialise_leaves(f"checkpoints/{cfg.name}.eqx", model)

    latent = jnp.zeros((0,))

    grid_res = 256
    grid_min = -1.0
    grid_max = 1.0
    # Smaller batch is somehow faster
    group_size = 4 * grid_res**2
    iter_size = grid_res**3 // group_size

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)

    @jit
    def infer_sdf():
        indices = jnp.linspace(grid_min, grid_max, grid_res)
        grid = jnp.stack(jnp.meshgrid(indices, indices, indices),
                         -1).reshape(iter_size, group_size, 3)

        query_data = {"grid": grid, "sdf": jnp.zeros((iter_size, group_size))}

        @jit
        def body_func(i, query_data):
            sdf = infer(query_data["grid"][i])[:, 0]
            query_data["sdf"] = query_data["sdf"].at[i].set(sdf)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["sdf"].reshape(grid_res, grid_res, grid_res)

    start_time = time.time()
    sdf = infer_sdf()
    print("Inference SDF", time.time() - start_time)
    start_time = time.time()

    # This step is unfortunately slow
    sdf_np = np.swapaxes(np.array(sdf), 0, 1)

    spacing = 1. / (grid_res - 1)
    V, F, _, _ = marching_cubes(sdf_np, 0., spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)

    V_vis, F_vis = vis_oct_field(proj_sh4_to_R3(infer(eval_samples)[:, 1:]),
                                 eval_samples, 0.01)

    # ps.init()
    # ps.register_surface_mesh("mesh", V, F)
    # ps.register_surface_mesh("Oct frames", V_vis, F_vis)
    # ps.show()

    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_mc.obj", V, F)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_cube.obj", V_vis, F_vis)


if __name__ == '__main__':
    for gap in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for angle in [10, 30, 90, 120, 150]:
            name = f"crease_{gap}_{angle}"

            config = json.load(open('configs/toy.json'))
            config['sdf_paths'] = [f"data/toy_sdf/{name}.npz"]

            cfg = Config(**config)
            cfg.name = name
            print(name)

            eval_samples = np.load(f"data/toy_eval/crease_{angle}.npy")

            eval(cfg, eval_samples, "output/toy")
