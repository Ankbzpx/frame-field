import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
import igl
import json

import model_jax
from config import Config
from config_utils import config_model
from common import vis_oct_field
from eval_jax import extract_surface
from sh_representation import proj_sh4_to_R3

import polyscope as ps
from icecream import ic

import time


def eval(cfg: Config, eval_samples, out_dir):
    model_key = jax.random.PRNGKey(0)
    model = config_model(cfg, model_key, 0)
    model: model_jax.MLP = eqx.tree_deserialise_leaves(
        f"checkpoints/{cfg.name}.eqx", model)

    latent = jnp.zeros((0,))

    @jit
    def infer(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)

    start_time = time.time()
    V, F = extract_surface(infer)
    print("Extract surface", time.time() - start_time)
    start_time = time.time()

    V_vis, F_vis = vis_oct_field(proj_sh4_to_R3(infer(eval_samples)[:, 1:]),
                                 eval_samples, 0.01)

    ps.init()
    ps.register_surface_mesh("mesh", V, F)
    ps.register_surface_mesh("Oct frames", V_vis, F_vis)
    ps.show()

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

            exit()
