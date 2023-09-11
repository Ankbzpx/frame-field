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
from sh_representation import proj_sh4_to_R3, rot6d_to_R3, R3_to_sh4_zonal

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
        return model(x, z)[:, 0]

    start_time = time.time()
    V, F = extract_surface(infer)
    print("Extract surface", time.time() - start_time)
    start_time = time.time()

    @jit
    def infer_aux(x):
        z = latent[None, ...].repeat(len(x), 0)
        return model(x, z)[:, 1:]

    def vis_oct(samples):
        aux = infer_aux(samples)
        if cfg.loss_cfg.rot6d:
            Rs = vmap(rot6d_to_R3)(aux[:, :6])
        else:
            sh4 = aux[:, :9]
            Rs = proj_sh4_to_R3(sh4)

        return vis_oct_field(Rs, samples, 0.01)

    V_vis_sup, F_vis_sup = vis_oct(eval_samples['samples'])
    V_vis_interp, F_vis_interp = vis_oct(eval_samples['samples_gap'])

    ps.init()
    ps.register_surface_mesh("mesh", V, F)
    ps.register_surface_mesh("Oct frames supervise", V_vis_sup, F_vis_sup)
    ps.register_surface_mesh("Oct frames interpolation", V_vis_interp,
                             F_vis_interp)
    ps.show()

    # V_cube = np.vstack([V_vis_sup, V_vis_interp])
    # F_cube = np.vstack([F_vis_sup, F_vis_interp + F_vis_sup.max() + 1])

    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_mc.obj", V, F)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_sup.obj", V_vis_sup,
                            F_vis_sup)
    igl.write_triangle_mesh(f"{out_dir}/{cfg.name}_interp.obj", V_vis_interp,
                            F_vis_interp)


if __name__ == '__main__':
    # 0.05, 0.1, 0.2, 0.3, 0.4
    # 10, 30, 90, 120, 150
    for gap in [0.4]:
        for angle in [120]:
            name = f"crease_{gap}_{angle}"

            config = json.load(open('configs/toy.json'))
            config['sdf_paths'] = [f"data/toy_sdf/{name}.npz"]

            cfg = Config(**config)
            cfg.name = name
            print(name)

            eval_samples = np.load(f"data/toy_eval/crease_{gap}_{angle}.npz")

            eval(cfg, eval_samples, "output/toy")

            # exit()
