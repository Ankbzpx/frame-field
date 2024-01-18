import equinox as eqx
import numpy as np
import jax
from jax import jit, numpy as jnp, vmap
import igl
import json
import argparse

import model_jax
from config import Config
from config_utils import config_model, config_latent, config_training_data
from train_jax import train
from eval_jax import eval

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/deep_geometric_prior.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    args = parser.parse_args()

    scan_list = ['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas']
    sample_size_list = [2500, 5000, 10000, -1]
    reg_cfg_list = ['reg', 'no_reg']

    for scan in scan_list:
        sdf_paths = [f"data/sdf/deep_geometric_prior/{scan}.npz"]

        for sample_size in sample_size_list:
            for reg_cfg in reg_cfg_list:
                name = f"{scan}_{sample_size if sample_size !=-1 else 'full'}_{reg_cfg}"

                config = json.load(open(args.config))
                config['sdf_paths'] = sdf_paths

                # 'loss_cfg' needs to be frozen
                if reg_cfg == 'reg':
                    config['loss_cfg']['regularize'] = 1e2

                cfg = Config(**config)
                cfg.name = name

                cfg.training.n_input_samples = sample_size
                cfg.training.close_scale = 1e-1

                model_key, data_key = jax.random.split(
                    jax.random.PRNGKey(cfg.training.seed), 2)

                latents, latent_dim = config_latent(cfg)
                model = config_model(cfg, model_key, latent_dim)

                if args.eval:
                    model: model_jax.MLP = eqx.tree_deserialise_leaves(
                        f"checkpoints/{cfg.name}.eqx", model)
                else:
                    # Debug
                    # cfg.training.n_steps = 101

                    data = config_training_data(cfg, data_key, latents)

                    model = train(cfg, model, data, 'checkpoints')

                tokens = '0_1_0'.split('_')
                # Interpolate latent
                i = int(tokens[0])
                j = int(tokens[1])
                t = float(tokens[2])
                latent = (1 - t) * latents[i] + t * latents[j]

                eval(cfg,
                     'output/deep_geometric_prior',
                     model,
                     latent,
                     vis_mc=args.vis,
                     geo_only=True)

                # exit()
