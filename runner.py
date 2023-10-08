import equinox as eqx
import os
import numpy as np
import argparse
from glob import glob
import json

import jax
from jax import numpy as jnp

from config import Config
from config_utils import config_latent, config_model, config_training_data

from train_jax import train
from eval_jax import eval

from icecream import ic

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_path', type=str, help='Path to input model sdf')
    parser.add_argument('--subfolder',
                        type=str,
                        default='',
                        help='Subfolder path.')
    parser.add_argument('--sample_size',
                        type=int,
                        default=10000,
                        help='Number of samples.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    args = parser.parse_args()

    subfolder = args.subfolder

    if args.sdf_path is not None:
        subfolder = '/'.join(args.sdf_path.split('/')[2:-1])
        sdf_paths = [args.sdf_path]
    else:
        sdf_paths = glob(os.path.join('data/sdf', subfolder, '*.npz'))

    # config_list = ['siren']
    config_list = ['siren', 'siren_reg']

    output_folder = os.path.join('output', subfolder)
    checkpoints_folder = os.path.join('checkpoints', subfolder)

    model_sharp_fea = ['fandisk', 'cube_twist']

    for sdf_path in sdf_paths:
        for base_cfg in config_list:

            config = json.load(open(f'configs/{base_cfg}.json'))
            config['sdf_paths'] = [sdf_path]

            model_name = sdf_path.split('/')[-1].split('.')[0]

            if model_name in model_sharp_fea:
                if 'regularize' in config['loss_cfg']:
                    config['loss_cfg']['regularize'] = 1e2

            cfg = Config(**config)
            cfg.name = f'{model_name}_{base_cfg}'

            model_key, data_key = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), 2)

            latents, latent_dim = config_latent(cfg)
            model = config_model(cfg, model_key, latent_dim)

            if args.eval:
                model = eqx.tree_deserialise_leaves(
                    os.path.join(checkpoints_folder, f'{cfg.name}.eqx'), model)
            else:
                data = config_training_data(cfg, data_key, latents)
                model = train(cfg, model, data, checkpoints_folder)

            eval(cfg, output_folder, model, jnp.zeros((0,)))
