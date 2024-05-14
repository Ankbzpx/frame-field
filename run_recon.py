import equinox as eqx
import jax
import json
import argparse
from glob import glob
import os

import model_jax
from config import Config
from config_utils import config_model, config_latent, config_training_data_pytorch
from train_jax import train
from eval_jax import eval

from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        nargs='*',
                        help='Path to pointcloud files.')
    parser.add_argument('--model_folder',
                        type=str,
                        default='data/sdf',
                        help='Path to pointcloud folder.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/octa.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    parser.add_argument('--skip',
                        action='store_true',
                        help='Skip existing output')
    args = parser.parse_args()

    if args.model is not None:
        tag = ''
        model_list = args.model
    else:
        # TODO; Maybe not hard coded
        tag = '_'.join(args.model_folder.split('/')[-2:])
        model_list = sorted(glob(os.path.join(args.model_folder, '*.ply')))

    for model in model_list:
        sdf_paths = [model]
        config = json.load(open(args.config))
        config['sdf_paths'] = sdf_paths

        cfg_name = args.config.split('/')[-1].split('.')[0]
        model_name = model.split('/')[-1].split('.')[0]
        name = model_name
        print(name)

        cfg = Config(**config)
        cfg.name = name
        cfg.out_dir = os.path.join(cfg.out_dir, cfg_name, tag)
        cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, cfg_name, tag)

        if args.skip:
            out_file = os.path.join(cfg.out_dir, f"{model_name}.obj")
            if os.path.exists(out_file):
                continue

        model_key, data_key = jax.random.split(
            jax.random.PRNGKey(cfg.training.seed), 2)

        latents, latent_dim = config_latent(cfg)
        model = config_model(cfg, model_key, latent_dim)

        if args.eval:
            model: model_jax.MLP = eqx.tree_deserialise_leaves(
                os.path.join(cfg.checkpoints_dir, f"{cfg.name}.eqx"), model)
        else:
            data = config_training_data_pytorch(cfg, latents)
            model = train(cfg, model, data)

        tokens = '0_1_0'.split('_')
        # Interpolate latent
        i = int(tokens[0])
        j = int(tokens[1])
        t = float(tokens[2])
        latent = (1 - t) * latents[i] + t * latents[j]

        eval(cfg, model, latent, vis_mc=args.vis)

        # exit()
