import equinox as eqx
import jax
import json
import argparse
from glob import glob

import model_jax
from config import Config
from config_utils import config_model, config_latent, config_training_data_pytorch
from train_jax import train
from eval_jax import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        nargs='*',
                        help='Path to pointcloud.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/octa.json',
                        help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--vis', action='store_true', help='Visualize')
    args = parser.parse_args()

    if args.model is None:
        model_list = sorted(glob('data/sdf/*.ply'))
    else:
        model_list = args.model

    for model in model_list:
        sdf_paths = [model]
        config = json.load(open(args.config))
        config['sdf_paths'] = sdf_paths

        cfg_name = args.config.split('/')[-1].split('.')[0]
        model_name = model.split('/')[-1].split('.')[0]
        name = f"{model_name}_{cfg_name}"
        print(name)

        cfg = Config(**config)
        cfg.name = name

        model_key, data_key = jax.random.split(
            jax.random.PRNGKey(cfg.training.seed), 2)

        latents, latent_dim = config_latent(cfg)
        model = config_model(cfg, model_key, latent_dim)

        if args.eval:
            model: model_jax.MLP = eqx.tree_deserialise_leaves(
                f"checkpoints/{cfg.name}.eqx", model)
        else:
            data = config_training_data_pytorch(cfg, latents)

            # Debug
            # total_steps = 1001
            total_steps = None

            model = train(cfg,
                          model,
                          data,
                          'checkpoints',
                          total_steps=total_steps)

        tokens = '0_1_0'.split('_')
        # Interpolate latent
        i = int(tokens[0])
        j = int(tokens[1])
        t = float(tokens[2])
        latent = (1 - t) * latents[i] + t * latents[j]

        eval(cfg, 'output', model, latent, vis_mc=args.vis)

        # exit()
