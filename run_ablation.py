from config import Config
from train_jax import train
from eval_jax import eval
import json
from tqdm import tqdm
import argparse
import os

import numpy as np
from icecream import ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    name = args.config.split('/')[-1].split('.')[0]
    out_dir = f"output/ablation_{name}"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    cfg_dict = json.load(open(args.config))
    ablation_cfg = [0.0, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

    for s in tqdm(ablation_cfg):
        cfg_dict_abl = cfg_dict.copy()
        cfg_dict_abl['loss_cfg']['smooth'] = s

        cfg = Config(**cfg_dict)
        cfg.name = f'smooth_{np.format_float_scientific(s)}'

        train(cfg)
        eval(cfg, out_dir)
