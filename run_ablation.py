from config import Config
from train_jax import train
from eval_jax import eval
import json
from tqdm import tqdm

import numpy as np
from icecream import ic

if __name__ == '__main__':
    cfg_dict = json.load(open('configs/bunny_dual_smooth.json'))
    ablation_cfg = [1e-2, 1e-1, 1.0, 1e1, 1e2]

    for s in tqdm(ablation_cfg):
        cfg_dict_abl = cfg_dict.copy()
        cfg_dict_abl['loss_cfg']['smooth'] = s

        cfg = Config(**cfg_dict)
        cfg.name = f'smooth_ablation_{np.format_float_scientific(s)}'

        train(cfg)
        eval(cfg, 'output/ablation')
