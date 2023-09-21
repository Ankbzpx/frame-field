import os
import subprocess

from icecream import ic

if __name__ == '__main__':

    config_list = os.listdir('configs')
    if 'toy.json' in config_list:
        config_list.remove('toy.json')

    for cfg in config_list:
        cfg_path = os.path.join('configs', cfg)
        subprocess.run(["python", "train_jax.py", cfg_path])
        subprocess.run(["python", "eval_jax.py", cfg_path])
