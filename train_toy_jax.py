import json

from config import Config
from train_jax import train

if __name__ == '__main__':
    for gap in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for angle in [10, 30, 90, 120, 150]:
            name = f"crease_{gap}_{angle}"

            config = json.load(open('configs/toy.json'))
            config['sdf_paths'] = [f"data/toy_sdf/{name}.npz"]

            cfg = Config(**config)
            cfg.name = name

            train(cfg)

            exit()
