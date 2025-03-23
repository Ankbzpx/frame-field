import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
from glob import glob
import json

from config import Config
from config_utils import config_training_data

from eval_pytorch import eval
import jax
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import torch
from train_pytorch import OctaGuidedSDF

from icecream import ic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="*", help="Path to pointcloud files."
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        default="data/sdf",
        help="Path to pointcloud folder.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/octa.json", help="Path to config file."
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--vis", action="store_true", help="Visualize")
    parser.add_argument("--skip", action="store_true", help="Skip existing output")
    args = parser.parse_args()

    if args.model is not None:
        tag = ""
        model_list = args.model
    else:
        # TODO; Maybe not hard coded
        tag = "_".join(args.model_folder.split("/")[-2:])
        model_list = sorted(glob(os.path.join(args.model_folder, "*.ply")))

    for model in model_list:
        sdf_paths = [model]
        config = json.load(open(args.config))
        config["sdf_paths"] = sdf_paths

        cfg_name = args.config.split("/")[-1].split(".")[0]
        model_name = model.split("/")[-1].split(".")[0]
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

        model = OctaGuidedSDF(cfg)

        if args.eval:
            checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{model_name}.ckpt")
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            model.cuda()
            model.eval()
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.checkpoints_dir, filename=cfg.name
            )
            dataloader = config_training_data(
                cfg,
                np.empty(
                    1,
                ),
                with_jax=False,
            )
            trainer = L.Trainer(
                max_steps=cfg.training.n_steps,
                max_epochs=cfg.training.n_epochs,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(model=model, train_dataloaders=dataloader)

        eval(cfg, model, vis_mc=args.vis)

        # exit()
