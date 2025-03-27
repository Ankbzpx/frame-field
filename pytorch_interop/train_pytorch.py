import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
import json

from config import Config, LossConfig
from config_utils import config_training_data
from loss import align_sh4_explicit, align_sh4_explicit_cosine

import jax
from jax import jit, vmap
from jax2torch import jax2torch
import lightning as L
from model_pytorch import gradient, HashMLP, LipschitzMLP, Siren, vector_gradient
import numpy as np
import optax
from sh_torch import R3_to_sh4_zonal
import torch

from icecream import ic


def eikonal(x):
    return torch.abs(torch.linalg.norm(x) - 1)


def cosine_similarity(x, y):
    demo = torch.linalg.norm(x) * torch.linalg.norm(y)
    return torch.dot(x, y) / torch.where(demo > 1e-8, demo, 1e-8)


def align_torch(sh4, normal):
    return jax2torch(align_sh4_explicit_cosine)(sh4, normal)


def reg_torch(sh4, normal):
    return jax2torch(align_sh4_explicit)(sh4, normal)


def normalize(x):
    return x / (torch.linalg.norm(x) + 1e-8)


def rot6d_to_R3(rot6d):
    a0 = rot6d[:3]
    a1 = rot6d[3:]
    b0 = normalize(a0)
    b1 = normalize(a1 - torch.dot(b0, a1) * b0)
    b2 = torch.linalg.cross(b0, b1)
    return torch.stack([b0, b1, b2]).T


def func_param(rot6d):
    basis = torch.vmap(rot6d_to_R3)(rot6d)
    sh4 = R3_to_sh4_zonal(basis)
    return sh4


def eval_param_jac(x, param_fun, eps=1e-3):
    eps_x = torch.tensor([eps, 0.0, 0.0], dtype=x.dtype, device=x.device)
    eps_y = torch.tensor([0.0, eps, 0.0], dtype=x.dtype, device=x.device)
    eps_z = torch.tensor([0.0, 0.0, eps], dtype=x.dtype, device=x.device)
    # Forward difference
    param = param_fun(x)
    param_x = param_fun(x + eps_x[None, :])
    param_y = param_fun(x + eps_y[None, :])
    param_z = param_fun(x + eps_z[None, :])

    dx = (param_x - param) / eps
    dy = (param_y - param) / eps
    dz = (param_z - param) / eps
    grad = torch.stack([dx, dy, dz], dim=-1)
    return grad


class OctaGuidedSDF(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        mlp_cfgs = cfg.mlp_cfgs
        self.sdf_mlp = Siren(**mlp_cfgs[0])
        self.octa_mlp = HashMLP(3, 256, 1, 3)
        self.cfg: Config = cfg

        self.smooth_schedule = jax2torch(
            jit(
                optax.linear_schedule(
                    0,
                    1,
                    int(0.1 * cfg.training.n_steps),
                    int(0.4 * cfg.training.n_steps),
                )
            )
        )

    def training_step(self, batch, batch_idx):
        loss_cfg = self.cfg.loss_cfg

        # On
        samples_on_sur: torch.Tensor = batch["samples_on_sur"][0]
        samples_on_sur.requires_grad_(True)
        normals_on_sur: torch.Tensor = batch["normals_on_sur"][0]

        pred_on_sur_sdf: torch.Tensor = self.sdf_mlp(samples_on_sur)
        pred_normals_on_sur: torch.Tensor = gradient(pred_on_sur_sdf, samples_on_sur)

        # Off
        samples_off_sur: torch.Tensor = batch["samples_off_sur"][0]
        pred_off_sur_sdf: torch.Tensor = self.sdf_mlp(samples_off_sur)

        # Siren
        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * torch.abs(pred_on_sur_sdf).mean()
        loss_normal = (
            loss_cfg.normal
            * (
                1 - torch.vmap(cosine_similarity)(pred_normals_on_sur, normals_on_sur)
            ).mean()
        )
        loss_off = (
            loss_cfg.off_sur * torch.exp(-1e2 * torch.abs(pred_off_sur_sdf)).mean()
        )
        loss_eikonal = (
            loss_cfg.eikonal * torch.vmap(eikonal)(pred_normals_on_sur).mean()
        )
        loss = loss_mse + loss_normal + loss_off + loss_eikonal

        def param_func(x):
            y: torch.Tensor = self.sdf_mlp(x)
            dydx: torch.Tensor = gradient(y, x)
            aux_on = self.octa_mlp(x)
            rot6d = torch.hstack([dydx, aux_on])
            sh4 = func_param(rot6d)
            return sh4

        smooth_weight = self.smooth_schedule(self.global_step)

        jac = eval_param_jac(samples_on_sur, param_func)
        loss_smooth = torch.linalg.matrix_norm(jac).mean()
        loss += smooth_weight * loss_smooth

        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split("/")[-1].split(".")[0]
    cfg.sdf_paths = [os.path.join("..", path) for path in cfg.sdf_paths]

    model = OctaGuidedSDF(cfg)

    dataloader = config_training_data(
        cfg,
        np.empty(
            1,
        ),
        with_jax=False,
    )

    trainer = L.Trainer(
        max_steps=cfg.training.n_steps, max_epochs=cfg.training.n_epochs
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
