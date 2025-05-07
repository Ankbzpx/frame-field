import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
import json

from config import Config
from config_utils import config_training_data

import lightning as L
from model_pytorch import gradient, hessian, Siren, vector_gradient
import numpy as np
from sh_pytorch import align_sh4_functional_grad
import torch


def eikonal(x):
    return torch.abs(torch.linalg.norm(x, dim=-1) - 1)


def linear_schedule(init_value, end_value, transition_steps, transition_begin):
    def get_value(step):
        if step < transition_begin:
            return init_value
        elif step > transition_steps + transition_begin:
            return end_value
        else:
            return (
                init_value
                + (end_value - init_value)
                * (step - transition_begin)
                / transition_steps
            )

    return get_value


class OctaGuidedSDF(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        mlp_cfgs = cfg.mlp_cfgs
        self.sdf_mlp = Siren(**mlp_cfgs[0])
        self.octa_mlp = Siren(**mlp_cfgs[1])
        self.cfg: Config = cfg

        self.regularize_schedule = linear_schedule(
            0,
            cfg.loss_cfg.regularize,
            int(0.2 * cfg.training.n_steps),
            int(cfg.loss_cfg.regularize_begin * cfg.training.n_steps),
        )
        self.hessian_schedule = linear_schedule(
            cfg.loss_cfg.hessian,
            cfg.loss_cfg.hessian_annealing * cfg.loss_cfg.hessian,
            int(0.1 * cfg.training.n_steps),
            0,
        )

    def training_step(self, batch, batch_idx):
        loss_cfg = self.cfg.loss_cfg

        smooth_weight = self.cfg.loss_cfg.smooth
        align_weight = self.cfg.loss_cfg.align
        regularize_weight = self.regularize_schedule(self.global_step)
        hessian_weight = self.hessian_schedule(self.global_step)

        # On
        samples_on_sur: torch.Tensor = batch["samples_on_sur"][0]
        samples_on_sur.requires_grad_(True)
        normals_on_sur: torch.Tensor = batch["normals_on_sur"][0]

        pred_on_sur_sdf: torch.Tensor = self.sdf_mlp(samples_on_sur)
        pred_normals_on_sur: torch.Tensor = gradient(pred_on_sur_sdf, samples_on_sur)
        aux_on: torch.Tensor = self.octa_mlp(samples_on_sur)

        # Off
        samples_off_sur: torch.Tensor = batch["samples_off_sur"][0]
        pred_off_sur_sdf: torch.Tensor = self.sdf_mlp(samples_off_sur)

        if smooth_weight > 0:
            samples_off_sur.requires_grad_(True)
            aux_off: torch.Tensor = self.octa_mlp(samples_off_sur)

        # Close
        samples_close_sur: torch.Tensor = batch["samples_close_sur"][0]
        samples_close_sur.requires_grad_(True)

        pred_close_sur_sdf: torch.Tensor = self.sdf_mlp(samples_close_sur)
        hessian_close = hessian(pred_close_sur_sdf, samples_close_sur)

        # Siren
        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * torch.abs(pred_on_sur_sdf).mean()
        loss_off = (
            loss_cfg.off_sur * torch.exp(-1e2 * torch.abs(pred_off_sur_sdf)).mean()
        )
        loss_eikonal = loss_cfg.eikonal * eikonal(pred_normals_on_sur).mean()
        loss = loss_mse + loss_off + loss_eikonal
        loss_dict = {
            "loss_mse": loss_mse,
            "loss_off": loss_off,
            "loss_eikonal": loss_eikonal,
        }

        sample_weight = torch.exp(-1e2 * torch.abs(pred_on_sur_sdf.detach()))

        # Align
        if align_weight > 0:
            normal_align = pred_normals_on_sur.detach()
            aux_align = aux_on
            loss_align = (
                align_weight
                * (
                    sample_weight * align_sh4_functional_grad(aux_align, normal_align)
                ).mean()
            )
            loss += loss_align
            loss_dict["loss_align"] = loss_align

        # Regularize
        if regularize_weight > 0:
            normal_reg = pred_normals_on_sur
            aux_reg = aux_on.detach()
            loss_reg = (
                regularize_weight
                * (
                    sample_weight * align_sh4_functional_grad(aux_reg, normal_reg)
                ).mean()
            )
            loss += loss_reg
            loss_dict["loss_reg"] = loss_reg

        if smooth_weight > 0:
            jac_on = vector_gradient(aux_on, samples_on_sur)
            jac_off = vector_gradient(aux_off, samples_off_sur)
            sh4_jac = torch.vstack([jac_on, jac_off])
            loss_smooth = (
                smooth_weight
                * (sample_weight * torch.linalg.matrix_norm(sh4_jac)).mean()
            )
            loss += loss_smooth
            loss_dict["loss_smooth"] = loss_smooth

        # Hessian
        if hessian_weight > 0:
            loss_hessian = (
                hessian_weight * 0.5 * torch.abs(torch.det(hessian_close)).mean()
            )
            loss += loss_hessian
            loss_dict["loss_hessian"] = loss_hessian

        self.log("loss", loss, prog_bar=True)
        self.log_dict(loss_dict, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
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
