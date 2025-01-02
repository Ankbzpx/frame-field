import torch
import numpy as np
import lightning as L

from jax import vmap, jit
from jax2torch import jax2torch
from config import Config, LossConfig
from config_utils import config_training_data
from model import Siren, LipschitzMLP, gradient, hessian
from loss import eikonal, align_sh4_explicit_cosine, align_sh4_explicit

import json
import argparse

from icecream import ic


@jit
def eikonal_vmapped(x):
    return vmap(eikonal)(x)


def eikonal_torch(normal):
    return jax2torch(eikonal_vmapped)(normal)


def align_torch(sh4, normal):
    return jax2torch(align_sh4_explicit_cosine)(sh4, normal)


def reg_torch(sh4, normal):
    return jax2torch(align_sh4_explicit)(sh4, normal)


class OctaGuidedSDF(L.LightningModule):

    def __init__(self, sdf_mlp, octa_mlp, cfg: Config):
        super().__init__()
        self.sdf_mlp: torch.nn.Module = sdf_mlp
        self.octa_mlp: torch.nn.Module = octa_mlp
        self.cfg: Config = cfg

    def training_step(self, batch, batch_idx):
        loss_cfg = self.cfg.loss_cfg

        # On
        samples_on_sur: torch.Tensor = batch['samples_on_sur'][0]
        samples_on_sur.requires_grad_(True)
        normals_on_sur: torch.Tensor = batch['normals_on_sur'][0]

        pred_on_sur_sdf: torch.Tensor = self.sdf_mlp(samples_on_sur)
        pred_normals_on_sur: torch.Tensor = gradient(pred_on_sur_sdf,
                                                     samples_on_sur)
        aux_on: torch.Tensor = self.octa_mlp(samples_on_sur)

        # Off
        samples_off_sur: torch.Tensor = batch['samples_off_sur'][0]

        pred_off_sur_sdf: torch.Tensor = self.sdf_mlp(samples_off_sur)

        # Close
        samples_close_sur: torch.Tensor = batch['samples_close_sur'][0]
        samples_close_sur.requires_grad_(True)

        pred_close_sur_sdf: torch.Tensor = self.sdf_mlp(samples_close_sur)
        hessian_close = hessian(pred_close_sur_sdf, samples_close_sur)

        # Siren
        # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L214
        loss_mse = loss_cfg.on_sur * torch.abs(pred_on_sur_sdf).mean()
        loss_off = loss_cfg.off_sur * torch.exp(
            -1e2 * torch.abs(pred_off_sur_sdf)).mean()
        loss_eikonal = loss_cfg.eikonal * eikonal_torch(
            pred_normals_on_sur).mean()
        loss = loss_mse + loss_off + loss_eikonal
        loss_dict = {
            'loss_mse': loss_mse,
            'loss_off': loss_off,
            'loss_eikonal': loss_eikonal
        }

        # Align
        if loss_cfg.align > 0:
            sample_weight = torch.exp(-1e2 *
                                      torch.abs(pred_on_sur_sdf.detach()))
            normal_align = pred_normals_on_sur.detach()
            aux_align = aux_on
            loss_align = loss_cfg.align * (
                sample_weight * align_torch(aux_align, normal_align)).mean()
            loss += loss_align
            loss_dict['loss_align'] = loss_align

        # Regularize
        if loss_cfg.regularize > 0:
            normal_reg = pred_normals_on_sur
            aux_reg = aux_on.detach()
            loss_reg = loss_cfg.regularize * reg_torch(aux_reg,
                                                       normal_reg).mean()
            loss += loss_reg
            loss_dict['loss_reg'] = loss_reg

        # Lip
        if loss_cfg.lip > 0:
            loss_lip = loss_cfg.lip * octa_mlp.get_lipschitz_loss()
            loss += loss_lip
            loss_dict['loss_lip'] = loss_lip

        # Hessian
        if loss_cfg.hessian > 0:
            loss_hessian = loss_cfg.hessian * 0.5 * torch.abs(
                torch.det(hessian_close)).mean()
            loss += loss_hessian
            loss_dict['loss_hessian'] = loss_hessian

        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]

    mlp_cfgs = cfg.mlp_cfgs
    sdf_mlp = Siren(**mlp_cfgs[0])
    octa_mlp = LipschitzMLP(**mlp_cfgs[1])
    model = OctaGuidedSDF(sdf_mlp, octa_mlp, cfg)

    dataloader = config_training_data(cfg, np.empty(1,), with_jax=False)

    trainer = L.Trainer(max_steps=cfg.training.n_steps,
                        max_epochs=cfg.training.n_epochs)
    trainer.fit(model=model, train_dataloaders=dataloader)
