import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import numpy as np
import lightning as L

from model_pytorch import HashMLP, VanillaMLP, Siren

from common import vis_oct_field, normalize_aabb
from jax import vmap, jit
from jax2torch import jax2torch
from config import Config, LossConfig
from config_utils import config_training_data, load_sdf
from sh_representation import R3_to_sh4_zonal

import json
import argparse

import polyscope as ps
from icecream import ic


def normalize(x):
    return x / (torch.linalg.norm(x) + 1e-8)


def cosine_similarity(x, y):
    demo = torch.linalg.norm(x) * torch.linalg.norm(y)

    return torch.dot(x, y) / torch.where(demo > 1e-8, demo, 1e-8)


def rot6d_to_R3(rot6d):
    a0 = rot6d[:3]
    a1 = rot6d[3:]
    b0 = normalize(a0)
    b1 = normalize(a1 - torch.dot(b0, a1) * b0)
    b2 = torch.linalg.cross(b0, b1)
    return torch.stack([b0, b1, b2]).T


def double_well_potential(x):
    return 16 * (x - 0.5)**4 - 8 * (x - 0.5)**2 + 1


def align_loss(rot6d, normal):
    basis = torch.vmap(rot6d_to_R3)(rot6d)
    dps = torch.einsum('bij,bi->bj', basis, torch.vmap(normalize)(normal))
    return double_well_potential(torch.abs(dps)).sum(-1)


def align_loss_axis(rot6d, normal):
    basis = torch.vmap(rot6d_to_R3)(rot6d)
    return 1 - torch.vmap(cosine_similarity)(basis[..., 0],
                                             torch.vmap(normalize)(normal))


def func_param_jax(basis):
    return vmap(R3_to_sh4_zonal)(basis)


def func_param(rot6d):
    basis = torch.vmap(rot6d_to_R3)(rot6d)
    sh4 = jax2torch(func_param_jax)(basis)
    return sh4


class HashOcta(L.LightningModule):

    def __init__(self):
        super().__init__()

        # Use rot6d
        self.octa_mlp = HashMLP(3, 256, 1, 6, interpolation="Nearest")
        # self.octa_mlp = VanillaMLP(3, 256, 4, 6)
        # self.octa_mlp = HashMLP(3, 256, 1, 6, interpolation="Linear")

    def training_step(self, batch, batch_idx):
        # On
        samples_on_sur: torch.Tensor = batch['samples_on_sur'][0]
        normals_on_sur: torch.Tensor = batch['normals_on_sur'][0]
        aux_on: torch.Tensor = self.octa_mlp(samples_on_sur)

        # Off
        samples_off_sur: torch.Tensor = batch['samples_off_sur'][0]
        samples_close_sur: torch.Tensor = batch['samples_close_sur'][0]

        # Align
        aux_align = aux_on
        normal_align = normals_on_sur
        loss_align = align_loss(aux_align, normal_align).mean()

        # Smooth
        # samples_smooth = torch.vstack([samples_off_sur, samples_close_sur])
        samples_smooth = samples_on_sur
        eps = 1e-3
        eps_x = torch.tensor([eps, 0., 0.],
                             dtype=samples_smooth.dtype,
                             device=samples_smooth.device)
        eps_y = torch.tensor([0., eps, 0.],
                             dtype=samples_smooth.dtype,
                             device=samples_smooth.device)
        eps_z = torch.tensor([0., 0., eps],
                             dtype=samples_smooth.dtype,
                             device=samples_smooth.device)

        # Forward difference
        param = func_param(self.octa_mlp(samples_smooth))
        param_x = func_param(self.octa_mlp(samples_smooth + eps_x[None, :]))
        param_y = func_param(self.octa_mlp(samples_smooth + eps_y[None, :]))
        param_z = func_param(self.octa_mlp(samples_smooth + eps_z[None, :]))

        dx = (param_x - param) / eps
        dy = (param_y - param) / eps
        dz = (param_z - param) / eps
        grad = torch.stack([dx, dy, dz], dim=-1)
        loss_smooth = torch.linalg.matrix_norm(grad).mean()

        loss = loss_align + 0.1 * loss_smooth
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    args = parser.parse_args()

    cfg = Config(**json.load(open(args.config)))
    cfg.name = args.config.split('/')[-1].split('.')[0]
    cfg.sdf_paths = [os.path.join('..', path) for path in cfg.sdf_paths]

    model = HashOcta()

    if args.eval:

        checkpoint_path = "lightning_logs/version_5/checkpoints/epoch=0-step=10000.ckpt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

        sdf_data = load_sdf(cfg.sdf_paths[0])
        sur_sample = sdf_data['samples_on_sur']
        sur_sample = normalize_aabb(sur_sample)
        x = torch.from_numpy(sur_sample).float().cuda()
        Rs = torch.vmap(rot6d_to_R3)(model.octa_mlp(x)).detach().cpu().numpy()

        ps.init()
        pc_viz = ps.register_point_cloud(f"pc", sur_sample)
        pc_viz.add_vector_quantity("v0", Rs[..., 0])
        pc_viz.add_vector_quantity("v1", Rs[..., 1])
        pc_viz.add_vector_quantity("v2", Rs[..., 2])
        ps.show()

        exit()

    dataloader = config_training_data(cfg, np.empty(1,), with_jax=False)

    trainer = L.Trainer(max_steps=cfg.training.n_steps,
                        max_epochs=cfg.training.n_epochs)
    trainer.fit(model=model, train_dataloaders=dataloader)
