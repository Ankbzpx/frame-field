import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import seed_worker
from transformers.training_args import TrainingArguments
from dataio import PointCloud
import models

import polyscope as ps
from icecream import ic


class SDFCallback(TrainerCallback):

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, **kwargs):
        ic(state.log_history)
        pass


class SDFTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self._train_batch_size,
                          drop_last=self.args.dataloader_drop_last,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=seed_worker)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model.forward_grad_batch(inputs['coords'])
        pred_normal, (pred_sdf, sh9) = outputs

        on_surface = inputs['on_surface']
        off_surface = torch.logical_not(on_surface)

        loss_mse = (pred_sdf[on_surface]).abs().mean()
        loss_off = torch.exp(-1e2 * pred_sdf[off_surface].abs()).mean()
        loss_normal = (1 - F.cosine_similarity(pred_normal[on_surface],
                                               inputs['normals'][on_surface],
                                               dim=-1)).mean()
        loss_eikonal = (pred_normal.norm(dim=-1) - 1).abs().mean()
        loss = 3e3 * loss_mse + 1e2 * loss_off + 1e2 * loss_normal + 5e1 * loss_eikonal

        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':

    mlp_cfg = {
        'in_features': 3,
        'hidden_features': 256,
        'hidden_layers': 4,
        'out_features': 10,
    }

    mlp = torch.compile(models.MLP(**mlp_cfg))

    output_dir = 'output'
    args = TrainingArguments(output_dir)
    args.set_training(learning_rate=1e-4, batch_size=128 * 3, num_epochs=100)

    pc_path = 'data/sdf/rocker_arm.npy'
    pc_sample_size = 512
    dataset = PointCloud(pc_path, pc_sample_size)

    trainer = SDFTrainer(model=mlp,
                         args=args,
                         train_dataset=dataset,
                         callbacks=[SDFCallback()])

    trainer.train()

    torch.save(mlp.state_dict(), 'model_weights.pth')
