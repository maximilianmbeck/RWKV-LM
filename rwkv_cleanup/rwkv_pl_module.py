from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import FusedAdam
from torch import nn

from rwkv_cleanup.rwkv_model import RWKV, RWKVConfig


@dataclass
class RWKVModuleConfig:
    """Configuration for RWKVModule."""
    model: RWKVConfig


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKVModel(pl.LightningModule):

    def __init__(self, cfg: RWKVModuleConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        args = kwargs.get('args', None)
        if args is None:
            args = {}
        else:
            args = vars(args)
        self.rwkv_args = args

        self.model = RWKV(cfg.model)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_idxes, y = batch
        y_hat = self(x_idxes)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log("train_loss", loss)
        # TODO what does the masking stuff do?
        # TODO make loss wrapper configurable

        return L2Wrap.apply(loss, y_hat)

    def configure_optimizers(self):
        betas = self.rwkv_args.get('betas', (0.9, 0.99))
        eps = self.rwkv_args.get('adam_eps', 1e-8)
        lr = self.rwkv_args.get('lr_init', 0.0008)
        print(f"Using lr={lr}, betas={betas}, eps={eps}")
        if self.rwkv_args.get('layerwise_lr', 0) > 0:
            # time_decay have lr 2x
            # time_first have lr 3x
            print("Using layerwise lr")
            lr_1x, lr_2x, lr_3x = set(), set(), set()
            param_dict = {}
            for name, param in self.model.named_parameters():
                param_dict[name] = param
                if 'time_decay' in name:
                    lr_2x.add(name)
                elif 'time_first' in name:
                    lr_3x.add(name)
                else:
                    lr_1x.add(name)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            print(f"lr_1x: {lr_1x}")
            print(f"lr_2x: {lr_2x}")
            print(f"lr_3x: {lr_3x}")
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 1.0
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 2.0
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 3.0
                },
            ]
        else:
            print("Using single lr")
            optim_groups = [{
                "params": [p for p in self.model.parameters()],
                "weight_decay": 0.0
            }]
        # optimizer = torch.optim.Adam(optim_groups,
        #                              lr=lr,
        #                              betas=betas,
        #                              eps=eps)

        optimizer = FusedAdam(optim_groups,
                              lr=lr,
                              betas=betas,
                              eps=eps,
                              bias_correction=True,
                              adam_w_mode=False,
                              weight_decay=0.0,
                              amsgrad=False)
        return optimizer

    def generate_init_weight(self):
        """For compatibility reasons with RWKV trainer. Simply return the state dict of the model."""
        self.model.reset_parameters()

        return self.state_dict()

    def training_step_end(self, batch_parts):
        """For compatibility reasons with RWKV trainer. Assing all losses to trainer.my_loss_all."""
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all