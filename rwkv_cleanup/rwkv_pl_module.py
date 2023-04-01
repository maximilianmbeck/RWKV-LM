from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch import nn

from rwkv_cleanup.rwkv_model import RWKV, RWKVConfig


@dataclass 
class RWKVModuleConfig:
    """Configuration for RWKVModule."""
    model : RWKVConfig

    

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
        # TODO use Loss wrapper

        return loss
    
    def configure_optimizers(self):
        betas = self.rwkv_args.get('betas', (0.9, 0.99))
        eps = self.rwkv_args.get('adam_eps', 1e-8)
        lr = self.rwkv_args.get('lr_init', 0.0008)
        print(f"Using lr={lr}, betas={betas}, eps={eps}")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps)
        return optimizer

    def generate_init_weight(self):
        """For compatibility reasons with RWKV trainer. Simply return the state dict of the model."""
        return self.state_dict()

    def training_step_end(self, batch_parts):
        """For compatibility reasons with RWKV trainer. Assing all losses to trainer.my_loss_all."""
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all