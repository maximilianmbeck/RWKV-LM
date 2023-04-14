import logging
import os

import dacite
import torch
from dacite import from_dict
from ml_utils.run_utils.runner import setup_directory
from omegaconf import DictConfig, OmegaConf
from rwkv.data.dataloaders import get_dataloader_creator
from rwkv.models import get_model
from rwkv.trainer import UniversalRwkvTrainer
from torch.utils import data

LOGGER = logging.getLogger(__name__)

def run_job(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    setup_directory('job', cfg)

    cfg = from_dict(data_class=UniversalRwkvTrainer.config_class,
                    data=OmegaConf.to_container(cfg.config))

    # create model
    model = get_model(cfg.model)
    # LOGGER.info('Use torch.compile')
    # model = torch.compile(model, mode='reduce-overhead')

    # create dataloader
    dataloader_creator = get_dataloader_creator(cfg.data.name)
    get_dataloaders = lambda: dataloader_creator(cfg.data.kwargs)

    # create metrics
    # TODO make this configurable
    def get_metrics():
        import torchmetrics
        perplexity = torchmetrics.Perplexity()
        train_metrics = torchmetrics.MetricCollection([perplexity])
        return train_metrics, None

    # TODO add a mapping from dataloader to model input

    trainer = UniversalRwkvTrainer(config=cfg,
                                   model=model,
                                   get_dataloaders=get_dataloaders,
                                   get_metrics=get_metrics)
    trainer.run()
