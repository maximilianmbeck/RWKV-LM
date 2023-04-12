import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

import torch
import torch.utils.data as data
from ml_utilities.config import Config
from ml_utilities.data.datasetgeneratorinterface import DatasetGeneratorInterface
from ml_utilities.logger import Logger, create_wandb_init_args
from ml_utilities.torch_utils.losses import get_loss
from ml_utilities.torch_utils.metrics import create_metrics
from ml_utilities.torch_utils.optimizer_scheduler import (
    create_optimizer_and_scheduler_from_config,
)
from ml_utilities.trainer.basetrainer import BaseTrainer
from torch import nn
from torchmetrics import MetricCollection

LOGGER = logging.getLogger(__name__)


class UniversalBaseTrainer(BaseTrainer):
    config_class = Config
    def __init__(self,
                 config: Config,
                 model_init_func: Callable,
                 datasetgenerator_init_func: Callable,
                 metrics_init_func: Callable = create_metrics):
        super().__init__(experiment_dir=config.experiment_data.experiment_dir,
                         seed=config.experiment_data.seed,
                         gpu_id=config.experiment_data.gpu_id,
                         n_steps=config.trainer.n_steps,
                         n_epochs=config.trainer.n_epochs,
                         val_every=config.trainer.val_every,
                         save_every=config.trainer.save_every,
                         save_every_idxes=config.trainer.save_every_idxes,
                         early_stopping_patience=config.trainer.early_stopping_patience,
                         num_workers=config.trainer.num_workers,
                         resume_training=config.trainer.resume_training)
        self.config = config

        self._log_train_step_every = self.config.trainer.additional_cfg.get('log_train_step_every', 10)

        exp_data = self.config.experiment_data
        wandb_args = create_wandb_init_args(self.config)

        self._logger = Logger(job_name=exp_data.job_name,
                              job_dir=exp_data.experiment_dir,
                              project_name=exp_data.project_name,
                              entity_name=exp_data.entity,
                              config=asdict(self.config),
                              wandb_args=wandb_args)
        self._logger.setup_logger()

        self._datasetgenerator = None

        self._model_init_func = model_init_func
        self._datasetgenerator_init_func = datasetgenerator_init_func
        self._metrics_init_func = metrics_init_func

    def _train_step(self, train_batch, batch_idx: int) -> Dict[str, Union[float, torch.Tensor]]:
        xs, ys = train_batch
        xs, ys = xs.to(self.device), ys.to(self.device)
        # forward pass
        ys_pred = self._model(xs)
        loss = self._loss(ys_pred, ys)
        loss_dict = {'loss': loss}

        # backward pass
        self._optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._model.parameters(), 2)

        self._optimizer.step()
        # log learning rate, assume only one parameter group
        loss_dict['lr'] = self._optimizer.param_groups[0]['lr']
        # metrics & logging
        with torch.no_grad():
            metric_vals = self._train_metrics(ys_pred, ys)
        # log step
        self._log_step(losses_step=loss_dict, metrics_step=metric_vals)
        return loss_dict

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        self._datasetgenerator = self._datasetgenerator_init_func(self.config.data)
        self._datasetgenerator.generate_dataset()
        train_set, val_set = self._datasetgenerator.train_split, self._datasetgenerator.val_split
        self._train_metrics, self._val_metrics = self._datasetgenerator.train_metrics, self._datasetgenerator.val_metrics
        LOGGER.info(f'Size of training/validation set: ({len(train_set)}/{len(val_set)})')
        self._datasets = dict(train=train_set, val=val_set)

    def _create_dataloaders(self) -> None:
        # for `pin_memory` see here: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        train_loader = data.DataLoader(dataset=self._datasets['train'],
                                       batch_size=self.config.trainer.batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       num_workers=self.config.trainer.num_workers,
                                       persistent_workers=True,
                                       pin_memory=True)
        val_loader = data.DataLoader(dataset=self._datasets['val'],
                                     batch_size=self.config.trainer.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=self.config.trainer.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True)
        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        loss_cls = get_loss(self.config.trainer.loss)
        loss_module = loss_cls(reduction='mean')
        self._loss = loss_module

    def _create_metrics(self) -> None:
        if self._train_metrics is None and self._val_metrics is None and self._metrics_init_func is not None:
            LOGGER.info('Creating metrics from config.')
            self._train_metrics, self._val_metrics = self._metrics_init_func(self.config.trainer.metrics)
        else:
            LOGGER.info('Using metrics from Datasetgenerator.')
        from ml_utilities.torch_utils.metrics import Loss
        LOGGER.info('Adding validation loss as first metric for early stopping.')
        self._val_metrics = MetricCollection(metrics=[Loss(self._loss), self._val_metrics])

    def _create_model(self) -> None:
        self._model = self._model_init_func(self.config.model)
        self._logger.watch_model(self._model)

    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        LOGGER.info('Creating optimizer and scheduler.')
        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler_from_config(
            model.parameters(),
            optimizer_cfg=self.config.trainer.optimizer,
            lr_scheduler_cfg=self.config.trainer.lr_scheduler)

    def _log_step(self,
                  losses_step: Dict[str, torch.Tensor],
                  metrics_step: Dict[str, torch.Tensor],
                  additional_logs_step: Dict[str, Any] = {}) -> None:
        if self._train_step_idx % self._log_train_step_every == 0:
            log_dict = {**losses_step, **metrics_step, **additional_logs_step}
            self._logger.log_keys_vals(prefix='train_step',
                                       train_step=self._train_step_idx,
                                       epoch=self._epoch_idx,
                                       keys_val=log_dict,
                                       log_to_console=False)
