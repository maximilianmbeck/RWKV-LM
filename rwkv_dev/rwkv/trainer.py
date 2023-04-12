import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

import torch
import torch.utils.data as data
from ml_utils.config import Config
from ml_utils.logger import Logger, create_wandb_init_args
from ml_utils.models.base_model import BaseModel
from ml_utils.torch_utils.optimizer_scheduler import (
    create_optimizer_and_scheduler_from_config,
)
from ml_utils.trainer.basetrainer import BaseTrainer
from torch import nn
from torchmetrics import MetricCollection

LOGGER = logging.getLogger(__name__)


class UniversalRwkvTrainer(BaseTrainer):
    config_class = Config

    def __init__(self, config: Config, model: BaseModel,
                 get_dataloaders: Callable, get_metrics: Callable):
        super().__init__(
            experiment_dir=config.experiment_data.experiment_dir,
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

        self._log_train_step_every = self.config.trainer.additional_cfg.get(
            'log_train_step_every', 1)

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

        self._get_dataloaders = get_dataloaders
        self._get_metrics = get_metrics
        self._model = model

    def _train_step(self, train_batch,
                    batch_idx: int) -> Dict[str, Union[float, torch.Tensor]]:
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
        if self._train_metrics is None:
            metric_vals = {}
        else:
            with torch.no_grad():
                metric_vals = self._train_metrics(ys_pred, ys)
        # log step
        self._log_step(losses_step=loss_dict, metrics_step=metric_vals)
        return loss_dict

    def _create_datasets(self) -> None:
        # unused use dataloaders directly
        pass

    def _create_dataloaders(self) -> None:
        # for `pin_memory` see here: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        train_loader, val_loader = self._get_dataloaders()

        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        self._loss = self._model.get_loss_func()

    def _create_metrics(self) -> None:
        from ml_utils.torch_utils.metrics import Loss
        if self._get_metrics is not None:
            LOGGER.info('Creating metrics.')
            self._train_metrics, self._val_metrics = self._get_metrics()
            self._val_metrics = MetricCollection(
                metrics=[Loss(self._loss), self._val_metrics])
        else:
            LOGGER.info('No metrics specified.')
            self._train_metrics = None
            self._val_metrics = MetricCollection(metrics=[Loss(self._loss)])

    def _create_model(self) -> None:
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
