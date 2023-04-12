import copy
import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from math import isnan
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from ml_utils.config import ResumeTrainingConfig
from ml_utils.logger import PREFIX_BEST_CHECKPOINT, Logger
from ml_utils.models.base_model import BaseModel
from ml_utils.output_loader.directories import JobDirectory
from ml_utils.run_utils.runner import Runner
from ml_utils.time_utils import Stopwatch
from ml_utils.utils import get_device, set_seed, setup_exception_logging
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

RUN_PROGRESS_MEASURE_STEP = 'train_step'
RUN_PROGRESS_MEASURE_EPOCH = 'epoch'


class BaseTrainer(Runner, ABC):

    def __init__(
            self,
            experiment_dir: str,
            n_steps: int = -1,
            n_epochs: int = -1,
            val_every: int = 1,
            save_every: int = 0,
            save_every_idxes: List[int] = [],
            early_stopping_patience: int = -1,
            seed: int = 0,
            gpu_id: int = 0,
            num_workers: int = 0,
            resume_training: ResumeTrainingConfig = None):
        """Base class for all pytorch supervised-trainers. Takes care of early stopping and checkpointing. 

        Args:
            experiment_dir (str): The directory where all training data is stored.
            n_steps (int, optional): Maximum number of steps to train for. Defaults to -1.
            n_epochs (int, optional): Maximum number of epochs to train for. Either `n_steps` or `n_epochs` must be specified. Defaults to -1.
            val_every (int, optional): Validate every `val_every` epochs. Defaults to 1.
            save_every (int, optional): Save the checkpoint every `save_every` epochs. Defaults to 0.
            save_every_idxes (List[int], optional): Save checkpoint at every index in this list. Defaults to [].
            early_stopping_patience (int, optional): Early stop training if validation metric has not improved for `early_stopping_patience` epochs. Defaults to -1.
            seed (int, optional): Seed of the experiment. Defaults to 0.
            gpu_id (int, optional): The GPU id where the experiment is run. Defaults to 0.
            num_workers (int, optional): Number of workers, for e.g. for dataloader. Defaults to 0.
            resume_training (ResumeTrainingConfig, optional): Resume training config. Contains location of checkpoint to resume training. Defaults to None.
        """
        super().__init__(runner_dir=experiment_dir)
        # parameters
        self._experiment_dir = experiment_dir
        self._seed = int(seed)
        self._gpu_id = int(gpu_id)
        self._num_workers = int(num_workers)
        self.device = get_device(self._gpu_id)
        self._resume_training = resume_training

        self._n_epochs = int(n_epochs)
        self._n_steps = int(n_steps)
        assert (self._n_steps >= 0 and not self._n_epochs >= 0) or (
            not self._n_steps >= 0 and self._n_epochs >= 0
        ), 'Must either specify maximum number of epochs or maximum number of steps, but not both.'
        self._val_every = int(val_every)
        self._save_every = int(save_every)
        self._save_every_idxes = save_every_idxes
        self._early_stopping_patience = int(early_stopping_patience)

        # member variables
        self._datasets = None
        self._loaders = None
        self._model: BaseModel = None
        self._optimizer: optim.Optimizer = None
        self._lr_scheduler: lr_scheduler._LRScheduler = None
        self._loss: nn.Module = None
        self._train_metrics: MetricCollection = None
        self._val_metrics: MetricCollection = None
        self._logger: Logger = Logger(job_dir=self._experiment_dir)

        # progress variables
        self._best_model: BaseModel = None
        self._progress_measure = RUN_PROGRESS_MEASURE_EPOCH if self._n_epochs > 0 else RUN_PROGRESS_MEASURE_STEP
        self._train_step_idx = 0
        self._epoch_idx = 0
        self._best_idx = 0
        self._best_val_score: float = None

        set_seed(self._seed)
        setup_exception_logging()
        LOGGER.info(f'Logging experiment data to directory: {self._experiment_dir}.')

    def _initialize(self):
        self._logger.setup_logger()
        self._create_datasets()
        self._create_dataloaders()
        self._create_model()
        self._create_loss()
        self._create_metrics()

        self._model.to(device=self.device)
        self._loss.to(device=self.device)
        self._train_metrics.to(device=self.device)
        self._val_metrics.to(device=self.device)

        self._create_optimizer_and_scheduler(self._model)
        self._train_step_idx = 0
        self._epoch_idx = 0

    def run(self) -> None:
        self.train()

    @abstractmethod
    def _create_datasets(self) -> None:
        pass

    @abstractmethod
    def _create_dataloaders(self) -> None:
        pass

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def _create_loss(self) -> None:
        """Create loss for optimization. 
        """
        pass

    @abstractmethod
    def _create_metrics(self) -> None:
        """Create a list of metrics for training and validation.
        The first entry in val_metric is used for early stopping.
        """
        pass

    def _reset_metrics(self, which: str = 'all') -> None:
        if self._train_metrics is not None and (which == 'all' or which == 'train'):
            self._train_metrics.reset()
        if self._val_metrics is not None and (which == 'all' or which == 'val'):
            self._val_metrics.reset()

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        losses_epoch: List[Dict[str, torch.Tensor]] = []

        # training loop (iterations per epoch)
        pbar = tqdm(self._loaders['train'], desc=f'Train epoch {epoch}', file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            self._model.train()
            with Stopwatch() as sw:
                loss_dict = self._train_step(train_batch=batch, batch_idx=batch_idx)
            if self._train_step_idx % 100 == 0: # do not log T_train_step every step
                self._logger.log_keys_vals(prefix='timer',
                                           epoch=self._epoch_idx,
                                           train_step=self._train_step_idx,
                                           keys_val={'T_train_step': sw.elapsed_seconds})
            self._train_step_idx += 1
            losses_epoch.append(loss_dict)

            if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
                if self._lr_scheduler is not None: self._lr_scheduler.step()
                if self._validation_and_early_stopping(idx=self._train_step_idx, specifier=self._progress_measure):
                    break
                elif self._train_step_idx >= (self._n_steps): # TODO check this maybe +/- 1?
                    break

        if self._progress_measure == RUN_PROGRESS_MEASURE_EPOCH:
            if self._lr_scheduler is not None: self._lr_scheduler.step()

        # log epoch
        metrics_epoch = self._train_metrics.compute()
        self._logger.log_keys_vals(prefix='train',
                                   epoch=self._epoch_idx,
                                   train_step=self._train_step_idx,
                                   keys_multiple_vals=losses_epoch,
                                   keys_val=metrics_epoch,
                                   log_to_console=True)

        self._reset_metrics('train')

    @abstractmethod
    def _train_step(self, train_batch, batch_idx: int) -> Dict[str, Union[float, torch.Tensor]]:
        return {}

    def _val_epoch(self, progress_idx: int, trained_model: nn.Module) -> float:
        """Implementation of one validation epoch.

        Args:
            progress_idx (int): Epoch or step index.
            trained_model (nn.Module): Model to validate

        Returns:
            float: Metric value used for early stopping
        """
        losses_epoch: List[Dict[str, torch.Tensor]] = []

        val_loader = self._loaders.get('val', None)
        if val_loader is None or self._val_metrics is None:
            return None
        
        pbar = tqdm(val_loader, desc=f'Val after {self._progress_measure} {progress_idx}', file=sys.stdout)
        with torch.no_grad():
            for xs, ys in pbar:
                xs, ys = xs.to(self.device), ys.to(self.device)

                y_pred = trained_model(xs)

                # loss = self._loss(y_pred, ys)
                # loss_dict = {'loss': loss}
                m_val = self._val_metrics(y_pred, ys) # val metrics contain the loss
                # losses_epoch.append(loss_dict)

        # compute mean metrics over dataset
        metrics_epoch = self._val_metrics.compute()
        self._logger.log_keys_vals(prefix='val',
                                   epoch=self._epoch_idx,
                                   train_step=self._train_step_idx,
                                   keys_val=metrics_epoch,
                                   log_to_console=True)
        val_score = metrics_epoch[next(iter(self._val_metrics.items()))[0]].item()
        self._reset_metrics('val')
        self._hook_on_val_epoch_end(progress_idx=progress_idx, trained_model=trained_model)
        return val_score

    def _val_lower_is_better(self) -> bool:
        """Return the value for the first validation metric in the metric collection."""
        # index 1 is the Metrics class
        return not next(iter(self._val_metrics.items()))[1].higher_is_better

    def _hook_before_initialization(self, *args, **kwargs) -> None:
        pass

    def _hook_on_training_start(self, *args, **kwargs) -> None:
        pass

    def _hook_on_training_end(self, *args, **kwargs) -> None:
        pass

    def _hook_on_val_epoch_end(self, *args, **kwargs) -> None:
        pass

    def _create_checkpoint(self) -> None:
        checkpoint = {}
        # model
        if isinstance(self._model, BaseModel):
            checkpoint.update(self._model.get_checkpoint_data())
        else:
            checkpoint['model_state_dict'] = self._model.state_dict()
        # optimizer
        checkpoint['optimizer_state_dict'] = self._optimizer.state_dict()
        # scheduler
        if self._lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self._lr_scheduler.state_dict()
        # trainer
        trainer_data = {'train_step_idx': self._train_step_idx, 'epoch_idx': self._epoch_idx}
        checkpoint['trainer_data'] = trainer_data
        # job_dir
        checkpoint['__job_directory'] = str(self._experiment_dir)

        idx = self._train_step_idx if self._progress_measure == RUN_PROGRESS_MEASURE_STEP else self._epoch_idx
        self._logger.save_checkpoint(checkpoint, idx=idx, specifier=self._progress_measure)

    def _resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # model
        self._model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        # trainer
        trainer_data = checkpoint['trainer_data']
        self._train_step_idx = trainer_data['train_step_idx']
        self._epoch_idx = trainer_data['epoch_idx']

    def train(self) -> Dict[str, Any]:
        """Train for n_epochs using early-stopping, epoch counter starts with 1.

        Returns:
            Dict[str, Any]: the final results
        """
        self._hook_before_initialization()
        self._initialize()
        if self._resume_training:
            LOGGER.info(f'Resume from checkpoint: {str(self._resume_training)}')
            checkpoint = JobDirectory.load_resume_checkpoint(**asdict(self._resume_training), device=self.device)
            self._resume_from_checkpoint(checkpoint=checkpoint)

        self._hook_on_training_start()
        if self._progress_measure == RUN_PROGRESS_MEASURE_STEP:
            # no number of epochs given
            # calculate from number of steps
            num_steps_per_epoch = len(self._loaders['train'])
            self._n_epochs = math.ceil(self._n_steps / num_steps_per_epoch)

        LOGGER.info(f'Starting training with progress measure `{self._progress_measure}`.')
        self._best_idx = self._train_step_idx if self._progress_measure == RUN_PROGRESS_MEASURE_STEP else self._epoch_idx
        self._best_val_score = float('inf') if self._val_lower_is_better() else -float('inf')

        # validate untrained model as baseline
        self._model.eval()
        with Stopwatch() as sw:  # TODO make this a decorator
            self._best_val_score = self._val_epoch(self._epoch_idx, self._model)
        self._logger.log_keys_vals(prefix='timer',
                                   epoch=self._epoch_idx,
                                   train_step=self._train_step_idx,
                                   keys_val={'T_val_epoch': sw.elapsed_seconds})

        # save initialized/untrained model
        self._create_checkpoint()
        self._best_model = copy.deepcopy(self._model).to(torch.device('cpu'))
        self._train_step_idx += 1
        self._epoch_idx += 1
        while self._epoch_idx <= (self._n_epochs):
            self._model.train()
            with Stopwatch() as sw:
                self._train_epoch(self._epoch_idx)
            self._logger.log_keys_vals(prefix='timer',
                                       epoch=self._epoch_idx,
                                       train_step=self._train_step_idx,
                                       keys_val={'T_train_epoch': sw.elapsed_seconds})

            if self._progress_measure == RUN_PROGRESS_MEASURE_EPOCH and self._validation_and_early_stopping(
                    idx=self._epoch_idx, specifier=self._progress_measure):
                break
            self._epoch_idx += 1
        
        # save trained model
        self._create_checkpoint()

        final_results = {
            f'{PREFIX_BEST_CHECKPOINT}{self._progress_measure}': self._best_idx,
            f'{PREFIX_BEST_CHECKPOINT}val_score': self._best_val_score
        }
        LOGGER.info(f"Final results: \n{pd.Series(final_results)}")

        if self._best_idx >= 0:
            self._logger.save_best_checkpoint_idx(specifier=self._progress_measure, best_idx=self._best_idx)

            if self._best_idx > 0:
                self._logger.save_checkpoint(self._best_model.get_checkpoint_data(),
                                             idx=self._best_idx,
                                             specifier=self._progress_measure,
                                             name='model')

        self._logger.finish(final_results=final_results)
        self._hook_on_training_end(final_results=final_results)

        return final_results

    def _validation_and_early_stopping(self, idx: int, specifier: str) -> bool:
        """Runs validation and early stopping.

        Args:
            idx (int): Current epoch or step index.
            specifier (str): `epoch` or `step`

        Returns:
            bool: True, if training should be early stopped.
        """
        model_saved = False
        if (self._save_every > 0 and idx % self._save_every == 0) or idx in self._save_every_idxes:
            self._create_checkpoint()
            model_saved = True

        if self._val_every > 0 and idx % self._val_every == 0:
            lower_is_better = self._val_lower_is_better()

            self._model.eval()
            with Stopwatch() as sw:
                val_score = self._val_epoch(progress_idx=idx, trained_model=self._model)
                if val_score is None:
                    return False # keep training
            self._logger.log_keys_vals(prefix='timer',
                                       epoch=self._epoch_idx,
                                       train_step=self._train_step_idx,
                                       keys_val={'T_val_epoch': sw.elapsed_seconds})
            assert isinstance(val_score, float)
            if isnan(val_score):
                raise RuntimeError(f'Validation score is NaN in {specifier} {idx}.')

            if (lower_is_better and val_score < self._best_val_score) or \
                    (not lower_is_better and val_score > self._best_val_score):
                LOGGER.info(
                    f"New best val score: {val_score} {'<' if lower_is_better else '>'} {self._best_val_score} (old best val score)"
                )
                self._best_idx = idx
                self._best_val_score = val_score
                self._best_model = copy.deepcopy(self._model).to(torch.device('cpu'))

            if self._early_stopping_patience > 0:
                if ((lower_is_better and val_score >= self._best_val_score)
                        or (not lower_is_better and val_score <= self._best_val_score)) \
                        and idx > self._best_idx + self._early_stopping_patience:
                    LOGGER.info('Early stopping patience exhausted. '
                                f'Best val score {self._best_val_score} in {specifier} {self._best_idx}.')
                    return True
        return False
