import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import torch
import wandb
from torch import nn

from rwkv_dev.ml_utils.config import Config
from rwkv_dev.ml_utils.output_loader.directories import JobDirectory
from rwkv_dev.ml_utils.utils import convert_dict_to_python_types, save_dict_as_yml

LOGGER = logging.getLogger(__name__)

PREFIX_BEST_CHECKPOINT = 'best_'
FN_BEST_CHECKPOINT = PREFIX_BEST_CHECKPOINT + '{specifier}.txt'
LOG_POLICY_LIVE = 'live'  # should write to files on the fly
LOG_POLICY_END = 'end'  # should write the files at the end

LOG_STEP_KEY = 'log_step'

FN_FINAL_RESULTS = 'final_results'
FN_DATA_LOG_PREFIX = 'stats_'
FN_DATA_LOG = FN_DATA_LOG_PREFIX + '{datasource}.csv'

FORMAT_LOGGING = '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'


def increment_log_step_on_call(method):
    """Decorator to increment the internal log step of a logging / save function of `Logger`:
    """

    def inner(self, *args, **kwargs):
        method(self, *args, **kwargs)
        self.log_step += 1

    return inner

def create_wandb_init_args(config: Config) -> Dict[str, Any]:
    if config.wandb is None:
        wandb_init_args = {'tags': [config.experiment_data.experiment_tag],'notes': config.experiment_data.experiment_notes, 
                       'group': config.experiment_data.experiment_tag, 'job_type': config.experiment_data.experiment_type}
    else:
        wandb_init_args = config.wandb['init']
    return {'init': wandb_init_args}

class Logger:
    """A logger that logs to wandb and to disc. 
    Logging to wandb can be disabled. There are two different strategies when to save the logs to disc: `live` and `end`.
    """

    def __init__(self,
                 job_dir: Union[str, Path, JobDirectory],
                 job_name: str = '',
                 project_name: str = '',
                 entity_name: str = '',
                 config: Dict[str, Any] = {},
                 wandb_args: Dict[str, Any] = {},
                 log_policy: str = LOG_POLICY_END):

        self.logger_directory = JobDirectory(job_dir)

        self._job_name = job_name
        self._entity_name = entity_name
        self._wandb_args = wandb_args
        self._project_name = project_name
        self._config = config
        self._log_policy = log_policy
        self._wandb_run = None

        if self._log_policy == LOG_POLICY_LIVE:
            # TODO implement log policy 'live'
            raise NotImplementedError(f'Log policy `{self._log_policy}` not supported yet!')

        self.log_to_wandb = bool(wandb_args)

        # the logs saved as lists
        self.logs: Dict[str, List[Dict[str, Any]]] = {}
        # counter for counting how often log_keys_vals was called
        self.log_step = 0

    def setup_logger(self):
        """Sets up wandb if necessary and creates directories."""
        self.logger_directory.create_directories()

        # start wandb if necessary
        if self.log_to_wandb:
            LOGGER.info('Starting wandb.')
            self._wandb_run = wandb.init(entity=self._entity_name,
                                        project=self._project_name,
                                        name=self._job_name,
                                        # dir=str(self.logger_directory.dir), # use default wandb dir to enable later wandb sync
                                        config=self._config,
                                        **self._wandb_args['init'],
                                        settings=wandb.Settings(start_method='fork'))
        else:
            LOGGER.info('Not logging to wandb. Logging to disc only.')

    @increment_log_step_on_call
    def log_keys_vals(self,
                      prefix: str = 'default',
                      epoch: int = -1,
                      train_step: int = -1,
                      keys_multiple_vals: Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]],
                                                List[Dict[str, torch.Tensor]]] = {},
                      keys_val: Dict[str, Any] = {},
                      log_to_console: bool = False,
                      specifier: str = 'epoch') -> None:
        if keys_multiple_vals or keys_val:
            # log mean value if multiple values are given
            keys_multiple_vals_df = pd.DataFrame(keys_multiple_vals)
            keys_multiple_vals_mean = keys_multiple_vals_df.mean(axis=0).to_dict()

            log_dict = {**keys_multiple_vals_mean, **keys_val}
            if epoch > -1:
                log_dict.update({'epoch': epoch})
            if train_step > -1:
                log_dict.update({'train_step': train_step})

            if self.log_to_wandb:
                wandb.log({f'{prefix}/': log_dict})

            # log to console
            log_dict_numeric_vals = convert_dict_to_python_types(log_dict)
            if log_to_console:
                LOGGER.info(f'{prefix} {specifier} \n{pd.Series(log_dict_numeric_vals, dtype=float)}')

            # add the internal log_step to every log
            log_dict_numeric_vals.update({LOG_STEP_KEY: self.log_step})

            # init empty list if not available yet
            if not prefix in self.logs:
                self.logs[prefix] = []
            self.logs[prefix].append(log_dict_numeric_vals)

    def watch_model(self, model: nn.Module):
        if self.log_to_wandb:
            watch_args = self._wandb_args.get('watch', {'log': None})
            wandb.watch(model, **watch_args)

    @increment_log_step_on_call
    def save_fig(self, *args, **kwargs) -> None:
        raise NotImplementedError()  # TODO

    @increment_log_step_on_call
    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        idx: int,
                        specifier: str = '',
                        name: str = 'checkpoint') -> None:
        self.logger_directory.save_checkpoint(checkpoint=checkpoint, idx=idx, specifier=specifier, name=name)

    def save_best_checkpoint_idx(self, specifier: str, best_idx: int) -> None:
        with open(self.logger_directory.checkpoints_folder / FN_BEST_CHECKPOINT.format(specifier=specifier), 'w') as fp:
            fp.write(str(best_idx))

    def finish(
        self,
        final_results: Dict[str, Any] = {},
    ):
        LOGGER.info(f'Finishing logging.')
        if self.log_to_wandb:
            wandb.run.summary.update(final_results)

        if final_results:
            save_dict_as_yml(self.logger_directory.stats_folder, filename=FN_FINAL_RESULTS, dictionary=final_results)

        # save data to disc
        for datasource, log_data in self.logs.items():
            filename = FN_DATA_LOG.format(datasource=datasource)
            LOGGER.info(f'Creating dump: {filename}')
            log_data_df = pd.DataFrame(log_data)
            log_data_df.to_csv(self.logger_directory.stats_folder / filename)

        if self._wandb_run is not None:
            LOGGER.info(f'Finishing wandb run.')
            self._wandb_run.finish()