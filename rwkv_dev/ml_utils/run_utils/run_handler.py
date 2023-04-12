import copy
import itertools
import logging
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wandb
from joblib import Parallel, delayed
from ml_utils.run_utils.runner import Runner
from ml_utils.run_utils.sweep import EXPERIMENT_CONFIG_KEY, OVERRIDE_PARAMS_KEY, Sweeper
from ml_utils.utils import (
    archive_code,
    convert_to_simple_str,
    get_config,
    hyp_param_cfg_to_str,
    make_str_filename,
    remove_toplevelkeys_from_dictconfig,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

EXP_NAME_DIVIDER = '--'
SWEEP_KEY = 'sweep'


class RunHandler(Runner):
    """A class to handle experiments with multiple runs, i.e. sweeps or multiple seeds.

    Args:
        config (Union[str, Path, dict, DictConfig]): The config.
        script_path (str): Path to the run script.
        log_wandb (bool, optional): Override to disable logging to wandb. 
                                    If True, logs to wandb if specified in config. Defaults to True.
        num_workers (int, optional): Number of workers used for config generation. Defaults to 4.
    """

    def __init__(self,
                 sweep_dir: Union[str, Path],
                 config: Union[str, Path, dict, DictConfig],
                 script_path: str,
                 log_wandb: bool = True,
                 num_workers: int = 4):
        super().__init__(runner_dir=sweep_dir)
        self.config = get_config(config)
        self.config_raw = copy.deepcopy(self.config)
        self.script_path = script_path
        self.log_wandb = log_wandb
        self._wandb_run = None
        self.num_workers = num_workers

    def run(self):
        """Run experiments."""
        hostname = socket.gethostname()
        self.config.run_config.hostname = hostname  # !< config is modified here

        run_config = self.config.run_config
        # init wandb
        if self.log_wandb and run_config.get('wandb', None):
            self.log_wandb = True
            LOGGER.info(f'Logging handler status to wandb.')
            exp_data = self.config.config.experiment_data
            run_handler_runname = f"{hostname}-{convert_to_simple_str(run_config.gpu_ids)}-" + self.runner_dir.name
            log_cfg = remove_toplevelkeys_from_dictconfig(copy.deepcopy(self.config), ['hydra'])
            self._wandb_run = wandb.init(entity=exp_data.get('entity', None),
                       project=exp_data.project_name,
                       name=run_handler_runname,
                       dir=str(Path.cwd()),
                       config=OmegaConf.to_container(log_cfg, resolve=True, throw_on_missing=True),
                       **run_config.wandb.init,
                       settings=wandb.Settings(start_method='fork'))
        else:
            self.log_wandb = False
        # snapshot code as .zip file
        # archive_code(repo_dir=self.runner_dir, save_dir=self.runner_dir) # TODO implement code snapshotting
        # run
        if run_config.exec_type == 'sequential':
            self._run_sequential()
        elif run_config.exec_type == 'parallel':
            self._run_parallel()
        if not self._wandb_run is None:
            self._wandb_run.finish()

    def _run_sequential(self):
        """Run experiments on the (first) gpu_id sequentially."""
        # get gpu_id
        gpu_ids = self.config.run_config.gpu_ids
        if isinstance(gpu_ids, list):
            gpu_id = gpu_ids[0]  # use the first gpu_id
        else:
            gpu_id = int(gpu_ids)
        self.__run(gpu_ids=gpu_id, runs_per_gpu=1)

    def _run_parallel(self):
        """Run experiments in parallel."""
        self.__run()

    def __run(self, gpu_ids: Optional[Union[int, List[int]]] = None, runs_per_gpu: Optional[int] = None):
        """Run experiments in separate processes."""
        config = copy.deepcopy(self.config)
        sweep_configs = self._extract_sweep(config)

        # get seeds
        seeds = self.config.get('seeds', None)
        if isinstance(seeds, (list, ListConfig)):
            assert len(seeds) > 0, f"No seeds are given to start runs."

            LOGGER.info(f'Starting every run with {len(seeds)} seeds.')

            def prepare_cfg_with_seed(cfg: DictConfig, seed: int) -> DictConfig:
                current_config = copy.deepcopy(cfg)
                current_config[EXPERIMENT_CONFIG_KEY].experiment_data.seed = seed
                return current_config

            experiment_configs = Parallel(n_jobs=self.num_workers)(
                delayed(prepare_cfg_with_seed)(cfg, seed) for seed, cfg in tqdm(
                    itertools.product(seeds, sweep_configs), desc='Overriding config seeds', file=sys.stdout))
        else:
            assert isinstance(seeds, int)
            LOGGER.info(f'Starting every run with a single seed={seeds}.')
            experiment_configs = sweep_configs

        schedule_runs(self.runner_dir,
                      experiment_configs,
                      self.script_path,
                      gpu_ids=gpu_ids,
                      runs_per_gpu=runs_per_gpu,
                      log_wandb=self.log_wandb, 
                      shuffle_configs=self.config.run_config.get('shuffle_configs', False))

    def _extract_sweep(self, config: DictConfig) -> List[DictConfig]:
        if config.get(SWEEP_KEY, None) is None:
            # no sweep specified
            LOGGER.warn(
                "No hyperparameter sweep specified, but experiment started through RunHandler. Using default configuration."
            )
            return [config]
        else:
            # get sweeper
            sweeper = Sweeper.create(config)
            # get configs
            if sweeper is None:
                LOGGER.info('Using default configuration.')
                return [config]
            else:
                return sweeper.generate_configs()


# FUNCTIONS:


def update_and_save_config(save_dir: Union[str, Path],
                           config: DictConfig,
                           gpu_id: int,
                           seed: int = None,
                           hostname: str = None) -> str:
    """Updates the config-file with seed and gpu_id, saves it in the current working directory and returns its name.

    Args:
        save_dir (Path) : The directory where the config is saved.
        config (DictConfig): The config to be updated and saved.
        seed (int): 
        gpu_id (int): 

    Returns:
        str: Name of the saved config file.
    """
    # save seed in config
    if seed is not None:
        config[EXPERIMENT_CONFIG_KEY].experiment_data.seed = seed  # !< config is modified here
    else:
        seed = config[EXPERIMENT_CONFIG_KEY].experiment_data.seed
    exp_name = config[EXPERIMENT_CONFIG_KEY].experiment_data.experiment_name
    wandb_group = copy.deepcopy(exp_name)
    # add hyperparameter values
    exp_name += EXP_NAME_DIVIDER + hyp_param_cfg_to_str(config.get(OVERRIDE_PARAMS_KEY, {}))
    exp_name = make_str_filename(exp_name)
    # wandb_job_type = copy.deepcopy(exp_name) # TODO: the max job_type lenght is 64 characters, exp_name can be longer
    exp_name += f"-seed-{seed}"
    config[EXPERIMENT_CONFIG_KEY].experiment_data.experiment_name = exp_name  # !< config is modified here
    # set device
    config[EXPERIMENT_CONFIG_KEY].experiment_data.gpu_id = gpu_id  # !< config is modified here
    if hostname is not None:
        config[EXPERIMENT_CONFIG_KEY].experiment_data.hostname = hostname  # !< config is modified here

    # set wandb group and job_type
    if config[EXPERIMENT_CONFIG_KEY].get('wandb', None) is not None:
        config[EXPERIMENT_CONFIG_KEY].wandb.init.group = wandb_group
        config[EXPERIMENT_CONFIG_KEY].wandb.init.job_type = config[EXPERIMENT_CONFIG_KEY].experiment_data.experiment_type 
        config[EXPERIMENT_CONFIG_KEY].wandb.init.notes = config[EXPERIMENT_CONFIG_KEY].experiment_data.get('experiment_notes', None)

    save_name = exp_name + '.yaml'
    OmegaConf.save(config, Path(save_dir) / save_name)
    return save_name


def schedule_runs(configs_save_dir: Path,
                  experiment_configs: List[DictConfig],
                  script_path: str,
                  gpu_ids: Optional[Union[int, List[int]]] = None,
                  runs_per_gpu: Optional[int] = None,
                  log_wandb: bool = True,
                  sleep_time: int = 1, 
                  shuffle_configs: bool = False):
    """Distribute multiple runs on different gpus of the same machine.

    Example:
    Given: 5 experiment configs, gpu id: 0 1, runs per gpu: 3
    Result: Starts runs on gpu 0 and 1 as long as number runs_per_gpu is not reached 
    and there are still runs in the experiment_configs list.

    Args:
        configs_save_dir (Path): Directory, where all configs are saved before started.
        experiment_configs (List[DictConfig]): List of experiment configs to schedule
        script_path (str): Full path to a hydra python script
        gpu_ids (Optional[Union[int, List[int]]], optional): The gpus to schedule runs on. Defaults to None (in this case value taken from config)
        runs_per_gpu (Optional[int], optional): The max runs per gpu. Defaults to None (in this case value taken from config).
        log_wandb (bool, optional): If true, logs status to wandb.
        sleep_time (int, optional): Time to wait until starting next run.
        shuffle_configs (bool, optional): If true, shuffles the list of configs before scheduling. Defaults to False.
    """
    assert len(experiment_configs) > 0, f"No experiments to schedule given."

    # * run config is same for every run:
    run_config = experiment_configs[0].run_config
    if gpu_ids is None:
        gpu_ids = run_config.gpu_ids
    if runs_per_gpu is None:
        runs_per_gpu = run_config.runs_per_gpu

    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    LOGGER.info(
        f'Distributing {len(experiment_configs)} runs across GPUs {gpu_ids}. Running {runs_per_gpu} in parallel on each GPU (total parallel: {len(gpu_ids)*runs_per_gpu})'
    )

    # for approximately equal memory usage during hyperparam tuning, randomly shuffle list of processes
    if shuffle_configs:
        random.shuffle(experiment_configs)

    # array to keep track on how many runs are currently running per GPU
    n_parallel_runs = len(gpu_ids) * runs_per_gpu
    gpu_counter = np.zeros((len(gpu_ids)), dtype=int)

    running_processes: Dict[Tuple[int, str, int], subprocess.Popen] = {}
    running_run_idxs: Dict[int, bool] = {}
    counter = 0
    if log_wandb:
        log_dict = _create_run_status_dict(experiment_configs, counter, running_run_idxs)
        wandb.log(log_dict[0])
        # wandb.log(log_dict[1])

    while True:

        # * start new runs
        for _ in range(n_parallel_runs - len(running_processes)):

            if counter >= len(experiment_configs):
                break
            # * determine which GPU to use
            node_id = int(np.argmin(gpu_counter))
            gpu_counter[node_id] += 1
            gpu_id = gpu_ids[node_id]

            # * prepare next experiment in list
            current_config = copy.deepcopy(experiment_configs[counter])
            config_name = update_and_save_config(save_dir=configs_save_dir,
                                                 config=current_config,
                                                 gpu_id=gpu_id,
                                                 hostname=socket.gethostname())

            # start run via subprocess call
            run_command = f'python {script_path} --config-path {str(configs_save_dir)} --config-name {config_name}'
            LOGGER.info(f'Starting run {counter}/{len(experiment_configs) - 1}: {run_command}')
            running_processes[(counter, run_command, node_id)] = subprocess.Popen(run_command,
                                                                                  stdout=subprocess.DEVNULL,
                                                                                  shell=True)
            running_run_idxs[counter] = True

            counter += 1
            if log_wandb:
                log_dict = _create_run_status_dict(experiment_configs, counter, running_run_idxs)
                wandb.log(log_dict[0])
                # wandb.log(log_dict[1])

            time.sleep(sleep_time)

        # check for completed runs
        for key, process in running_processes.items():
            if process.poll() is not None:
                LOGGER.info(f'Finished run {key[0]} ({key[1]})')
                gpu_counter[key[2]] -= 1
                LOGGER.info('Cleaning up...\n')
                running_run_idxs[key[0]] = False
                if log_wandb:
                    log_dict = _create_run_status_dict(experiment_configs, counter, running_run_idxs)
                    wandb.log(log_dict[0])
                    # wandb.log(log_dict[1])
                try:
                    _ = process.communicate(timeout=5)
                except TimeoutError:
                    LOGGER.warning('')
                    LOGGER.warning(f'WARNING: PROCESS {key} COULD NOT BE REAPED!')
                    LOGGER.warning('')
                running_processes[key] = None

        # delete possibly finished runs
        running_processes = {key: val for key, val in running_processes.items() if val is not None}
        running_run_idxs = {key: val for key, val in running_run_idxs.items() if val}
        time.sleep(sleep_time)

        if (len(running_processes) == 0) and (counter >= len(experiment_configs)):
            break

    LOGGER.info(f'Done. # of finished runs: {counter}')
    sys.stdout.flush()


def _create_run_status_dict(experiment_configs: List[DictConfig], counter: int,
                            running_run_idxs: Dict[int, bool]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # finished runs: run_indexes < counter and not in running_run_idxs
    finished_run_idxs = list(range(counter))
    for run_idx, running in running_run_idxs.items():
        if running:
            finished_run_idxs.remove(run_idx)

    # running runs: running_run_idxs, where running_run_idxs[run_idx] == True
    running_run_idxs = [idx for idx in running_run_idxs if running_run_idxs[idx]]

    # remaining runs: run_indices >= counter and not in running_run_idxs
    remaining_run_idxs = list(range(counter, len(experiment_configs)))

    # assert set(list(range(len(experiment_configs)))
    #            ) - set(finished_run_idxs) - set(running_run_idxs) - set(remaining_run_idxs) == set()

    def create_run_table(run_indexes: List[int], experiment_configs: List[DictConfig]) -> Dict[str, Any]:
        data = {'run_index': [], 'experiment_name': []}
        for idx in run_indexes:
            data['run_index'].append(idx)
            data['experiment_name'].append(experiment_configs[idx].config.experiment_data.experiment_name)
        df = pd.DataFrame(data)
        return df

    finished_table = wandb.Table(dataframe=create_run_table(finished_run_idxs, experiment_configs))
    running_table = wandb.Table(dataframe=create_run_table(running_run_idxs, experiment_configs))
    remaining_data = create_run_table(remaining_run_idxs, experiment_configs)
    remaining_table = wandb.Table(dataframe=remaining_data)

    log_dict_counts = {
        'n_finished': len(finished_run_idxs),
        'n_running': len(running_run_idxs),
        'n_remaining': len(remaining_run_idxs)
    }
    log_dict_tables = {
        'finished_runs': finished_table,
        'running_runs': running_table,
        'remaining_table': remaining_table
    }
    return {'run_scheduler/': log_dict_counts}, log_dict_tables
