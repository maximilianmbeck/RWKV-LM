import copy
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from ml_utils.models.base_model import BaseModel
from ml_utils.run_utils.run_handler import EXP_NAME_DIVIDER
from ml_utils.utils import (
    convert_dict_to_python_types,
    convert_listofdicts_to_dictoflists,
    flatten_hierarchical_dict,
)
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .directories import JobDirectory, SweepDirectory
from .model_loader import get_best_model_idx, load_model_from_idx

DATA_LOG_COMMON_COLS = ['epoch', 'train_step', 'log_step']
PARALLELIZE_THRESHOLD = 20


class JobResult:
    """Class providing access to results of a finished job.
    """

    def __init__(self, job_dir: Union[str, Path]):
        """
        Args:
            job_dir (Union[str, Path]): The run directory of the job.
        """
        self._job_dir = JobDirectory(job_dir)
        self._job_dir.is_directory()
        self._config = None

    @property
    def directory(self) -> Path:
        """The job directory."""
        return self._job_dir.dir

    @property
    def config(self) -> DictConfig:
        """The job config."""
        if not self._config:
            self._config = self._job_dir.load_config()
        return self._config

    @property
    def experiment_data(self) -> Dict[str, Any]:
        """The experiment data."""
        return self.config.config.experiment_data

    @property
    def seed(self) -> int:
        "The job seed."
        return self.experiment_data.seed

    @property
    def experiment_name(self) -> str:
        """The raw experiment name without hyperparameters."""
        return self.experiment_data.experiment_name.split(EXP_NAME_DIVIDER)[0]

    @property
    def override_hpparams(self) -> Dict[str, Any]:
        """Return the hyper-parameters that where overriden by a sweep."""
        from ml_utils.run_utils.sweep import OVERRIDE_PARAMS_KEY
        cfg = self.config
        override_hpparams = cfg.get(OVERRIDE_PARAMS_KEY, {})
        override_hpparams = flatten_hierarchical_dict(override_hpparams)
        override_hpparams = convert_dict_to_python_types(override_hpparams, objects_as_str=True)
        return override_hpparams

    @property
    def is_successful_job(self) -> bool:
        """Returns true, if the job has run sucessfully."""
        valid_job = True
        try:
            best_idx, specifier = get_best_model_idx(self._job_dir)
        except:
            valid_job = False
        return valid_job

    @property
    def data_log_sources(self) -> List[str]:
        """The data log sources."""
        log_dir = self._job_dir.stats_folder
        from ml_utils.logger import FN_DATA_LOG_PREFIX
        return [p.stem.replace(FN_DATA_LOG_PREFIX, '') for p in log_dir.glob(pattern=f'{FN_DATA_LOG_PREFIX}*')]

    @property
    def progress_measure(self) -> str:
        """The progress measure used for this run. For e.g. early stopping depends on this."""
        _, progress_measure = get_best_model_idx(self._job_dir)
        return progress_measure

    @property
    def available_model_checkpoint_indices(self) -> List[int]:
        """The available model checkpoints."""
        return self._job_dir.get_checkpoint_indices()

    @property
    def best_model_idx(self) -> int:
        """The best model index."""
        best_idx, _ = get_best_model_idx(self._job_dir)
        return best_idx

    @property
    def highest_model_idx(self) -> int:
        """Highest model index, or longest trained model."""
        highest_idx = max(self.available_model_checkpoint_indices)
        return highest_idx

    @property
    def best_model(self) -> BaseModel:
        """The best model."""
        return self.get_model_idx(-1)

    @property
    def all_log_columns(self) -> Dict[str, List[str]]:
        """The available log columns per data log.
        """
        log_cols = {}
        # get all log columns
        for dls in self.data_log_sources:
            # get raw data log dataframe
            log_df = self.get_data_log(source=dls, common_cols=[])
            log_cols[dls] = list(log_df.columns)
        return log_cols

    def get_model_idx(
        self,
        idx: int = -1,
        device: Union[torch.device, str, int] = 'cpu',
    ) -> BaseModel:
        """Return a model given a checkpoint index.

        Args:
            idx (int, optional): The model checkpoint index. If idx is -1, return best model. Defaults to -1.
            device (Union[torch.device, str, int], optional): Device to where the model is loaded to. Defaults to "auto".

        Returns:
            BaseModel: The model.
        """

        if idx == -1:
            idx, _ = get_best_model_idx(self._job_dir)
        return load_model_from_idx(job_dir=self._job_dir, idx=idx, device=device)

    def get_summary(self,
                    log_source: str = '',
                    row_sel: Union[Tuple[str, int], Tuple[str, List[int]]] = (),
                    col_sel: Union[str, List[str]] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        """Return a summary of the run. 

        Args:
            log_source (str, optional): The log source. If specified, it adds metrics from the logsource. Defaults to ''.
            row_sel (Union[Tuple[str, int], Tuple[str, List[int]]], optional): Select a row / multiple rows in the log source, 
                                                                               if unspecified use the best epoch / step. 
                                                                               The str is the progress measure, e.g. `epoch` or `train_step`. 
                                                                               Defaults to ().
            col_sel (Union[str, List[str]], optional): The columns. Defaults to [].
            append_override_hpparams (bool, optional): Add override params. Defaults to True.
            append_seed (bool, optional): Add a seed column. Defaults to True.

        Returns:
            pd.DataFrame: The summary table.
        """
        from ml_utils.logger import FN_FINAL_RESULTS
        from ml_utils.trainer.basetrainer import RUN_PROGRESS_MEASURE_STEP

        summary_dict = OmegaConf.to_container(OmegaConf.load(self._job_dir.stats_folder / f'{FN_FINAL_RESULTS}.yaml'))
        if log_source:
            assert col_sel, 'Must provide a column selection.'
            if not isinstance(col_sel, list):
                col_sel = [col_sel]

            if not row_sel:
                idx, progress_measure = get_best_model_idx(self._job_dir)
                if progress_measure == RUN_PROGRESS_MEASURE_STEP:
                    progress_measure = 'train_step'
                row_sel = (progress_measure, idx)

            if not isinstance(row_sel[1], list):
                row_sel = (row_sel[0], [row_sel[1]])

            log_df = self.get_data_log(source=log_source)
            for row in row_sel[1]:
                row_df = log_df[log_df[row_sel[0]] == row][col_sel]
                if row_df.empty:
                    raise ValueError(f'No row `{row_sel[0]}-{row}` found.')
                log_dict = row_df.transpose().to_dict()  # this is a dictionary {index: {col: vals}}
                # remove index
                _, log_dict = next(iter(log_dict.items()))
                # add row indicator
                log_dict = {f'{k}-{row_sel[0]}-{row}': v for k, v in log_dict.items()}
                summary_dict.update(log_dict)
        if append_override_hpparams:
            summary_dict.update(self.override_hpparams)
        if append_seed:
            cfg = self.config
            seed = cfg.config.experiment_data.seed
            summary_dict.update({'seed': seed})
        return pd.DataFrame(summary_dict, index=[self._job_dir.dir.name])

    def get_data_log(self,
                     source: Union[str, List[str]] = [],
                     common_cols: List[str] = DATA_LOG_COMMON_COLS) -> pd.DataFrame:
        """Returns the data log table.

        Args:
            source (Union[str, List[str]], optional): The specifier(s) of the log source, e.g. `val` or `train`. 
                                                      Can be either a single value or a list of values. 
                                                      If list is empty return data from all logs. Defaults to [].
            common_cols (List[str], optional): Column names of columns that are shared across all data logs. 

        Raises:
            ValueError: If the data log file does not exist.

        Returns:
            pd.DataFrame: The log table.
        """
        from ml_utils.logger import FN_DATA_LOG, LOG_STEP_KEY
        if source == []:
            source = self.data_log_sources
        elif not isinstance(source, list):
            source = [source]
        assert source, f'No data log sources found / given.'

        data_logs = []
        for src in source:
            filename = FN_DATA_LOG.format(datasource=src)
            log_file = self._job_dir.stats_folder / filename
            if not log_file.exists():
                raise ValueError(f'Log file for source `{src}` does not exist at `{str(self._job_dir.stats_folder)}`!')

            src_df = pd.read_csv(log_file, index_col=0)
            # move log_step column to first place
            ls_col = src_df.pop(LOG_STEP_KEY)
            src_df.insert(0, LOG_STEP_KEY, ls_col)

            if len(source) > 1:
                # we are concatening multiple data log sources
                # rename columns
                new_names = {cn: f'{src}-{cn}' for cn in src_df.columns if cn not in common_cols}
                src_df.rename(columns=new_names, inplace=True)

            data_logs.append(src_df)

        if len(data_logs) > 1:
            # merge dataframes
            data_log_df = reduce(lambda left, right: pd.merge(left, right, on=common_cols, how='outer'), data_logs)
        else:
            data_log_df = data_logs[0]

        data_log_df.sort_values(LOG_STEP_KEY, inplace=True)
        return data_log_df

    def __str__(self):
        return str(self._job_dir)

    def __repr__(self):
        return f'JobResult({str(self)})'


class SweepResult:
    """Class providing access to a finished hyperparameter sweep.
    """

    def __init__(self, sweep_dir: Union[str, Path], num_loader_worker: int = 6):
        self._sweep_dir = SweepDirectory(sweep_dir)
        self._sweep_dir.is_directory()
        self._sweep_runs_dir = self._sweep_dir.jobs_folder
        self._num_workers = num_loader_worker
        if not self._sweep_runs_dir.exists():
            raise FileNotFoundError(
                f'Run folder `{self._sweep_runs_dir.dir.stem}` does not exist in sweep `{self._sweep_dir}`.')
        self._joblist = self._get_joblist()

        self._config: DictConfig = None
        self._sweep_hpparams: Dict[str, List[Any]] = None
        self._failed_sweep_hpparams: Dict[str, List[Any]] = None
        self._failed_jobs: List[JobResult] = None
        self._query_summary: pd.DataFrame = None

    @property
    def config(self) -> DictConfig:
        """Return the template config for each job in the sweep."""
        if not self._config:
            self._config = self._sweep_dir.load_config()
        return self._config

    @property
    def directory(self) -> Path:
        """The sweep directory."""
        return self._sweep_dir.dir

    @property
    def job_directory(self) -> Path:
        """The job directory. Folder containing all job folders."""
        return self._sweep_runs_dir

    @property
    def seeds(self) -> List[int]:
        """The seeds for which each hyperparameter config is run."""
        seeds = self.config.seeds
        if isinstance(seeds, int):
            seeds = [seeds]
        return seeds

    @property
    def sweep_params(self) -> List[str]:
        """Parameters that were modified during sweep."""
        cfg = self.config
        from ml_utils.run_utils.sweep import SWEEP_AXES_KEY
        return [ax.parameter for ax in cfg.sweep[SWEEP_AXES_KEY]]

    @property
    def sweep_cfg(self) -> Dict[str, Any]:
        """Sweep config dictionary."""
        return self.config.sweep

    @property
    def sweep_str(self) -> str:
        """Sweep parameters in a string representation."""
        from ml_utils.run_utils.sweep import SWEEP_AXES_KEY, SWEEP_TYPE_KEY
        sweep_param_cfg = self.sweep_cfg
        if not sweep_param_cfg:
            return 'No sweep.'
        sweep_type = sweep_param_cfg[SWEEP_TYPE_KEY]
        sweep_axes = sweep_param_cfg[SWEEP_AXES_KEY]
        sweep_str = f'Sweep type: {sweep_type}\n'
        if sweep_axes:
            for sax in sweep_axes:
                param: str = sax['parameter']
                vals = sax['vals']
                param_name = param  # Show full parameter, instead of `param.split('.')[-1]`
                val_str = vals
                sweep_str += f'  {param_name}: {val_str}\n'
        else:
            sweep_str += '  No sweep axes.\n'
        return sweep_str

    @property
    def available_log_columns(self) -> Dict[str, List[str]]:
        """A dictionary listing all available colomns (values) per data log (keys)."""
        log_cols = self[0].all_log_columns
        available_log_cols = {}
        available_log_cols['_common_cols'] = DATA_LOG_COMMON_COLS
        for dls, cols in log_cols.items():
            available_log_cols[dls] = list(set(cols) - set(DATA_LOG_COMMON_COLS))
        return available_log_cols

    def get_sweep_param_values(self, param_searchstr: str = '') -> Dict[str, Any]:
        """Returns (deepcopy) the hyperparameter values of all successful runs as dictionary. 
        Allows hyperparameter filtering with `param_searchstr`.

        Args:
            param_searchstr (str, optional): The search string. Defaults to ''.

        Returns:
            Dict[str, Any]: Dictionary containing hyperparameters as keys and list of values as values.
        """
        if self._sweep_hpparams is None:
            _ = self.get_failed_jobs()

        sweep_hpparams = copy.deepcopy(self._sweep_hpparams)

        # no search string given: return values for all parameters as dict
        if not param_searchstr:
            return sweep_hpparams

        ret_dict = {}
        matching_hpparams = [k for k in sweep_hpparams if param_searchstr in k]
        # search string given: return only matching parameter values as dict
        ret_dict = {param: sweep_hpparams[param] for param in matching_hpparams}

        return ret_dict

    def _get_joblist(self, searchstr: str = '') -> List[Path]:
        if searchstr:
            return list(self._sweep_runs_dir.glob(f'*{searchstr}*'))
        else:
            return list(self._sweep_runs_dir.iterdir())

    def find_jobs(self, searchstr: str = '') -> List[str]:
        """Get matching job names.
        """
        return [str(j) for j in self._get_joblist(searchstr)]

    def get_jobs(self, searchstr: str = '') -> List[JobResult]:
        """Get jobs with matching job name."""
        joblist = self._get_joblist(searchstr)
        if len(joblist) > PARALLELIZE_THRESHOLD:  # parallelize only for many jobs
            return Parallel(n_jobs=self._num_workers)(delayed(JobResult)(j) for j in joblist)
        return [JobResult(j) for j in joblist]

    def query_jobs(self,
                   hypparam_sel: Dict[str, Any] = {},
                   combine_operator: str = '&',
                   return_joblist: bool = True,
                   float_eps: float = 1e-3) -> Tuple[pd.DataFrame, Optional[List[JobResult]]]:
        """Query the jobs by hyperparameters in the standard run summary.

        Args:
            hypparam_sel (Dict[str, Any]): Dict with the hyperparameters as keys and the query values as values.
                                           If {}, return the standard run summary. Defaults to {}.
            combine_operator (str, optional): Bool operator combining the hyperparameter query. Defaults to '&'.
            return_joblist (bool, optional): Return a list of found job results.. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, Optional[List[JobResult]]]: The query result dataframe and optionally the Jobresult list.
        """

        def generate_query(hypparam_sel, combine_operator, float_eps) -> str:
            query = ''
            query_col_val_pair = '`{column}` == {value}'
            query_col_val_pair_float = '({value}-{eps}) <= `{column}` <= ({value}+{eps})'
            for column, value in hypparam_sel.items():
                if query != '':
                    query += f' {combine_operator} '
                if isinstance(value, float):
                    query += query_col_val_pair_float.format(column=column, value=value, eps=float_eps)
                else:
                    query += query_col_val_pair.format(column=column, value=value)
            return query

        if self._query_summary is None:
            self._query_summary = self.get_summary(append_override_hpparams=True, append_seed=True)

        query = generate_query(hypparam_sel=hypparam_sel, combine_operator=combine_operator, float_eps=float_eps)
        if query:
            query_df = self._query_summary.query(query)
        else:
            query_df = self._query_summary

        joblist = None
        # desc = 'Creating jobresults'
        if return_joblist:
            jobnames = list(query_df.index)
            if len(jobnames) > PARALLELIZE_THRESHOLD:
                joblist = Parallel(n_jobs=self._num_workers)(
                    delayed(JobResult)(self._sweep_runs_dir / jobname) for jobname in jobnames)
            else:
                joblist = [JobResult(self._sweep_runs_dir / jobname) for jobname in jobnames]

        return query_df, joblist

    def get_failed_jobs(self) -> List[str]:
        """Get failed jobs (and collects hpparams of succesful runs on-the-fly)."""
        _desc = 'Collecting failed jobs'

        def extract_is_successful_and_override_hpparams(jp: Path) -> Tuple[bool, Dict[str, Any]]:
            job = JobResult(jp)
            hpparams = job.override_hpparams
            hpparams['seed'] = job.seed
            return job.is_successful_job, job, hpparams

        if self._failed_jobs is None:
            joblist = self._joblist
            if len(joblist) == 0:
                raise ValueError('No jobs found in sweep directory.')
            if len(joblist) > PARALLELIZE_THRESHOLD:
                failed_jobs_and_hpparams = Parallel(n_jobs=self._num_workers)(
                    delayed(extract_is_successful_and_override_hpparams)(jp) for jp in tqdm(joblist, desc=_desc))
            else:
                failed_jobs_and_hpparams = [
                    extract_is_successful_and_override_hpparams(jp) for jp in tqdm(joblist, desc=_desc)
                ]

            # rows: jobs, columns: [is_successful, JobResult, override_hpparams]
            fh_array = np.array(failed_jobs_and_hpparams)
            successful_runs = fh_array[:, 0].astype(bool)
            # extract failed runs
            failed_jobs = fh_array[np.logical_not(successful_runs), 1].tolist()
            # extract hpparams
            successful_hpparams = convert_listofdicts_to_dictoflists(fh_array[successful_runs, 2].tolist())
            failed_hpparams = convert_listofdicts_to_dictoflists(fh_array[np.logical_not(successful_runs), 2].tolist())
            for hp_dict in [successful_hpparams, failed_hpparams]:
                for hp in hp_dict:
                    hp_dict[hp] = list(set(hp_dict[hp]))  # make hp values unique
                    hp_dict[hp].sort()
            self._sweep_hpparams = successful_hpparams
            self._failed_sweep_hpparams = failed_hpparams
            self._failed_jobs = failed_jobs

        return self._failed_jobs, self._failed_sweep_hpparams

    def get_summary(self,
                    searchstr: str = '',
                    log_source: str = '',
                    row_sel: Union[Tuple[str, int], Tuple[str, List[int]]] = (),
                    col_sel: Union[str, List[str]] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        """Return a summary table with the job name as index. 
        Calls the `get_summary()` method of each job in the sweep.

        Args:
            searchstr (str, optional): A string to prefilter the runs in the sweep. Defaults to ''.
            log_source (str, optional): The log source. If specified, it adds metrics from the logsource. Defaults to ''.
            row_sel (Union[Tuple[str, int], Tuple[str, List[int]]], optional): Select a row / multiple rows in the log source, 
                                                                               if unspecified use the best epoch / step. Defaults to ().
            col_sel (Union[str, List[str]], optional): The columns. Defaults to [].
            append_override_hpparams (bool, optional): Add override params. Defaults to True.
            append_seed (bool, optional): Add a seed column. Defaults to True.

        Example:
        ```python
        swr = SweepResult(sweep_dir)
        swr.get_summary(log_source='val', col_sel='Accuracy-top-1', row_sel=('epoch', 10))
        ``` 

        Returns:
            pd.DataFrame: The summary table.
        """

        def get_job_result_summary(job_dir: Path) -> pd.DataFrame:
            try:
                ret = JobResult(job_dir).get_summary(log_source=log_source,
                                                     row_sel=row_sel,
                                                     col_sel=col_sel,
                                                     append_override_hpparams=append_override_hpparams,
                                                     append_seed=append_seed)
            except FileNotFoundError as e:
                print(f'Could not find job {job_dir}')
                ret = pd.DataFrame()
            return ret

        _desc = 'Collecting summaries'
        joblist = self._get_joblist(searchstr=searchstr)
        if len(joblist) > PARALLELIZE_THRESHOLD:
            summaries = Parallel(n_jobs=self._num_workers)(
                delayed(get_job_result_summary)(job_dir) for job_dir in tqdm(sorted(joblist), desc=_desc))
        else:
            summaries = [get_job_result_summary(job_dir) for job_dir in tqdm(sorted(joblist), desc=_desc)]

        return pd.concat(summaries)

    def __len__(self) -> int:
        return len(self._joblist)

    def __getitem__(self, key) -> JobResult:
        return JobResult(job_dir=self._joblist[key])

    def __str__(self) -> str:
        from .repo import FORMAT_CFG_DATETIME, KEY_CFG_UPDATED

        sweep_result_str = 'Exp. Tag(start_num): {tag_num}\nExp. Name: {exp_name}\nTraining setup: {training_setup}\nModel name: {model_name}\nDataset name: {dataset_name}\n{sweep_str}Seeds: {seeds}\nNum. jobs: {num_jobs}\nConfig updated: {cfg_last_updated}\nSweep started:  {sweep_started}\n'

        cfg = self.config
        tagnum = f'{cfg.config.experiment_data.experiment_tag}({self.config.start_num})'
        expname = cfg.config.experiment_data.experiment_name.split(EXP_NAME_DIVIDER)[0]
        sweep_started = datetime.strptime(self.directory.name.split(EXP_NAME_DIVIDER)[-1],
                                          '%y%m%d_%H%M%S').strftime(FORMAT_CFG_DATETIME)
        model_name = cfg.config.model.get('name', None)
        if model_name is None:
            model_name = cfg.config.model.model_cfg
        cfg_last_updated = cfg.get(KEY_CFG_UPDATED, '')
        trainingsetup = cfg.config.trainer.training_setup
        dataset_name = cfg.config.data.get('dataset', None)
        if dataset_name is None:
            dataset_name = cfg.config.data.get('name', None)
        if dataset_name is None:
            dataset_name = 'Unknown. Check config!'
        ss = sweep_result_str.format(tag_num=tagnum,
                                     exp_name=expname,
                                     training_setup=trainingsetup,
                                     model_name=model_name,
                                     dataset_name=dataset_name,
                                     sweep_str=self.sweep_str,
                                     seeds=self.seeds,
                                     num_jobs=len(self),
                                     cfg_last_updated=cfg_last_updated,
                                     sweep_started=sweep_started)
        return ss

    def __repr__(self) -> str:
        return f'SweepResult({self.directory.name})'
