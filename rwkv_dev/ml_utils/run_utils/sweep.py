import copy
import itertools
import logging
import sys
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rwkv_dev.ml_utils.run_utils.sweep_searchspaces import SearchSpaceDim
from rwkv_dev.ml_utils.run_utils.value_parser import parse_list_str
from rwkv_dev.ml_utils.utils import flatten_hierarchical_dict, zip_strict

SWEEP_TYPE_KEY = 'type'
SWEEP_AXES_KEY = 'axes'
SWEEP_TYPE_GRIDVAL = 'grid'  # create a grid of all given parameter values
SWEEP_TYPE_LINEVAL = 'line'  # create new configs where val indices correspond to each other
SWEEP_TYPE_RANDOMVAL = 'random'  # sample random parameters according to specified distributions
SWEEP_TYPE_RANDOMGRIDVAL = 'random_grid'  # sample from gridsearch config
SWEEP_TYPE_SKIPVAL = 'skip'  # skip sweeping and use default parameters
EXPERIMENT_CONFIG_KEY = 'config'
OVERRIDE_PARAMS_KEY = f'{EXPERIMENT_CONFIG_KEY}_override_params'

LOGGER = logging.getLogger(__name__)


class Sweeper(ABC):
    """This class is passed the config and it generates multiple configs according to the sweep configuration."""

    def __init__(self, config: DictConfig = None, sweep_config: DictConfig = None):
        self._config = config
        try:
            self.sweep_config = self._config.sweep
        except:
            if sweep_config:
                self.sweep_config = sweep_config
            else:
                raise ValueError(f'No sweep config provided.')

        assert OmegaConf.is_list(self.sweep_config[SWEEP_AXES_KEY])
        # TODO set seed (optional)

    @property
    def sweep_params(self) -> List[str]:
        """Names of sweep parameters."""
        return [ax.parameter for ax in self.sweep_config[SWEEP_AXES_KEY]]

    @abstractmethod
    def _generate_sweep_axes(self, sweep_config: DictConfig) -> List[Iterator[DictConfig]]:
        """Creates an iterator for each sweep axes. Each iterator produces a new set of parameter values for 
        the respective sweep axis, when next() is called on it.
        The output of all iterators in then combined to a new search point in the hyperparameter search space.

        Args:
            sweep_config (DictConfig): The sweep config dictionary.

        Returns:
            List[Iterator[DictConfig]]: Generators for each sweep axis. In combination they produce 
                                        a new search point in hyperparameter search space.
        """
        pass

    @abstractmethod
    def _generate_axis_combinations(self, sweep_axes: List[Iterator[DictConfig]]) -> Iterable[Tuple[DictConfig, ...]]:
        """Produces a combination of each sweep axis iterator. 

        Args:
            sweep_axes (List[Iterator[DictConfig]]): The list of sweep axis iterators producing parameter values.

        Returns:
            Iterable[Tuple[DictConfig, ...]]: An iterator for sweep axis combinations.
        """
        pass

    def drop_axes(self, ax_param_names: Union[str, List[str]]) -> List[DictConfig]:
        """Remove axes from the sweep.

        Args:
            ax_param_names (Union[str, List[str]]): The (exact) parameter names. Hierarchy separated by dots.

        Returns:
            List[DictConfig]: The removed axis configs.
        """
        if isinstance(ax_param_names, str):
            ax_param_names = [ax_param_names]
        drop_axes = []
        all_axes = self.sweep_config[SWEEP_AXES_KEY]
        for i, axis in enumerate(all_axes):
            if axis.parameter in ax_param_names:
                drop_axes.append(all_axes.pop(i))
        return drop_axes


    def generate_sweep_parameter_combinations(self, flatten_hierarchical_dicts: bool = True) -> List[DictConfig]:
        """Generate a list of axis combinations.

        Args:
            flatten_hierarchical_dicts (bool, optional): Transform hierarchical dicts to dicts with dot-separated keys. Defaults to True.

        Returns:
            List[DictConfig]: List of sweep parameter combinations.
        """
        if flatten_hierarchical_dicts:
            flatten = lambda ac: flatten_hierarchical_dict(ac)
        else:
            flatten = lambda ac: ac
        return [
            flatten(OmegaConf.merge(*ac))
            for ac in self._generate_axis_combinations(self._generate_sweep_axes(self.sweep_config))
        ]

    def generate_configs(self) -> List[DictConfig]:
        """Generates a list of DictConfigs specified by the sweep. 

        Returns:
            List[DictConfig]: The DictConfig list with all configs in the sweep.
        """
        if self._config is None:
            raise ValueError('No template config file given. Cannot generate configs for sweep!')

        self.sweep_axes = self._generate_sweep_axes(self.sweep_config)

        # iterate through all possible grid combinations
        hypsearch_sweep_configs = []
        for axis_combination in tqdm(self._generate_axis_combinations(self.sweep_axes),
                                     file=sys.stdout,
                                     desc='Generating configs'):
            # contains only the parameters to be updated in the 'main' config:
            hypsearch_point_config = OmegaConf.merge(*axis_combination)

            # create new config with updated params
            tmp_config = copy.deepcopy(self._config)
            tmp_config[EXPERIMENT_CONFIG_KEY] = OmegaConf.merge(tmp_config[EXPERIMENT_CONFIG_KEY],
                                                                hypsearch_point_config)

            # add override parameters as extra entry
            # need `open_dict` as DictConfig is marked as `struct`, which does not allow to create new fields,
            # read more: https://omegaconf.readthedocs.io/en/2.2_branch/usage.html?highlight=read%20only#struct-flag
            with open_dict(tmp_config):
                tmp_config[OVERRIDE_PARAMS_KEY] = hypsearch_point_config

            hypsearch_sweep_configs.append(tmp_config)

        LOGGER.info(f"Sweep with {len(self.sweep_axes)} axes and {len(hypsearch_sweep_configs)} runs generated.")
        return hypsearch_sweep_configs

    @staticmethod
    def create(config: DictConfig = None, sweep_config: DictConfig = None) -> Optional['Sweeper']:
        try:
            sweep_type = config.sweep.type
        except:
            sweep_type = sweep_config.type
        LOGGER.info(f'Generating sweep type: {sweep_type}')
        if sweep_type == SWEEP_TYPE_GRIDVAL:
            return GridSweep(config, sweep_config)
        elif sweep_type == SWEEP_TYPE_LINEVAL:
            return LineSweep(config, sweep_config)
        elif sweep_type == SWEEP_TYPE_RANDOMVAL:
            return RandomSweep(config, sweep_config)
        elif sweep_type == SWEEP_TYPE_RANDOMGRIDVAL:
            return RandomGridSweep(config, sweep_config)
        elif sweep_type == SWEEP_TYPE_SKIPVAL:
            return None
        else:
            raise ValueError(f"Unsupported sweep type: '{sweep_type}'")


class DeterministicSweep(Sweeper):

    def _generate_sweep_axes(self, sweep_config: DictConfig) -> List[Iterator[DictConfig]]:
        # generate list of axis generators
        return [sweep_axis_grid(ax) for ax in sweep_config[SWEEP_AXES_KEY]]


class GridSweep(DeterministicSweep):
    # TODO use range or deterministic search spaces
    def _generate_axis_combinations(self, sweep_axes: List[Iterator[DictConfig]]) -> Iterable[Tuple[DictConfig, ...]]:
        return itertools.product(*sweep_axes)


class RandomGridSweep(DeterministicSweep):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.num_samples = self.sweep_config.num_runs

    def _generate_axis_combinations(self, sweep_axes: List[Iterator[DictConfig]]) -> Iterable[Tuple[DictConfig, ...]]:
        # pre-generate the full grid and sample from the axes combinations. Return the samples.
        axis_combinations_full_grid = [ac for ac in itertools.product(*sweep_axes)]
        axis_combinations_idxes = np.arange(len(axis_combinations_full_grid))
        np.random.default_rng().shuffle(axis_combinations_idxes)
        axis_combinations_sampled = [
            axis_combinations_full_grid[idx] for idx in axis_combinations_idxes[:self.num_samples]
        ]
        return axis_combinations_sampled


class LineSweep(DeterministicSweep):

    def _generate_axis_combinations(self, sweep_axes: List[Iterator[DictConfig]]) -> Iterable[Tuple[DictConfig, ...]]:
        return zip_strict(*sweep_axes)


class RandomSweep(Sweeper):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.num_samples = self.sweep_config.num_runs

    def _generate_sweep_axes(self, sweep_config: DictConfig) -> List[Iterator[DictConfig]]:
        # generate list of axis generators
        return [sweep_axis_random(ax, self.num_samples, random_grid=False) for ax in sweep_config[SWEEP_AXES_KEY]]

    def _generate_axis_combinations(self, sweep_axes: List[Iterator[DictConfig]]) -> Iterable[Tuple[DictConfig, ...]]:
        # TODO later: add gridsearch searchspace (each entry is sampled num_samples / len(gridsearch) times) < hard, if even possible with current design
        return zip_strict(*sweep_axes)


#  a generator is an iterator: https://www.geeksforgeeks.org/generators-in-python/
def sweep_axis_grid(axis_config: DictConfig) -> Iterator[DictConfig]:
    """Receives a DictConfig object with keys `parameter` and `vals` and generates parameter->value dicts. 
    It also supports creating an axis, where multiple parameters are varied at once. In this case axis.parameter contains a list of paramters
    and axis.vals contains a zipped list of the parameter values for the paramters in axis.parameter.

    For examples see `example_config.yaml`.
    
    Args:
        axis_config (DictConfig): Sweep config for one axis, as specified by user in the config .yaml file.

    Yields:
        Iterator[DictConfig]: A generator, that produces single DictConfigs with the parameter as keys and its respective values.
    """
    # check num parameters
    parameters = axis_config.parameter
    if OmegaConf.is_list(parameters):
        num_parameters = len(parameters)
    else:
        num_parameters = 1
        parameters = [parameters]

    vals = axis_config.vals
    if isinstance(vals, str):
        # convert expression to list
        vals = parse_list_str(vals)
    elif OmegaConf.is_list(vals):
        vals = list(vals)
    else:
        raise ValueError(
            f'Axis config parameter values for parameter `{axis_config.parameter}` are of unsupported type: {type(vals)}'
        )
    assert isinstance(vals, list)
    num_vals = len(vals)

    assert num_vals % num_parameters == 0, f"Number of specified values ({num_vals}) must be divisible by the number of parameters ({num_parameters})!"

    # iterate over vars
    for val_index in range(0, num_vals, num_parameters):
        sweep_axis_param = OmegaConf.create()
        for param_index, param in enumerate(parameters):
            OmegaConf.update(sweep_axis_param, param, vals[val_index + param_index])
        yield sweep_axis_param


def sweep_axis_random(axis_config: DictConfig, num_samples: int, random_grid: bool = False) -> Iterator[DictConfig]:
    if random_grid:
        with open_dict(axis_config):
            OmegaConf.update(axis_config, 'searchspace', 'choice')
    searchspace_dim = SearchSpaceDim.create(axis_config)
    # make sure to sample only num_samples
    for i in range(num_samples):
        yield searchspace_dim.sample_parameter()