import inspect
import sys
from abc import ABC, abstractmethod
from math import isclose
from typing import Any, List, Tuple, Type, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from rwkv_dev.ml_utils.run_utils.value_parser import parse_list_str
from rwkv_dev.ml_utils.utils import convert_to_python_types

"""These search spaces are based on the API of ray.tune. See [#].

References:
    .. [#] https://docs.ray.io/en/latest/tune/api_docs/search_space.html#search-space-api 
"""

DEFAULT_LOG_DISTR_BASE = 10.0

# TODO add Hyperparameter sweeps with Halton sequences of quasi-random numbers: 
# https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/halton.py

class SearchSpaceDim(ABC):

    def __init__(self, axis_config: DictConfig):
        self._config = axis_config

        assert isinstance(self._config.parameter, str), f'`parameter` must be a dot separated string.'
        self.parameter: str = self._config.parameter

        assert OmegaConf.is_list(self._config.vals), f'`vals` must be a list.'
        self.vals = self._config.vals

    def _create_axis_param_dict(self, param: str, val: Any) -> DictConfig:
        axis_param_dict = OmegaConf.create({})
        OmegaConf.update(axis_param_dict, param, val)
        return axis_param_dict

    @abstractmethod
    def _sample_value(self, vals: List[Any]) -> Any:
        pass

    def sample_parameter(self) -> DictConfig:
        value = self._sample_value(self.vals)
        return self._create_axis_param_dict(self.parameter, value)

    def _get_n_arguments(self, vals: List[Any], n_arguments: int, last_optional: bool = False) -> Tuple[float, ...]:
        assert n_arguments > 0
        l_o = int(last_optional)
        assert len(vals) == n_arguments or len(vals) == n_arguments - 1
        vals = [float(v) for v in vals]
        assert vals[0] <= vals[1], f'Second argument {vals[1]} must be smaller than first argument {vals[0]}!'
        return tuple(vals)

    def _quantize_value(self, value: float, lower: float, upper: float, q: float) -> float:
        if lower > float("-inf") and not isclose(lower / q, round(lower / q)):
            raise ValueError(f"Your lower variable bound {lower} is not divisible by "
                             f"quantization factor {q}.")
        if upper < float("inf") and not isclose(upper / q, round(upper / q)):
            raise ValueError(f"Your upper variable bound {upper} is not divisible by "
                             f"quantization factor {q}.")
        quantized = np.round(np.divide(value, q)) * q
        return quantized.item()

    def _check_log_distr_inputs(self, lower: float, upper: float) -> None:
        if not lower > 0:
            raise ValueError("LogDistribution (e.g. LogUniform) requires a lower bound greater than 0."
                             f"Got: {self.lower}. Did you pass a variable that has "
                             "been log-transformed? If so, pass the non-transformed value "
                             "instead.")
        if not 0 < upper < float("inf"):
            raise ValueError("LogDistribution (e.g. LogUniform) requires a upper bound greater than 0. "
                             f"Got: {self.lower}. Did you pass a variable that has "
                             "been log-transformed? If so, pass the non-transformed value "
                             "instead.")

    @staticmethod
    def create(axis_config: DictConfig) -> 'SearchSpaceDim':
        searchspace_name = axis_config.searchspace
        searchspace_cls = _searchspace_registry[searchspace_name]
        return searchspace_cls(axis_config)


class Uniform(SearchSpaceDim):
    name = 'uniform'

    def _sample_value(self, vals: List[Any]) -> float:
        lower, upper = self._get_n_arguments(vals, 2)
        return np.random.default_rng().uniform(low=lower, high=upper)


class QUniform(SearchSpaceDim):
    name = 'quniform'

    def _sample_value(self, vals: List[Any]) -> float:
        lower, upper, quantization = self._get_n_arguments(vals, 3)
        sample_val = np.random.default_rng().uniform(low=lower, high=upper)
        return self._quantize_value(sample_val, lower, upper, quantization)


class LogUniform(SearchSpaceDim):
    name = 'loguniform'

    def _sample_value(self, vals: List[Any]) -> Any:
        args = self._get_n_arguments(vals, 3, last_optional=True)
        if len(args) == 2:
            lower, upper = args
            base = DEFAULT_LOG_DISTR_BASE
        else:
            lower, upper, base = args
        logmin = np.log(lower) / np.log(base)
        logmax = np.log(upper) / np.log(base)
        sample_val = base**np.random.default_rng().uniform(low=logmin, high=logmax)
        return sample_val.item()


class QLogUniform(SearchSpaceDim):
    name = 'qloguniform'

    def _sample_value(self, vals: List[Any]) -> Any:
        args = self._get_n_arguments(vals, 4, last_optional=True)
        if len(args) == 2:
            lower, upper, quantization = args
            base = DEFAULT_LOG_DISTR_BASE
        else:
            lower, upper, quantization, base = args
        self._check_log_distr_inputs(lower, upper)
        logmin = np.log(lower) / np.log(base)
        logmax = np.log(upper) / np.log(base)
        sample_val = base**np.random.default_rng().uniform(low=logmin, high=logmax)
        return self._quantize_value(sample_val, lower, upper, quantization)


class Normal(SearchSpaceDim):
    name = 'normal'

    def _sample_value(self, vals: List[Any]) -> float:
        mean, std = self._get_n_arguments(vals, 2)
        sample_val = np.random.default_rng().normal(loc=mean, scale=std)
        return sample_val.item()


class QNormal(SearchSpaceDim):
    name = 'qnormal'

    def _sample_value(self, vals: List[Any]) -> float:
        mean, std, quantization = self._get_n_arguments(vals, 3)
        sample_val = np.random.default_rng().normal(loc=mean, scale=std)
        return self._quantize_value(sample_val, lower=float('-inf'), upper=float('inf'), q=quantization)


class UniformInt(SearchSpaceDim):
    name = 'uniformint'

    def _sample_value(self, vals: List[Any]) -> int:
        lower, upper = tuple(map(lambda x: int(x), self._get_n_arguments(vals, 2)))  # cast all values to integers
        sample_val = np.random.default_rng().integers(low=lower, high=upper)
        return sample_val.item()


class QUniformInt(SearchSpaceDim):
    name = 'quniformint'

    def _sample_value(self, vals: List[Any]) -> int:
        lower, upper, quantization = tuple(map(lambda x: int(x),
                                               self._get_n_arguments(vals, 3)))  # cast all values to integers
        sample_val = np.random.default_rng().integers(low=lower, high=upper)
        return int(self._quantize_value(sample_val, lower, upper, quantization))


class LogUniformInt(SearchSpaceDim):
    name = 'loguniformint'

    def _sample_value(self, vals: List[Any]) -> Any:
        args = self._get_n_arguments(vals, 3, last_optional=True)
        if len(args) == 2:
            lower, upper = args
            base = DEFAULT_LOG_DISTR_BASE
        else:
            lower, upper, base = args
        logmin = np.log(lower) / np.log(base)
        logmax = np.log(upper) / np.log(base)
        sample_val = base**np.random.default_rng().uniform(low=logmin, high=logmax)
        sample_val = np.floor(sample_val).astype(int)
        return sample_val.item()


class QLogUniformInt(SearchSpaceDim):
    name = 'qloguniformint'

    def _sample_value(self, vals: List[Any]) -> Any:
        args = self._get_n_arguments(vals, 4, last_optional=True)
        if len(args) == 2:
            lower, upper, quantization = args
            base = DEFAULT_LOG_DISTR_BASE
        else:
            lower, upper, quantization, base = args
        self._check_log_distr_inputs(lower, upper)
        logmin = np.log(lower) / np.log(base)
        logmax = np.log(upper) / np.log(base)
        sample_val = base**np.random.default_rng().uniform(low=logmin, high=logmax)
        sample_val = np.floor(sample_val).astype(int)
        return int(self._quantize_value(sample_val, lower, upper, quantization))


class Choice(SearchSpaceDim):
    name = 'choice'

    def _sample_value(self, vals: Union[List[Any], str]) -> Any:
        if isinstance(vals, str):
            vals = parse_list_str(vals)
        val_array = np.array(vals)
        sample_choice = np.random.default_rng().choice(val_array)
        if isinstance(sample_choice, (np.ndarray, np.generic)):
            return convert_to_python_types(sample_choice)
        else:
            return sample_choice

# TODO add gridsearch search space

# every class in this module is a SearchSpace, get all of them but the abstract base 'SearchSpace'
searchspace_classes: Tuple[str, Type['SearchSpaceDim']] = inspect.getmembers(
    sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__ and member.__name__
    != SearchSpaceDim.__name__)

_searchspace_registry = {ss_cls[1].name: ss_cls[1] for ss_cls in searchspace_classes}
