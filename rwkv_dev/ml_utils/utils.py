import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    # For `use_deterministic_algorithms` need to set CUBLAS_WORKSPACE_CONFIG environment variable
    # when using CUDA version 10.2 or greater
    # https://pytorch.org/docs/stable/notes/randomness#avoiding-nondeterministic-algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = str(':4096:8')
    torch.use_deterministic_algorithms(mode=True)

    np.random.seed(seed)  # only sets seed for old API
    # take care when using the new API, i.e. default_rng()!
    # default_rng(seed=None) always pulls a fresh, unpredictable entropy from the OS
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[torch.device, str, int]) -> torch.device:
    if device == "auto":
        device = "cuda"
    if isinstance(device, int):
        if device < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device}")
    else:
        device = torch.device(device)

    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        LOGGER.warn(f"Device '{str(device)}' is not available! Using cpu now.")
        return torch.device("cpu")
    return device


def archive_code(repo_dir: Path = None,
                 save_dir: Path = None,
                 code_modules: Union[str, List[str]] = None,
                 filename: str = 'code',
                 format: str = 'zip') -> None:
    """Archives the code folders / files. 

    Args:
        repo_dir (Path): Location of the repo (toplevel directory). 
                         If None uses the original working directory (working directory of the main run file). 
                         Defaults to None. 
        save_dir (Path): Location, where the archive file will be saved to. 
                         If None, uses the current working directory (which is typically set by hydra). Defaults to None.
        code_modules (Union[str, List[str]], optional): Name of the code modules to archive. If None, infers the code module name from repo name. Defaults to None.
        filename (str, optional): Name of the archive file (without file format-specific extension). Defaults to 'code'.
        format (str, optional): Archive format. Defaults to 'zip'.
    """
    if not repo_dir:
        repo_dir = Path().cwd()
    # preprocess code_modules arguments to be of type List[str]
    if not code_modules:
        # no code modules to archive given -> assume module has same name as repo
        code_modules = [repo_dir.stem]
    else:
        # code modules were given, make sure that it is a list
        if isinstance(code_modules, str):
            code_modules = [code_modules]
        assert isinstance(code_modules, list) and isinstance(code_modules[0], str)
    if len(code_modules) > 1:
        # TODO use zipfile instead of shutil.make_archive
        raise NotImplementedError('Archiving multiple folders/files not supported yet!')

    if not save_dir:
        save_dir = Path().cwd()

    root_dir = repo_dir / code_modules[0]
    shutil.make_archive(base_name=save_dir / filename, format=format, root_dir=root_dir)


def get_git_hash() -> Optional[str]:
    """Try to get the git hash of the current repository.

    Returns
    -------
    Optional[str]
        Git hash if git is installed and the project is cloned via git, None otherwise.
    """
    current_dir = str(Path(__file__).absolute().parent)
    try:
        if subprocess.call(['git', '-C', current_dir, 'branch'], stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0:
            git_output = subprocess.check_output(['git', '-C', current_dir, 'describe', '--always'])
            return git_output.strip().decode('ascii')
    except OSError:
        pass  # git is probably not installed
    return None


def setup_experiment_dir(experiment_name: str,
                         base_dir: Union[str, Path] = None,
                         outputs_folder_name: str = 'outputs') -> Path:
    """Creates a run directory with unique name.

    Args:
        experiment_name (str): Name of the experiment
        outputs_dir (Union[str, Path], optional): Path to the project directory. If unspecified, uses cwd / 'outputs' / ... Defaults to None.
        outputs_folder_name (str, optional): Name of the outputs folder. Defaults to 'outputs'.

    Returns:
        Path: the path to the created run directory
    """
    from rwkv_dev.ml_utils.time_utils import FORMAT_DATETIME_SHORT
    now = datetime.now().strftime(FORMAT_DATETIME_SHORT)

    run_name = f'{experiment_name}--{now}'

    if base_dir is None:
        exp_dir = Path().cwd() / outputs_folder_name / run_name
    else:
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        exp_dir = base_dir / outputs_folder_name / run_name

    exp_dir.mkdir(parents=True, exist_ok=False)

    return exp_dir


def setup_logging(logfile: str = "output.log"):
    """Initialize logging to `log_file` and stdout.     

    Args:
        log_file (str, optional): Name of the log file. Defaults to "output.log".
    """
    from rwkv_dev.ml_utils.logger import FORMAT_LOGGING

    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(handlers=[file_handler, stdout_handler], level=logging.INFO, format=FORMAT_LOGGING, force=True)

    setup_exception_logging()

    LOGGER.info(f'Logging to {logfile} initialized.')


def setup_exception_logging():
    """Make sure that uncaught exceptions are logged with the logging."""

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception('Uncaught exception', exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging


def save_dict_as_json(path: Union[str, Path], filename: str, dictionary: dict) -> None:
    """Save a dictionary as .json file."""
    if isinstance(path, str):
        path = Path(path)
    save_path = path / (filename + ".json")
    with open(save_path, "w") as f:
        json.dump(dictionary, f)


def save_dict_as_yml(path: Union[str, Path], filename: str, dictionary: dict) -> None:
    """Save a dictionary as .yaml file."""
    if isinstance(path, str):
        path = Path(path)
    save_path = path / (filename + ".yaml")
    OmegaConf.save(OmegaConf.create(dictionary), save_path)


def get_config(config: Union[str, Path, dict, DictConfig]) -> DictConfig:
    """Creates a config from different sources."""
    if isinstance(config, (dict, DictConfig)):
        return OmegaConf.create(config)
    elif isinstance(config, (str, Path)):
        return OmegaConf.load(config)
    else:
        raise ValueError(f'Config type {type(config)} not supported.')


def get_config_file_from_cli(script_file: Path, config_folder: str = '') -> Path:
    """Read the config file from command line arguments `--config-name` or `-cn` and `--config-path` or `-cp`. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', '-cn', type=str)
    parser.add_argument('--config-path', '-cp', type=str, default=None)
    args = vars(parser.parse_args())

    config_folder_dir = None
    if config_folder:
        file_dir = script_file.parent.resolve()
        config_folder_dir = file_dir / config_folder
    if args['config_path'] is not None:
        config_folder_dir = Path(args['config_path'])
        assert config_folder_dir.is_absolute()
    assert config_folder_dir is not None, 'No config folder directory provided. Specify via function or command-line arguments.'

    config_file = config_folder_dir / args['config_name']
    if not config_file.exists():
        raise ValueError(f'Config file or folder not found: `{str(config_file)}`')
    return config_file


def convert_dict_to_python_types(d: Dict[str, Any],
                                 single_vals_as_list: bool = False,
                                 objects_as_str: bool = True) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Tries to convert a dictionary of ints, floats, tensors, arrays and other objects to a dict of plane python types.
    
    Args:
        single_vals_as_list (bool, optional): If true, single values will be inserted in a list with a single entry.
        objects_as_str (bool, optional): If true, objects will be converted to string.
    """
    return {k: convert_to_python_types(v, single_vals_as_list, objects_as_str) for k, v in d.items()}


def convert_to_python_types(x: Any,
                            single_vals_as_list: bool = False,
                            objects_as_str: bool = True) -> Union[Any, List[Any]]:
    """Takes an array or tensor as input and returns the content as (nested) list(s).
    
    Args:
        single_vals_as_list (bool, optional): If true, single values will be inserted in a list with a single entry. 
        objects_as_str (bool, optional): If true, objects will be converted to string.
    """
    if isinstance(x, (float, int, str, bool)):
        val = x
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        try:
            val = x.item()
        except ValueError:
            val = x.tolist()  # avoid recursion and use standard API
    else:
        if objects_as_str:
            val = str(x)
        else:
            raise ValueError(f'Object of type {type(x)} not supported.')
    if single_vals_as_list and not isinstance(val, list):
        val = [val]
    return val


def zip_strict(*iterables: Iterable) -> Iterable:
    """``zip()`` function but enforces that iterables are of equal length.
    Raises:
        ValueError: If iterables not of equal length.
    """
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def remove_toplevelkeys_from_dictconfig(dict_cfg: DictConfig, keys: List[str]) -> DictConfig:
    assert isinstance(dict_cfg,
                      DictConfig), f'The given dict is not a `DictConfig`. For python dicts consider .pop() or del.'

    new_dict_cfg = OmegaConf.create({})
    for k in dict_cfg:
        if not k in keys:
            OmegaConf.update(new_dict_cfg, key=k, value=dict_cfg[k])
    return new_dict_cfg


def convert_to_simple_str(l: Iterable, separator: str = '_') -> str:
    """Can be used to make a string out of a list."""
    if not isinstance(l, Iterable):
        l = [l]
    s = ''
    for it in l:
        s += str(it) if s == '' else f'{separator}{str(it)}'
    return s


def convert_listofdicts_to_dictoflists(l: List[Dict[str, Any]],
                                       key_prefix: str = '',
                                       convert_vals_to_python_types: bool = True,
                                       sort_lists: bool = False) -> Dict[str, List[Any]]:
    # TODO jit
    if not l:  # list empty
        return {}

    keys = set(l[0].keys())
    ret_dict = {f'{key_prefix}{k}': [] for k in keys}
    for d in l:
        assert keys == set(d.keys()), 'Dicts in given list have different keys!'
        for k in keys:
            v = convert_to_python_types(d[k]) if convert_vals_to_python_types else d[k]
            ret_dict[k].append(v)
    if sort_lists:
        for k in ret_dict:
            ret_dict[k].sort()
    return ret_dict


def make_str_filename(possible_filename: str) -> str:
    """Removes or replaces all unallowed tokens in a file path."""
    return possible_filename.replace('[', '').replace(']', '').replace(' ', '').replace(',', '_').replace("'", '')


def flatten_hierarchical_dict(hierarchical_dict: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Flattens a hierarchical dict.

    Args:
        hierarchical_dict (Union[DictConfig, dict]): Hierarchical dictionary with string keys only.

    Returns:
        Dict[str, Any]: Dictionary with keys as dot-separated strings.

    Example:
    Input: 
    `{'data': {'dataset_kwargs': {'rotation_angle': 0.0}}, 'trainer': {'batch_size': 128}}`
    Output: 
    `{'data.dataset_kwargs.rotation_angle': 0.0, 'trainer.batch_size': 128}`
    """
    dotdict = {}
    for k, v in hierarchical_dict.items():
        assert isinstance(k, str)
        if isinstance(v, (DictConfig, dict)):
            retdict = flatten_hierarchical_dict(v)
            dotdict.update({f'{k}.{rk}': rv for rk, rv in retdict.items()})
        else:
            dotdict[k] = v
    return dotdict


def hyp_param_cfg_to_str(hyp_param_cfg: Union[DictConfig, Dict[str, Any]], string_len_limit_for_val: int = 100) -> str:
    """Create a string representation for a single hyperparameter configuration in  a sweep.
    Any parameters that are stored as `bool`, `int` or `float` are converted to a string
    with pythons float presentation `g`. See https://docs.python.org/3/library/string.html#format-specification-mini-language. 

    Args:
        hyp_param_cfg (Union[DictConfig, Dict[str, Any]]): They hyperparameter config.
        string_len_limit_for_val (int, optional): Limit the length of the string representation of a value. Defaults to 100.

    Returns:
        str: String representation for hyperparameters.
    """
    if not hyp_param_cfg:
        return ''
    hyp_params = flatten_hierarchical_dict(hyp_param_cfg)
    hyp_params = convert_dict_to_python_types(hyp_params)
    param_str = ''
    for param, val in hyp_params.items():
        assert isinstance(param, str), f'Hyperparameter `{param}` is not of type `str`!'
        # convert bool, int and float to string with proper float representation
        if isinstance(val, (bool, int, float)):
            val = f'{val:g}'
        if isinstance(val, (str)):
            if param_str:
                param_str += '-'
            if len(val) > string_len_limit_for_val:
                LOGGER.warning(f'Value for parameter {param} is longer than {string_len_limit_for_val} characters. Truncating.')
            param_str += f"{param.split('.')[-1]}-{val[:string_len_limit_for_val]}"
        else:
            raise ValueError(
                f'Cannot create parameter string. Parameter {param} has a value which is not of type bool, str, int or float. (Value: {val}).'
            )
    return param_str


def sweep_param_cfg_to_str(sweep_param_cfg: Union[DictConfig, Dict[str, Any]]) -> str:
    """Create a string representation for a sweep configuration. 

    Args:
        sweep_param_cfg (Union[DictConfig, Dict[str, Any]]): The sweep configuration

    Returns:
        str: String representation for sweep.
    """
    from rwkv_dev.ml_utils.run_utils.sweep import SWEEP_AXES_KEY, SWEEP_TYPE_KEY

    sweep_type = sweep_param_cfg[SWEEP_TYPE_KEY]
    sweep_axes = sweep_param_cfg[SWEEP_AXES_KEY]
    sweep_str = f'{sweep_type}'
    for sax in sweep_axes:
        param: str = sax['parameter']
        vals = sax['vals']
        param_name = param.split('.')[-1]
        val_str = vals
        # if isinstance(vals, (list, ListConfig)):
        #     val_str = convert_to_simple_str(vals)
        sweep_str += f'--{param_name}-{val_str}'
    return sweep_str


def match_number_list_to_interval(number_list: List[Union[int, float]],
                                  interval: Union[int, float],
                                  endpoint: bool = True,
                                  number_list_sorted: bool = True) -> List[Union[int, float]]:
    assert number_list, 'Empty list given!'
    if number_list_sorted and len(number_list) > 1:
        assert number_list[1] >= number_list[0], 'List not sorted!'
    else:
        number_list.sort()
    matched_list = []
    matching_number = number_list[0]
    i = 0

    for i, n in enumerate(number_list):
        if i == len(number_list) - 1:
            if endpoint:
                matched_list.append(n)
            break
        if n <= matching_number and matching_number <= number_list[i + 1]:
            matched_list.append(n)
            matching_number = n + interval

    return matched_list
