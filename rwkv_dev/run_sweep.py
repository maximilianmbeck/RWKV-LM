from pathlib import Path

from omegaconf import DictConfig

from rwkv_dev.ml_utils.run_utils.runner import run_sweep
from rwkv_dev.ml_utils.utils import get_config, get_config_file_from_cli


def run(cfg: DictConfig):
    run_sweep(cfg)

if __name__=='__main__':
    cfg_file = get_config_file_from_cli(config_folder='configs', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run(cfg)