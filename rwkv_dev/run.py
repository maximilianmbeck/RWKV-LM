from pathlib import Path

from ml_utils.utils import get_config, get_config_file_from_cli
from rwkv.run_rwkv import run_job

if __name__=='__main__':
    import wandb
    wandb.login(host="https://wandb.ml.jku.at")
    # wandb.login(host="https://api.wandb.ai")
    cfg_file = get_config_file_from_cli(config_folder='configs', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run_job(cfg)