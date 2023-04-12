import dacite
from dacite import from_dict
from ml_utils.run_utils.runner import setup_directory
from omegaconf import DictConfig, OmegaConf
from rwkv.models import get_model
from rwkv.trainer import UniversalRwkvTrainer
from torch.utils import data


def run_job(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    setup_directory('job', cfg)

    cfg = from_dict(data_class=UniversalRwkvTrainer.config_class,
                    data=OmegaConf.to_container(cfg.config))

    # create model
    model = get_model(cfg.model)

    # create dataloader
    # TODO make this configurable
    def get_dataloaders():
        from rwkv.data.enwik8_dataset import EnWik8
        ds = EnWik8(
            datafile=
            '/system/user/beck/pwbeck/projects/rwkv/RWKV-LM/data/enwik8')
        data_loader = data.DataLoader(ds,
                                      shuffle=False,
                                      pin_memory=True,
                                      batch_size=12,
                                      num_workers=1,
                                      persistent_workers=True,
                                      drop_last=True)
        return data_loader, None

    # create metrics
    # TODO make this configurable
    def get_metrics():
        return None, None

    # TODO add a mapping from dataloader to model input

    trainer = UniversalRwkvTrainer(config=cfg,
                                   model=model,
                                   get_dataloaders=get_dataloaders,
                                   get_metrics=get_metrics)
    trainer.run()
