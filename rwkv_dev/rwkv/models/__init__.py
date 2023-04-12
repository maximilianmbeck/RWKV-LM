

from typing import Dict, Type, Union

from dacite import from_dict
from ml_utils.config import NameAndKwargs
from ml_utils.models.base_model import BaseModel
from omegaconf import DictConfig, OmegaConf

from .rwkv_model import RWKV

_model_registry = {'rwkv': RWKV}

def get_model_class(name: str) -> Type[BaseModel]:
    if name in _model_registry:
        return _model_registry[name]
    else:
        assert False, f"Unknown model name \"{name}\". Available models are: {str(_model_registry.keys())}"

def get_model(config: Union[Dict, DictConfig, NameAndKwargs]) -> BaseModel:
    if not isinstance(config, NameAndKwargs):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)
        cfg = from_dict(data_class=NameAndKwargs, data=config)
    else:
        cfg = config
    model_class = get_model_class(cfg.name)
    model_cfg = from_dict(data_class=model_class.config_class, data=cfg.kwargs)
    return model_class(model_cfg)
    

