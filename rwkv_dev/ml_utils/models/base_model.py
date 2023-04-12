import copy
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from ml_utils.utils import get_device
from torch import nn

FN_MODEL_PREFIX = 'model_'
FN_MODEL_FILE_EXT = '.p'


class BaseModel(nn.Module, ABC):
    """BaseModel class
    Takes care of easy saving and loading.
    """

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.config = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loss_func(self, **kwargs):
        pass

    def _get_constructor_parameters(self) -> dict:
        if isinstance(self.config, dict):
            return self.config
        return asdict(self.config)

    def reset_parameters(self):
        self.apply(self.get_init_fn())

    def get_init_fn(self) -> Callable[[torch.Tensor], None]:
        return None

    @property
    def num_parameters(self) -> int:
        return torch.tensor([p.numel() for p in self.parameters()]).sum().item()

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def copy_to_cpu(self) -> "BaseModel":
        """Copy the model to CPU."""
        return copy.deepcopy(self).to(torch.device('cpu'))

    def get_checkpoint_data(self, dict_key_prefix: str = 'model_') -> Dict[str, Any]:
        checkpoint_dict = {
            f"{dict_key_prefix}state_dict": self.state_dict(),
            f"{dict_key_prefix}data": self._get_constructor_parameters(),
            f"{dict_key_prefix}name": self.__class__.__name__
        }
        return checkpoint_dict

    def save(self,
             path: Union[str, Path],
             model_name: str,
             file_extension: Optional[str] = FN_MODEL_FILE_EXT,
             dict_key_prefix: str = 'model_') -> None:
        if isinstance(path, str):
            path = Path(path)
        save_path = path / (model_name + file_extension)
        torch.save(self.get_checkpoint_data(dict_key_prefix), save_path)


    @staticmethod
    def model_save_name(idx: int, specifier: str = 'epoch', num_digits: int = -1) -> str:
        """Get a consistnet the model save name.

        Args:
            epoch (int): Epoch / iteration number.
            specifier (str, optional): A specifier for the idx. Defaults to epoch.
            num_digits (int, optional): The number of digits in the save name. Unused by default, 
                since this causes overrides when we have an overflow. Defaults to -1.

        Returns:
            str: Model save name.
        """
        if num_digits == -1:
            return f"{FN_MODEL_PREFIX}{specifier}_{idx}"
        else:
            return f"{FN_MODEL_PREFIX}{specifier}_{idx:0{num_digits}}"

    @classmethod
    def load(cls,
             path: Union[str, Path],
             model_name: str = None,
             file_extension: Optional[str] = '.p',
             device: Union[torch.device, str, int] = 'auto',
             dict_key_prefix: str = 'model_') -> 'BaseModel':
        device = get_device(device)
        if isinstance(path, str):
            path = Path(path)
        if model_name is None:
            save_path = path
        else:
            save_path = path / (model_name + file_extension)
        checkpoint = torch.load(save_path, map_location=device)
        return BaseModel.class_and_params_from_checkpoint(checkpoint=checkpoint, dict_key_prefix=dict_key_prefix)

    @classmethod
    def params_from_checkpoint(cls, checkpoint: Dict[str, Any], dict_key_prefix: str = 'model_') -> 'BaseModel':
        model = cls(**checkpoint[f"{dict_key_prefix}data"])
        model.load_state_dict(checkpoint[f"{dict_key_prefix}state_dict"])
        return model

    @staticmethod
    def class_and_params_from_checkpoint(checkpoint: Dict[str, Any], dict_key_prefix: str = 'model_') -> 'BaseModel':
        from . import get_model_class
        model_class = get_model_class(checkpoint[f"{dict_key_prefix}name"])
        model = model_class.params_from_checkpoint(checkpoint=checkpoint, dict_key_prefix=dict_key_prefix)
        return model
    