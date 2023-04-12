from typing import Any, Dict, Iterable, Tuple, Type

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from ml_utilities.config import NameAndKwargs

_optim_registry = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop
}

_lr_scheduler_registry = {
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    # "linear_lr": lr_scheduler.LinearLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
}

def get_optim(optim_name: str) -> Type[optim.Optimizer]:
    """Returns the optimizer class given its name.

    Args:
        optim_name (str): The name of the optimizer.

    Returns:
        optim.Optimizer: The pytorch optimizer class.
    """
    if optim_name in _optim_registry:
        return _optim_registry[optim_name]
    else:
        assert False, f"Unkown optimizer name \"{optim_name}\". Available optimizers are: {str(_optim_registry.keys())}"


def get_lr_scheduler(lr_scheduler_name: str) -> Type[lr_scheduler._LRScheduler]:
    """Returns the learning rate scheduler class given its name.

    Args:
        lr_scheduler_name (str): The name of the lr scheduler.

    Returns:
        lr_scheduler._LRScheduler: The pytorch lr scheduler class.
    """
    if lr_scheduler_name in _lr_scheduler_registry:
        return _lr_scheduler_registry[lr_scheduler_name]
    else:
        assert False, f"Unkown lr scheduler name \"{lr_scheduler_name}\". Available lr scheduler are: {str(_lr_scheduler_registry.keys())}"


def create_optimizer_and_scheduler_from_config(model_params: Iterable[nn.parameter.Parameter],
                                   optimizer_cfg: NameAndKwargs, lr_scheduler_cfg: NameAndKwargs = None) -> Tuple[torch.optim.
                                                                                        Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Creates an optimizer and lr scheduler.

    Args:
        model_params (Iterable[nn.parameter.Parameter]): The model parameters obtained by model.parameters()
        optimizer (NameAndKwargs): 
        lr_scheduler (NameAndKwargs, optional): Defaults to None.

    Raises:
        ValueError: If lr scheduler name is given, but lr_scheduler_kwargs are missing.

    Returns:
        Tuple[torch.optim. Optimizer, torch.optim.lr_scheduler._LRScheduler]: Optimizer and lr scheduler.
    """
    optim_class = get_optim(optim_name=optimizer_cfg.name)
    optimizer_cfg = optim_class(model_params, **optimizer_cfg.kwargs)
    # HACK: set initial lr manually. 
    # This is usually done in the learning rate scheduler, when it is initialized and no last epoch is given
    # In this way it is ensured that you only resume to a learning rate schedule, when the optimizer is initialized
    # via load_state_dict. However here we want to do it in a hacky way where we also want to resume 
    # from a pretrained model checkpoint but with freshly initialized optimizer states. 
    # We can still load the optimizer state dict afterwards, then this manual setting of the initial_lr
    # will be overriden. 
    # See torch.optim.lr_scheduler class _LRScheduler
    for group in optimizer_cfg.param_groups:
        group.setdefault('initial_lr', group['lr'])

    if lr_scheduler_cfg is not None:
        lr_scheduler_class = get_lr_scheduler(lr_scheduler_cfg.name)
        if lr_scheduler_cfg.kwargs is None:
            raise ValueError('Scheduler name given, but scheduler kwargs are unspecified!')
        lr_scheduler_cfg = lr_scheduler_class(optimizer=optimizer_cfg, **lr_scheduler_cfg.kwargs)
    else:
        lr_scheduler_cfg = None

    return optimizer_cfg, lr_scheduler_cfg