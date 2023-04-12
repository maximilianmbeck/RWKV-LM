from typing import Type

from torch import nn

_loss_registry = {
    "crossentropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
}

def get_loss(loss_name: str) -> Type[nn.Module]:
    """Returns the loss class given its name.

    Args:
        loss_name (str): The name of the loss.

    Returns:
        Type[nn.Module]: The loss class
    """
    if loss_name in _loss_registry:
        return _loss_registry[loss_name]
    else:
        assert False, f"Unknown loss name \"{loss_name}\". Available loss functions are: {str(_loss_registry.keys())}"