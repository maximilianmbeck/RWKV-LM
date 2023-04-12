from typing import Type

from .basetrainer import BaseTrainer
from .universalbasetrainer import UniversalBaseTrainer

_trainer_registry = {'supervised-universal': UniversalBaseTrainer}

def get_trainer_class(training_setup: str) -> Type[BaseTrainer]:
    if training_setup in _trainer_registry:
        return _trainer_registry[training_setup]
    else:
        assert False, f"Unknown training setup \"{training_setup}\". Available training setups are: {str(_trainer_registry.keys())}"

def register_trainer(training_setup: str, trainer_class: Type[BaseTrainer]):
    _trainer_registry[training_setup] = trainer_class