from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class NameAndKwargs:
    name: str
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    entity: Optional[str]
    project_name: str
    experiment_tag: Optional[str]
    experiment_type: Optional[str]
    experiment_name: str
    experiment_dir: Optional[str]
    experiment_notes: Optional[str]
    seed: int
    gpu_id: int
    job_name: Optional[str]
    hostname: Optional[str]

@dataclass
class ResumeTrainingConfig:
    job_dir: str
    checkpoint_idx: int

@dataclass
class TrainerConfig:
    optimizer: NameAndKwargs
    training_setup: Optional[str] = None
    batch_size: Optional[int] = -1
    n_steps: Optional[Union[int, float]] = -1
    n_epochs: Optional[Union[int, float]] = -1
    val_every: Union[int, float] = 1
    save_every: Union[int, float] = 0
    save_every_idxes: List[Union[int, float]] = field(default_factory=list)
    early_stopping_patience: Union[int, float] = -1
    num_workers: int = 0
    resume_training: ResumeTrainingConfig = None
    lr_scheduler: Optional[NameAndKwargs] = None
    loss: str = ''
    metrics: List[Any] = field(default_factory=list)
    additional_cfg: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    experiment_data: ExperimentConfig
    trainer: TrainerConfig
    model: NameAndKwargs
    data: NameAndKwargs
    wandb: Optional[Dict[str, Any]] = None