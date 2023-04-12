from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from ml_utils.utils import get_device
from omegaconf import DictConfig, OmegaConf

FN_CONFIG = 'config.yaml'
DIR_OUTPUT_FOLDER_NAME = 'outputs'


@dataclass
class Directory:

    dir: Union[str, Path] = '.'

    config_file_name: str = FN_CONFIG
    log_file_name: str = 'output.log'
    _directories: List[Path] = field(default_factory=list)

    def _assign_dir(self) -> None:
        if isinstance(self.dir, JobDirectory):
            self.dir = Path(self.dir.dir).resolve()
        else:
            self.dir = Path(self.dir).resolve()

    def create_directories(self, exist_ok: bool = True) -> None:
        for d in self._directories:
            d.mkdir(exist_ok=exist_ok)

    def is_directory(self) -> bool:
        for d in self._directories:
            if not d.exists():
                raise ValueError(f'Directory {d} does not exist for {self.__class__.__name__}!')
        if not (self.dir / self.config_file_name).exists():
            raise ValueError(f'Config file {self.config_file_name} does not exist for {self.__class__.__name__}!')
        return True

    def save_config(self, config: Union[dict, DictConfig]) -> None:
        OmegaConf.save(config, self.dir / self.config_file_name)

    def load_config(self) -> DictConfig:
        return OmegaConf.load(self.dir / self.config_file_name)

    @property
    def log_file(self) -> Path:
        return self.dir / self.log_file_name

    def __str__(self) -> str:
        return str(self.dir)


@dataclass
class SweepDirectory(Directory):

    jobs_folder_name = DIR_OUTPUT_FOLDER_NAME

    def __post_init__(self):
        self._assign_dir()
        self.jobs_folder = self.dir / self.jobs_folder_name
        self._directories = [self.jobs_folder]

    def __repr__(self) -> str:
        return f'SweepDirectory({str(self)})'


@dataclass
class JobDirectory(Directory):

    stats_folder_name: str = 'statistics'
    figures_folder_name: str = 'figures'
    checkpoints_folder_name: str = 'checkpoints'

    def __post_init__(self):
        self._assign_dir()
        d = self.dir
        self.stats_folder = d / self.stats_folder_name
        self.figures_folder = d / self.figures_folder_name
        self.checkpoints_folder = d / self.checkpoints_folder_name

        self._directories = [self.stats_folder, self.figures_folder, self.checkpoints_folder]

    def __repr__(self) -> str:
        return f'JobDirectory({str(self)})'

    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        idx: int = -1,
                        specifier: str = '',
                        name: str = 'checkpoint',
                        file_extension: str = '.p',
                        override_path: Union[str, Path] = '') -> None:
        """Store checkpoint with pattern `{name}-{specifier}--{idx}{file_extension}`.

        Args:
            specifier (str, optional): Typically the progress measure: step or epoch. Defaults to ''.
            name (str, optional): The Filename. Defaults to 'checkpoint'.
            idx (int, optional): The index. Defaults to -1.
            file_extension (str, optional): Defaults to '.p'.
            override_path (Union[str, Path], optional): Save at this path if specified. Use standard checkpoints folder otherwise. 
                                                        Defaults to ''.
        """
        filename = self.get_checkpoint_filename(idx=idx, specifier=specifier, name=name, file_extension=file_extension)
        save_path = self.checkpoints_folder
        if override_path:
            save_path = Path(override_path)

        file = save_path / filename
        torch.save(checkpoint, file)

    def load_checkpoint(self,
                        idx: int = -1,
                        specifier: str = '',
                        name: str = 'checkpoint',
                        file_extension: str = '.p',
                        override_path: Union[str, Path] = '',
                        device: Union[torch.device, str, int] = 'cpu',
                        load_kwargs: Dict[str, Any] = {}) -> Dict[str, Any]:
        # default: load from checkpoints folder
        load_path = self.checkpoints_folder
        if override_path:
            load_path = Path(override_path)
        # construct filename
        if idx >= 0 and specifier:
            filename = self.get_checkpoint_filename(idx=idx,
                                                    specifier=specifier,
                                                    name=name,
                                                    file_extension=file_extension)
        elif idx >= 0:
            pattern = f'{name}*--{idx}{file_extension}'
            filenames = [f.name for f in load_path.glob(pattern)]
            assert len(
                filenames
            ) > 0, f'No checkpoints found for pattern `{pattern}` in directory `{str(load_path)}: {str(filenames)}`!'
            filename = filenames[0]

        if not device is None:
            device = get_device(device)

        file = load_path / filename
        return torch.load(file, map_location=device, **load_kwargs)

    def get_checkpoint_filename(self,
                                idx: int = -1,
                                specifier: str = '',
                                name: str = 'checkpoint',
                                file_extension: str = '.p') -> str:
        """Return the filename in the pattern `{name}-{specifier}--{idx}{file_extension}`."""
        filename = f'{name}'
        if specifier:
            filename += f'-{specifier}'
        if idx >= 0:
            filename += f'--{idx}'
        filename += file_extension
        return filename

    def get_checkpoint_indices(self) -> List[int]:
        idxes = [int(f.stem.split('--')[-1]) for f in self.checkpoints_folder.iterdir() if len(f.stem.split('--')) > 1]
        idxes.sort()
        return idxes

    @staticmethod
    def load_resume_checkpoint(job_dir: str, checkpoint_idx: int, device: Union[str, int] = 'cpu') -> Dict[str, Any]:
        job_directory = JobDirectory(job_dir)
        checkpoint = job_directory.load_checkpoint(idx=checkpoint_idx, device=device)
        return checkpoint