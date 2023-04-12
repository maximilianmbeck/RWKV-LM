from typing import Union
from pathlib import Path
from .result_loader import SweepResult, JobResult


def create_job_output_loader(output_dir: Path) -> Union[SweepResult, JobResult]:
    """Create an output loader. 

    Args:
        output_dir (Path): The output directory.

    Returns:
        Union[SweepResult, JobResult]: If directory contains sweep results, return SweepResult object 
                                       or JobResult object, otherwise
    """
    try:
        output_result = SweepResult(output_dir)
    except:
        output_result = JobResult(output_dir)
    return output_result