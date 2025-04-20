from pathlib import Path

from aces.config import TaskExtractorConfig

def load_task_cfg(criteria_fp: Path, predicates_fp: Path | None = None) -> TaskExtractorConfig:
    """
    Load the input ACES task configuration from the specified file paths.

    Args:
        criteria_fp: Path to the original task configuration file. This file is used for specifying the
            task inclusion and exclusion criteria; predicates can be included or can be loaded from a separate
            file.
        predicates_fp: Path to the predicates file. If None, only the predicates in the constraints file are
            used.

    Returns:
        The loaded task configuration object, as an ACES TaskExtractorConfig instance.

    Raises:
        FileNotFoundError: If the specified file paths do not exist.
    """
    raise NotImplementedError("This is a placeholder for the actual loading code.")

def validate_task_cfg(task_cfg: TaskExtractorConfig):
    """Validates that the given task configuration is usable in the zero-shot labeling context.

    Validation checks include:
      - Checking that the task configuration has a prediction time and label defined.
      - Checking that the task configuration is a future-prediction task.

    Args:
        task_cfg: The task configuration to validate, in ACES format.

    Raises:
        ValueError: If the task configuration is not valid for zero-shot labeling.
    """
    raise NotImplementedError("This is a placeholder for the actual validation code.")

def convert_to_zero_shot(task_cfg: TaskExtractorConfig, labeler_cfg: DictConfig):
    """Converts the task configuration to a zero-shot format by removing past and future criteria.

    Args:
        task_cfg: The task configuration to convert.
        labeler_cfg: The labeler configuration to use.

    Returns:
        A new task configuration object in zero-shot format.

    Raises:
        ValueError: If the task configuration is not valid for zero-shot labeling.
    """
    raise NotImplementedError("This is a placeholder for the actual conversion code.")


def resolve_zero_shot_task_cfg(task_cfg: DictConfig, labeler_cfg: DictConfig):
    """Resolves the task configuration for 0-shot prediction by removing past & (optionally) future criteria.

    Args:
        task_cfg: The task configuration to resolve.
        labeler_cfg: The labeler configuration to use.

    Returns:
        ???

    Raises:
        FileNotFoundError: If the specified file paths do not exist.
        ValueError: If the task configuration is invalid or cannot be resolved.
    """

    orig_cfg = load_task_cfg(Path(task_cfg.criteria_fp), task_cfg.predicates_fp)

    validate_task_cfg(orig_cfg)

    return convert_to_zero_shot(orig_cfg, labeler_cfg)
