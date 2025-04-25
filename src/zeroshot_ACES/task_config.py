"""Utilities to define and manipulate zero-shot labeling task configurations from ACES configurations."""

from io import StringIO

from aces.config import TaskExtractorConfig
from bigtree import print_tree
from omegaconf import DictConfig

from .aces_utils import _resolve_node


def validate_task_cfg(task_cfg: TaskExtractorConfig):
    """Validates that the given task configuration is usable in the zero-shot labeling context.

    Validation checks include:
      - Checking that the task configuration has a prediction time and label defined.
      - Checking that the task configuration is a future-prediction task.

    Args:
        task_cfg: The task configuration to validate, in ACES format.

    Raises:
        ValueError: If the task configuration is not valid for zero-shot labeling.

    Examples:
        >>> validate_task_cfg(sample_ACES_cfg)
    """

    for k in ("label_window", "index_timestamp_window"):
        val = getattr(task_cfg, k)
        if val is None:
            raise ValueError(f"The task configuration must have a {k} defined.")
        elif val == "trigger":
            raise ValueError(f"The task configuration must not have a {k} set to 'trigger'.")

    label_window_root = _resolve_node(task_cfg, task_cfg.label_window)
    prediction_time_window_root = _resolve_node(task_cfg, task_cfg.index_timestamp_window)

    label_window_node = task_cfg.window_nodes[label_window_root.node_name]
    prediction_time_window_node = task_cfg.window_nodes[prediction_time_window_root.node_name]

    if prediction_time_window_node not in label_window_node.ancestors:
        strio = StringIO()
        tree_str = print_tree(task_cfg.window_tree, file=strio)
        raise ValueError(
            "zeroshot_ACES only supports task configs where the prediction time node is an ancestor of the "
            f"label node. Here, the prediction time window node ({prediction_time_window_node.node_name}) "
            f"is not an ancestor of the label window node ({label_window_node.node_name}). Got tree:\n"
            f"{tree_str.get_value()}"
        )


def convert_to_zero_shot(task_cfg: TaskExtractorConfig, labeler_cfg: DictConfig | None = None):
    """Converts the task configuration to a zero-shot format by removing past and future criteria.

    For zero-shot conversion, we construct an implicit "zero-shot task config" that functionally describes the
    sequence of window endpoints from the prediction time to the end of (a) the task config, or (b) the label
    config, if the labeler arguments indicate that future criteria should be ignored.

    > [!WARNING]
    > this implementation is not correct yet.

    Args:
        task_cfg: The task configuration to convert.
        labeler_cfg: The labeler configuration to use.

    Returns:
        A new task configuration object in zero-shot format.
    """

    raise NotImplementedError("This is not supported yet.")


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

    Examples:
        >>> task_cfg = DictConfig({
        ...     "criteria_fp": str(sample_task_criteria_fp), "predicates_fp": str(sample_predicates_fp)
        ... })
        >>> labeler_cfg = DictConfig({})
        >>> resolve_zero_shot_task_cfg(task_cfg, labeler_cfg)
    """

    orig_cfg = TaskExtractorConfig.load(task_cfg.criteria_fp, task_cfg.predicates_fp)

    validate_task_cfg(orig_cfg)

    return convert_to_zero_shot(orig_cfg, labeler_cfg)
