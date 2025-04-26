"""Utilities to define and manipulate zero-shot labeling task configurations from ACES configurations."""

import copy
from io import StringIO

from aces.config import TaskExtractorConfig
from bigtree import print_tree
from omegaconf import DictConfig

from .aces_utils import WindowNode, ZeroShotTaskConfig, _resolve_node


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


def convert_to_zero_shot(
    task_cfg: TaskExtractorConfig, labeler_cfg: DictConfig | None = None
) -> ZeroShotTaskConfig:
    """Converts the task configuration to a zero-shot format by removing past and future criteria.

    For zero-shot conversion, we construct an implicit "zero-shot task config" that functionally describes the
    sequence of window endpoints from the prediction time to the end of (a) the task config, or (b) the label
    config, if the labeler arguments indicate that future criteria should be ignored.

    > [!WARNING]
    > this implementation is not complete yet.

    Args:
        task_cfg: The task configuration to convert.
        labeler_cfg: The labeler configuration to use.

    Returns:
        A new task configuration object in zero-shot format.
    """

    if labeler_cfg:
        raise NotImplementedError("This is not supported yet.")

    prediction_time_window_name = task_cfg.index_timestamp_window
    prediction_time_window_cfg = task_cfg.windows[prediction_time_window_name]
    prediction_time_window = WindowNode(
        prediction_time_window_name, prediction_time_window_cfg.index_timestamp
    )
    new_root = _resolve_node(task_cfg, root_node=prediction_time_window)

    label_window = task_cfg.label_window
    label_window_node = _resolve_node(task_cfg, root_node=WindowNode(label_window, "end"))

    new_task_cfg = ZeroShotTaskConfig(
        predicates=copy.deepcopy(task_cfg.predicates),
        windows=copy.deepcopy(task_cfg.windows),
        trigger=copy.deepcopy(task_cfg.trigger),
        label_window=copy.deepcopy(task_cfg.label_window),
        index_timestamp_window=copy.deepcopy(task_cfg.index_timestamp_window),
    )

    new_root_node = new_task_cfg.window_nodes[new_root.node_name]
    new_root_node.name = prediction_time_window.node_name
    new_root_node.endpoint_expr = None

    if prediction_time_window.root == "end":
        new_task_cfg.windows[prediction_time_window_name].has = None

    new_task_cfg.window_tree = new_root_node

    new_label_node = new_task_cfg.window_nodes[label_window_node.node_name]

    label_rel_nodes = set(new_label_node.ancestors) | {new_label_node} | set(new_label_node.descendants)
    prediction_time_rel_nodes = set(new_root_node.descendants) | {new_root_node}

    allowed_nodes = prediction_time_rel_nodes & label_rel_nodes

    new_task_cfg.window_nodes = {
        v.node_name: v for v in new_task_cfg.window_nodes.values() if v in allowed_nodes
    }

    for node in new_task_cfg.window_nodes.values():
        node.parent = node.parent if node.parent in allowed_nodes else None
        node.children = [n for n in node.children if n in allowed_nodes]

    return new_task_cfg


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
