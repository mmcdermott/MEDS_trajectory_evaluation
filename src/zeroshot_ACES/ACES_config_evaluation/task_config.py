"""Utilities to define and manipulate zero-shot labeling task configurations from ACES configurations."""

import copy
from io import StringIO

from aces.config import TaskExtractorConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import print_tree
from omegaconf import DictConfig

from ..aces_utils import WindowNode, ZeroShotTaskConfig, _resolve_node


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


def _strip_to_rel_windows(task_cfg: TaskExtractorConfig) -> ZeroShotTaskConfig:
    """Strips all windows not direct descendants of the prediction time node and the tree for zero-shot use.

    This ensures the zero-shot config does not rely on other windows outside the scope of the zero-shot
    evaluation setting.

    Args:
        task_cfg: The task configuration to convert.

    Returns:
        A zero-shot task configuration with only the relevant windows for zero-shot labeling.
    """

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
    new_root_node.constraints = {}

    if prediction_time_window.root == "end":
        new_task_cfg.windows[prediction_time_window_name].has = {}

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


def collapse_temporal_gap_windows(task_cfg: ZeroShotTaskConfig) -> ZeroShotTaskConfig:
    """Collapses the temporal gap windows in the task configuration between the prediction time and the label.

    Args:
        task_cfg: The task configuration to collapse.

    Returns:
        The collapsed task configuration with the temporal gap windows maximally collapsed.
    """

    label_window_node = _resolve_node(task_cfg, root_node=WindowNode(task_cfg.label_window, "end"))
    root_to_label = list(task_cfg.window_nodes[label_window_node.node_name].node_path)

    label_to_root = root_to_label[::-1]

    new_label_to_root = [label_to_root.pop(0)]

    while label_to_root:
        node = label_to_root[0]

        if node.constraints:
            break

        match getattr(node, "endpoint_expr", None):
            case ToEventWindowBounds() | None:
                break
            case TemporalWindowBounds() as time_bound:
                label_to_root.pop(0)
                new_label_to_root[-1].parent = node.parent
                node.parent.children = tuple(n for n in node.parent.children if n != node)
                if isinstance(new_label_to_root[-1].endpoint_expr, TemporalWindowBounds):
                    new_label_to_root[-1].endpoint_expr.window_size += time_bound.window_size
            case _ as other:
                raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")

    new_label_to_root.extend(label_to_root)

    return task_cfg


def remove_post_label_windows(task_cfg: ZeroShotTaskConfig) -> ZeroShotTaskConfig:
    """Removes all windows that are descendants of the label window end node.

    This is useful for removing any censoring protection nodes that are not needed for zero-shot labeling.

    Args:
        task_cfg: The task configuration to modify.

    Returns:
        The modified task configuration with the post-label windows removed.
    """

    label_window_node = _resolve_node(task_cfg, root_node=WindowNode(task_cfg.label_window, "end"))
    label_window_node = task_cfg.window_nodes[label_window_node.node_name]

    descendant_names = [n.node_name for n in label_window_node.descendants]

    label_window_node.children = ()
    for n in descendant_names:
        task_cfg.window_nodes.pop(n, None)

    return task_cfg


ALLOWED_RELAXATIONS = {"remove_all_criteria", "collapse_temporal_gap_windows", "remove_post_label_windows"}


def convert_to_zero_shot(
    task_cfg: TaskExtractorConfig, labeler_cfg: DictConfig | None = None
) -> ZeroShotTaskConfig:
    """Converts the given task configuration to a zero-shot task configuration.

    This function modifies the task configuration to ensure it is suitable for zero-shot labeling. This
    includes both window/tree pruning and relationship modification per the labeler config.

    Args:
        task_cfg: The task configuration to convert.
        labeler_cfg: The labeler configuration to use.

    Returns:
        A zero-shot task configuration with the relevant windows and relationships for zero-shot labeling.
    """

    zero_shot_cfg = _strip_to_rel_windows(task_cfg)

    if labeler_cfg is None:
        labeler_cfg = {}
    labeler_cfg = copy.deepcopy(labeler_cfg)

    if labeler_cfg.keys() - ALLOWED_RELAXATIONS:
        raise ValueError(f"Unexpected keys in labeler config: {labeler_cfg.keys() - ALLOWED_RELAXATIONS}")

    if labeler_cfg.get("remove_all_criteria", False):
        for window in zero_shot_cfg.windows.values():
            window.has = {}
        for node in zero_shot_cfg.window_nodes.values():
            node.constraints = {}

    if labeler_cfg.get("collapse_temporal_gap_windows", False):
        zero_shot_cfg = collapse_temporal_gap_windows(zero_shot_cfg)

    if labeler_cfg.get("remove_post_label_windows", False):
        zero_shot_cfg = remove_post_label_windows(zero_shot_cfg)

    return zero_shot_cfg


def resolve_zero_shot_task_cfg(task_cfg: DictConfig, labeler_cfg: DictConfig) -> ZeroShotTaskConfig:
    """Resolves the task configuration for 0-shot prediction by removing past & (optionally) future criteria.

    Args:
        task_cfg: The task configuration to resolve.
        labeler_cfg: The labeler configuration to use.

    Returns:
        A zero-shot task configuration with the relevant windows and relationships for zero-shot labeling.

    Raises:
        FileNotFoundError: If the specified file paths do not exist.
        ValueError: If the task configuration is invalid or cannot be resolved.
    """

    orig_cfg = TaskExtractorConfig.load(task_cfg.criteria_fp, task_cfg.predicates_fp)

    validate_task_cfg(orig_cfg)

    return convert_to_zero_shot(orig_cfg, labeler_cfg)
