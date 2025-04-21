from typing import Literal, NamedTuple

from aces.config import TaskExtractorConfig
from bigtree import print_tree
from omegaconf import DictConfig

WINDOW_NODE = tuple[str, Literal["start", "end"]] | tuple[Literal["trigger"], None]


class WindowNode(NamedTuple):
    name: str | Literal["trigger"]
    root: Literal["start", "end"] | None

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self.name if self.root is None else f"{self.name}.{self.root}"


def _ACES_config_timeline(task_cfg: TaskExtractorConfig) -> dict[WindowNode, set[WindowNode]]:
    """Produces a set of temporal order guarantees for the nodes in the ACES task configuration.

    Args:
        task_cfg: The task configuration to analyze.

    Returns:
        A mapping of window nodes to a set of all nodes that guaranteeably occur before them in time.
    """
    raise NotImplementedError("This is a placeholder for the actual implementation.")


def _resolve_node(task_cfg: TaskExtractorConfig, window_name: str) -> WindowNode:
    """Resolves a node in the task configuration based on the window name.

    Args:
        task_cfg: The task configuration to resolve the node from.
        window_name: The name of the window to resolve.

    Returns:
        The resolved node.

    Raises:
        ValueError: If the window name is not found in the task configuration.

    Examples:
        >>> from bigtree import print_tree
        >>> print_tree(sample_ACES_cfg.window_tree)
        trigger
        ├── input.end
        │   └── input.start
        └── gap.end
            └── target.end

    On this tree, the windows depend on the following nodes:

        >>> _resolve_node(sample_ACES_cfg, "gap")
        WindowNode(name='trigger', root=None)
        >>> _resolve_node(sample_ACES_cfg, "input")
        WindowNode(name='input', root='end')
        >>> _resolve_node(sample_ACES_cfg, "target")
        WindowNode(name='gap', root='end')

    If we pass a non-existent window name, it raises a ValueError:

        >>> _resolve_node(sample_ACES_cfg, "nonexistent")
        Traceback (most recent call last):
            ...
        ValueError: Window 'nonexistent' not found in task configuration.
    """
    if window_name not in task_cfg.windows:
        raise ValueError(f"Window '{window_name}' not found in task configuration.")

    def _get_referenced_node(window: WindowNode) -> WindowNode:
        window_cfg = task_cfg.windows[window.name]

        bound = window_cfg._parsed_start if window.root == "start" else window_cfg._parsed_end

        ref = bound["referenced"]

        if ref == "trigger":
            return WindowNode("trigger", None)
        elif ref in {"start", "end"}:
            return WindowNode(window.name, ref)
        else:
            return WindowNode(*ref.split("."))

    def _node_in_tree(window: WindowNode) -> bool:
        if window.root is None:
            return True

        window_cfg = task_cfg.windows[window.name]

        if window.root == "start":
            return window_cfg.start_endpoint_expr is not None
        else:
            return window_cfg.end_endpoint_expr is not None

    window_cfg = task_cfg.windows[window_name]
    root_node = WindowNode(window_name, window_cfg.root_node)

    while not _node_in_tree(root_node):
        root_node = _get_referenced_node(root_node)

    return root_node


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

    task_node_timeline = _ACES_config_timeline(task_cfg)

    if label_window_root not in task_node_timeline[prediction_time_window_root]:
        raise ValueError(
            f"Cannot guarantee that label node {label_window_root.node_name} occurs before "
            f"prediction time node {prediction_time_window_root.node_name} in the task configuration."
        )

    label_window_node = task_cfg.window_nodes[label_window_root.node_name]
    prediction_time_window_node = task_cfg.window_nodes[prediction_time_window_root.node_name]

    print(f"Label window node: {label_window_node.node_name}")
    print(f"Prediction time window node: {prediction_time_window_node.node_name}")
    print(f"Label window ancestors: {list(label_window_node.ancestors)}")
    print(f"Prediction time window ancestors: {list(prediction_time_window_node.ancestors)}")
    print(print_tree(task_cfg.window_tree))
    return

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
