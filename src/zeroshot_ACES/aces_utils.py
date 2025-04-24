"""Utilities to work with ACES configuration objects more easily."""

from typing import Literal, NamedTuple

from aces.config import TaskExtractorConfig, WindowConfig


class WindowNode(NamedTuple):
    """A window endpoint node in the ACES configuration file.

    You can reference `node_name` to retrieve the resolved name of the node in the ACES tree.

    Attributes:
        name: The name of the window.
        root: The root node of the window, which can be "start", "end", or None. None is only allowed in the
            case of the trigger node.

    Examples:
        >>> N = WindowNode("foo", "end")
        >>> print(f"name: {N.name}, root: {N.root}, node_name: {N.node_name}")
        name: foo, root: end, node_name: foo.end
        >>> N = WindowNode("trigger", None)
        >>> print(f"name: {N.name}, root: {N.root}, node_name: {N.node_name}")
        name: trigger, root: None, node_name: trigger
    """

    name: str | Literal["trigger"]
    root: Literal["start", "end"] | None

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self.name if self.root is None else f"{self.name}.{self.root}"


def _get_referenced_node(windows: dict[str, WindowConfig], window: WindowNode) -> WindowNode:
    """Identifies the node in the windows mapping that the given window node refers to.

    Args:
        windows: A dictionary mapping window name to ACES window configurations.
        window: The query window node (tuple of window name and endpoint).

    Returns:
        All (non-trigger) nodes in the ACES windows tree refer to another node. This function returns the
        identifier of the window node to which the passed window node refers. This returned node need not be
        preserved in the final tree; this merely resolves the start vs. end syntax used in the ACES task
        configuration.

    Examples:
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ...     "post_target": WindowConfig("target.end", "start + 24h", False, True),
        ...     "weird_window": WindowConfig("end - 24h", "post_target.start", False, False),
        ... }
        >>> _get_referenced_node(windows, WindowNode("weird_window", "start"))
        WindowNode(name='weird_window', root='end')
        >>> _get_referenced_node(windows, WindowNode("weird_window", "end"))
        WindowNode(name='post_target', root='start')
        >>> _get_referenced_node(windows, WindowNode("post_target", "start"))
        WindowNode(name='target', root='end')
        >>> _get_referenced_node(windows, WindowNode("input", "start"))
        WindowNode(name='input', root='end')
        >>> _get_referenced_node(windows, WindowNode("gap", "end"))
        WindowNode(name='gap', root='start')
        >>> _get_referenced_node(windows, WindowNode("target", "start"))
        WindowNode(name='gap', root='end')
        >>> _get_referenced_node(windows, WindowNode("gap", "start"))
        WindowNode(name='trigger', root=None)
        >>> _get_referenced_node(windows, WindowNode("post_target", "end"))
        WindowNode(name='post_target', root='start')
    """

    window_cfg = windows[window.name]

    bound = window_cfg._parsed_start if window.root == "start" else window_cfg._parsed_end

    ref = bound["referenced"]

    if ref == "trigger":
        return WindowNode("trigger", None)
    elif ref in {"start", "end"}:
        return WindowNode(window.name, ref)
    else:
        return WindowNode(*ref.split("."))


def _node_in_tree(windows: dict[str, WindowConfig], window: WindowNode) -> bool:
    """Checks if the node will be in the ACES final tree.

    ACES deletes nodes from the tree if they are "equality" nodes that point directly to other nodes in the
    tree. This checks if the given node is such a node or not.

    Args:
        windows: A dictionary mapping window name to ACES window configurations.
        window: The query window node (tuple of window name and endpoint).

    Returns:
        True if the node is in the tree, False otherwise.

    Examples:
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ...     "post_target": WindowConfig("target.end", "start + 24h", False, True),
        ...     "weird_window": WindowConfig("end - 24h", "post_target.start", False, False),
        ... }
        >>> _node_in_tree(windows, WindowNode("weird_window", "start"))
        True
        >>> _node_in_tree(windows, WindowNode("weird_window", "end"))
        False
        >>> _node_in_tree(windows, WindowNode("post_target", "start"))
        False
        >>> _node_in_tree(windows, WindowNode("input", "start"))
        True
        >>> _node_in_tree(windows, WindowNode("gap", "end"))
        True
        >>> _node_in_tree(windows, WindowNode("target", "start"))
        False
        >>> _node_in_tree(windows, WindowNode("gap", "start"))
        False
        >>> _node_in_tree(windows, WindowNode("post_target", "end"))
        True
    """

    if window.root is None:
        return True

    if window.root == "start":
        return windows[window.name].start_endpoint_expr is not None
    else:
        return windows[window.name].end_endpoint_expr is not None


def _resolve_node(
    task_cfg: TaskExtractorConfig,
    window_name: str | None = None,
    root_node: WindowNode | None = None,
) -> WindowNode:
    """Resolves a node in the task configuration based on the window name or an input root node.

    Args:
        task_cfg: The task configuration to resolve the node from.
        window_name: The name of the window to resolve. If None, the root_node will be used.
        root_node: The root node to resolve. If None, the window_name will be used.

    Returns:
        The node that exists in the final output tree that corresponds to the passed node.

    Raises:
        ValueError: If the window name is not found in the task configuration.

    Examples:
        >>> from bigtree import print_tree
        >>> print_tree(sample_ACES_cfg.window_tree)
        trigger
        └── input.end
            └── gap.end
                └── target.end

    On this tree, the windows depend on the following nodes:

        >>> _resolve_node(sample_ACES_cfg, window_name="gap")
        WindowNode(name='input', root='end')
        >>> _resolve_node(sample_ACES_cfg, window_name="input")
        WindowNode(name='trigger', root=None)
        >>> _resolve_node(sample_ACES_cfg, window_name="target")
        WindowNode(name='gap', root='end')

    If we pass a non-existent window name, it raises a ValueError:

        >>> _resolve_node(sample_ACES_cfg, window_name="nonexistent")
        Traceback (most recent call last):
            ...
        ValueError: Window 'nonexistent' not found in task configuration.

    We can also pass a node directly, rather than resolving the window to the root node:

        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("input", "start"))
        WindowNode(name='trigger', root=None)
        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("input", "end"))
        WindowNode(name='input', root='end')
        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("target", "start"))
        WindowNode(name='gap', root='end')
        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("target", "end"))
        WindowNode(name='target', root='end')

    We must pass exactly one of window_name or root_node:

        >>> _resolve_node(sample_ACES_cfg, window_name="input", root_node=WindowNode("target", "start"))
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of window_name or root_node must be provided.
        >>> _resolve_node(sample_ACES_cfg, window_name=None, root_node=None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of window_name or root_node must be provided.
    """

    if (window_name is None and root_node is None) or (window_name is not None and root_node is not None):
        raise ValueError("Exactly one of window_name or root_node must be provided.")

    if window_name is not None:
        if window_name not in task_cfg.windows:
            raise ValueError(f"Window '{window_name}' not found in task configuration.")

        root_node = WindowNode(window_name, task_cfg.windows[window_name].root_node)

    while not _node_in_tree(task_cfg.windows, root_node):
        root_node = _get_referenced_node(task_cfg.windows, root_node)

    return root_node
