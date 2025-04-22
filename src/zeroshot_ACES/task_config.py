import itertools
from collections import defaultdict
from datetime import timedelta
from typing import Literal, NamedTuple

from aces.config import TaskExtractorConfig, WindowConfig
from aces.utils import parse_timedelta
from omegaconf import DictConfig

WINDOW_NODE = tuple[str, Literal["start", "end"]] | tuple[Literal["trigger"], None]


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


def _singleton_to_ancestral_relationships(singleton_relationships: dict) -> dict:
    """Aggregates tree relationships at the direct child level to the all ancestors level.

    Assumes that the tree relationships are either a boolean property (indicated with a dictionary mapping
    nodes to sets, where presence in the set indicates True) or a summable property (indicated with a
    dictionary mapping nodes to dictionaries mapping nodes to a summable type).

    Args:
        tree_relationships: A dictionary mapping nodes to their direct child relationships.

    Returns:
        The tree relationships aggregated to the all ancestors level.

    Examples:
        >>> singleton_relationships = {
        ...     "A": {"B", "C"},
        ...     "B": {"D", "E"},
        ...     "C": {"F"},
        ...     "D": {"G"},
        ...     "E": {},
        ...     "F": {},
        ...     "G": {},
        ... }
        >>> relationships = _singleton_to_ancestral_relationships(singleton_relationships)
        >>> for k, v in relationships.items():
        ...     print(f"{k}: {sorted(v)}")
        A: ['B', 'C', 'D', 'E', 'F', 'G']
        B: ['D', 'E', 'G']
        C: ['F']
        D: ['G']
        E: []
        F: []
        G: []
        >>> singleton_relationships = {
        ...     "A": {"B": 1, "C": 2},
        ...     "B": {"D": 3, "E": 1},
        ...     "C": {"F": 10},
        ...     "D": {"G": 1},
        ...     "F": {},
        ...     "G": {},
        ...     "R": {"S": 1},
        ... }
        >>> relationships = _singleton_to_ancestral_relationships(singleton_relationships)
        >>> for k, v in relationships.items():
        ...     print(f"{k}: {v}")
        A: {'C': 2, 'F': 12, 'B': 1, 'E': 2, 'D': 4, 'G': 5}
        B: {'E': 1, 'D': 3, 'G': 4}
        C: {'F': 10}
        D: {'G': 1}
        R: {'S': 1}
        F: {}
        G: {}
        >>> singleton_relationships = {
        ...     "A": {"B": timedelta(days=1), "C": timedelta(days=2)},
        ...     "B": {"D": timedelta(hours=3), "E": timedelta(minutes=1)},
        ... }
        >>> relationships = _singleton_to_ancestral_relationships(singleton_relationships)
        >>> for k, v in relationships.items():
        ...     print(f"{k}: {v}")
        A: {'C': datetime.timedelta(days=2),
            'B': datetime.timedelta(days=1),
            'E': datetime.timedelta(days=1, seconds=60),
            'D': datetime.timedelta(days=1, seconds=10800)}
        B: {'E': datetime.timedelta(seconds=60),
            'D': datetime.timedelta(seconds=10800)}
    """

    if not singleton_relationships:
        return {}

    val_type = type(next(iter(singleton_relationships.values())))

    total = defaultdict(val_type)

    for res_node, children_relationships in singleton_relationships.items():
        to_parse = list(children_relationships)
        to_parse_meta = {**children_relationships} if val_type is dict else {}
        parsed = set()

        while to_parse:
            current = to_parse.pop()

            if val_type is dict:
                current_val = to_parse_meta.get(current, 0)

            if current in parsed:
                continue

            if val_type is set:
                total[res_node].add(current)
            elif val_type is dict:
                total[res_node][current] = current_val

            for child in singleton_relationships.get(current, val_type()):
                if child not in parsed:
                    to_parse.append(child)
                    if val_type is dict:
                        to_parse_meta[child] = current_val + singleton_relationships[current][child]

            parsed.add(current)

    for node in singleton_relationships:
        if node not in total:
            if val_type is set:
                total[node] = set()
            elif val_type is dict:
                total[node] = {}

    return dict(total)


def _get_direct_temporal_offsets(
    task_cfg: TaskExtractorConfig,
) -> dict[WindowNode, dict[WindowNode, timedelta]]:
    """Returns a mapping of the direct time deltas between parent-child nodes in the task configuration.

    Args:
        task_cfg: The task configuration to analyze.

    Returns:
        A mapping of all direct, parent-child timedelta offset relationships between nodes in the task
        configuration.

    Examples:
        >>> from aces.config import PlainPredicateConfig, EventConfig
        >>> predicates = {
        ...     "icu_admission": PlainPredicateConfig("ICU_ADMISSION"),
        ...     "discharge_or_death": PlainPredicateConfig("DISCHARGE_OR_DEATH"),
        ... }
        >>> trigger = EventConfig("icu_admission")
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_tree(cfg.window_tree)
        trigger
        ├── input.end
        │   └── input.start
        └── gap.end
            └── target.end

        >>> offsets = _get_direct_temporal_offsets(cfg)
        >>> for k, v in offsets.items():
        ...     for kk, vv in v.items():
        ...         print(f"{k.node_name} -> {kk.node_name}: {vv}")
        gap.end -> trigger: 2 days, 0:00:00
        trigger -> gap.end: -2 days, 0:00:00
        trigger -> input.end: -1 day, 0:00:00
        input.end -> trigger: 1 day, 0:00:00
    """

    window_nodes = [WindowNode("trigger", None)]
    window_nodes.extend(WindowNode(*e) for e in itertools.product(task_cfg.windows.keys(), ("start", "end")))

    singleton_past_offsets = defaultdict(dict)

    for in_node in window_nodes:
        res_node = _resolve_node(task_cfg, root_node=in_node)

        if in_node.root is None:
            continue

        referenced_node = _get_referenced_node(task_cfg.windows, in_node)
        referenced_node = _resolve_node(task_cfg, root_node=referenced_node)

        window_cfg = task_cfg.windows[in_node.name]
        bound = window_cfg._parsed_start if in_node.root == "start" else window_cfg._parsed_end

        if bound["offset"] is not None:
            # A positive offset means that referenced_node occurs after res_node.
            singleton_past_offsets[res_node][referenced_node] = parse_timedelta(bound["offset"])
            singleton_past_offsets[referenced_node][res_node] = -parse_timedelta(bound["offset"])

    return singleton_past_offsets


def _get_all_temporal_offsets(task_cfg: TaskExtractorConfig) -> dict[WindowNode, dict[WindowNode, timedelta]]:
    """Returns a mapping from a node to a dictionary of all nodes that have known timedeltas from it.

    Args:
        task_cfg: The task configuration to analyze.

    Returns:
        A mapping of all, implicit or explicit, known temporal relationships between nodes in the task
        configuration.

    Examples:
        >>> from aces.config import PlainPredicateConfig, EventConfig
        >>> predicates = {
        ...     "icu_admission": PlainPredicateConfig("ICU_ADMISSION"),
        ...     "discharge_or_death": PlainPredicateConfig("DISCHARGE_OR_DEATH"),
        ... }
        >>> trigger = EventConfig("icu_admission")
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_tree(cfg.window_tree)
        trigger
        ├── input.end
        │   └── input.start
        └── gap.end
            └── target.end
        >>> offsets = _get_all_temporal_offsets(cfg)
        >>> for k, v in offsets.items():
        ...     for kk, vv in v.items():
        ...         print(f"{k.node_name} -> {kk.node_name}: {vv}")
        gap.end -> trigger: 2 days, 0:00:00
        gap.end -> input.end: 1 day, 0:00:00
        trigger -> input.end: -1 day, 0:00:00
        trigger -> gap.end: -2 days, 0:00:00
        input.end -> trigger: 1 day, 0:00:00
        input.end -> gap.end: -1 day, 0:00:00
    """

    window_nodes = [WindowNode("trigger", None)]
    window_nodes.extend(WindowNode(*e) for e in itertools.product(task_cfg.windows.keys(), ("start", "end")))

    # 1. Get the direct parent-child temporal relationships.
    singleton_past_offsets = _get_direct_temporal_offsets(task_cfg)

    # 2. Iterate through all temporal offsets and aggregate over all ancestors.
    total_past_offsets = _singleton_to_ancestral_relationships({**singleton_past_offsets})

    # total past offsets maps each node to the set of all nodes that have a known temporal offset to it. E.g.,
    # total_past_offsets["A"] = {"B": time of B - time of A, ...}.

    # 3. Examine the offsets between children of a common root in the tree to find non-tree temporal ordering
    # guarantees.

    for unresolved_node in window_nodes:
        node = _resolve_node(task_cfg, root_node=unresolved_node)
        tree_node = task_cfg.window_nodes[node.node_name]

        if len(tree_node.children) <= 1:
            # The node's children have no siblings, so all temporal constraints would be caught by the above
            # loop.
            continue

        sibling_offsets_from_node = {}
        for child in tree_node.children:
            child_node = _resolve_node(task_cfg, root_node=WindowNode(*child.node_name.split(".")))

            if child_node in singleton_past_offsets.get(node, {}):
                sibling_offsets_from_node[child_node] = singleton_past_offsets[node][child_node]

        siblings = list(sibling_offsets_from_node.keys())
        for sibling_node in siblings:
            to_parse = [n for n in sibling_offsets_from_node if n != sibling_node]

            while to_parse:
                comparison_node = to_parse.pop()

                if comparison_node in total_past_offsets[sibling_node]:
                    continue

                delta = sibling_offsets_from_node[sibling_node] - sibling_offsets_from_node[comparison_node]

                total_past_offsets[sibling_node][comparison_node] = delta
                total_past_offsets[comparison_node][sibling_node] = -delta

                for comp_child in singleton_past_offsets[comparison_node]:
                    sibling_offsets_from_node[comp_child] = (
                        sibling_offsets_from_node[comparison_node]
                        + singleton_past_offsets[comparison_node][comp_child]
                    )
                    to_parse.append(comp_child)

    out = {}
    for node, offsets in total_past_offsets.items():
        out[node] = {}
        for offset_node, offset in offsets.items():
            if offset_node != node:
                out[node][offset_node] = offset

    return out


def _ACES_config_timeline(task_cfg: TaskExtractorConfig) -> dict[WindowNode, set[WindowNode]]:
    """Produces a set of temporal order guarantees for the nodes in the ACES task configuration.

    This function works in four steps:

      1. For the immediate parent-child relationships between nodes, we identify (a) all known instances where
         node A deterministically occurs before node B, and (b) (when possible), what temporal offset exists
         between A and B. This is only possible to know for temporal nodes.
      2. We iterate through all temporal offsets and aggregate over all ancestors. E.g., in this step, we
         identify that if A is 3d before B, and B is 1d before C, then A is 4d before C.
      3. We use all known temporal offsets across all levels of tree relationships to link sibling-level nodes
         with temporal ordering guarantees.
      4. We use all identified temporal ordering guarantees to identify all nodes that are guaranteed to occur
         before other nodes, regardless of the tree structure.

    Args:
        task_cfg: The task configuration to analyze.

    Returns:
        A mapping of window nodes to a set of all nodes that guaranteeably occur before them in time.

    Examples:
        >>> from bigtree import print_tree
        >>> print_tree(sample_ACES_cfg.window_tree)
        trigger
        ├── input.end
        │   └── input.start
        └── gap.end
            └── target.end

    The tree does not uniquely determine the timeline, but for this example, we have the following temporal
    relationships:

        >>> temporal_constraints = _ACES_config_timeline(sample_ACES_cfg)
        >>> for node, all_before in temporal_constraints.items():
        ...     print(f"{node.node_name} is guaranteeably after: {sorted(n.node_name for n in all_before)}")
        input.end is guaranteeably after: ['trigger']
        gap.end is guaranteeably after: ['input.end', 'trigger']
        target.end is guaranteeably after: ['gap.end', 'input.end', 'trigger']
    """

    window_nodes = [WindowNode("trigger", None)]
    window_nodes.extend(WindowNode(*e) for e in itertools.product(task_cfg.windows.keys(), ("start", "end")))

    # 1. Get the immediate, within tree, parent-children temporal relationships.
    singleton_past = defaultdict(set)

    for in_node in window_nodes:
        res_node = _resolve_node(task_cfg, root_node=in_node)

        if in_node.root is None:
            continue

        referenced_node = _get_referenced_node(task_cfg.windows, in_node)
        referenced_node = _resolve_node(task_cfg, root_node=referenced_node)

        window_cfg = task_cfg.windows[in_node.name]
        bound = window_cfg._parsed_start if in_node.root == "start" else window_cfg._parsed_end

        occurs_before_referenced = bound["occurs_before"]
        if occurs_before_referenced is None or occurs_before_referenced:
            continue

        singleton_past[res_node].add(referenced_node)

    # 2. Iterate through all temporal offsets and aggregate over all ancestors.
    total_past_offsets = _get_all_temporal_offsets(task_cfg)

    # total past offsets maps each node to the set of all nodes that have a known temporal offset to it. E.g.,
    # total_past_offsets["A"] = {"B": time of B - time of A, ...}.

    for node, temporal_offsets in total_past_offsets.items():
        for alt_node, offset in temporal_offsets.items():
            if offset > timedelta(0):
                singleton_past[node].add(alt_node)

    # 4. Traverse all singleton relationships to find all nodes that are guaranteed to occur before each node.
    total_past = _singleton_to_ancestral_relationships(singleton_past)

    return total_past


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
        ├── input.end
        │   └── input.start
        └── gap.end
            └── target.end

    On this tree, the windows depend on the following nodes:

        >>> _resolve_node(sample_ACES_cfg, window_name="gap")
        WindowNode(name='trigger', root=None)
        >>> _resolve_node(sample_ACES_cfg, window_name="input")
        WindowNode(name='input', root='end')
        >>> _resolve_node(sample_ACES_cfg, window_name="target")
        WindowNode(name='gap', root='end')

    If we pass a non-existent window name, it raises a ValueError:

        >>> _resolve_node(sample_ACES_cfg, window_name="nonexistent")
        Traceback (most recent call last):
            ...
        ValueError: Window 'nonexistent' not found in task configuration.

    We can also pass a node directly, rather than resolving the window to the root node:

        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("input", "start"))
        WindowNode(name='input', root='start')
        >>> _resolve_node(sample_ACES_cfg, root_node=WindowNode("target", "start"))
        WindowNode(name='gap', root='end')

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

    if prediction_time_window_root not in task_node_timeline[label_window_root]:
        raise ValueError(
            f"Cannot guarantee that label node {label_window_root.node_name} occurs before "
            f"prediction time node {prediction_time_window_root.node_name} in the task configuration."
        )


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
