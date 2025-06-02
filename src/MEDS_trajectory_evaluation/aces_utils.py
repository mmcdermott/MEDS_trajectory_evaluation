"""Utilities to work with ACES configuration objects more easily."""

import tempfile
from collections import defaultdict
from datetime import timedelta
from typing import Literal, NamedTuple

import polars as pl
from aces.config import TaskExtractorConfig, WindowConfig
from aces.predicates import PlainPredicateConfig, get_predicates_df
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import Node, yield_tree
from omegaconf import DictConfig


class ZeroShotTaskConfig(TaskExtractorConfig):
    @property
    def window_tree(self) -> Node:
        if not hasattr(self, "_root_node"):
            return self.window_nodes["trigger"]
        else:
            return self.window_nodes[self._root_node]

    @window_tree.setter
    def window_tree(self, node: Node):
        self._root_node = node.node_name


def get_constraint_str(window_cfg: WindowConfig) -> str:
    """Return a simple string representing the constraints of a window.

    Args:
        window_cfg: The configuration of the window whose constraints should be printed.

    Returns:
        A string representation of the constraints of a window.

    Examples:
        >>> get_constraint_str(WindowConfig("foo", "start + 1d", False, False))
        ''
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(None, 0)", "bar": "(None, 0)"}),
        ... )
        'no foo, bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(None, 4)", "bar": "(None, 2)"}),
        ... )
        'no more than 4 foo, 2 bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(5, None)", "bar": "(5, None)"}),
        ... )
        'at least 5 foo, 5 bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(5, 6)", "bar": "(1, 3)"}),
        ... )
        'between 5-6 foo, 1-3 bar'
        >>> get_constraint_str(
        ...     WindowConfig(
        ...         "b", "start + 1d", False, False,
        ...         has={
        ...             "A": "(5, 6)",
        ...             "_ANY_EVENT": "(1, None)",
        ...             "C": "(2, 4)",
        ...             "D": "(None, 0)",
        ...             "E": "(2, None)",
        ...             "F": "(None, 0)",
        ...             "G": "(None, 2)",
        ...             "H": "(None, 0)",
        ...         },
        ...     ),
        ... )
        'between 5-6 A, 2-4 C; at least 1 event(s), 2 E; no D, F, H; no more than 2 G'
    """
    if not window_cfg.has:
        return ""

    by_prefix = defaultdict(list)
    for k_raw, constraint in window_cfg.has.items():
        left, right = constraint

        k = "event(s)" if k_raw == "_ANY_EVENT" else k_raw

        if left is None and right == 0:
            prefix = "no"
            str_rep = k
        elif left is None:
            prefix = "no more than"
            str_rep = f"{right} {k}"
        elif right is None:
            prefix = "at least"
            str_rep = f"{left} {k}"
        else:
            prefix = "between"
            str_rep = f"{left}-{right} {k}"

        by_prefix[prefix].append(str_rep)

    out_parts = [f"{prefix} {', '.join(strs)}" for prefix, strs in by_prefix.items()]
    return "; ".join(out_parts)


def print_ACES(task_cfg: TaskExtractorConfig, **kwargs):
    """A pretty printer for ACES task configurations.

    This is purely for development / debugging purposes.

    Args:
        task_cfg: The task configuration to print.
        **kwargs: Additional arguments to pass to the print function.

    Examples:
        >>> print_ACES(sample_ACES_cfg)
        trigger
        └── (+1 day, 0:00:00) input.end (no icu_admission, discharge_or_death); **Prediction Time**
            └── (+1 day, 0:00:00) gap.end (no icu_admission, discharge_or_death)
                └── (next discharge_or_death) target.end; **Label: Presence of death**

        >>> from aces.config import PlainPredicateConfig, EventConfig
        >>> predicates = {
        ...     "admission": PlainPredicateConfig("ADMISSION"),
        ...     "discharge": PlainPredicateConfig("DISCHARGE"),
        ... }
        >>> trigger = EventConfig("admission")
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "discharge": WindowConfig("gap.end", "start -> discharge", False, True),
        ...     "target": WindowConfig("discharge.end + 1d", "start + 29d", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_ACES(cfg)
        trigger
        ├── (+1 day, 0:00:00) input.end
        │   └── (start of record) input.start
        └── (+2 days, 0:00:00) gap.end
            └── (next discharge) discharge.end
                └── (+1 day, 0:00:00) target.start
                    └── (+29 days, 0:00:00) target.end
        >>> windows = {
        ...     "prior_hospitalization": WindowConfig("end <- admission", "trigger <- discharge", True, True),
        ...     "pre_data": WindowConfig("end - 12d", "trigger-2d", True, False),
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "discharge": WindowConfig("gap.end", "start -> discharge", False, True),
        ...     "target": WindowConfig("discharge.end + 1d", "start + 29d", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_ACES(cfg)
        trigger
        ├── (prior discharge) prior_hospitalization.end
        │   └── (prior admission) prior_hospitalization.start
        ├── (-2 days, 0:00:00) pre_data.end
        │   └── (-12 days, 0:00:00) pre_data.start
        ├── (+1 day, 0:00:00) input.end
        │   └── (start of record) input.start
        └── (+2 days, 0:00:00) gap.end
            └── (next discharge) discharge.end
                └── (+1 day, 0:00:00) target.start
                    └── (+29 days, 0:00:00) target.end
    """

    index_ts_window_name = getattr(task_cfg, "index_timestamp_window", None)

    if index_ts_window_name:
        index_ts_window = task_cfg.windows[index_ts_window_name]
        index_ts_node = _resolve_node(
            task_cfg, root_node=WindowNode(index_ts_window_name, index_ts_window.index_timestamp)
        )
    else:
        index_ts_node = None

    for branch, stem, node in yield_tree(task_cfg.window_tree):
        window_name = node.node_name.split(".")[0]

        if index_ts_node and (node.node_name == index_ts_node.node_name):
            index_ts_str = "; **Prediction Time**"
        else:
            index_ts_str = ""

        if window_name == "trigger":
            print(f"{branch}{stem}{node.node_name}{index_ts_str}", **kwargs)
            continue

        match getattr(node, "endpoint_expr", None):
            case ToEventWindowBounds() as event_bound:
                if event_bound.end_event == "-_RECORD_START":
                    event_str = "start of record"
                elif event_bound.end_event == "_RECORD_END":
                    event_str = "end of record"
                elif event_bound.end_event.startswith("-"):
                    event_str = f"prior {event_bound.end_event[1:]}"
                else:
                    event_str = f"next {event_bound.end_event}"

                if event_bound.offset:
                    raise NotImplementedError("Offset not supported.")
                else:
                    bound = f"({event_str}) "
            case TemporalWindowBounds() as time_bound:
                sign = "+" if time_bound.window_size >= timedelta(0) else ""
                if time_bound.offset:
                    raise NotImplementedError("Offset not supported.")
                else:
                    bound = f"({sign}{time_bound.window_size}) "
            case None:
                bound = ""
            case _ as other:
                raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")

        window_cfg = task_cfg.windows[window_name]

        side = node.node_name.split(".")[1]

        is_end_node = side != window_cfg.root_node

        if not is_end_node:
            print(f"{branch}{stem}{bound}{node.node_name}{index_ts_str}", **kwargs)
            continue

        c_str = get_constraint_str(window_cfg)
        constraint_str = f" ({get_constraint_str(window_cfg)})" if c_str else ""
        label_str = f"; **Label: Presence of {window_cfg.label}**" if window_cfg.label else ""
        print(f"{branch}{stem}{bound}{node.node_name}{constraint_str}{label_str}{index_ts_str}", **kwargs)


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

    if root_node.node_name == task_cfg.window_tree.node_name:
        return root_node

    while not _node_in_tree(task_cfg.windows, root_node):
        root_node = _get_referenced_node(task_cfg.windows, root_node)

    return root_node


def get_MEDS_predicates(
    MEDS_df: pl.DataFrame,
    task_cfg: TaskExtractorConfig,
) -> pl.DataFrame:
    """Gets the predicate realizations for a MEDS dataframe.

    TODO(mmd): This is very stupid. We should just modify ACES to be able to get the predicates from a MEDS
    dataframe directly.

    Args:
        MEDS_df: The MEDS dataframe to get the predicates from.
        task_cfg: The task configuration to use for the predicates.
    """

    with tempfile.NamedTemporaryFile(suffix=".parquet") as data_fp:
        MEDS_df.write_parquet(data_fp.name, use_pyarrow=True)
        return get_predicates_df(task_cfg, DictConfig({"path": data_fp.name, "standard": "meds"}))


def get_MEDS_plain_predicates(
    MEDS_df: pl.DataFrame,
    predicates: dict[str, PlainPredicateConfig],
) -> pl.DataFrame:
    """Gets the plain predicate realizations for a MEDS dataframe.

    Args:
        MEDS_df: The MEDS dataframe to get the predicates from.
        predicates: A dictionary of ACES plain predicates to be extracted from the MEDS data.

    Examples:
        >>> MEDS_df = pl.DataFrame({
        ...     'subject_id': [
        ...         1, 1, 1, 1,
        ...         2,
        ...         3, 3, 3,
        ...     ],
        ...     'time': [
        ...         datetime(2020, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 2), datetime(2022, 1, 4),
        ...         datetime(2022, 1, 1),
        ...         datetime(2001, 1, 1), datetime(2002, 1, 2), datetime(2002, 1, 3),
        ...     ],
        ...     'code': [
        ...         'icd9//150.1', 'icd9//400', 'icd9//250.3', 'icd9//250.5',
        ...         'icd9//250.2',
        ...         'icd9//250.1', 'icd9//402', 'icd9//400',
        ...     ],
        ... })
        >>> predicates = {
        ...     '250.*': PlainPredicateConfig(code={"regex": "250.*"}),
        ...     '400': PlainPredicateConfig(code="icd9//400"),
        ... }
        >>> get_MEDS_plain_predicates(MEDS_df, predicates)
        shape: (8, 5)
        ┌────────────┬─────────────────────┬─────────────┬───────┬───────┐
        │ subject_id ┆ time                ┆ code        ┆ 250.* ┆ 400   │
        │ ---        ┆ ---                 ┆ ---         ┆ ---   ┆ ---   │
        │ i64        ┆ datetime[μs]        ┆ str         ┆ bool  ┆ bool  │
        ╞════════════╪═════════════════════╪═════════════╪═══════╪═══════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ icd9//150.1 ┆ false ┆ false │
        │ 1          ┆ 2022-01-02 00:00:00 ┆ icd9//400   ┆ false ┆ true  │
        │ 1          ┆ 2022-01-02 00:00:00 ┆ icd9//250.3 ┆ true  ┆ false │
        │ 1          ┆ 2022-01-04 00:00:00 ┆ icd9//250.5 ┆ true  ┆ false │
        │ 2          ┆ 2022-01-01 00:00:00 ┆ icd9//250.2 ┆ true  ┆ false │
        │ 3          ┆ 2001-01-01 00:00:00 ┆ icd9//250.1 ┆ true  ┆ false │
        │ 3          ┆ 2002-01-02 00:00:00 ┆ icd9//402   ┆ false ┆ false │
        │ 3          ┆ 2002-01-03 00:00:00 ┆ icd9//400   ┆ false ┆ true  │
        └────────────┴─────────────────────┴─────────────┴───────┴───────┘
    """

    predicate_exprs = {n: (p.MEDS_eval_expr() > 0) for n, p in predicates.items()}
    return MEDS_df.with_columns(**predicate_exprs)
