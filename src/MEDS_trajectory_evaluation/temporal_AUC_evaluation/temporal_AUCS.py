from collections.abc import Callable, Collection
from datetime import timedelta

import polars as pl
import polars.selectors as cs
from meds import LabelSchema


def _reprefix_fntr(new_prefix: str) -> Callable[[str], str]:
    """Create a function that replaces the prefix of a string with a new prefix.

    This function is used to change the prefix of a string that starts with "*/" to a new prefix.

    Args:
        new_prefix: The new prefix to use.

    Returns:
        A function that takes a string and replaces its prefix with the new prefix, where prefix vs. suffix is
        determined by the first slash in the string.

    Examples:
        >>> fn = _reprefix_fntr("new_prefix")
        >>> fn("old_prefix/some/path")
        'new_prefix/some/path'
        >>> fn("another_prefix/foo")
        'new_prefix/foo'
        >>> fn("no_prefix")
        Traceback (most recent call last):
            ...
        ValueError: String 'no_prefix' does not have a valid prefix to replace.
    """

    def reprefix_fntr(s: str) -> str:
        """Replace the prefix of a string with a new prefix."""
        parts = s.split("/")
        if len(parts) < 2:
            raise ValueError(f"String '{s}' does not have a valid prefix to replace.")
        suffix = "/".join(parts[1:])
        return f"{new_prefix}/{suffix}"

    return reprefix_fntr


def df_AUC(df: pl.DataFrame) -> pl.DataFrame:
    """Given a DataFrame with sorted cols "true" and "false" containing lists of probabilities, compute AUC.

    This uses polars expressions and the probabilistic interpretation of AUC as the probability that a
    randomly chosen positive example has a higher score than a randomly chosen negative example to
    efficiently compute the AUC across many tasks at once.

    The input dataframe is structured with a collection of task index columns, followed by two columns: `true`
    and `false`, which contain lists of probabilities for the positive and negative classes, respectively.
    These can be full distributions or subsamples of the distributions, but _must_ be sorted in ascending
    order. A search algorithm is then used to determine where each positive sample would fit in the ordered
    set of negative samples to determine the number of correctly ordered pairs, which is divided by the total
    number of pairs to compute the AUC.

    This algorithm could be made even more efficient by leveraging the sorting of both columns to limit the
    search space of the search algorithm for each positive sample, but this is not implemented yet and may
    result in slowdowns due to the fact that it would limit parallelization.

    Arguments:
        df: A DataFrame with task row-index columns followed by sets of two columns "true/$task_col" and
            "false/$task_col" containing lists of probabilities for the positive and negative classes,
            respectively, for the task indexed by the given row and columq. The lists must be sorted in
            ascending order.

    Returns:
        A DataFrame with the same task index columns and a new column "AUC/$task_col" containing the computed
        AUC for each task row X column combination.

    Examples:
        >>> df = pl.DataFrame({
        ...     "task": ["task1", "task2"],
        ...     "true/A": [[0.1, 0.5, 0.8], [0.3, 0.5]],
        ...     "false/A": [[0.2, 0.3], [0.4, 0.9]],
        ...     "true/B": [[0.1, 0.34], [0.8, 0.9]],
        ...     "false/B": [[0.02, 0.3], [0.4, 0.5]],
        ... })

    In this example, the DataFrame `df` contains two task rows and columns with their respective true and
    false probabilities.
    Task 1.A has true probabilities of [0.1, 0.5, 0.8] and false probabilities of [0.2, 0.3], which results in
    four correctly ordered pairs (0.5 > 0.2, 0.5 > 0.3, 0.8 > 0.2, 0.8 > 0.3) out of a total of six pairs, for
    an AUC of 4/6 = 0.6667.
    Task 2.A has true probabilities of [0.3, 0.5] and false probabilities of [0.4, 0.9], which results in
    one correctly ordered pair (0.5 > 0.4) out of a total of four pairs, for an AUC of 1/4 = 0.25.
    Task 1.B has true probabilities of [0.1, 0.34] and false probabilities of [0.02, 0.3], which results in
    three correctly ordered pairs (0.34 > 0.02, 0.34 > 0.3, 0.1 > 0.02) out of a total of four pairs, for an
    AUC of 3/4 = 0.75.
    Task 2.B has true probabilities of [0.8, 0.9] and false probabilities of [0.4, 0.5], which results in an
    AUC of 1.0, since both true probabilities are greater than both false probabilities.

        >>> df_AUC(df)
        shape: (2, 3)
        ┌───────┬──────────┬───────┐
        │ task  ┆ AUC/A    ┆ AUC/B │
        │ ---   ┆ ---      ┆ ---   │
        │ str   ┆ f64      ┆ f64   │
        ╞═══════╪══════════╪═══════╡
        │ task1 ┆ 0.666667 ┆ 0.75  │
        │ task2 ┆ 0.25     ┆ 1.0   │
        └───────┴──────────┴───────┘

    Let's look at another example, where there are duplicates in the true and false distributions. In this
    example, Task 1 has true probabilities of [0.2, 0.2, 0.4, 0.6] and false probabilities of [0.1, 0.3, 0.3],
    which results in eight correctly ordered pairs:
        0.2 > 0.1, 0.2 > 0.1,
        0.4 > 0.1, 0.4 > 0.3, 0.4 > 0.3,
        0.6 > 0.1, 0.6 > 0.3, 0.6 > 0.3
    out of a total of twelve pairs, for an AUC of 8/12 = 0.6667.
    Task 2 has true probabilities of [0.8] and false probabilities of [0.1, 0.4, 0.4, 0.9], which results in
    three correctly ordered pairs (0.8 > 0.1, 0.8 > 0.4, 0.8 > 0.4) out of a total of four pairs, for an AUC
    of 3/4 = 0.75.

        >>> df = pl.DataFrame({
        ...     "task": ["task1", "task2"],
        ...     "true/A": [[0.2, 0.2, 0.4, 0.6], [0.8]],
        ...     "false/A": [[0.1, 0.3, 0.3], [0.1, 0.4, 0.4, 0.9]]
        ... })
        >>> df_AUC(df)
        shape: (2, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC/A    │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.666667 │
        │ task2 ┆ 0.75     │
        └───────┴──────────┘

    If we have an empty distribution, the AUC returned is `null`:

        >>> df_empty = pl.DataFrame({
        ...     "task": ["task1", "task2", "task3"],
        ...     "true/A": [[0.1, 0.5, 0.8], [], [0.3, 0.5]],
        ...     "false/A": [[0.2, 0.3], [0.4, 0.9], []]
        ... })
        >>> df_AUC(df_empty)
        shape: (3, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC/A    │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.666667 │
        │ task2 ┆ null     │
        │ task3 ┆ null     │
        └───────┴──────────┘

    Ties should contribute 1/2 an ordered pair to the AUC. In the example below, Task 1 has true probabilities
    of [0.2, 0.3, 0.5] and false probabilities of [0.2, 0.3, 0.3], which results in:
      * 4 correctly ordered pairs (0.3 > 0.2, 0.5 > 0.2, 0.5 > 0.3, 0.5 > 0.3)
      * 3 ties (0.2 == 0.2, 0.3 == 0.3, 0.3 == 0.3)
      * of a total of 9 pairs.
    This gives an AUC of (4 + 0.5 * 3) / 9 = 5.5 / 9 = 0.6111.
    Task 2 has true probabilities of [0.9] and false probabilities of [0.4, 0.9], which results in:
      * 1 correctly ordered pair (0.9 > 0.4)
      * 1 tie (0.9 == 0.9)
      * of a total of 2 pairs.
    This gives an AUC of (1 + 0.5 * 1) / 2 = 0.75.

        >>> df_ties = pl.DataFrame({
        ...     "task": ["task1", "task2"],
        ...     "true/A": [[0.2, 0.3, 0.5], [0.9]],
        ...     "false/A": [[0.2, 0.3, 0.3], [0.4, 0.9]]
        ... })
        >>> df_AUC(df_ties)
        shape: (2, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC/A    │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.611111 │
        │ task2 ┆ 0.75     │
        └───────┴──────────┘
    """

    tasks = [c.split("/")[1] for c in df.columns if c.startswith("true/")]
    ids = [c for c in df.columns if c.split("/")[0] not in {"true", "false"}]

    structs = [
        pl.struct(true=pl.col(f"true/{t}"), false=pl.col(f"false/{t}")).alias(f"dist/{t}") for t in tasks
    ]

    df = df.select(*ids, *structs).with_row_index("__idx")

    dists = cs.starts_with("dist/")
    T = dists.struct.field("true")
    F = dists.struct.field("false")

    num_pairs = T.list.len() * F.list.len()
    num_F_lt_T_pairs = F.list.explode().search_sorted(T.list.explode(), side="left").sum().over("__idx")
    num_F_lte_T_pairs = F.list.explode().search_sorted(T.list.explode(), side="right").sum().over("__idx")

    AUC = ((num_F_lt_T_pairs + num_F_lte_T_pairs) / 2) / num_pairs
    AUC = pl.when(AUC.is_infinite() | AUC.is_nan()).then(None).otherwise(AUC)

    return df.select(*ids, AUC.name.map(_reprefix_fntr("AUC")))


def resolution_grid(ttes_df: pl.DataFrame, resolution: str) -> pl.Series:
    raise NotImplementedError


def random_grid(ttes_df: pl.DataFrame, n: int | None) -> pl.Series:
    raise NotImplementedError


def get_grid(
    ttes_df: pl.DataFrame,
    grid: str | int | None | list[timedelta] = 10000,
) -> list[timedelta]:
    match grid:
        case str() as resolution:
            return resolution_grid(ttes_df, resolution)
        case int() | None as n:
            return random_grid(ttes_df, n)
        case list() as seq:
            if not all(isinstance(x, timedelta) for x in seq):
                raise ValueError("All elements in the sequence must be of type 'timedelta'.")
            return seq


def add_labels_from_true_tte(
    df: pl.DataFrame,
    *,
    offset: timedelta = timedelta(0),
    offset_col: str | None = None,
) -> pl.DataFrame:
    """Convert the true time-to-predicate values into a label for a given duration window.

    Given a dataframe with a set of columns prefaced with `tte/` containing the true time-to-predicate value
    and a column `duration` containing the duration for which the AUC should be computed, this function
    computes the label of whether or not the predicate occurred within the given duration. The output columns
    are named `label/<n>` for each `tte/<n>` column in the input.

    Arguments:
        df: A DataFrame with columns "tte/*" (a time-to-predicate value) and ``duration``
            (the duration for which the label should be computed). ``offset`` sets a constant
            offset from the prediction time. If ``offset_col`` is provided, its values are
            added to ``offset`` on a per-row basis.

    Returns:
        A DataFrame with the same columns as the input except for `tte/*`, plus a new column "label/*"
        containing the labels for each row and tte column.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [timedelta(days=5), timedelta(days=10), None],
        ...     "tte/B": [timedelta(days=2), timedelta(days=3), timedelta(days=15)],
        ...     "duration": [timedelta(days=7), timedelta(days=8), timedelta(days=8)]
        ... })
        >>> add_labels_from_true_tte(df)
        shape: (3, 5)
        ┌────────────┬─────────────────────┬──────────────┬─────────┬─────────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ label/A ┆ label/B │
        │ ---        ┆ ---                 ┆ ---          ┆ ---     ┆ ---     │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ bool    ┆ bool    │
        ╞════════════╪═════════════════════╪══════════════╪═════════╪═════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ true    ┆ true    │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ false   ┆ true    │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ false   ┆ false   │
        └────────────┴─────────────────────┴──────────────┴─────────┴─────────┘
        >>> add_labels_from_true_tte(df, offset=timedelta(days=3))
        shape: (3, 5)
        ┌────────────┬─────────────────────┬──────────────┬─────────┬─────────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ label/A ┆ label/B │
        │ ---        ┆ ---                 ┆ ---          ┆ ---     ┆ ---     │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ bool    ┆ bool    │
        ╞════════════╪═════════════════════╪══════════════╪═════════╪═════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ true    ┆ false   │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ true    ┆ false   │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ false   ┆ false   │
        └────────────┴─────────────────────┴──────────────┴─────────┴─────────┘
    """

    tte_cols = cs.starts_with("tte/")

    start = pl.lit(offset)
    if offset_col is not None:
        start = start + pl.col(offset_col)
    label_expr = (
        ((tte_cols > start) & (tte_cols <= start + pl.col("duration")))
        .fill_null(False)
        .name.map(_reprefix_fntr("label"))
    )

    return df.with_columns(label_expr).drop(tte_cols)


def add_probs_from_pred_ttes(
    df: pl.DataFrame,
    *,
    offset: timedelta = timedelta(0),
    offset_col: str | None = None,
) -> pl.DataFrame:
    """Convert the list of predicted time-to-predicate values into a probability distribution.

    Given a dataframe with a column `tte_pred` containing lists of predicted time-to-predicate values and a
    column `duration` containing the duration for which the AUC should be computed, this function computes
    the probability (proportion of sampled trajectories which satisfy) that the time-to-predicate is less than
    or equal to the duration for each trajectory.

    Arguments:
        df: A DataFrame with columns prefixed with ``tte_pred/`` (each a list of predicted
            time-to-predicate values) and ``duration``. ``offset`` sets a constant offset from
            the prediction time. If ``offset_col`` is provided, its values are added to ``offset``
            on a per-row basis.

    Returns:
        A DataFrame with the same columns as the input except for `tte_pred/*`, plus new columns "prob/*"
        containing the computed probabilities for each row and tte_pred column.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte_pred/A": [
        ...         [timedelta(days=5), timedelta(days=10), None],
        ...         [timedelta(days=7), timedelta(days=8)],
        ...         [timedelta(days=8), timedelta(days=9), timedelta(days=10)]
        ...     ],
        ...     "tte_pred/B": [
        ...         [timedelta(days=5), timedelta(days=1), timedelta(days=24)],
        ...         [timedelta(days=7), timedelta(days=2)],
        ...         [timedelta(days=80), timedelta(days=9), timedelta(days=10)]
        ...     ],
        ...     "duration": [timedelta(days=7), timedelta(days=8), timedelta(days=8)]
        ... })
        >>> add_probs_from_pred_ttes(df)
        shape: (3, 5)
        ┌────────────┬─────────────────────┬──────────────┬──────────┬──────────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ prob/A   ┆ prob/B   │
        │ ---        ┆ ---                 ┆ ---          ┆ ---      ┆ ---      │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ f64      ┆ f64      │
        ╞════════════╪═════════════════════╪══════════════╪══════════╪══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ 0.333333 ┆ 0.666667 │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ 1.0      ┆ 1.0      │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ 0.333333 ┆ 0.0      │
        └────────────┴─────────────────────┴──────────────┴──────────┴──────────┘
        >>> add_probs_from_pred_ttes(df, offset=timedelta(days=3))
        shape: (3, 5)
        ┌────────────┬─────────────────────┬──────────────┬──────────┬──────────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ prob/A   ┆ prob/B   │
        │ ---        ┆ ---                 ┆ ---          ┆ ---      ┆ ---      │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ f64      ┆ f64      │
        ╞════════════╪═════════════════════╪══════════════╪══════════╪══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ 0.666667 ┆ 0.333333 │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ 1.0      ┆ 0.5      │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ 1.0      ┆ 0.666667 │
        └────────────┴─────────────────────┴──────────────┴──────────┴──────────┘
    """

    tte_pred_cols = cs.starts_with("tte_pred/")

    sorted_lists = tte_pred_cols.list.sort(descending=False, nulls_last=True)

    offset_expr = pl.lit(offset)
    if offset_col is not None:
        offset_expr = offset_expr + pl.col(offset_col)

    upper = (
        sorted_lists.explode()
        .fill_null(pl.lit(timedelta.max))
        .search_sorted(offset_expr + pl.col("duration"), side="right")
    )
    lower = sorted_lists.explode().fill_null(pl.lit(timedelta.max)).search_sorted(offset_expr, side="left")

    num_trajectories_within_duration = upper - lower

    prob_expr = num_trajectories_within_duration / (tte_pred_cols.list.len())

    return (
        df.with_row_index("__idx")
        .with_columns(prob_expr.over("__idx").name.map(_reprefix_fntr("prob")))
        .drop("__idx", cs.starts_with("tte_pred/"))
    )


def temporal_aucs(
    true_tte: pl.DataFrame,
    pred_ttes: pl.DataFrame,
    duration_grid: str | int | None | list[timedelta] = 10000,
    AUC_dist_approx: int = -1,
    seed: int = 0,
    *,
    offset: timedelta = timedelta(0),
    exclude_history: bool | Collection[str] = False,
) -> pl.DataFrame:
    """Compute the AUC over different prediction windows for the first occurrence of a predicate.

    Parameters:
        true_tte: The true time-to-first-predicate, with columns "subject_id", "prediction_time", and "tte/*".
            The suffix of the tte column represents the task/predicate in question. A `null` value in any
            "tte/*" column indicates that that predicate did not occur.
        pred_ttes: The predicted time-to-first-predicate, with columns "subject_id", "prediction_time", and
            "tte/*", the latter being a list of observed time-to-predicate values for the different generated
            trajectories for each studied task. The set of tasks (suffixes of the "tte/*" columns) must be the
            same as the `true_tte` dataframe "tte/*" column suffixes. A `null` value in any "tte/*" column
            indicates that that predicate did not occur.
        duration_grid: The temporal resolution for the windowing grid within which the AUC should be
            computed. If a string, builds a regular grid at the specified resolution (e.g., "1d" for one day).
            If an integer, it samples that many time-points at at random at which an event is observed in
            either the real or predicted data to use as the grid boundary points. If `None`, all change-points
            are used as grid boundary points. If a sequence of `timedelta` objects, these are used as the grid
            boundary points.
        AUC_dist_approx: If greater than 0, the number of samples to use for approximating the AUC
            distribution. If -1, the full distribution is used. This can be useful for large datasets to
            reduce the cost of computing the AUC, but may result in a less accurate estimate.
        seed: The random seed to use for sampling the AUC distribution if `AUC_dist_approx` is greater than 0.
        offset: Offset from the prediction time at which the evaluation window begins.
        exclude_history: If ``True`` or an iterable of task names, rows where the subject has a historical
            instance of the predicate prior to the prediction time are removed for the specified tasks.

    Examples:
        >>> duration_grid = [timedelta(days=1), timedelta(days=5), timedelta(days=10), timedelta(days=15)]

    We'll begin with a duration grid spanning 1, 5, 10, and 15 days. This means that we will output 4 AUCs per
    task:: one for a prediction window of 1 day, one for 5 days, one for 10 days, and one for 15 days. We'll
    construct a setting which has two tasks (A & B) with true labels of:
    - Task A:
      * Duration 1 day: subject 1 is False, subject 2 is False, subject 3 is False.
      * Duration 5 days: subject 1 is True, subject 2 is False, subject 3 is False.
      * Duration 10 days: subject 1 is True, subject 2 is True, subject 3 is False.
      * Duration 15 days: subject 1 is True, subject 2 is True, subject 3 is False.
    - Task B:
      * Duration 1 day: subject 1 is False, subject 2 is False, subject 3 is False.
      * Duration 5 days: subject 1 is False, subject 2 is True, subject 3 is False.
      * Duration 10 days: subject 1 is True, subject 2 is True, subject 3 is False.
      * Duration 15 days: subject 1 is True, subject 2 is True, subject 3 is True.

        >>> true_tte = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [timedelta(days=5), timedelta(days=10), None],
        ...     "tte/B": [timedelta(days=8), timedelta(days=2), timedelta(days=12)],
        ... })

    We'll also construct a set of predicted time-to-event values, which are lists of observed
    time-to-predicate values for the different generated trajectories. In this case, we have for task A:
      * Subject 1: 3 trajectories with TTEs of 3, 6, and 11 days, yielding...
        - 1 day: 0 True, 3 False, for a probability of 0/3 = 0.0
        - 5 days: 1 True, 2 False, for a probability of 1/3 = 0.3333
        - 10 days: 2 True, 1 False, for a probability of 2/3 = 0.6667
        - 15 days: 3 True, 0 False, for a probability of 1.0
      * Subject 2: 1 trajectory with a TTE of 11 days and one that never observes the predicate, yielding...
        - 1 day: 0 True, 3 False, for a probability of 0/3 = 0.0
        - 5 days: 0 True, 2 False, for a probability of 0/2 = 0.0
        - 10 days: 0 True, 2 False, for a probability of 0/2 = 0.0
        - 15 days: 1 True, 1 False, for a probability of 1/2 = 0.5
      * Subject 3: 1 trajectory with a TTE of 12 days, yielding...
        - 1 day: 0 True, 3 False, for a probability of 0/3 = 0.0
        - 5 days: 0 True, 1 False, for a probability of 0/1 = 0.0
        - 10 days: 0 True, 1 False, for a probability of 0/1 = 0.0
        - 15 days: 1 True, 0 False, for a probability of 1.0
    For task B, we have:
      * Subject 1: 3 trajectories with TTEs of 1, 5, and 24 days, yielding...
        - 1 day: 1 True, 2 False, for a probability of 1/3 = 0.3333
        - 5 days: 2 True, 1 False, for a probability of 2/3 = 0.6667
        - 10 days: 2 True, 1 False, for a probability of 2/3 = 0.6667
        - 15 days: 2 True, 1 False, for a probability of 2/3 = 0.6667
      * Subject 2: 3 trajectories with no occurrences, yielding uniform probabilities of 0.0 for all
        durations.
      * Subject 3: 2 trajectories with TTEs of 1 day and never observed, yielding uniform probabilities of 0.5
        for all durations.

        >>> pred_ttes = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [
        ...         [timedelta(days=3), timedelta(days=6), timedelta(days=11)],
        ...         [timedelta(days=11), None],
        ...         [timedelta(days=12)],
        ...     ],
        ...     "tte/B": [
        ...         [timedelta(days=1), timedelta(days=5), timedelta(days=24)],
        ...         [None, None, None],
        ...         [timedelta(days=1), None],
        ...     ],
        ... })

    This means that we have the following distributions and AUCs:
    - For task A:
      * Duration 1 day: (False, 0.0), (False, 0.0), (False, 0.0), for an AUC of null (no positive examples)
      * Duration 5 days: (True, 0.3333), (False, 0.0), (False, 0.0), for an AUC of 1.0
      * Duration 10 days: (True, 0.6667), (True, 0.0), (False, 0.0), for an AUC of 0.75
      * Duration 15 days: (True, 1.0), (True, 0.5), (False, 1.0), for an AUC of 0.25
    - For task B:
      * Duration 1 day: (False, 0.3333), (False, 0.0), (False, 0.5), for an AUC of null (no positive examples)
      * Duration 5 days: (False, 0.6667), (True, 0.0), (False, 0.5), for an AUC of 0.0
      * Duration 10 days: (True, 0.6667), (True, 0.0), (False, 0.5), for an AUC of 0.5
      * Duration 15 days: (True, 0.6667), (True, 0.0), (True, 0.5), for an AUC of null (no negative examples)

        >>> temporal_aucs(true_tte, pred_ttes, duration_grid)
        shape: (4, 3)
        ┌──────────────┬───────┬───────┐
        │ duration     ┆ AUC/A ┆ AUC/B │
        │ ---          ┆ ---   ┆ ---   │
        │ duration[μs] ┆ f64   ┆ f64   │
        ╞══════════════╪═══════╪═══════╡
        │ 1d           ┆ null  ┆ null  │
        │ 5d           ┆ 1.0   ┆ 0.0   │
        │ 10d          ┆ 0.75  ┆ 0.5   │
        │ 15d          ┆ 0.25  ┆ null  │
        └──────────────┴───────┴───────┘
        >>> temporal_aucs(true_tte, pred_ttes, duration_grid, offset=timedelta(days=2))
        shape: (4, 3)
        ┌──────────────┬───────┬───────┐
        │ duration     ┆ AUC/A ┆ AUC/B │
        │ ---          ┆ ---   ┆ ---   │
        │ duration[μs] ┆ f64   ┆ f64   │
        ╞══════════════╪═══════╪═══════╡
        │ 1d           ┆ null  ┆ null  │
        │ 5d           ┆ 1.0   ┆ null  │
        │ 10d          ┆ 0.25  ┆ 0.75  │
        │ 15d          ┆ 0.25  ┆ 0.75  │
        └──────────────┴───────┴───────┘
        >>> true_tte_hist = true_tte.with_columns([
        ...     pl.Series("history/A", [False, True, False]),
        ...     pl.Series("history/B", [False, False, True]),
        ... ])
        >>> temporal_aucs(
        ...     true_tte_hist,
        ...     pred_ttes,
        ...     [timedelta(days=5), timedelta(days=10)],
        ...     exclude_history=True,
        ... )
        shape: (2, 3)
        ┌──────────────┬───────┬───────┐
        │ duration     ┆ AUC/A ┆ AUC/B │
        │ ---          ┆ ---   ┆ ---   │
        │ duration[μs] ┆ f64   ┆ f64   │
        ╞══════════════╪═══════╪═══════╡
        │ 5d           ┆ 1.0   ┆ 0.0   │
        │ 10d          ┆ 1.0   ┆ null  │
        └──────────────┴───────┴───────┘
    """

    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

    tasks = [c.split("/")[1] for c in true_tte.columns if c.startswith("tte/")]
    # TODO(mmd): Error validation of the input dataframes to ensure they have the expected columns and types.

    joint = true_tte.join(pred_ttes, on=ids, how="left", maintain_order="left", coalesce=True, suffix="/pred")

    def name_remapper(n: str) -> str:
        if not (n.startswith("tte/") and n.endswith("/pred")):
            return n

        task_name = n.split("/")[1]
        return f"tte_pred/{task_name}"

    # Rename the columns to be tte/task_name and tte_pred/task_name
    joint = joint.rename(name_remapper)

    tte_cols = cs.starts_with("tte/")
    tte_pred_cols = cs.starts_with("tte_pred/")

    duration_grid = get_grid(joint.select(tte_cols | tte_pred_cols), duration_grid)

    with_duration = joint.with_columns(
        pl.lit(duration_grid).alias("duration"),
        pl.lit(offset).alias("offset"),
    ).explode("duration")
    with_labels = add_labels_from_true_tte(with_duration, offset_col="offset")
    with_probs = add_probs_from_pred_ttes(with_labels, offset_col="offset")

    if exclude_history:
        exclude_set = set(tasks) if exclude_history is True else set(exclude_history)
    else:
        exclude_set = set()

    dfs_by_task = []
    for task in tasks:
        df_task = with_probs
        if task in exclude_set and f"history/{task}" in df_task.columns:
            df_task = df_task.filter(~pl.col(f"history/{task}"))

        dfs_by_task.append(
            df_task.select(
                *ids,
                "duration",
                pl.lit(task).alias("task"),
                cs.ends_with(f"/{task}").name.map(lambda n: n.split("/")[0]),
            )
        )

    df = pl.concat(dfs_by_task, how="vertical")

    if AUC_dist_approx > 0:
        n_expr = pl.min_horizontal(pl.col("prob").len(), AUC_dist_approx)
        prob_dist_expr = pl.col("prob").sample(n=n_expr, seed=seed)
    else:
        prob_dist_expr = pl.col("prob")

    def resolve_pivot_col_names(n: str) -> str:
        if not (n.startswith("{") and n.endswith("}")):
            return n

        n = n[1:-1]
        label, task = n.split(",")
        task = task[1:-1]  # Remove the quotes around the task name
        return f"{label}/{task}"

    aucs = df_AUC(
        df.group_by("duration", "task", "label")
        .agg(prob_dist_expr.sort().name.keep())
        .pivot(on=["label", "task"], index="duration", values="prob", aggregate_function=None)
        .rename(resolve_pivot_col_names)
        .sort("duration", descending=False)
    )

    out_cols = [f"AUC/{task}" for task in tasks]

    included_cols = []
    for c in out_cols:
        if c not in aucs:
            print(f"Warning: missing column {c}")
        else:
            included_cols.append(c)

    return aucs.select("duration", *included_cols)  # re-order columns
