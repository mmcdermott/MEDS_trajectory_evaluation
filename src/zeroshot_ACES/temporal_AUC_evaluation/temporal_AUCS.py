from datetime import timedelta

import polars as pl
from meds import LabelSchema


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
        df: A DataFrame with task index columns followed by two columns "true" and "false" containing lists of
            probabilities for the positive and negative classes, respectively. The lists must be sorted in
            ascending order.

    Returns:
        A DataFrame with the same task index columns and a new column "AUC" containing the computed AUC for
        each task.

    Examples:
        >>> df = pl.DataFrame({
        ...     "task": ["task1", "task2"],
        ...     "true": [[0.1, 0.5, 0.8], [0.3, 0.5]],
        ...     "false": [[0.2, 0.3], [0.4, 0.9]]
        ... })

    In this example, the DataFrame `df` contains two tasks with their respective true and false probabilities.
    Task 1 has true probabilities of [0.1, 0.5, 0.8] and false probabilities of [0.2, 0.3], which results in
    four correctly ordered pairs (0.5 > 0.2, 0.5 > 0.3, 0.8 > 0.2, 0.8 > 0.3) out of a total of six pairs, for
    an AUC of 4/6 = 0.6667.
    Task 2 has true probabilities of [0.3, 0.5] and false probabilities of [0.4, 0.9], which results in
    one correctly ordered pair (0.5 > 0.4) out of a total of four pairs, for an AUC of 1/4 = 0.25.

        >>> df_AUC(df)
        shape: (2, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC      │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.666667 │
        │ task2 ┆ 0.25     │
        └───────┴──────────┘

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
        ...     "true": [[0.2, 0.2, 0.4, 0.6], [0.8]],
        ...     "false": [[0.1, 0.3, 0.3], [0.1, 0.4, 0.4, 0.9]]
        ... })
        >>> df_AUC(df)
        shape: (2, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC      │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.666667 │
        │ task2 ┆ 0.75     │
        └───────┴──────────┘

    If we have an empty distribution, the AUC returned is `null`:

        >>> df_empty = pl.DataFrame({
        ...     "task": ["task1", "task2", "task3"],
        ...     "true": [[0.1, 0.5, 0.8], [], [0.3, 0.5]],
        ...     "false": [[0.2, 0.3], [0.4, 0.9], []]
        ... })
        >>> df_AUC(df_empty)
        shape: (3, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC      │
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
        ...     "true": [[0.2, 0.3, 0.5], [0.9]],
        ...     "false": [[0.2, 0.3, 0.3], [0.4, 0.9]]
        ... })
        >>> df_AUC(df_ties)
        shape: (2, 2)
        ┌───────┬──────────┐
        │ task  ┆ AUC      │
        │ ---   ┆ ---      │
        │ str   ┆ f64      │
        ╞═══════╪══════════╡
        │ task1 ┆ 0.611111 │
        │ task2 ┆ 0.75     │
        └───────┴──────────┘
    """

    ids = [c for c in df.columns if c not in {"true", "false"}]

    ordered_pairs = pl.col("false").search_sorted(pl.col("true").first(), side="right").first()
    equal_pairs = (pl.col("false") == pl.col("true")).sum()

    AUC = (pl.col("num_ordered_pairs") - (pl.col("num_equal_pairs") / 2)) / pl.col("num_pairs")

    def non_num(col: str) -> pl.Expr:
        return pl.col(col).is_infinite() | pl.col(col).is_nan()

    return (
        df.with_columns((pl.col("true").list.len() * pl.col("false").list.len()).alias("num_pairs"))
        .explode("true")
        .with_row_index("__idx")
        .explode("false")
        .group_by([*ids, "num_pairs", "__idx", "true"], maintain_order=True)
        .agg(ordered_pairs.alias("num_ordered_pairs"), equal_pairs.alias("num_equal_pairs"))
        .group_by([*ids, "num_pairs"], maintain_order=True)
        .agg(
            pl.col("num_ordered_pairs").sum().alias("num_ordered_pairs"),
            pl.col("num_equal_pairs").sum().alias("num_equal_pairs"),
        )
        .select(*ids, AUC.alias("AUC"))
        .select(*ids, pl.when(non_num("AUC")).then(None).otherwise(pl.col("AUC")).alias("AUC"))
    )


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


def add_labels_from_true_tte(df: pl.DataFrame) -> pl.DataFrame:
    """Convert the true time-to-predicate values into a label for a given duration window.

    Given a dataframe with a column `tte` containing the true time-to-predicate value and a column `duration`
    containing the duration for which the AUC should be computed, this function computes the label of whether
    or not the predicate occurred within the given duration.

    Arguments:
        df: A DataFrame with columns "tte" (a time-to-predicate value) and "duration"
            (the duration for which the label should be computed).

    Returns:
        A DataFrame with the same columns as the input except for `tte`, plus a new column "label"
        containing the label for each row.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte": [timedelta(days=5), timedelta(days=10), None],
        ...     "duration": [timedelta(days=7), timedelta(days=8), timedelta(days=8)]
        ... })
        >>> add_labels_from_true_tte(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬───────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ label │
        │ ---        ┆ ---                 ┆ ---          ┆ ---   │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ bool  │
        ╞════════════╪═════════════════════╪══════════════╪═══════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ true  │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ false │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ false │
        └────────────┴─────────────────────┴──────────────┴───────┘
    """

    label_expr = (pl.col("tte") <= pl.col("duration")).fill_null(False)

    return df.with_columns(label_expr.alias("label")).drop("tte")


def add_probs_from_pred_ttes(df: pl.DataFrame) -> pl.DataFrame:
    """Convert the list of predicted time-to-predicate values into a probability distribution.

    Given a dataframe with a column `tte_pred` containing lists of predicted time-to-predicate values and a
    column `duration` containing the duration for which the AUC should be computed, this function computes
    the probability (proportion of sampled trajectories which satisfy) that the time-to-predicate is less than
    or equal to the duration for each trajectory.

    Arguments:
        df: A DataFrame with columns "tte_pred" (a list of predicted time-to-predicate values) and "duration"
            (the duration for which the probabilities should be computed).

    Returns:
        A DataFrame with the same columns as the input except for `tte_pred`, plus a new column "prob"
        containing the computed probabilities for each row.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte_pred": [
        ...         [timedelta(days=5), timedelta(days=10), None],
        ...         [timedelta(days=7), timedelta(days=8)],
        ...         [timedelta(days=8), timedelta(days=9), timedelta(days=10)]
        ...     ],
        ...     "duration": [timedelta(days=7), timedelta(days=8), timedelta(days=8)]
        ... })
        >>> add_probs_from_pred_ttes(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────┐
        │ subject_id ┆ prediction_time     ┆ duration     ┆ prob     │
        │ ---        ┆ ---                 ┆ ---          ┆ ---      │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ f64      │
        ╞════════════╪═════════════════════╪══════════════╪══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 7d           ┆ 0.333333 │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 8d           ┆ 1.0      │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ 8d           ┆ 0.333333 │
        └────────────┴─────────────────────┴──────────────┴──────────┘
    """

    num_trajectories_within_duration = (
        pl.col("tte_pred")
        .list.sort(descending=False, nulls_last=True)
        .explode()
        .fill_null(pl.lit(timedelta.max))
        .search_sorted(pl.col("duration"), side="right")
    )
    prob_expr = num_trajectories_within_duration / (pl.col("tte_pred").list.len())

    return (
        df.with_row_count("__idx")
        .with_columns(prob_expr.over("__idx").alias("prob"))
        .drop("__idx", "tte_pred")
    )


def temporal_aucs(
    true_tte: pl.DataFrame,
    pred_ttes: pl.DataFrame,
    duration_grid: str | int | None | list[timedelta] = 10000,
    AUC_dist_approx: int = -1,
    seed: int = 0,
) -> pl.DataFrame:
    """Compute the AUC over different prediction windows for the first occurrence of a predicate.

    Parameters:
        true_tte: The true time-to-first-predicate, with columns "subject_id", "prediction_time", and "tte". A
            `null` value in the "tte" column indicates that the predicate did not occur.
        pred_ttes: The predicted time-to-first-predicate, with columns "subject_id", "prediction_time", and
            "tte", the latter being a list of observed time-to-predicate values for the different generated
            trajectories. A `null` value in the "tte" column indicates that the predicate did not occur.
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

    Examples:
        >>> duration_grid = [timedelta(days=1), timedelta(days=5), timedelta(days=10), timedelta(days=15)]

    We'll begin with a duration grid spanning 1, 5, 10, and 15 days. This means that we will output 4 AUCs:
    one for a prediction window of 1 day, one for 5 days, one for 10 days, and one for 15 days. We'll
    construct a setting which has true labels of:
      * Duration 1 day: subject 1 is False, subject 2 is False, subject 3 is False.
      * Duration 5 days: subject 1 is True, subject 2 is False, subject 3 is False.
      * Duration 10 days: subject 1 is True, subject 2 is True, subject 3 is False.
      * Duration 15 days: subject 1 is True, subject 2 is True, subject 3 is False.

        >>> true_tte = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte": [timedelta(days=5), timedelta(days=10), None]
        ... })

    We'll also construct a set of predicted time-to-event values, which are lists of observed
    time-to-predicate values for the different generated trajectories. In this case, we have
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

        >>> pred_ttes = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte": [
        ...         [timedelta(days=3), timedelta(days=6), timedelta(days=11)],
        ...         [timedelta(days=11), None],
        ...         [timedelta(days=12)],
        ...     ],
        ... })

    This means that we have the following distributions and AUCs:
      * Duration 1 day: (False, 0.0), (False, 0.0), (False, 0.0), for an AUC of null (no positive examples)
      * Duration 5 days: (True, 0.3333), (False, 0.0), (False, 0.0), for an AUC of 1.0
      * Duration 10 days: (True, 0.6667), (True, 0.0), (False, 0.0), for an AUC of 0.75
      * Duration 15 days: (True, 1.0), (True, 0.5), (False, 1.0), for an AUC of 0.25

        >>> temporal_aucs(true_tte, pred_ttes, duration_grid)
        shape: (4, 2)
        ┌──────────────┬──────┐
        │ duration     ┆ AUC  │
        │ ---          ┆ ---  │
        │ duration[μs] ┆ f64  │
        ╞══════════════╪══════╡
        │ 1d           ┆ null │
        │ 5d           ┆ 1.0  │
        │ 10d          ┆ 0.75 │
        │ 15d          ┆ 0.25 │
        └──────────────┴──────┘
    """

    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

    joint = true_tte.join(pred_ttes, on=ids, how="left", maintain_order="left", coalesce=True, suffix="_pred")

    duration_grid = get_grid(joint.select("tte", "tte_pred"), duration_grid)

    with_duration = joint.with_columns(pl.lit(duration_grid).alias("duration")).explode("duration")
    with_labels = add_labels_from_true_tte(with_duration)
    with_probs = add_probs_from_pred_ttes(with_labels)

    if AUC_dist_approx > 0:
        prob_dist_expr = pl.col("prob").sample(n=AUC_dist_approx, seed=seed)
    else:
        prob_dist_expr = pl.col("prob")

    return df_AUC(
        with_probs.group_by("duration", "label", maintain_order=True)
        .agg(prob_dist_expr.sort().alias("prob"))
        .pivot(on="label", index="duration", values="prob", aggregate_function=None, maintain_order=True)
    )
