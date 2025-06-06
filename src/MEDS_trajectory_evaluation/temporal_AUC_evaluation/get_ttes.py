"""Utilities to construct true and trajectory observed time-to-events for ACES predicates."""

from collections.abc import Callable

import polars as pl
import polars.selectors as cs
from aces.config import PlainPredicateConfig
from meds import DataSchema, LabelSchema

from ..aces_utils import get_MEDS_plain_predicates

PREDICATES_T = dict[str, PlainPredicateConfig]

POSSIBLE_IDS = {DataSchema.subject_id_name, LabelSchema.prediction_time_name}


def get_all_predicate_times(
    MEDS_df: pl.DataFrame,
    predicates: PREDICATES_T,
) -> pl.DataFrame:
    """Extracts all predicate times for the given MEDS dataset and predicates.

    Args:
        MEDS_df: A MEDS data schema DataFrame containing the patient data.
        predicates: A dictionary of ACES predicates to be extracted from the MEDS data.

    Returns:
        A DataFrame keyed by subject ID (and or prediction time, if present) with columns for each predicate,
        containing a list of the times when each predicate occurs in the data corresponding to those IDs.

    Examples:
        >>> MEDS_df = pl.DataFrame({
        ...     'subject_id': [
        ...         1, 1, 1, 1,
        ...         2,
        ...         3, 3, 3,
        ...     ],
        ...     'time': [
        ...         datetime(2020, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3), datetime(2022, 1, 4),
        ...         datetime(2022, 1, 1),
        ...         datetime(2001, 1, 1), datetime(2002, 1, 2), datetime(2002, 1, 3),
        ...     ],
        ...     'code': [
        ...         'icd9//150.1', 'icd9//400', 'icd9//250.3', 'icd9//250.5',
        ...         'icd9//250.2',
        ...         'icd9//250.1', 'icd9//400.1', 'icd9//400',
        ...     ],
        ... })
        >>> predicates = {
        ...     '250.*': PlainPredicateConfig(code={"regex": "250.*"}),
        ...     '400': PlainPredicateConfig(code="icd9//400"),
        ... }

    With these inputs
      * subject 1 has 250.* at 2022-1-3 and 2022-1-4 and has 400 at 2022-1-2
      * subject 2 has 250.* at 2022-1-1 and no 400,
      * subject 3 has 250.* at 2001-1-1 and 400 at 2002-1-3

        >>> get_all_predicate_times(MEDS_df, predicates)
        shape: (3, 3)
        ┌────────────┬─────────────────────────────────┬───────────────────────┐
        │ subject_id ┆ 250.*                           ┆ 400                   │
        │ ---        ┆ ---                             ┆ ---                   │
        │ i64        ┆ list[datetime[μs]]              ┆ list[datetime[μs]]    │
        ╞════════════╪═════════════════════════════════╪═══════════════════════╡
        │ 1          ┆ [2022-01-03 00:00:00, 2022-01-… ┆ [2022-01-02 00:00:00] │
        │ 2          ┆ [2022-01-01 00:00:00]           ┆ []                    │
        │ 3          ┆ [2001-01-01 00:00:00]           ┆ [2002-01-03 00:00:00] │
        └────────────┴─────────────────────────────────┴───────────────────────┘
    """

    predicate_names = list(predicates.keys())

    id_cols = list(POSSIBLE_IDS.intersection(MEDS_df.columns))

    return (
        get_MEDS_plain_predicates(MEDS_df, predicates)
        .filter(pl.any_horizontal(pl.col(name) for name in predicate_names))
        .select(
            *id_cols,
            *[pl.when(pl.col(n)).then(pl.col(DataSchema.time_name)).alias(n) for n in predicate_names],
        )
        .group_by(id_cols, maintain_order=True)
        .agg(
            *[pl.col(n).drop_nulls().alias(n) for n in predicate_names],
        )
    )


def get_raw_tte(
    MEDS_df: pl.DataFrame,
    index_df: pl.DataFrame,
    predicates: PREDICATES_T,
    *,
    include_history: bool = False,
) -> pl.DataFrame:
    """Extracts the time-to-predicate values for the given MEDS dataset, index dataframe, and predicates.

    Args:
        MEDS_df: A MEDS data schema DataFrame containing the patient data.
        index_df: DataFrame containing the index of patients and their trajectories. This is in the MEDS Label
            schema (though it need not have any labels in it, merely the subject IDs and prediction times).
        predicates: A dictionary of ACES predicates to be extracted from the MEDS data.

    Returns:
        A DataFrame in the same order as the index dataframe, with the subject ID and prediction times
        included, and with a column `tte/${predicate}` containing the time-to-first predicate for each
        predicate in the config.
        If ``include_history`` is ``True`` an additional column ``history/${predicate}``
        is included which indicates whether the subject had an instance of the predicate at or before the
        prediction time.

    Examples:
        >>> MEDS_df = pl.DataFrame({
        ...     'subject_id': [
        ...         1, 1, 1, 1,
        ...         2,
        ...         3, 3, 3, 3,
        ...     ],
        ...     'time': [
        ...         datetime(2020, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3), datetime(2022, 1, 4),
        ...         datetime(2022, 1, 1),
        ...         datetime(2001, 1, 1), datetime(2002, 1, 2), datetime(2002, 1, 3), datetime(2002, 1, 3),
        ...     ],
        ...     'code': [
        ...         'icd9//150.1', 'icd9//400', 'icd9//250.3', 'icd9//250.5',
        ...         'icd9//250.2',
        ...         'icd9//250.1', 'icd9//400.1', 'icd9//400', 'icd9//400',
        ...     ],
        ... })
        >>> index_df = pl.DataFrame({
        ...     'subject_id': [1, 1, 2, 3, 3],
        ...     'prediction_time': [
        ...         datetime(2022, 1, 1), datetime(2022, 1, 3, 1),
        ...         datetime(2022, 1, 1),
        ...         datetime(2022, 1, 1), datetime(2000, 1, 1),
        ...     ],
        ... })
        >>> predicates = {
        ...     '250.*': PlainPredicateConfig(code={"regex": "250.*"}),
        ...     '400': PlainPredicateConfig(code="icd9//400"),
        ... }

    With these inputs, we want to identify that the first time after the (shared) index date of 2022-1-1 for
    subject 1 occurs at 2022-1-3 for 250.* and at 2022-1-2 for 400, while for subject 2, the first time after
    2022-1-1 occurs at 2022-1-1 for 250.* and there is no time after 2022-1-1 for 400, and for subject 3,
    there are no times after 2022-1-1 for either predicate.

        >>> get_raw_tte(MEDS_df, index_df, predicates)
        shape: (5, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ prediction_time     ┆ tte/250.*    ┆ tte/400      │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ duration[μs] │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ 2d           ┆ 1d           │
        │ 1          ┆ 2022-01-03 01:00:00 ┆ 23h          ┆ null         │
        │ 2          ┆ 2022-01-01 00:00:00 ┆ null         ┆ null         │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ null         ┆ null         │
        │ 3          ┆ 2000-01-01 00:00:00 ┆ 366d         ┆ 733d         │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
        >>> get_raw_tte(MEDS_df, index_df, predicates, include_history=True)
        shape: (5, 6)
        ┌────────────┬─────────────────────┬───────────────┬─────────────┬──────────────┬──────────────┐
        │ subject_id ┆ prediction_time     ┆ history/250.* ┆ history/400 ┆ tte/250.*    ┆ tte/400      │
        │ ---        ┆ ---                 ┆ ---           ┆ ---         ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ bool          ┆ bool        ┆ duration[μs] ┆ duration[μs] │
        ╞════════════╪═════════════════════╪═══════════════╪═════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ false         ┆ false       ┆ 2d           ┆ 1d           │
        │ 1          ┆ 2022-01-03 01:00:00 ┆ true          ┆ true        ┆ 23h          ┆ null         │
        │ 2          ┆ 2022-01-01 00:00:00 ┆ true          ┆ true        ┆ null         ┆ null         │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ true          ┆ true        ┆ null         ┆ null         │
        │ 3          ┆ 2000-01-01 00:00:00 ┆ false         ┆ false       ┆ 366d         ┆ 733d         │
        └────────────┴─────────────────────┴───────────────┴─────────────┴──────────────┴──────────────┘
    """

    df = get_all_predicate_times(MEDS_df, predicates).join(
        index_df.select(LabelSchema.subject_id_name, LabelSchema.prediction_time_name),
        on=DataSchema.subject_id_name,
        how="right",
        maintain_order="right",
        coalesce=True,
    )

    predicate_idx_exprs = [
        (
            (pl.col(name).explode().search_sorted(pl.col(LabelSchema.prediction_time_name), side="right"))
            .over(LabelSchema.subject_id_name, pl.col(LabelSchema.prediction_time_name))
            .alias(f"{name}/idx")
        )
        for name in predicates
    ]

    predicate_time_exprs = {
        name: pl.col(name).list.get(pl.col(f"{name}/idx"), null_on_oob=True) for name in predicates
    }

    predicate_tte_exprs = [
        (time_expr - pl.col(LabelSchema.prediction_time_name)).alias(f"tte/{name}")
        for name, time_expr in predicate_time_exprs.items()
    ]

    history_exprs = []
    if include_history:
        history_exprs = [(pl.col(f"{name}/idx") > 0).alias(f"history/{name}") for name in predicates]

    return df.with_columns(*predicate_idx_exprs).select(
        LabelSchema.subject_id_name,
        LabelSchema.prediction_time_name,
        *history_exprs,
        *predicate_tte_exprs,
    )


def get_trajectory_tte(
    trajectory_df: pl.DataFrame,
    predicates: PREDICATES_T,
) -> pl.DataFrame:
    """Similar to `get_raw_tte`, but for a trajectory DataFrame.

    Args:
        trajectory_df: A DataFrame containing the trajectory data, which should have columns for subject ID,
            prediction time, and future MEDS formatted events.
        predicates: A dictionary of ACES predicates to be extracted from the trajectory data.

    Returns:
        A DataFrame in the same format as `get_raw_tte`, with the subject IDs, prediction times, and
        time-to-predicate values for each predicate in the config.

    Examples:
        >>> trajectory_df = pl.DataFrame({
        ...     'subject_id': [
        ...         1, 1, 1, 1,
        ...         3, 3, 3, 3,
        ...     ],
        ...     'prediction_time': [
        ...         datetime(2022, 1, 1), datetime(2022, 1, 1),
        ...         datetime(2022, 1, 3, 1), datetime(2022, 1, 3, 1),
        ...         datetime(2022, 1, 1), datetime(2022, 1, 1),
        ...         datetime(2000, 1, 1), datetime(2000, 1, 1),
        ...     ],
        ...     'time': [
        ...         datetime(2022, 1, 2), datetime(2022, 1, 3),
        ...         datetime(2022, 1, 4), datetime(2022, 1, 4),
        ...         datetime(2022, 1, 2), datetime(2022, 1, 3),
        ...         datetime(2001, 1, 1), datetime(2002, 1, 2),
        ...     ],
        ...     'code': [
        ...         'icd9//250.3', 'icd9//400',
        ...         'icd9//250.5', 'icd9//250.1',
        ...         'icd9//400', 'icd9//251.2',
        ...         'icd9//150.1', 'icd9//400.3',
        ...     ],
        ... })
        >>> predicates = {
        ...     '250.*': PlainPredicateConfig(code={"regex": "250.*"}),
        ...     '400': PlainPredicateConfig(code="icd9//400"),
        ... }
        >>> get_trajectory_tte(trajectory_df, predicates)
        shape: (4, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ prediction_time     ┆ tte/250.*    ┆ tte/400      │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ duration[μs] ┆ duration[μs] │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2022-01-01 00:00:00 ┆ 1d           ┆ 2d           │
        │ 1          ┆ 2022-01-03 01:00:00 ┆ 23h          ┆ null         │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ null         ┆ 1d           │
        │ 3          ┆ 2000-01-01 00:00:00 ┆ null         ┆ null         │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
    """

    tte_exprs = [
        (
            pl.col(DataSchema.time_name).filter(pl.col(name)).first()
            - pl.col(LabelSchema.prediction_time_name).first()
        ).alias(f"tte/{name}")
        for name in predicates
    ]

    return (
        get_MEDS_plain_predicates(trajectory_df, predicates)
        .group_by(LabelSchema.subject_id_name, LabelSchema.prediction_time_name, maintain_order=True)
        .agg(*tte_exprs)
    )


def merge_pred_ttes(pred_tte_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Merges multiple predicted time-to-predicate DataFrames into one with a list column.

    Args:
        pred_tte_dfs: A list of DataFrames, each containing predicted time-to-predicate values for different
            predicates, and identifier columns for subject ID and prediction time.

    Returns:
        A single DataFrame with the subject ID and prediction time columns, and a list column for each
        predicate containing the predicted time-to-event values.

    Examples:
        >>> df_1 = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [timedelta(days=5), timedelta(days=10), None],
        ...     "tte/B": [timedelta(days=2), timedelta(days=3), timedelta(days=15)],
        ... })
        >>> df_2 = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [timedelta(days=3), timedelta(days=1), timedelta(days=20)],
        ...     "tte/B": [timedelta(days=10), timedelta(days=45), None],
        ... })
        >>> df_3 = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "tte/A": [None, timedelta(days=1), None],
        ...     "tte/B": [timedelta(days=1), timedelta(days=3), timedelta(days=5)],
        ... })
        >>> merge_pred_ttes([df_1, df_2, df_3])
        shape: (3, 4)
        ┌────────────┬─────────────────────┬────────────────────┬────────────────────┐
        │ subject_id ┆ prediction_time     ┆ tte/A              ┆ tte/B              │
        │ ---        ┆ ---                 ┆ ---                ┆ ---                │
        │ i64        ┆ datetime[μs]        ┆ list[duration[μs]] ┆ list[duration[μs]] │
        ╞════════════╪═════════════════════╪════════════════════╪════════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ [5d, 3d, null]     ┆ [2d, 10d, 1d]      │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ [10d, 1d, 1d]      ┆ [3d, 45d, 3d]      │
        │ 3          ┆ 2021-01-03 00:00:00 ┆ [null, 20d, null]  ┆ [15d, null, 5d]    │
        └────────────┴─────────────────────┴────────────────────┴────────────────────┘
    """

    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]
    df_chunks = [pred_tte_dfs[0].select(*ids)]

    pfx = "tte/"
    tasks = [c.removeprefix(pfx) for c in pred_tte_dfs[0].columns if c not in ids]

    def add_suffix_fntr(sfx: str) -> Callable[[str], str]:
        """Returns a function to add a suffix to a column name."""
        return lambda n: f"{n}/{sfx}"

    for i, df in enumerate(pred_tte_dfs):
        df_chunks.append(df.drop(*ids).rename(add_suffix_fntr(i)))

    df = pl.concat(df_chunks, how="horizontal")

    task_exprs = [pl.concat_list(cs.starts_with(f"tte/{task}/")).alias(f"tte/{task}") for task in tasks]

    return df.select(*ids, *task_exprs)
