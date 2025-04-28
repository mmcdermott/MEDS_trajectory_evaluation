import tempfile

import polars as pl
from aces.extract_subtree import extract_subtree
from aces.predicates import get_predicates_df
from meds import prediction_time_field, subject_id_field, time_field
from omegaconf import DictConfig

from .task_config import ZeroShotTaskConfig


def get_input_subtree_anchor_realizations(raw_trajectories: pl.DataFrame) -> pl.DataFrame:
    """Formats the raw input trajectories as subtree anchor realizations for ACES.

    Args:
        raw_trajectories: The raw input trajectories as a Polars DataFrame.

    Returns:
        A Polars DataFrame containing the subtree anchor realizations. Subtree anchor realization dataframes
        in ACES include a "subtree_anchor_timestamp" column, which is the timestamp of the subtree anchor
        realization. No rows where this is null are included, and all rows with this populated are assumed to
        be valid subtree anchor realizations. This and subject_id are the only columns.

    Examples:
        >>> raw_trajectories = sample_labeled_trajectories_dfs["trajectory_0.parquet"]
        >>> raw_trajectories
        shape: (9, 5)
        ┌─────────────────────────┬───────────────┬───────────────┬────────────┬─────────────────────────┐
        │ time                    ┆ code          ┆ numeric_value ┆ subject_id ┆ prediction_time         │
        │ ---                     ┆ ---           ┆ ---           ┆ ---        ┆ ---                     │
        │ datetime[μs, UTC]       ┆ str           ┆ f64           ┆ i32        ┆ datetime[μs, UTC]       │
        ╞═════════════════════════╪═══════════════╪═══════════════╪════════════╪═════════════════════════╡
        │ 1993-01-01 12:00:00 UTC ┆ LAB_1         ┆ 1.0           ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
        │ 1993-01-01 13:00:00 UTC ┆ LAB_2         ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
        │ 1993-01-01 14:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
        │ 1993-01-22 00:00:00 UTC ┆ MEDS_DEATH    ┆ null          ┆ 1          ┆ 1993-01-01 00:00:00 UTC │
        │ 1993-02-20 00:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
        │ 1995-01-01 00:00:00 UTC ┆ LAB_23        ┆ 1.2           ┆ 1          ┆ 1993-01-20 00:00:00 UTC │
        │ 1999-01-01 13:00:00 UTC ┆ LAB_3         ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
        │ 1999-01-01 14:00:00 UTC ┆ ICU_DISCHARGE ┆ null          ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
        │ 1999-01-04 14:00:00 UTC ┆ LAB_4         ┆ 1.1           ┆ 2          ┆ 1999-01-01 00:00:00 UTC │
        └─────────────────────────┴───────────────┴───────────────┴────────────┴─────────────────────────┘
        >>> get_input_subtree_anchor_realizations(raw_trajectories)
        shape: (3, 2)
        ┌─────────────────────────────┬──────────────────────────┐
        │ subject_id                  ┆ subtree_anchor_timestamp │
        │ ---                         ┆ ---                      │
        │ struct[2]                   ┆ datetime[μs, UTC]        │
        ╞═════════════════════════════╪══════════════════════════╡
        │ {1,1993-01-01 00:00:00 UTC} ┆ 1993-01-01 12:00:00 UTC  │
        │ {1,1993-01-20 00:00:00 UTC} ┆ 1993-02-20 00:00:00 UTC  │
        │ {2,1999-01-01 00:00:00 UTC} ┆ 1999-01-01 13:00:00 UTC  │
        └─────────────────────────────┴──────────────────────────┘
    """

    new_subj_id = pl.struct(subject_id_field, prediction_time_field)

    return (
        raw_trajectories.with_row_index("__idx")
        .filter(pl.col("__idx") == pl.col("__idx").min().over(subject_id_field, prediction_time_field))
        .select(
            new_subj_id.alias(subject_id_field),
            pl.col(time_field).alias("subtree_anchor_timestamp"),
            # pl.col(prediction_time_field).alias("subtree_anchor_timestamp")
        )
    )


def label_trajectories(
    trajectories: pl.DataFrame,
    zero_shot_task_cfg: ZeroShotTaskConfig,
) -> pl.DataFrame:
    """Takes a dataframe of trajectories and a zero-shot task configuration and returns the labels.

    Args:
        trajectories: A dataframe containing the trajectories to be labeled.
        zero_shot_task_cfg: The zero-shot task configuration to use for labeling.

    Returns:
        A dataframe with the labels each trajectory evaluates to for the given config.
    """

    new_subj_id = pl.struct(subject_id_field, prediction_time_field)

    reformatted_trajectories = trajectories.select(
        new_subj_id.alias(subject_id_field),
        *[c for c in trajectories.columns if c not in {subject_id_field, prediction_time_field}],
    )

    subtree_anchor_realizations = get_input_subtree_anchor_realizations(trajectories)

    with tempfile.NamedTemporaryFile(suffix=".parquet") as data_fp:
        # TODO: This is very stupid. We should just modify ACES to be able to get the predicates from a MEDS
        # dataframe directly.

        reformatted_trajectories.write_parquet(data_fp.name, use_pyarrow=True)

        data_config = DictConfig(
            {
                "path": data_fp.name,
                "standard": "meds",
            }
        )

        predicates_df = get_predicates_df(zero_shot_task_cfg, data_config)

    label_window = zero_shot_task_cfg.label_window
    label_predicate = zero_shot_task_cfg.windows[label_window].label

    label_col = pl.col(f"{label_window}.end_summary").struct.field(label_predicate)

    aces_results = (
        extract_subtree(zero_shot_task_cfg.window_tree, subtree_anchor_realizations, predicates_df)
        .unnest(subject_id_field)
        .select(
            subject_id_field,
            prediction_time_field,
            pl.lit(True).alias("valid"),
            label_col.is_not_null().alias("determinable"),
            pl.when(label_col.is_not_null()).then(label_col > 0).alias("label"),
        )
    )

    none_lit = pl.lit(None, dtype=pl.Boolean)

    return (
        subtree_anchor_realizations.unnest(subject_id_field)
        .select(
            subject_id_field,
            prediction_time_field,
            none_lit.alias("valid"),
            none_lit.alias("determinable"),
            none_lit.alias("label"),
        )
        .update(
            aces_results,
            on=[subject_id_field, prediction_time_field],
            how="left",
            maintain_order="left",
        )
        .select(
            subject_id_field,
            prediction_time_field,
            pl.col("valid").fill_null(False).alias("valid"),
            pl.col("determinable"),
            pl.col("label"),
        )
    )
