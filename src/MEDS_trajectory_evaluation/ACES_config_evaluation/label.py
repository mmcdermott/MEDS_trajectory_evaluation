import polars as pl
from aces.config import TaskExtractorConfig
from aces.extract_subtree import extract_subtree
from meds import LabelSchema

from ..aces_utils import get_MEDS_predicates
from .task_config import ZeroShotTaskConfig

SUBJ_AND_PRED_TIME = pl.struct(LabelSchema.subject_id_name, LabelSchema.prediction_time_name).alias(
    LabelSchema.subject_id_name
)


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
        │ {1,1993-01-01 00:00:00 UTC} ┆ 1993-01-01 00:00:00 UTC  │
        │ {1,1993-01-20 00:00:00 UTC} ┆ 1993-01-20 00:00:00 UTC  │
        │ {2,1999-01-01 00:00:00 UTC} ┆ 1999-01-01 00:00:00 UTC  │
        └─────────────────────────────┴──────────────────────────┘
    """

    return raw_trajectories.select(
        SUBJ_AND_PRED_TIME,
        pl.col(LabelSchema.prediction_time_name).alias("subtree_anchor_timestamp"),
    ).unique(maintain_order=True)


def get_predicates_and_anchor_realizations(
    trajectories: pl.DataFrame,
    task_cfg: TaskExtractorConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extracts predicates and subtree anchor realizations from the input trajectories.

    Performs the following:
      1. reformats the subject-ID and prediction-time inputs into a new subject ID for independent trajectory
         labeling
      2. extracts subtree anchor realizations given the prediction timestamps in the input trajectories
      3. extracts predicates from the generated trajectories
      4. merges the anchor realizations in with the predicates as needed.

    Args:
        trajectories: A Polars DataFrame containing the input trajectories.
        zero_shot_task_cfg: The zero-shot task configuration to use for labeling.

    Returns:
        The subtree anchor realizations and predicates extracted from the input trajectories.

    Examples:
        >>> _ = pl.Config.set_tbl_rows(-1)
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
        >>> sample_ACES_cfg.predicates
        {'icu_admission': PlainPredicateConfig(code='ICU_ADMISSION', ...),
         'icu_discharge': PlainPredicateConfig(code='ICU_DISCHARGE', ...),
         'death': PlainPredicateConfig(code={'regex': 'MEDS_DEATH.*'}, ...),
         'discharge_or_death': DerivedPredicateConfig(expr='or(icu_discharge, death)', ...)}
        >>> realizations, preds = get_predicates_and_anchor_realizations(raw_trajectories, sample_ACES_cfg)
        >>> realizations
        shape: (3, 2)
        ┌─────────────────────────────┬──────────────────────────┐
        │ subject_id                  ┆ subtree_anchor_timestamp │
        │ ---                         ┆ ---                      │
        │ struct[2]                   ┆ datetime[μs, UTC]        │
        ╞═════════════════════════════╪══════════════════════════╡
        │ {1,1993-01-01 00:00:00 UTC} ┆ 1993-01-01 00:00:00 UTC  │
        │ {1,1993-01-20 00:00:00 UTC} ┆ 1993-01-20 00:00:00 UTC  │
        │ {2,1999-01-01 00:00:00 UTC} ┆ 1999-01-01 00:00:00 UTC  │
        └─────────────────────────────┴──────────────────────────┘
        >>> preds
        shape: (12, 6)
        ┌───────────────┬─────────────────────┬───────────────┬───────────────┬───────┬────────────────────┐
        │ subject_id    ┆ timestamp           ┆ icu_admission ┆ icu_discharge ┆ death ┆ discharge_or_death │
        │ ---           ┆ ---                 ┆ ---           ┆ ---           ┆ ---   ┆ ---                │
        │ struct[2]     ┆ datetime[μs, UTC]   ┆ i64           ┆ i64           ┆ i64   ┆ i64                │
        ╞═══════════════╪═════════════════════╪═══════════════╪═══════════════╪═══════╪════════════════════╡
        │ {1,1993-01-01 ┆ 1993-01-01 00:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-01 ┆ 1993-01-01 12:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-01 ┆ 1993-01-01 13:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-01 ┆ 1993-01-01 14:00:00 ┆ 0             ┆ 1             ┆ 0     ┆ 1                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-01 ┆ 1993-01-22 00:00:00 ┆ 0             ┆ 0             ┆ 1     ┆ 1                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-20 ┆ 1993-01-20 00:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-20 ┆ 1993-02-20 00:00:00 ┆ 0             ┆ 1             ┆ 0     ┆ 1                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {1,1993-01-20 ┆ 1995-01-01 00:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {2,1999-01-01 ┆ 1999-01-01 00:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {2,1999-01-01 ┆ 1999-01-01 13:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {2,1999-01-01 ┆ 1999-01-01 14:00:00 ┆ 0             ┆ 1             ┆ 0     ┆ 1                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        │ {2,1999-01-01 ┆ 1999-01-04 14:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 00:00:00 UTC} ┆ UTC                 ┆               ┆               ┆       ┆                    │
        └───────────────┴─────────────────────┴───────────────┴───────────────┴───────┴────────────────────┘
    """

    subtree_anchor_realizations = get_input_subtree_anchor_realizations(trajectories)

    reformatted_trajectories = trajectories.with_columns(SUBJ_AND_PRED_TIME).drop(
        LabelSchema.prediction_time_name
    )

    predicates_df = (
        get_MEDS_predicates(reformatted_trajectories, task_cfg)
        .join(
            subtree_anchor_realizations.rename({"subtree_anchor_timestamp": "timestamp"}),
            on=[LabelSchema.subject_id_name, "timestamp"],
            coalesce=True,
            how="full",
            maintain_order="left",
        )
        .sort((LabelSchema.subject_id_name, "timestamp"), maintain_order=True)
        .fill_null(0)
    )

    return subtree_anchor_realizations, predicates_df


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

    subtree_anchor_realizations, predicates_df = get_predicates_and_anchor_realizations(
        trajectories, zero_shot_task_cfg
    )

    label_window = zero_shot_task_cfg.label_window
    label_predicate = zero_shot_task_cfg.windows[label_window].label

    label_col = pl.col(f"{label_window}.end_summary").struct.field(label_predicate)

    aces_results = (
        extract_subtree(zero_shot_task_cfg.window_tree, subtree_anchor_realizations, predicates_df)
        .unnest(LabelSchema.subject_id_name)
        .select(
            LabelSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            pl.lit(True).alias("valid"),
            label_col.is_not_null().alias("determinable"),
            pl.when(label_col.is_not_null()).then(label_col > 0).alias("label"),
        )
    )

    none_lit = pl.lit(None, dtype=pl.Boolean)

    return (
        subtree_anchor_realizations.unnest(LabelSchema.subject_id_name)
        .select(
            LabelSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            none_lit.alias("valid"),
            none_lit.alias("determinable"),
            none_lit.alias("label"),
        )
        .update(
            aces_results,
            on=[LabelSchema.subject_id_name, LabelSchema.prediction_time_name],
            how="left",
            maintain_order="left",
        )
        .select(
            LabelSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            pl.col("valid").fill_null(False).alias("valid"),
            pl.col("determinable"),
            pl.col("label"),
        )
    )
