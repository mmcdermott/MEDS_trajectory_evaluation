import polars as pl
from meds import LabelSchema
from meds_evaluation.schema import PredictionSchema


def aggregate_predictions(labels_df: pl.DataFrame, undetermined_probability: float = 0.0) -> pl.DataFrame:
    """Aggregate per-trajectory labels into a single predicted probability per (subject, prediction_time).

    Only trajectories that are both valid and determinable contribute to the predicted probability.
    The probability is the fraction of qualifying trajectories whose label is ``True``. When no
    qualifying trajectories exist for a given (subject, prediction_time), the probability falls
    back to ``undetermined_probability``.

    Args:
        labels_df: Output of ``label_trajectories`` concatenated across trajectory files. Expected
            columns are ``subject_id``, ``prediction_time``, ``valid``, ``determinable``, and ``label``.
        undetermined_probability: Probability to assign when no valid and determinable trajectories
            are present for a (subject, prediction_time) group. Defaults to 0.0.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, and
        ``predicted_boolean_probability``.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "prediction_time": [
        ...         datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1),
        ...         datetime(2021, 1, 2), datetime(2021, 1, 2),
        ...     ],
        ...     "valid": [True, True, False, True, True],
        ...     "determinable": [True, True, None, True, False],
        ...     "label": [True, False, None, False, None],
        ... })

    Subject 1 at 2021-01-01 has two valid+determinable trajectories (True, False), so prob = 0.5.
    Subject 2 at 2021-01-02 has one valid+determinable trajectory (False), so prob = 0.0.

        >>> aggregate_predictions(df).sort("subject_id")
        shape: (2, 3)
        ┌────────────┬─────────────────────┬───────────────────────────────┐
        │ subject_id ┆ prediction_time     ┆ predicted_boolean_probability │
        │ ---        ┆ ---                 ┆ ---                           │
        │ i64        ┆ datetime[μs]        ┆ f64                           │
        ╞════════════╪═════════════════════╪═══════════════════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0.5                           │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 0.0                           │
        └────────────┴─────────────────────┴───────────────────────────────┘

    When no trajectories are determinable, the fallback probability is used:

        >>> df_undet = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 1)],
        ...     "valid": [False, True],
        ...     "determinable": [None, False],
        ...     "label": [None, None],
        ... })
        >>> aggregate_predictions(df_undet, undetermined_probability=0.3)
        shape: (1, 3)
        ┌────────────┬─────────────────────┬───────────────────────────────┐
        │ subject_id ┆ prediction_time     ┆ predicted_boolean_probability │
        │ ---        ┆ ---                 ┆ ---                           │
        │ i64        ┆ datetime[μs]        ┆ f64                           │
        ╞════════════╪═════════════════════╪═══════════════════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0.3                           │
        └────────────┴─────────────────────┴───────────────────────────────┘
    """

    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

    # Only valid and determinable trajectories contribute to the probability.
    usable = labels_df.filter(pl.col("valid") & pl.col("determinable").fill_null(False))

    prob_col = PredictionSchema.predicted_boolean_probability_name

    if usable.height == 0:
        # All groups are undetermined — return one row per group with the fallback.
        return (
            labels_df.select(*ids)
            .unique(maintain_order=True)
            .with_columns(pl.lit(undetermined_probability).alias(prob_col))
        )

    probs = usable.group_by(*ids, maintain_order=True).agg(
        pl.col("label").cast(pl.Float64).mean().alias(prob_col)
    )

    # Ensure every (subject, prediction_time) from the input is present, filling groups that
    # had no usable trajectories with the fallback probability.
    all_ids = labels_df.select(*ids).unique(maintain_order=True)
    return all_ids.join(probs, on=ids, how="left", coalesce=True).with_columns(
        pl.col(prob_col).fill_null(undetermined_probability)
    )
