import polars as pl
from meds import LabelSchema
from meds_evaluation import schema as eval_schema


def aggregate_predictions(labels_df: pl.DataFrame, undetermined_probability: float) -> pl.DataFrame:
    """Aggregate labeled trajectories into prediction probabilities.

    Args:
        labels_df: Output of ``label_trajectories`` for many trajectories.
        undetermined_probability: Probability to assign when no determinable trajectories are present.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time`` and ``predicted_boolean_probability``.
    """
    if labels_df.is_empty():
        return pl.DataFrame(
            schema={
                eval_schema.SUBJECT_ID_FIELD: pl.Int64,
                eval_schema.PREDICTION_TIME_FIELD: pl.Datetime("us"),
                eval_schema.BOOLEAN_VALUE_FIELD: pl.Boolean,
                eval_schema.PREDICTED_BOOLEAN_PROBABILITY_FIELD: pl.Float64,
            }
        )

    agg = (
        labels_df.group_by(LabelSchema.subject_id_name, LabelSchema.prediction_time_name)
        .agg(
            pl.col("label")
            .first()
            .alias(eval_schema.BOOLEAN_VALUE_FIELD),
            .alias(eval_schema.PREDICTED_BOOLEAN_PROBABILITY_FIELD),
            pl.col(eval_schema.PREDICTED_BOOLEAN_PROBABILITY_FIELD).fill_null(
            pl.col(LabelSchema.subject_id_name)
            .cast(pl.Int64)
            .alias(eval_schema.SUBJECT_ID_FIELD),
            .cast(pl.Datetime("us"))
            .alias(eval_schema.PREDICTION_TIME_FIELD),
        )
        .select(
            eval_schema.SUBJECT_ID_FIELD,
            eval_schema.PREDICTION_TIME_FIELD,
            eval_schema.BOOLEAN_VALUE_FIELD,
            eval_schema.PREDICTED_BOOLEAN_PROBABILITY_FIELD,
        )
            [eval_schema.SUBJECT_ID_FIELD, eval_schema.PREDICTION_TIME_FIELD],
            pl.col(PredictionSchema.predicted_boolean_probability_name).fill_null(undetermined_probability)
        )
        .sort(
            [LabelSchema.subject_id_name, LabelSchema.prediction_time_name],
            maintain_order=True,
        )
    )

    return agg
