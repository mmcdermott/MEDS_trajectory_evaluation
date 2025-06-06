import polars as pl
from meds import LabelSchema
from meds_evaluation.schema import PredictionSchema


def aggregate_predictions(labels_df: pl.DataFrame, undetermined_probability: float) -> pl.DataFrame:
    """Aggregate labeled trajectories into prediction probabilities.

    Parameters
    ----------
    labels_df: DataFrame
        Output of ``label_trajectories`` for many trajectories.
    undetermined_probability: float
        Probability to assign when no determinable trajectories are present.

    Returns
    -------
    DataFrame
        Columns ``subject_id``, ``prediction_time`` and
        ``predicted_boolean_probability``.
    """
    if labels_df.is_empty():
        return pl.DataFrame(
            schema={
                LabelSchema.subject_id_name: pl.Int64,
                LabelSchema.prediction_time_name: pl.Datetime("us"),
                PredictionSchema.predicted_boolean_probability_name: pl.Float32,
            }
        )

    agg = (
        labels_df.group_by(LabelSchema.subject_id_name, LabelSchema.prediction_time_name)
        .agg(
            pl.col("label").mean().cast(pl.Float32).alias(PredictionSchema.predicted_boolean_probability_name)
        )
        .with_columns(
            pl.col(PredictionSchema.predicted_boolean_probability_name).fill_null(undetermined_probability)
        )
        .sort(
            [LabelSchema.subject_id_name, LabelSchema.prediction_time_name],
            maintain_order=True,
        )
    )

    return agg
