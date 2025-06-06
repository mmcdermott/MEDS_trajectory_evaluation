import polars as pl
from meds import LabelSchema
from meds_evaluation.schema import PredictionSchema


def aggregate_predictions(labels_df: pl.DataFrame, undetermined_probability: float) -> pl.DataFrame:
    """Aggregate labeled trajectories into prediction probabilities.

    Args:
        labels_df: Output of ``label_trajectories`` for many trajectories.
        undetermined_probability: Probability to assign when no determinable trajectories are present.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time`` and ``predicted_boolean_probability``.
    """

    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]

    return labels_df.group_by(*ids).agg(
        pl.col("label").mean().alias(PredictionSchema.predicted_boolean_probability_name)
    )
