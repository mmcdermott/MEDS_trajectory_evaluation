"""Tests for the aggregate_predictions function."""

from datetime import UTC, datetime

import polars as pl
import pytest
from meds_evaluation.schema import PredictionSchema

from MEDS_trajectory_evaluation.ACES_config_evaluation.aggregate import aggregate_predictions

PROB_COL = PredictionSchema.predicted_boolean_probability_name


def _labels_df(rows: list[tuple]) -> pl.DataFrame:
    """Helper to build a labels DataFrame from compact row tuples.

    Each row is (subject_id, prediction_time, valid, determinable, label).
    """
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "valid": pl.Boolean,
            "determinable": pl.Boolean,
            "label": pl.Boolean,
        },
        orient="row",
    )


class TestAggregateBasicBehavior:
    """Tests for core aggregation: only valid+determinable rows should contribute."""

    def test_only_valid_determinable_rows_contribute(self):
        """Invalid or indeterminate trajectories should be excluded from the probability."""
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
                (1, datetime(2021, 1, 1, tzinfo=UTC), False, None, None),  # invalid
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, False, None),  # not determinable
            ]
        )
        result = aggregate_predictions(df)
        assert result.height == 1
        assert result[PROB_COL][0] == pytest.approx(0.5)

    def test_all_true_labels(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
            ]
        )
        result = aggregate_predictions(df)
        assert result[PROB_COL][0] == pytest.approx(1.0)

    def test_all_false_labels(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
            ]
        )
        result = aggregate_predictions(df)
        assert result[PROB_COL][0] == pytest.approx(0.0)

    def test_multiple_groups(self):
        """Each (subject, prediction_time) should be aggregated independently."""
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
                (2, datetime(2021, 1, 2, tzinfo=UTC), True, True, True),
                (2, datetime(2021, 1, 2, tzinfo=UTC), True, True, True),
                (2, datetime(2021, 1, 2, tzinfo=UTC), True, True, True),
            ]
        )
        result = aggregate_predictions(df).sort("subject_id")
        assert result.height == 2
        assert result[PROB_COL][0] == pytest.approx(0.5)  # subject 1
        assert result[PROB_COL][1] == pytest.approx(1.0)  # subject 2


class TestAggregateUndeterminedFallback:
    """Tests for the undetermined_probability fallback."""

    def test_all_undetermined_uses_fallback(self):
        """When no valid+determinable trajectories exist, use the fallback probability."""
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), False, None, None),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, False, None),
            ]
        )
        result = aggregate_predictions(df, undetermined_probability=0.42)
        assert result.height == 1
        assert result[PROB_COL][0] == pytest.approx(0.42)

    def test_default_fallback_is_zero(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), False, None, None),
            ]
        )
        result = aggregate_predictions(df)
        assert result[PROB_COL][0] == pytest.approx(0.0)

    def test_mixed_determined_and_undetermined_groups(self):
        """One group has usable trajectories, another does not."""
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (2, datetime(2021, 1, 2, tzinfo=UTC), False, None, None),
                (2, datetime(2021, 1, 2, tzinfo=UTC), True, False, None),
            ]
        )
        result = aggregate_predictions(df, undetermined_probability=0.5).sort("subject_id")
        assert result.height == 2
        assert result[PROB_COL][0] == pytest.approx(1.0)  # subject 1: determined
        assert result[PROB_COL][1] == pytest.approx(0.5)  # subject 2: fallback


class TestAggregateOutputSchema:
    """Tests that the output conforms to PredictionSchema."""

    def test_output_has_required_columns(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
            ]
        )
        result = aggregate_predictions(df)
        assert "subject_id" in result.columns
        assert "prediction_time" in result.columns
        assert PROB_COL in result.columns

    def test_output_validates_against_prediction_schema(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
                (2, datetime(2021, 1, 2, tzinfo=UTC), True, True, False),
            ]
        )
        result = aggregate_predictions(df)
        # Should not raise
        PredictionSchema.validate(result.to_arrow())

    def test_fallback_output_validates_against_prediction_schema(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), False, None, None),
            ]
        )
        result = aggregate_predictions(df, undetermined_probability=0.5)
        PredictionSchema.validate(result.to_arrow())


class TestAggregateEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe(self):
        df = _labels_df([])
        result = aggregate_predictions(df)
        assert result.height == 0
        assert PROB_COL in result.columns

    def test_single_trajectory_true(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
            ]
        )
        result = aggregate_predictions(df)
        assert result[PROB_COL][0] == pytest.approx(1.0)

    def test_single_trajectory_false(self):
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, False),
            ]
        )
        result = aggregate_predictions(df)
        assert result[PROB_COL][0] == pytest.approx(0.0)

    def test_same_subject_different_prediction_times(self):
        """Same subject at two different prediction times should produce two rows."""
        df = _labels_df(
            [
                (1, datetime(2021, 1, 1, tzinfo=UTC), True, True, True),
                (1, datetime(2021, 1, 2, tzinfo=UTC), True, True, False),
            ]
        )
        result = aggregate_predictions(df).sort("prediction_time")
        assert result.height == 2
        assert result[PROB_COL][0] == pytest.approx(1.0)
        assert result[PROB_COL][1] == pytest.approx(0.0)
