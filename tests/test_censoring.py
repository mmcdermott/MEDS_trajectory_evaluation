"""Tests for censoring-aware survival analysis functionality."""

from datetime import UTC, datetime, timedelta

import polars as pl
from aces.config import PlainPredicateConfig

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.get_ttes import get_raw_tte
from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import (
    add_labels_from_true_tte,
    temporal_aucs,
)


def test_get_raw_tte_followup_time():
    """Test that get_raw_tte correctly calculates follow-up times."""

    # Create MEDS data with varying follow-up times
    MEDS_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 3, 3],
            "time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 5, tzinfo=UTC),  # Subject 1: 5 days total
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),  # Subject 2: 2 days total
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 10, tzinfo=UTC),  # Subject 3: 10 days total
            ],
            "code": ["event_A", "event_B", "event_A", "event_A", "event_B", "event_A", "event_B"],
        }
    )

    index_df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2020, 1, 2, tzinfo=UTC),  # Subject 1: 3 days follow-up
                datetime(2020, 1, 1, tzinfo=UTC),  # Subject 2: 1 day follow-up
                datetime(2020, 1, 5, tzinfo=UTC),  # Subject 3: 5 days follow-up
            ],
        }
    )

    predicates = {
        "A": PlainPredicateConfig(code="event_A"),
        "B": PlainPredicateConfig(code="event_B"),
    }

    result = get_raw_tte(MEDS_df, index_df, predicates, include_followup_time=True)

    # Verify follow-up times are calculated correctly
    expected_followups = [
        timedelta(days=3),  # Subject 1: last event 2020-01-05, prediction 2020-01-02
        timedelta(days=1),  # Subject 2: last event 2020-01-02, prediction 2020-01-01
        timedelta(days=5),  # Subject 3: last event 2020-01-10, prediction 2020-01-05
    ]

    assert result.select("max_followup_time").to_series().to_list() == expected_followups

    # Verify TTE calculations are still correct
    expected_tte_a = [
        timedelta(days=3),  # Subject 1: event_A at 2020-01-05, prediction at 2020-01-02
        None,  # Subject 2: no event_A after prediction time
        None,  # Subject 3: no event_A after prediction time
    ]

    assert result.select("tte/A").to_series().to_list() == expected_tte_a


def test_add_labels_censoring_aware():
    """Test censoring-aware labeling logic with comprehensive examples."""

    # Create test data with various censoring scenarios
    df = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "prediction_time": [datetime(2021, 1, 1, tzinfo=UTC)] * 5,
            "tte/A": [
                timedelta(days=3),  # Event at 3 days
                timedelta(days=15),  # Event at 15 days (outside 10-day window)
                None,  # No event observed
                None,  # No event observed
                timedelta(days=25),  # Event at 25 days (way outside window)
            ],
            "duration": [timedelta(days=10)] * 5,  # All evaluate 10-day window
            "max_followup_time": [
                timedelta(days=20),  # Adequate follow-up (20 > 10)
                timedelta(days=20),  # Adequate follow-up (20 > 10)
                timedelta(days=5),  # Insufficient follow-up (5 < 10) - CENSORED
                timedelta(days=15),  # Adequate follow-up (15 > 10)
                timedelta(days=30),  # Adequate follow-up (30 > 10)
            ],
        }
    )

    # Test censoring-aware labeling
    result_censored = add_labels_from_true_tte(df, handle_censoring=True)
    expected_labels = [
        True,  # Subject 1: Event at 3d, window=10d, adequate follow-up → True
        False,  # Subject 2: Event at 15d (outside 10d window), adequate follow-up → False
        None,  # Subject 3: No event, insufficient follow-up (5d < 10d) → Censored
        False,  # Subject 4: No event, adequate follow-up (15d > 10d) → False
        False,  # Subject 5: Event at 25d (outside 10d window), adequate follow-up → False
    ]

    actual_labels = result_censored.select("label/A").to_series().to_list()
    assert actual_labels == expected_labels, f"Expected {expected_labels}, got {actual_labels}"

    # Test legacy behavior (no censoring)
    result_legacy = add_labels_from_true_tte(df, handle_censoring=False)
    expected_legacy = [
        True,  # Event at 3d, window=10d → True
        False,  # Event at 15d (outside window) → False
        False,  # No event → False (not censored)
        False,  # No event → False
        False,  # Event at 25d (outside window) → False
    ]

    actual_legacy = result_legacy.select("label/A").to_series().to_list()
    assert actual_legacy == expected_legacy, f"Expected {expected_legacy}, got {actual_legacy}"


def test_temporal_aucs_censoring_exclusion():
    """Test that temporal_aucs properly excludes censored cases from AUC calculation."""

    # Create test data where censoring affects the outcome
    true_tte = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [datetime(2021, 1, 1, tzinfo=UTC)] * 4,
            "tte/A": [
                timedelta(days=2),  # Positive case: event at 2d
                None,  # True negative: no event, adequate follow-up
                None,  # Censored: no event, inadequate follow-up
                timedelta(days=12),  # Negative case: event outside window
            ],
            "max_followup_time": [
                timedelta(days=20),  # Adequate
                timedelta(days=20),  # Adequate
                timedelta(days=3),  # Inadequate (3 < 5) - will be censored
                timedelta(days=20),  # Adequate
            ],
        }
    )

    # Create trajectory predictions
    pred_ttes = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [datetime(2021, 1, 1, tzinfo=UTC)] * 4,
            "tte/A": [
                [timedelta(days=1), timedelta(days=3)],  # High prob (1 of 2 < 5d)
                [timedelta(days=10), timedelta(days=20)],  # Low prob (0 of 2 < 5d)
                [timedelta(days=2), timedelta(days=4)],  # High prob (2 of 2 < 5d) - but censored
                [timedelta(days=8), timedelta(days=15)],  # Low prob (0 of 2 < 5d)
            ],
        }
    )

    duration_grid = [timedelta(days=5)]

    # Test with censoring (subject 3 should be excluded)
    result_with_censoring = temporal_aucs(true_tte, pred_ttes, duration_grid, handle_censoring=True)

    # Expected: Only subjects 1, 2, 4 included
    # Subject 1: True, prob=0.5 (1 of 2 trajectories < 5d)
    # Subject 2: False, prob=0.0 (0 of 2 trajectories < 5d)
    # Subject 4: False, prob=0.0 (0 of 2 trajectories < 5d)
    # AUC = (1*2) / (1*2) = 1.0 (perfect separation)

    auc_with_censoring = result_with_censoring["AUC/A"][0]
    assert auc_with_censoring == 1.0, f"Expected AUC=1.0 with censoring, got {auc_with_censoring}"

    # Test without censoring (all subjects included, including censored case as negative)
    result_without_censoring = temporal_aucs(true_tte, pred_ttes, duration_grid, handle_censoring=False)

    # Expected: All subjects included
    # Subject 1: True, prob=0.5
    # Subject 2: False, prob=0.0
    # Subject 3: False (censored treated as negative), prob=1.0
    # Subject 4: False, prob=0.0
    # This creates an impossible case: False label with prob=1.0, leading to AUC < 1.0

    auc_without_censoring = result_without_censoring["AUC/A"][0]
    assert auc_without_censoring < 1.0, f"Expected AUC<1.0 without censoring, got {auc_without_censoring}"

    print(f"AUC with censoring handling: {auc_with_censoring:.3f}")
    print(f"AUC without censoring handling: {auc_without_censoring:.3f}")


def test_censoring_edge_cases():
    """Test edge cases in censoring logic."""

    # Test with zero follow-up time
    df_zero_followup = pl.DataFrame(
        {
            "subject_id": [1],
            "tte/A": [None],
            "duration": [timedelta(days=1)],
            "max_followup_time": [timedelta(days=0)],  # Zero follow-up
        }
    )

    result = add_labels_from_true_tte(df_zero_followup, handle_censoring=True)
    assert result["label/A"][0] is None  # Should be censored

    # Test with negative follow-up time (prediction after last event)
    df_negative_followup = pl.DataFrame(
        {
            "subject_id": [1],
            "tte/A": [None],
            "duration": [timedelta(days=1)],
            "max_followup_time": [timedelta(days=-5)],  # Negative follow-up
        }
    )

    result = add_labels_from_true_tte(df_negative_followup, handle_censoring=True)
    assert result["label/A"][0] is None  # Should be censored

    # Test with event occurring after follow-up ends (but we still observed it)
    df_late_event = pl.DataFrame(
        {
            "subject_id": [1],
            "tte/A": [timedelta(days=10)],  # Event at 10 days (we observed it!)
            "duration": [timedelta(days=15)],  # Evaluation window: 15 days
            "max_followup_time": [timedelta(days=8)],  # Follow-up ends at 8 days
        }
    )

    result = add_labels_from_true_tte(df_late_event, handle_censoring=True)
    # Event occurred within evaluation window (10d < 15d), so it's True
    # Follow-up time doesn't matter when we actually observed the event
    assert result["label/A"][0] is True

    # Test a true censoring case: event outside evaluation window, insufficient follow-up
    df_true_censoring = pl.DataFrame(
        {
            "subject_id": [1],
            "tte/A": [timedelta(days=20)],  # Event at 20 days (outside 15d window)
            "duration": [timedelta(days=15)],  # Evaluation window: 15 days
            "max_followup_time": [timedelta(days=10)],  # Follow-up ends at 10 days < 15 days
        }
    )

    result_censoring = add_labels_from_true_tte(df_true_censoring, handle_censoring=True)
    # Event occurred outside window (20d > 15d), but we don't have adequate follow-up
    # to confidently say it didn't occur within the window (follow-up 10d < window 15d)
    assert result_censoring["label/A"][0] is None


def test_censoring_with_offset():
    """Test censoring logic with time offsets."""

    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "tte/A": [timedelta(days=8), None],
            "duration": [timedelta(days=5), timedelta(days=5)],  # 5-day evaluation windows
            "max_followup_time": [timedelta(days=10), timedelta(days=3)],  # Varying follow-up
            "offset": [timedelta(days=2), timedelta(days=2)],  # 2-day offset
        }
    )

    result = add_labels_from_true_tte(df, offset_col="offset", handle_censoring=True)

    # Subject 1: Event at 8d, window from 2d to 7d, adequate follow-up → False (event outside window)
    # Subject 2: No event, window from 2d to 7d, inadequate follow-up (3d < 7d) → Censored
    expected = [False, None]
    actual = result.select("label/A").to_series().to_list()
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_multiple_tasks_censoring():
    """Test censoring logic with multiple predicates/tasks."""

    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "tte/A": [timedelta(days=3), None],
            "tte/B": [None, timedelta(days=8)],
            "duration": [timedelta(days=5)] * 2,
            "max_followup_time": [
                timedelta(days=10),
                timedelta(days=4),
            ],  # Subject 2 has insufficient follow-up
        }
    )

    result = add_labels_from_true_tte(df, handle_censoring=True)

    # Subject 1: A=True (event at 3d < 5d), B=False (no event, adequate follow-up)
    # Subject 2: A=None (no event, inadequate follow-up), B=None (event at 8d > 5d, but inadequate follow-up)
    expected_a = [True, None]
    expected_b = [False, None]

    actual_a = result.select("label/A").to_series().to_list()
    actual_b = result.select("label/B").to_series().to_list()

    assert actual_a == expected_a, f"Task A: Expected {expected_a}, got {actual_a}"
    assert actual_b == expected_b, f"Task B: Expected {expected_b}, got {actual_b}"


if __name__ == "__main__":
    # Run tests with verbose output for debugging
    test_get_raw_tte_followup_time()
    print("✅ test_get_raw_tte_followup_time passed")

    test_add_labels_censoring_aware()
    print("✅ test_add_labels_censoring_aware passed")

    test_temporal_aucs_censoring_exclusion()
    print("✅ test_temporal_aucs_censoring_exclusion passed")

    test_censoring_edge_cases()
    print("✅ test_censoring_edge_cases passed")

    test_censoring_with_offset()
    print("✅ test_censoring_with_offset passed")

    test_multiple_tasks_censoring()
    print("✅ test_multiple_tasks_censoring passed")

    print("\n🎉 All censoring tests passed!")
