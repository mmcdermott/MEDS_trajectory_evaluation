import math
from datetime import UTC, datetime, timedelta

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import df_AUC, temporal_aucs
from tests.helpers import _manual_auc


def test_temporal_aucs_handles_high_prevalence_task_at_single_duration():
    """Regression test for https://github.com/mmcdermott/MEDS_trajectory_evaluation/issues/31.

    The `temporal_aucs` docstring guarantees: "Missing columns (e.g., when all labels at a
    horizon are the same class) are omitted with a logged warning." This holds for the
    "all-negatives" case but not for the symmetric "all-positives" case — `df_AUC` crashes
    with `ColumnNotFoundError: unable to find column "false/<task>"`.

    Real-world scenario: predicting "any inpatient encounter within 1 year" for an elderly
    cohort. By a long enough horizon, every patient has had an encounter; at that single
    horizon, every label is True so the pivot produces only `true/encounter`, not
    `false/encounter`. A reasonable user request — "give me the year-1 AUC" — therefore
    crashes with no useful diagnostic.
    """

    base = datetime(2023, 1, 1, tzinfo=UTC)
    # Four elderly patients, each with one inpatient encounter within the year.
    true_tte = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [base] * 4,
            "tte/encounter": [
                timedelta(days=15),
                timedelta(days=120),
                timedelta(days=200),
                timedelta(days=350),
            ],
            "max_followup_time": [timedelta(days=400)] * 4,
        }
    )
    # One predicted trajectory per subject (e.g., a single autoregressive sample).
    pred_ttes = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [base] * 4,
            "tte/encounter": [
                [timedelta(days=10)],
                [timedelta(days=100)],
                [timedelta(days=180)],
                [timedelta(days=300)],
            ],
        }
    )

    # Single horizon at 365d — by which point every subject is positive.
    result = temporal_aucs(true_tte, pred_ttes, [timedelta(days=365)])

    # The call must not crash. AUC at this horizon is undefined (no negatives), so the
    # contract is "row exists, AUC is null OR column omitted" — matching the all-negatives
    # case the docstring already advertises.
    assert "duration" in result.columns
    assert result["duration"].to_list() == [timedelta(days=365)]
    if "AUC/encounter" in result.columns:
        assert result["AUC/encounter"][0] is None


def test_df_AUC_handles_task_with_only_true_column():
    """Direct unit test for the `df_AUC` precondition violated by issue #31.

    `df_AUC` derives the task list from `true/*` columns and unconditionally references
    `pl.col(f"false/{t}")`. Inputs where a task has positives at a horizon but no negatives
    at that horizon produce only `true/<t>` in the pivot — `df_AUC` should treat that as an
    undefined-AUC case (return None / drop), not crash.
    """

    df = pl.DataFrame({"task": ["task1"], "true/A": [[0.5, 0.7, 0.9]]})
    # Currently raises ColumnNotFoundError; should produce a result row with AUC/A null.
    result = df_AUC(df)
    assert result.height == 1
    if "AUC/A" in result.columns:
        assert result["AUC/A"][0] is None


@st.composite
def _df_inputs(draw):
    tasks = draw(st.lists(st.sampled_from(["A", "B", "C", "D"]), min_size=1, max_size=3, unique=True))
    n_rows = draw(st.integers(min_value=1, max_value=4))
    float_strategy = st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
    rows = []
    for _ in range(n_rows):
        row = {}
        for task in tasks:
            true_vals = sorted(draw(st.lists(float_strategy, min_size=0, max_size=5)))
            false_vals = sorted(draw(st.lists(float_strategy, min_size=0, max_size=5)))
            row[f"true/{task}"] = true_vals
            row[f"false/{task}"] = false_vals
        rows.append(row)
    return pl.DataFrame(rows), tasks


@st.composite
def _df_unsorted_inputs(draw):
    tasks = draw(st.lists(st.sampled_from(["A", "B", "C", "D"]), min_size=1, max_size=3, unique=True))
    n_rows = draw(st.integers(min_value=1, max_value=4))
    float_strategy = st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
    rows = []
    for _ in range(n_rows):
        row = {}
        for task in tasks:
            true_vals = draw(st.lists(float_strategy, min_size=0, max_size=5))
            false_vals = draw(st.lists(float_strategy, min_size=0, max_size=5))
            row[f"true/{task}"] = true_vals
            row[f"false/{task}"] = false_vals
        rows.append(row)
    return pl.DataFrame(rows), tasks


@given(_df_inputs())
def test_df_auc_matches_manual(data):
    df, tasks = data
    result = df_AUC(df)
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        for task in tasks:
            auc_val = result[f"AUC/{task}"][row_idx]
            expected = _manual_auc(row[f"true/{task}"], row[f"false/{task}"])
            if expected is None:
                assert auc_val is None
            else:
                assert math.isclose(auc_val, expected, rel_tol=1e-9)
                assert 0.0 <= auc_val <= 1.0


@given(_df_unsorted_inputs())
def test_df_auc_in_bounds_unsorted(data):
    df, tasks = data
    result = df_AUC(df)
    for row_idx in range(df.height):
        for task in tasks:
            auc_val = result[f"AUC/{task}"][row_idx]
            if auc_val is not None:
                assert 0.0 <= auc_val <= 1.0


def test_df_auc_large_datasets_precision():
    """Test for potential precision/overflow issues with large datasets.

    This test checks if the AUC calculation maintains precision and stays within bounds [0,1] when dealing
    with larger datasets that might cause integer overflow in the counting operations.
    """
    # Create a case with many values that could cause precision issues
    import numpy as np

    np.random.seed(42)  # For reproducibility

    # Large dataset with many similar values
    n_true = 1000
    n_false = 1000

    # Generate values with slight differences that might cause precision issues
    true_vals = [0.5 + i * 1e-6 for i in range(n_true)]
    false_vals = [0.5 - i * 1e-6 for i in range(n_false)]

    df = pl.DataFrame(
        {
            "task": ["large_test"],
            "true/A": [true_vals],
            "false/A": [false_vals],
        }
    )

    result = df_AUC(df)
    computed_auc = result["AUC/A"][0]

    # For this case, all true values should be > all false values, so AUC should be 1.0
    expected_auc = 1.0

    assert computed_auc is not None, "AUC should not be None for valid inputs"
    assert 0.0 <= computed_auc <= 1.0, f"AUC {computed_auc} is outside valid range [0,1]"
    assert math.isclose(computed_auc, expected_auc, rel_tol=1e-9), (
        f"Expected AUC {expected_auc}, got {computed_auc}"
    )


def test_df_auc_input_validation_edge_cases():
    """Test edge cases that might cause AUC values outside [0,1] due to input validation issues or numerical
    precision problems."""
    # Test case 1: Very large numbers that might cause overflow
    df1 = pl.DataFrame(
        {
            "task": ["overflow_test"],
            "true/A": [[1e15, 2e15]],
            "false/A": [[1e14, 2e14]],
        }
    )

    result1 = df_AUC(df1)
    auc1 = result1["AUC/A"][0]
    assert auc1 is not None, "AUC should not be None for valid inputs"
    assert 0.0 <= auc1 <= 1.0, f"Large number AUC {auc1} is outside valid range [0,1]"

    # Test case 2: Very small numbers near zero
    df2 = pl.DataFrame(
        {
            "task": ["underflow_test"],
            "true/A": [[1e-15, 2e-15]],
            "false/A": [[1e-16, 2e-16]],
        }
    )

    result2 = df_AUC(df2)
    auc2 = result2["AUC/A"][0]
    assert auc2 is not None, "AUC should not be None for valid inputs"
    assert 0.0 <= auc2 <= 1.0, f"Small number AUC {auc2} is outside valid range [0,1]"

    # Test case 3: Mixed very large and very small numbers
    df3 = pl.DataFrame(
        {
            "task": ["mixed_scale_test"],
            "true/A": [[1e15, 1e-15]],
            "false/A": [[1e14, 1e-14]],
        }
    )

    result3 = df_AUC(df3)
    auc3 = result3["AUC/A"][0]
    assert auc3 is not None, "AUC should not be None for valid inputs"
    assert 0.0 <= auc3 <= 1.0, f"Mixed scale AUC {auc3} is outside valid range [0,1]"
