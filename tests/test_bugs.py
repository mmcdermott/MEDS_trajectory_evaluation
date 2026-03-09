"""Tests that expose known bugs.

Each test in this file documents a specific bug and is expected to **fail** until the corresponding fix is
applied.
"""

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import temporal_aucs

# ---------------------------------------------------------------------------
# Bug: Cross-task censoring filter uses ``all_horizontal`` which drops a row
# from *every* task when it is censored for *any* task, producing incorrect
# AUC values for tasks where the subject is not actually censored.
# ---------------------------------------------------------------------------


def test_censored_subject_excluded_only_from_tasks_it_is_censored_for():
    """A subject who is censored for one task but not another should still contribute to the AUC of the non-
    censored task.

    Scenario
    --------
    Three subjects, two tasks (A and B), evaluated at duration = 7 days.

    * Subject 1: event A at 5 d, event B at 3 d, follow-up = 20 d
          → label/A = True,  label/B = True
    * Subject 2: no event A,     no event B,     follow-up = 10 d
          → label/A = False, label/B = False
    * Subject 3: event A at 3 d, no event B,     follow-up = 4 d
          → label/A = True,  label/B = censored (4 d follow-up < 7 d window)

    Predicted TTE (single trajectory each → deterministic probabilities):
    * Subject 1: pred A = [4 d], pred B = [2 d]  → prob/A = 1.0, prob/B = 1.0
    * Subject 2: pred A = [∞],   pred B = [∞]    → prob/A = 0.0, prob/B = 0.0
    * Subject 3: pred A = [∞],   pred B = [∞]    → prob/A = 0.0, prob/B = 0.0

    Expected AUCs
    ~~~~~~~~~~~~~
    * Task A — all three subjects have definitive labels:
      Positives (prob): sub1=1.0, sub3=0.0.  Negatives (prob): sub2=0.0.
      Pairs: (1.0 vs 0.0)=win, (0.0 vs 0.0)=tie → AUC = (1 + 0.5)/2 = 0.75

    * Task B — sub3 is censored, so only sub1 and sub2 contribute:
      Positives (prob): sub1=1.0.  Negatives (prob): sub2=0.0.
      AUC = 1.0

    What the bug produces
    ~~~~~~~~~~~~~~~~~~~~~
    The ``all_horizontal`` filter at line ~850 of ``temporal_AUCS.py`` drops
    sub3 from **both** tasks because sub3's ``label/B`` is null.  Task A then
    only sees sub1 (True, 1.0) and sub2 (False, 0.0), yielding AUC = 1.0
    instead of the correct 0.75.
    """

    true_tte = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
                datetime(2021, 1, 3, tzinfo=UTC),
            ],
            "tte/A": [timedelta(days=5), None, timedelta(days=3)],
            "tte/B": [timedelta(days=3), None, None],
            "max_followup_time": [
                timedelta(days=20),
                timedelta(days=10),
                timedelta(days=4),
            ],
        }
    )

    pred_ttes = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
                datetime(2021, 1, 3, tzinfo=UTC),
            ],
            "tte/A": [[timedelta(days=4)], [None], [None]],
            "tte/B": [[timedelta(days=2)], [None], [None]],
        }
    )

    result = temporal_aucs(
        true_tte,
        pred_ttes,
        duration_grid=[timedelta(days=7)],
        handle_censoring=True,
    )

    auc_a = result["AUC/A"][0]
    auc_b = result["AUC/B"][0]

    # --- Task A assertions ---
    # Subject 3 is NOT censored for task A (event occurred at 3d, well within
    # both the 7d evaluation window and the 4d follow-up), so subject 3 must
    # participate in task A's AUC.  With all three subjects, AUC = 0.75.
    assert auc_a is not None, "Task A should have a computable AUC"
    assert math.isclose(auc_a, 0.75, rel_tol=1e-9), (
        f"Task A AUC should be 0.75 (subject 3 contributes as a True positive), got {auc_a}"
    )

    # --- Task B assertions ---
    # Subject 3 IS censored for task B (no event, only 4d of follow-up for a
    # 7d window).  With only sub1 and sub2, AUC = 1.0.
    assert auc_b is not None, "Task B should have a computable AUC"
    assert math.isclose(auc_b, 1.0, rel_tol=1e-9), f"Task B AUC should be 1.0, got {auc_b}"


def test_censoring_contamination_varies_by_duration():
    """The same subject can be uncensored at short horizons and censored at long horizons for one task while
    always uncensored for another.  The bug must corrupt the long-horizon AUC of the unaffected task while
    leaving the short-horizon AUC correct.

    Scenario: 3 subjects, 2 tasks (A and B), 2 durations (3 d and 10 d).

    * Subject 1: event A at 2 d, event B at 1 d, follow-up = 20 d
          → label/A = True  at both.  label/B = True  at both.
    * Subject 2: no events,                      follow-up = 20 d
          → label/A = False at both.  label/B = False at both.
    * Subject 3: event A at 2 d, no event B,     follow-up = 4 d
          → label/A = True  at both (event at 2d).
             label/B = False at 3 d  (4d follow-up ≥ 3d).
             label/B = censored at 10 d (4d < 10d).

    Predicted TTE (single trajectory each):
    * Sub 1: pred A=[1d], pred B=[1d] → prob = 1.0 everywhere
    * Sub 2: pred A=[∞],  pred B=[∞]  → prob = 0.0 everywhere
    * Sub 3: pred A=[∞],  pred B=[∞]  → prob = 0.0 everywhere

    Expected AUCs
    ~~~~~~~~~~~~~
    Duration 3 d (no censoring for either task — all labels definitive):
    * Task A: True=[1.0,0.0], False=[0.0] → AUC = 0.75
    * Task B: True=[1.0],     False=[0.0,0.0] → AUC = 1.0

    Duration 10 d (sub3 censored for B only):
    * Task A: still all 3 subjects → AUC = 0.75
    * Task B: sub3 excluded        → AUC = 1.0

    Bug behaviour at duration 10 d
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``all_horizontal`` drops sub3 from both tasks.  Task A sees only sub1
    and sub2, giving AUC = 1.0 instead of 0.75.
    """

    true_tte = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
                datetime(2021, 1, 3, tzinfo=UTC),
            ],
            "tte/A": [timedelta(days=2), None, timedelta(days=2)],
            "tte/B": [timedelta(days=1), None, None],
            "max_followup_time": [
                timedelta(days=20),
                timedelta(days=20),
                timedelta(days=4),
            ],
        }
    )

    pred_ttes = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 2, tzinfo=UTC),
                datetime(2021, 1, 3, tzinfo=UTC),
            ],
            "tte/A": [[timedelta(days=1)], [None], [None]],
            "tte/B": [[timedelta(days=1)], [None], [None]],
        }
    )

    result = temporal_aucs(
        true_tte,
        pred_ttes,
        duration_grid=[timedelta(days=3), timedelta(days=10)],
        handle_censoring=True,
    )

    # --- Duration 3 d: no censoring anywhere, both tests should agree ---
    row_3d = result.filter(pl.col("duration") == timedelta(days=3))
    auc_a_3d = row_3d["AUC/A"][0]
    auc_b_3d = row_3d["AUC/B"][0]

    assert auc_a_3d is not None
    assert math.isclose(auc_a_3d, 0.75, rel_tol=1e-9), f"Task A at 3d: expected 0.75, got {auc_a_3d}"
    assert auc_b_3d is not None
    assert math.isclose(auc_b_3d, 1.0, rel_tol=1e-9), f"Task B at 3d: expected 1.0, got {auc_b_3d}"

    # --- Duration 10 d: sub3 censored for B but NOT for A ---
    row_10d = result.filter(pl.col("duration") == timedelta(days=10))
    auc_a_10d = row_10d["AUC/A"][0]
    auc_b_10d = row_10d["AUC/B"][0]

    # Task A must still include sub3 (uncensored for A) → AUC = 0.75
    assert auc_a_10d is not None, "Task A at 10d should have a computable AUC"
    assert math.isclose(auc_a_10d, 0.75, rel_tol=1e-9), (
        f"Task A at 10d: expected 0.75 (sub3 not censored for A), got {auc_a_10d}"
    )

    # Task B excludes sub3 (censored) → AUC = 1.0
    assert auc_b_10d is not None, "Task B at 10d should have a computable AUC"
    assert math.isclose(auc_b_10d, 1.0, rel_tol=1e-9), f"Task B at 10d: expected 1.0, got {auc_b_10d}"
