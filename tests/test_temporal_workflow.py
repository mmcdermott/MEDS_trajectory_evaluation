import math
from datetime import UTC, datetime, timedelta

import polars as pl
from aces.config import PlainPredicateConfig
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from meds import LabelSchema

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.get_ttes import (
    get_raw_tte,
    get_trajectory_tte,
    merge_pred_ttes,
)
from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import (
    temporal_aucs,
)
from MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC import (
    temporal_auc_from_trajectory_files,
)
from tests.helpers import _manual_auc


def test_temporal_auc_from_trajectory_files_exclude_history_actually_filters(tmp_path):
    """Regression test for https://github.com/mmcdermott/MEDS_trajectory_evaluation/issues/33.

    `temporal_auc_from_trajectory_files` accepts `exclude_history` and passes it to
    `temporal_aucs`, but never asks `get_raw_tte` to compute the `history/<task>` columns
    that the filter depends on. The filter at `temporal_AUCS.py:1036` is gated on
    `f"history/{task}" in df_task.columns`, so it silently no-ops — the user sees no
    error, no warning, and the same AUC they would have gotten without `exclude_history`.

    Real-world scenario: predicting *first* diabetes diagnosis. Excluding patients with
    prior diabetes is the entire point of the feature — those patients are not in the
    population the model is meant to address.
    """

    base = datetime(2022, 1, 1, tzinfo=UTC)  # noqa: F841 — kept for reading clarity
    pt = datetime(2022, 4, 1, tzinfo=UTC)

    # 4 patients, all admitted on 2022-04-01.
    #   subj 1, 2: prior diabetes diagnosis before admission — should be excluded.
    #              For both, the model predicts perfectly (the easy case).
    #   subj 3, 4: no prior diabetes — should remain in the cohort.
    #              Subject 3 has a true post-admission event but the model predicts late.
    #              Subject 4 never has a true event but the model predicts an early event.
    #   So among the no-history cohort, the model is *anti-correlated* (AUC=0).
    MEDS_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 4],
            "time": [
                datetime(2021, 6, 1, tzinfo=UTC),  # subj 1: prior dx
                datetime(2022, 4, 5, tzinfo=UTC),  # subj 1: future dx (4d post-pt)
                datetime(2021, 8, 1, tzinfo=UTC),  # subj 2: prior dx
                datetime(2022, 4, 6, tzinfo=UTC),  # subj 2: future dx (5d post-pt)
                datetime(2022, 4, 4, tzinfo=UTC),  # subj 3: future dx only (3d post-pt)
                datetime(2022, 5, 1, tzinfo=UTC),  # subj 4: lab only — no diabetes ever
            ],
            "code": [
                "ICD10//E11.9",
                "ICD10//E11.9",
                "ICD10//E11.9",
                "ICD10//E11.9",
                "ICD10//E11.9",
                "LAB_GLUCOSE",  # ensures subj 4 is in MEDS but with no diabetes events
            ],
        }
    )

    # Single trajectory: a sampled-future timeline per subject.
    #   subj 1, 2: predict the right event timing (high predicted probability).
    #   subj 3 (true positive): predict no event in window (low prob -> wrong).
    #   subj 4 (true negative): predict an in-window event (high prob -> wrong).
    traj = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "prediction_time": [pt] * 8,
            "time": [
                pt,
                datetime(2022, 4, 5, tzinfo=UTC),  # subj 1: prediction matches truth
                pt,
                datetime(2022, 4, 6, tzinfo=UTC),  # subj 2: prediction matches truth
                pt,
                datetime(2022, 5, 1, tzinfo=UTC),  # subj 3: predicts late (out of 10d window)
                pt,
                datetime(2022, 4, 5, tzinfo=UTC),  # subj 4: predicts in-window (wrong)
            ],
            "code": [
                "DUMMY",
                "ICD10//E11.9",
                "DUMMY",
                "ICD10//E11.9",
                "DUMMY",
                "ICD10//E11.9",
                "DUMMY",
                "ICD10//E11.9",
            ],
        }
    )
    traj.write_parquet(tmp_path / "traj_1.parquet")

    predicates = {"diabetes": PlainPredicateConfig(code="ICD10//E11.9")}

    # Same call, two different `exclude_history` settings. With the bug, both produce
    # the same AUC (the filter silently no-ops). After the fix, the two should differ
    # because subjects 1 and 2 (the two history-positive easy cases) are removed.
    auc_no_filter = temporal_auc_from_trajectory_files(
        MEDS_df,
        tmp_path,
        predicates,
        duration_grid=[timedelta(days=10)],
        exclude_history=False,
    )
    auc_filtered = temporal_auc_from_trajectory_files(
        MEDS_df,
        tmp_path,
        predicates,
        duration_grid=[timedelta(days=10)],
        exclude_history=True,
    )

    no_filter_val = auc_no_filter["AUC/diabetes"][0]
    filtered_val = auc_filtered["AUC/diabetes"][0] if "AUC/diabetes" in auc_filtered.columns else None

    # No-filter AUC includes all 4 subjects:
    #   labels: subj 1 True, subj 2 True, subj 3 True (true tte=3d ≤ 10d), subj 4 False
    #   probs:  subj 1 = 1.0, subj 2 = 1.0, subj 3 = 0.0, subj 4 = 1.0
    #   positives [1.0, 1.0, 0.0], negative [1.0] → AUC ≈ 0.333
    #
    # If exclude_history actually fires, subjects 1 and 2 are removed, leaving only
    # subj 3 (True, prob 0.0) and subj 4 (False, prob 1.0) → an anti-correlated pair
    # with AUC = 0.0.
    #
    # The bug: filtered_val == no_filter_val because the filter silently no-ops.
    assert filtered_val != no_filter_val, (
        f"`exclude_history=True` is a silent no-op (#33): both calls returned AUC={no_filter_val}. "
        "Expected the filtered call to drop subjects 1 and 2 and yield a different AUC "
        "(specifically AUC=0.0 once subjects 1 and 2 are excluded)."
    )


def _duration_tds(min_days: int, max_days: int) -> st.SearchStrategy[timedelta]:
    return st.timedeltas(
        min_value=timedelta(days=min_days),
        max_value=timedelta(days=max_days),
    )


@st.composite
def _workflow_inputs(draw):
    base_time = datetime(2022, 1, 1, tzinfo=UTC)
    subject_count = draw(st.integers(min_value=2, max_value=20))
    subjects = list(range(1, subject_count + 1))
    n_trajs = draw(st.integers(min_value=1, max_value=20))
    tasks = ["A", "B"]

    pred_times = {s: base_time + timedelta(days=draw(st.integers(0, 5))) for s in subjects}

    # true MEDS events
    meds_rows = []
    true_ttes = {}
    for s in subjects:
        for task in tasks:
            tte = draw(st.none() | _duration_tds(1, 30))
            true_ttes[(s, task)] = tte
            if tte is not None:
                meds_rows.append(
                    {
                        "subject_id": s,
                        "time": pred_times[s] + tte,
                        "code": task,
                    }
                )
    # ensure at least one subject has no upcoming events for all tasks
    assume(any(all(true_ttes[(s, t)] is None for t in tasks) for s in subjects))
    if meds_rows:
        MEDS_df = pl.DataFrame(meds_rows)
    else:
        MEDS_df = pl.DataFrame(
            {
                "subject_id": pl.Series([], dtype=pl.Int64),
                "time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
                "code": pl.Series([], dtype=pl.Utf8),
            }
        )

    # predicted trajectories
    pred_dfs = []
    for _ in range(n_trajs):
        rows = []
        for s in subjects:
            pt = pred_times[s]
            # dummy row to ensure group exists
            rows.append(
                {
                    "subject_id": s,
                    "prediction_time": pt,
                    "time": pt,
                    "code": "DUMMY",
                }
            )
            for task in tasks:
                pred_tte = draw(st.none() | _duration_tds(1, 15))
                if pred_tte is not None:
                    rows.append(
                        {
                            "subject_id": s,
                            "prediction_time": pt,
                            "time": pt + pred_tte,
                            "code": task,
                        }
                    )
        pred_dfs.append(pl.DataFrame(rows))

    duration_grid = sorted(set(draw(st.lists(_duration_tds(1, 30), min_size=1, max_size=5))))

    return MEDS_df, pred_dfs, duration_grid, tasks, true_ttes


def _manual_workflow(MEDS_df, pred_dfs, duration_grid, tasks, true_ttes):
    predicates = {t: PlainPredicateConfig(code=t) for t in tasks}

    pred_tte_dfs = [get_trajectory_tte(df, predicates) for df in pred_dfs]
    merged_pred = merge_pred_ttes(pred_tte_dfs)

    index_df = pred_dfs[0].select(LabelSchema.subject_id_name, LabelSchema.prediction_time_name).unique()
    subjects = index_df[LabelSchema.subject_id_name].to_list()
    true_tte = get_raw_tte(MEDS_df, index_df, predicates, include_followup_time=False)

    auc_df = temporal_aucs(true_tte, merged_pred, duration_grid, handle_censoring=False)

    manual = {task: [] for task in tasks}
    for duration in duration_grid:
        for task in tasks:
            positives = []
            negatives = []
            for s in subjects:
                true_tte_val = true_ttes[(s, task)]
                label = true_tte_val is not None and true_tte_val <= duration
                prob_list = merged_pred.filter(pl.col(LabelSchema.subject_id_name) == s)[f"tte/{task}"][0]
                prob = sum(1 for p in prob_list if p is not None and p <= duration) / len(prob_list)
                if label:
                    positives.append(prob)
                else:
                    negatives.append(prob)
            manual_auc = _manual_auc(positives, negatives)
            manual[task].append(manual_auc)
    return auc_df, manual


@settings(deadline=None, max_examples=50)
@given(_workflow_inputs())
def test_full_temporal_auc_workflow(data):
    MEDS_df, pred_dfs, duration_grid, tasks, true_ttes = data
    auc_df, manual = _manual_workflow(MEDS_df, pred_dfs, duration_grid, tasks, true_ttes)
    for i, duration in enumerate(duration_grid):
        assert auc_df["duration"][i] == duration
        for task in tasks:
            auc_val = auc_df[f"AUC/{task}"][i] if f"AUC/{task}" in auc_df.columns else None
            expected = manual[task][i]
            if expected is None:
                assert auc_val is None
            else:
                assert math.isclose(auc_val, expected, rel_tol=1e-9)
                assert 0.0 <= auc_val <= 1.0
