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
from tests.helpers import _manual_auc


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
