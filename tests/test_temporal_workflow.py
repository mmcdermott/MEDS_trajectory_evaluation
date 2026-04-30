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


def test_merge_pred_ttes_aligns_by_subject_not_position():
    """Regression test for https://github.com/mmcdermott/MEDS_trajectory_evaluation/issues/32.

    `merge_pred_ttes` uses `pl.concat(how="horizontal")` after dropping the id columns —
    so it aligns rows positionally. The id columns are taken from the *first* dataframe
    only. Any per-trajectory dataframe whose (subject_id, prediction_time) ordering differs
    from the first will silently scramble TTE values across subjects.

    This is the realistic case for `temporal_auc_from_trajectory_files`: each trajectory
    parquet is independently sampled, and per-file row order is whatever the upstream
    sampler chose. There is no guarantee orderings match.
    """

    base = datetime(2022, 1, 1, tzinfo=UTC)
    pt1 = base
    pt2 = base + timedelta(days=10)

    # Trajectory 1: subjects in order [1, 2]
    traj_1 = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": [pt1, pt2],
            "tte/A": [timedelta(days=5), timedelta(days=10)],
        }
    )
    # Trajectory 2: SAME subjects, same prediction times — different row order [2, 1].
    # In a real pipeline this happens whenever the sampler streams trajectories in a
    # different per-file order (parallel workers, shuffled batches, etc.).
    traj_2 = pl.DataFrame(
        {
            "subject_id": [2, 1],
            "prediction_time": [pt2, pt1],
            "tte/A": [timedelta(days=99), timedelta(days=30)],
        }
    )

    merged = merge_pred_ttes([traj_1, traj_2])
    by_subject = dict(zip(merged["subject_id"].to_list(), merged["tte/A"].to_list(), strict=True))

    # Subject 1's predicted TTEs across the two trajectories are 5d (from traj_1) and 30d
    # (from traj_2), in some order. Subject 2's are 10d and 99d. The exact list order
    # within each subject's cell is unspecified, but the *set* must match.
    assert set(by_subject[1]) == {timedelta(days=5), timedelta(days=30)}, (
        "merge_pred_ttes mis-attributed subject 1's TTEs (#32 — positional concat)"
    )
    assert set(by_subject[2]) == {timedelta(days=10), timedelta(days=99)}, (
        "merge_pred_ttes mis-attributed subject 2's TTEs (#32 — positional concat)"
    )


def test_temporal_auc_from_trajectory_files_handles_unordered_parquets(tmp_path):
    """End-to-end regression test for #32 via `temporal_auc_from_trajectory_files`.

    Two trajectory parquets are written for the same (subject, prediction_time) pairs, but with different row
    orderings — the realistic case where each trajectory file came from an independent sampling pass. The
    misalignment causes downstream AUCs to silently differ from a pipeline that simply happened to keep the
    orderings aligned.
    """

    base = datetime(2022, 1, 1, tzinfo=UTC)
    subjects = [1, 2, 3, 4]
    pred_times = {s: base + timedelta(days=s) for s in subjects}

    # Ground-truth MEDS: each subject has one event_A at a unique day after their pred_time
    true_offsets = {1: 4, 2: 8, 3: 12, 4: 16}
    MEDS_df = pl.DataFrame(
        {
            "subject_id": subjects,
            "time": [pred_times[s] + timedelta(days=true_offsets[s]) for s in subjects],
            "code": ["event_A"] * len(subjects),
        }
    )

    # Helper to build a trajectory dataframe with a given subject ordering.
    def build_traj_df(ordering: list[int], pred_offsets: dict[int, int]) -> pl.DataFrame:
        rows = []
        for s in ordering:
            # Dummy row to anchor the (subject_id, prediction_time) group.
            rows.append(
                {"subject_id": s, "prediction_time": pred_times[s], "time": pred_times[s], "code": "DUMMY"}
            )
            rows.append(
                {
                    "subject_id": s,
                    "prediction_time": pred_times[s],
                    "time": pred_times[s] + timedelta(days=pred_offsets[s]),
                    "code": "event_A",
                }
            )
        return pl.DataFrame(rows)

    # Two trajectories. The per-subject offsets differ, so positional misalignment
    # produces visibly different downstream AUCs.
    # traj_a, in subject order [1,2,3,4], assigns near-perfect predictions.
    traj_a = build_traj_df([1, 2, 3, 4], {1: 3, 2: 7, 3: 11, 4: 15})
    # traj_b is in REVERSE subject order [4,3,2,1] but assigns predictions that, when
    # aligned correctly by subject, are also near-perfect. If `merge_pred_ttes` aligns
    # positionally instead, subject 1 gets traj_b's *subject 4* TTE (and vice-versa).
    traj_b = build_traj_df([4, 3, 2, 1], {1: 5, 2: 9, 3: 13, 4: 17})

    (tmp_path / "traj_a.parquet").write_bytes(b"")  # ensure dir exists
    traj_a.write_parquet(tmp_path / "traj_a.parquet")
    traj_b.write_parquet(tmp_path / "traj_b.parquet")

    predicates = {"A": PlainPredicateConfig(code="event_A")}

    # Compute AUCs through the public entry point.
    actual = temporal_auc_from_trajectory_files(
        MEDS_df,
        tmp_path,
        predicates,
        duration_grid=[timedelta(days=10)],
    )

    # Correct (subject-aligned) predicted probabilities at duration=10d:
    #   subj 1: traj_a=3d in, traj_b=5d in  -> prob 1.0
    #   subj 2: traj_a=7d in, traj_b=9d in  -> prob 1.0
    #   subj 3: traj_a=11d not, traj_b=13d not -> prob 0.0
    #   subj 4: traj_a=15d not, traj_b=17d not -> prob 0.0
    # Labels at 10d: subj 1 True, subj 2 True, subj 3 False, subj 4 False
    # Positives [1.0, 1.0], Negatives [0.0, 0.0]  ->  AUC = 1.0
    #
    # If positional concat scrambles traj_b's TTEs, subject 1 gets traj_b's
    # subject-4 prediction (17d) and subject 4 gets traj_b's subject-1 prediction (5d).
    # That puts subject 4 (a negative) at higher predicted probability than subject 1
    # (a positive) → AUC drops below 1.0.
    assert actual["AUC/A"][0] == 1.0, (
        "AUC mismatch indicates trajectory rows were mis-attributed across subjects "
        "by `merge_pred_ttes` (#32). Actual: " + str(actual["AUC/A"][0])
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
