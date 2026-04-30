from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from aces.config import PlainPredicateConfig
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.get_ttes import get_raw_tte


def test_history_false_for_subject_with_no_matching_predicate_events():
    """Regression test for https://github.com/mmcdermott/MEDS_trajectory_evaluation/issues/30.

    Real-world scenario: an ICU 30-day mortality cohort. A "diabetes" predicate is computed
    for every patient at admission. Most patients in a cohort like this have *no* prior
    diabetes diagnosis at all — those subjects are entirely absent from the predicate-times
    intermediate (since `get_all_predicate_times` filters via `any_horizontal` over predicates).

    Pre-fix: the right-join in `get_raw_tte` re-introduced those subjects with `null`
    predicate-list cells, and `null.explode().search_sorted(...)` returned 1 — flagging
    them as if they had history. This silently corrupted any downstream `exclude_history`
    cohort filter. The fix masks `idx > 0` by `list.len() > 0` so empty/null lists
    yield `False`.
    """

    # Three patients admitted on the same day. Two have prior diabetes events; one doesn't.
    # Subject 30 has only an unrelated `LAB_GLUCOSE` event in MEDS — no diabetes events
    # at all. They are exactly the case `get_all_predicate_times` drops before the right-join.
    MEDS_df = pl.DataFrame(
        {
            "subject_id": [10, 10, 20, 30, 30],
            "time": [
                datetime(2022, 1, 1, tzinfo=UTC),  # subj 10: prior diabetes dx
                datetime(2022, 6, 1, tzinfo=UTC),  # subj 10: post-admission diabetes
                datetime(2022, 1, 15, tzinfo=UTC),  # subj 20: prior diabetes dx
                datetime(2022, 5, 1, tzinfo=UTC),  # subj 30: only unrelated lab events
                datetime(2022, 6, 15, tzinfo=UTC),
            ],
            "code": [
                "ICD10//E11.9",
                "ICD10//E11.9",
                "ICD10//E11.9",
                "LAB_GLUCOSE",
                "LAB_GLUCOSE",
            ],
        }
    )

    # All three are admitted (predicted on) 2022-04-01.
    index_df = pl.DataFrame(
        {
            "subject_id": [10, 20, 30],
            "prediction_time": [datetime(2022, 4, 1, tzinfo=UTC)] * 3,
        }
    )

    predicates = {"diabetes": PlainPredicateConfig(code="ICD10//E11.9")}

    result = get_raw_tte(MEDS_df, index_df, predicates, include_history=True)
    history_by_subject = dict(
        zip(result["subject_id"].to_list(), result["history/diabetes"].to_list(), strict=False)
    )

    assert history_by_subject[10] is True, "subject 10 had a prior diabetes dx"
    assert history_by_subject[20] is True, "subject 20 had a prior diabetes dx"
    assert history_by_subject[30] is False, (
        "subject 30 has zero diabetes events of any kind — `history/diabetes` must be False"
    )


def _duration_tds(min_days: int, max_days: int) -> st.SearchStrategy[timedelta]:
    return st.integers(min_value=min_days, max_value=max_days).map(lambda d: timedelta(days=d))


@st.composite
def _raw_inputs(draw):
    base_time = datetime(2023, 1, 1, tzinfo=UTC)
    subject_count = draw(st.integers(min_value=1, max_value=5))
    subjects = list(range(1, subject_count + 1))
    tasks = draw(st.lists(st.sampled_from(["A", "B", "C"]), min_size=1, max_size=3, unique=True))

    # generate MEDS events
    meds_rows: list[dict[str, object]] = []
    events_by_subject_task: dict[tuple[int, str], list[datetime]] = {}
    for s in subjects:
        for task in tasks:
            n_events = draw(st.integers(min_value=0, max_value=3))
            event_durations = draw(st.lists(_duration_tds(-5, 30), min_size=n_events, max_size=n_events))
            times = [base_time + d for d in sorted(event_durations)]
            events_by_subject_task[(s, task)] = times
            for t in times:
                meds_rows.append({"subject_id": s, "time": t, "code": task})
        # ensure each subject has at least one event to avoid null joins
        if all(len(events_by_subject_task[(s, t)]) == 0 for t in tasks):
            events_by_subject_task[(s, tasks[0])] = [base_time + timedelta(days=1)]
            meds_rows.append({"subject_id": s, "time": base_time + timedelta(days=1), "code": tasks[0]})

    MEDS_df = pl.DataFrame(meds_rows)

    # generate index dataframe
    index_rows = []
    for s in subjects:
        n_index = draw(st.integers(min_value=1, max_value=2))
        pred_durations = draw(
            st.lists(_duration_tds(-2, 10), min_size=n_index, max_size=n_index, unique=True)
        )
        for d in sorted(pred_durations):
            index_rows.append({"subject_id": s, "prediction_time": base_time + d})
    index_df = pl.DataFrame(index_rows)

    # compute manual ttes and histories replicating get_raw_tte semantics
    manual = {}
    has_none = False
    for row in index_rows:
        s = row["subject_id"]
        pt = row["prediction_time"]
        for task in tasks:
            events = sorted(events_by_subject_task.get((s, task), []))
            idx = 0
            while idx < len(events) and events[idx] <= pt:
                idx += 1
            # history = "any predicate event at or before the prediction time"; subjects
            # with no events for this predicate (empty or null list) have history=False.
            history = idx > 0 and len(events) > 0
            if idx < len(events):
                tte = events[idx] - pt
            else:
                tte = None
                has_none = True
            manual[(s, pt, task)] = (tte, history)
    assume(has_none)

    return MEDS_df, index_df, tasks, manual


@settings(deadline=None, max_examples=50)
@given(_raw_inputs())
def test_get_raw_tte_matches_manual(data):
    MEDS_df, index_df, tasks, manual = data
    preds = {t: PlainPredicateConfig(code=t) for t in tasks}

    result = get_raw_tte(MEDS_df, index_df, preds)
    result_hist = get_raw_tte(MEDS_df, index_df, preds, include_history=True)

    for i in range(index_df.height):
        sid = index_df["subject_id"][i]
        pt = index_df["prediction_time"][i]
        assert result["subject_id"][i] == sid
        assert result["prediction_time"][i] == pt
        for task in tasks:
            expected_tte, expected_hist = manual[(sid, pt, task)]
            assert result[f"tte/{task}"][i] == expected_tte
            assert result_hist[f"tte/{task}"][i] == expected_tte
            assert result_hist[f"history/{task}"][i] == expected_hist
