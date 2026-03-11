from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from aces.config import PlainPredicateConfig
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.get_ttes import get_raw_tte


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

    # compute manual ttes and histories replicating get_raw_tte logic
    manual = {}
    has_none = False
    for row in index_rows:
        s = row["subject_id"]
        pt = row["prediction_time"]
        for task in tasks:
            events = sorted(events_by_subject_task.get((s, task), []))
            if not events:
                idx = 1
            else:
                idx = 0
                while idx < len(events) and events[idx] <= pt:
                    idx += 1
            history = idx > 0
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
