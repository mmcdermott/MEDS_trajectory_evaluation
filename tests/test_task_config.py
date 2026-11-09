"""Tests for task configuration validation utilities."""

from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO

import polars as pl
import pytest
from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from bigtree import print_tree

from MEDS_trajectory_evaluation.ACES_config_evaluation.label import label_trajectories
from MEDS_trajectory_evaluation.ACES_config_evaluation.task_config import (
    convert_to_zero_shot,
    validate_task_cfg,
)


def test_validate_task_cfg_error_includes_tree_when_prediction_not_ancestor():
    """Ensures the validation error message contains the window tree when misconfigured."""
    task_cfg = TaskExtractorConfig(
        predicates={
            "trigger_pred": PlainPredicateConfig("TRIGGER"),
            "label_pred": PlainPredicateConfig("LABEL"),
        },
        trigger=EventConfig("trigger_pred"),
        windows={
            "prediction": WindowConfig(
                start="trigger + 1h",
                end="start + 1h",
                start_inclusive=True,
                end_inclusive=True,
                has={},
                index_timestamp="start",
            ),
            "label": WindowConfig(
                start="end - 1h",
                end="trigger + 4h",
                start_inclusive=False,
                end_inclusive=True,
                has={},
                label="label_pred",
            ),
        },
    )

    with pytest.raises(ValueError) as excinfo:
        validate_task_cfg(task_cfg)

    strio = StringIO()
    with redirect_stdout(strio):
        print_tree(task_cfg.window_tree)
    tree_str = strio.getvalue()

    message = str(excinfo.value)
    assert "prediction time window node (prediction.start)" in message
    assert tree_str in message


def _label_window_at_prediction_time_cfg(input_end: str) -> TaskExtractorConfig:
    """A task whose label window starts exactly at the prediction time: ``(t, t + 30d]``."""
    return TaskExtractorConfig(
        predicates={"label_pred": PlainPredicateConfig("LABEL")},
        trigger=EventConfig("_ANY_EVENT"),
        windows={
            "input": WindowConfig(
                start=None,
                end=input_end,
                start_inclusive=True,
                end_inclusive=True,
                has={},
                index_timestamp="end",
            ),
            "target": WindowConfig(
                start="input.end",
                end="start + 30d",
                start_inclusive=False,
                end_inclusive=True,
                has={},
                label="label_pred",
            ),
        },
    )


@pytest.mark.parametrize("input_end", ["trigger", "trigger + 24h"])
def test_validate_task_cfg_accepts_label_window_rooted_at_prediction_time(input_end):
    """A label window rooted at the prediction-time node itself is future-only and must be accepted.

    The window tree has no separate node for the label window's start: it *is* the prediction-time node, which
    is not its own ancestor.
    """
    validate_task_cfg(_label_window_at_prediction_time_cfg(input_end))


@pytest.mark.parametrize("input_end", ["trigger", "trigger + 24h"])
@pytest.mark.parametrize(
    "relaxations",
    [
        {},
        {
            "remove_all_criteria": True,
            "collapse_temporal_gap_windows": True,
            "remove_post_label_windows": True,
        },
    ],
)
def test_label_window_rooted_at_prediction_time_labels_window_edges(input_end, relaxations):
    """Labels follow ``(t, t + 30d]`` exactly: open at the prediction time, closed at the window end."""
    zero_shot_cfg = convert_to_zero_shot(_label_window_at_prediction_time_cfg(input_end), relaxations)

    t = datetime(2020, 1, 1, tzinfo=UTC)
    us, window = timedelta(microseconds=1), timedelta(days=30)
    # subject -> offset of its only LABEL event from the prediction time, and the expected label
    cases = {1: (us, True), 2: (window / 2, True), 3: (window, True), 4: (window + us, False)}
    rows = []
    for subject_id, (offset, _) in cases.items():
        rows.append((subject_id, t + offset, "LABEL", t))
        rows.append((subject_id, t + window + timedelta(days=5), "OTHER", t))
    trajectories = pl.DataFrame(
        rows, schema=["subject_id", "time", "code", "prediction_time"], orient="row"
    ).with_columns(pl.lit(None, dtype=pl.Float32).alias("numeric_value"))
    trajectories = trajectories.select("subject_id", "time", "code", "numeric_value", "prediction_time")

    labels = label_trajectories(trajectories, zero_shot_cfg).sort("subject_id")

    assert labels["valid"].to_list() == [True] * len(cases)
    assert labels["determinable"].to_list() == [True] * len(cases)
    assert labels["label"].to_list() == [expected for _, expected in cases.values()]
