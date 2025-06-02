"""Test set-up and fixtures code."""

import json
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from aces.config import TaskExtractorConfig
from meds import LabelSchema
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def sample_task_schema() -> pl.DataFrame:
    """A fixture that provides sample trajectories for testing."""

    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "prediction_time": [
                datetime(1993, 1, 1, tzinfo=UTC),
                datetime(1993, 1, 20, tzinfo=UTC),
                datetime(1999, 1, 1, tzinfo=UTC),
            ],
            "boolean_value": [True, False, True],
        }
    )


class Label(NamedTuple):
    valid: bool | None = None
    determinable: bool | None = None
    label: bool | None = None


class LabeledTrajectory(NamedTuple):
    trajectory: pl.DataFrame
    labels_by_relaxation: dict[frozenset[str], Label]


@pytest.fixture
def sample_labeled_trajectories(
    sample_task_schema: pl.DataFrame,
) -> dict[tuple[int, datetime], list[LabeledTrajectory]]:
    """A fixture that provides sample generated trajectories for testing.

    Recall that the sample configuration file has this structure:

    trigger
    └── (+1 day, 0:00:00) input.end (no icu_admission, discharge_or_death); **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no icu_admission, discharge_or_death)
            └── (next discharge_or_death) target.end; **Label: Presence of death**

    The structure of the labels by relaxation is that for any set of relaxation options, the label is given by
    the largest dictionary key set that is a subset of the query set of relaxations.
    """

    return {
        (1, datetime(1993, 1, 1, tzinfo=UTC)): [
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [
                            datetime(1993, 1, 1, 12, tzinfo=UTC),
                            datetime(1993, 1, 1, 13, tzinfo=UTC),
                            datetime(1993, 1, 1, 14, tzinfo=UTC),
                            datetime(1993, 1, 22, tzinfo=UTC),
                        ],
                        "code": ["LAB_1", "LAB_2", "ICU_DISCHARGE", "MEDS_DEATH"],
                        "numeric_value": [1.0, None, None, None],
                    }
                ),
                {
                    frozenset(): Label(valid=False),
                    frozenset({"remove_all_criteria"}): Label(label=True),
                    frozenset({"remove_all_criteria", "collapse_temporal_gap_windows"}): Label(label=False),
                },
            ),
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(1993, 1, 1, 12, tzinfo=UTC), datetime(1993, 1, 4, tzinfo=UTC)],
                        "code": ["LAB_1", "MEDS_DEATH"],
                        "numeric_value": [1.0, None],
                    }
                ),
                {frozenset(): Label(label=True)},  # Label is always true
            ),
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(1993, 1, 1, 12, tzinfo=UTC), datetime(1993, 1, 1, 13, tzinfo=UTC)],
                        "code": ["ICU_DISCHARGE", "ICU_ADMISSION"],
                        "numeric_value": [None, None],
                    }
                ),
                {
                    frozenset(): Label(valid=False),  # There is a discharge in the gap window.
                    frozenset({"remove_all_criteria"}): Label(determinable=False),
                    frozenset({"remove_all_criteria", "collapse_temporal_gap_windows"}): Label(label=False),
                },
            ),
        ],
        (1, datetime(1993, 1, 20, tzinfo=UTC)): [
            # Trajectory one is valid under the raw config and has a negative label. It has some unused future
            # data.
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(1993, 2, 20, tzinfo=UTC), datetime(1995, 1, 1, tzinfo=UTC)],
                        "code": ["ICU_DISCHARGE", "LAB_23"],
                        "numeric_value": [None, 1.2],
                    }
                ),
                {frozenset(): Label(label=False)},  # Label is always false.
            ),
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(1998, 1, 1, tzinfo=UTC), datetime(2000, 1, 1, tzinfo=UTC)],
                        "code": ["LAB_1", "LAB_3"],
                        "numeric_value": [1.1, 1.2],
                    }
                ),
                {frozenset(): Label(determinable=False)},  # Determinable will always be false here.
            ),
            LabeledTrajectory(
                # Trajectory 3 is empty here, to assess robustness.
                pl.DataFrame({"time": [], "code": [], "numeric_value": []}),
                {frozenset(): Label(determinable=False)},  # Determinable will always be false here.
            ),
        ],
        (2, datetime(1999, 1, 1, tzinfo=UTC)): [
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [
                            datetime(1999, 1, 1, 13, tzinfo=UTC),
                            datetime(1999, 1, 1, 14, tzinfo=UTC),
                            datetime(1999, 1, 4, 14, tzinfo=UTC),
                        ],
                        "code": ["LAB_3", "ICU_DISCHARGE", "LAB_4"],
                        "numeric_value": [None, None, 1.1],
                    }
                ),
                {
                    frozenset(): Label(valid=False),
                    frozenset({"remove_all_criteria"}): Label(determinable=False),
                    frozenset({"remove_all_criteria", "collapse_temporal_gap_windows"}): Label(label=False),
                },
            ),
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(1999, 1, 1, 12, tzinfo=UTC), datetime(1999, 2, 1, tzinfo=UTC)],
                        "code": ["ICU_ADMISSION", "MEDS_DEATH"],
                        "numeric_value": [None, None],
                    }
                ),
                {frozenset(): Label(valid=False), frozenset({"remove_all_criteria"}): Label(label=True)},
            ),
            LabeledTrajectory(
                pl.DataFrame(
                    {
                        "time": [datetime(2005, 1, 1, tzinfo=UTC)],
                        "code": ["MEDS_DEATH"],
                        "numeric_value": [None],
                    }
                ),
                {frozenset(): Label(label=True)},
            ),
        ],
    }


@pytest.fixture
def sample_labeled_trajectories_dfs(
    sample_labeled_trajectories: dict[tuple[int, datetime], list[LabeledTrajectory]],
) -> dict[str, pl.DataFrame]:
    df_parts = defaultdict(list)

    for (subject_id, prediction_time), labeled_trajectories in sample_labeled_trajectories.items():
        for i, labeled_trajectory in enumerate(labeled_trajectories):
            fn = f"trajectory_{i}.parquet"
            trajectory_df = labeled_trajectory.trajectory.with_columns(
                pl.lit(subject_id).alias(LabelSchema.subject_id_name),
                pl.lit(prediction_time).alias(LabelSchema.prediction_time_name),
            )
            df_parts[fn].append(trajectory_df)

    return {k: pl.concat(dfs, how="vertical_relaxed") for k, dfs in df_parts.items()}


@pytest.fixture
def sample_labeled_trajectories_on_disk(
    sample_labeled_trajectories_dfs: dict[str, pl.DataFrame],
) -> Path:
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        for fn, df in sample_labeled_trajectories_dfs.items():
            df.write_parquet(root / fn, use_pyarrow=True)

        yield root


@pytest.fixture
def sample_task_criteria_cfg() -> DictConfig:
    """A sample task definition."""

    return DictConfig(
        {
            "predicates": {
                "icu_admission": "???",
                "icu_discharge": "???",
                "death": {"code": {"regex": "MEDS_DEATH.*"}},
                "discharge_or_death": {"expr": "or(icu_discharge, death)"},
            },
            "trigger": "icu_admission",
            "windows": {
                "input": {
                    "start": "trigger",
                    "end": "start + 24h",
                    "start_inclusive": True,
                    "end_inclusive": True,
                    "index_timestamp": "end",
                    "has": {
                        "icu_admission": "(None, 0)",
                        "discharge_or_death": "(None, 0)",
                    },
                },
                "gap": {
                    "start": "input.end",
                    "end": "start + 24h",
                    "start_inclusive": False,
                    "end_inclusive": True,
                    "has": {
                        "icu_admission": "(None, 0)",
                        "discharge_or_death": "(None, 0)",
                    },
                },
                "target": {
                    "start": "gap.end",
                    "end": "start -> discharge_or_death",
                    "start_inclusive": False,
                    "end_inclusive": True,
                    "label": "death",
                },
            },
        }
    )


@pytest.fixture
def sample_predicates_cfg() -> DictConfig:
    """A sample predicates definition."""

    return DictConfig(
        {
            "predicates": {
                "icu_admission": {"code": "ICU_ADMISSION"},
                "icu_discharge": {"code": "ICU_DISCHARGE"},
            },
        }
    )


@pytest.fixture
def sample_task_criteria_fp(sample_task_criteria_cfg: DictConfig, tmp_path: Path) -> Path:
    """A sample task criteria file path."""

    criteria_fp = tmp_path / "task_criteria.yaml"
    OmegaConf.save(sample_task_criteria_cfg, criteria_fp)
    return criteria_fp


@pytest.fixture
def sample_predicates_fp(sample_predicates_cfg: DictConfig, tmp_path: Path) -> Path:
    """A sample predicates file path."""

    predicates_fp = tmp_path / "predicates.yaml"
    OmegaConf.save(sample_predicates_cfg, predicates_fp)
    return predicates_fp


@pytest.fixture
def sample_ACES_cfg(sample_task_criteria_fp: Path, sample_predicates_fp: Path) -> TaskExtractorConfig:
    """A sample ACES configuration."""
    return TaskExtractorConfig.load(sample_task_criteria_fp, sample_predicates_fp)


@contextmanager
def print_warnings(caplog: pytest.LogCaptureFixture):
    """Captures all logged warnings within this context block and prints them upon exit.

    This is useful in doctests, where you want to show printed outputs for documentation and testing purposes.
    """

    n_current_records = len(caplog.records)

    with caplog.at_level("WARNING"):
        yield
    # Print all captured warnings upon exit
    for record in caplog.records[n_current_records:]:
        print(f"Warning: {record.getMessage()}")


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    sample_task_criteria_cfg: DictConfig,
    sample_predicates_cfg: DictConfig,
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
    sample_ACES_cfg: TaskExtractorConfig,
    sample_labeled_trajectories_dfs: dict[str, pl.DataFrame],
) -> None:
    doctest_namespace.update(
        {
            "sample_labeled_trajectories_dfs": sample_labeled_trajectories_dfs,
            "sample_ACES_cfg": sample_ACES_cfg,
            "Path": Path,
            "sample_task_criteria_cfg": sample_task_criteria_cfg,
            "sample_predicates_cfg": sample_predicates_cfg,
            "sample_task_criteria_fp": sample_task_criteria_fp,
            "sample_predicates_fp": sample_predicates_fp,
            "DictConfig": DictConfig,
            "MagicMock": MagicMock,
            "patch": patch,
            "print_warnings": partial(print_warnings, caplog),
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "timedelta": timedelta,
            "tempfile": tempfile,
        }
    )
