"""Test set-up and fixtures code."""

import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from aces.config import TaskExtractorConfig
from MEDS_transforms.utils import print_directory_contents
from omegaconf import DictConfig, OmegaConf


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
) -> None:
    doctest_namespace.update(
        {
            "sample_ACES_cfg": sample_ACES_cfg,
            "Path": Path,
            "sample_task_criteria_cfg": sample_task_criteria_cfg,
            "sample_predicates_cfg": sample_predicates_cfg,
            "sample_task_criteria_fp": sample_task_criteria_fp,
            "sample_predicates_fp": sample_predicates_fp,
            "DictConfig": DictConfig,
            "MagicMock": MagicMock,
            "patch": patch,
            "print_directory_contents": print_directory_contents,
            "print_warnings": partial(print_warnings, caplog),
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "tempfile": tempfile,
        }
    )
