from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from aces.config import PlainPredicateConfig
from omegaconf import OmegaConf

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC import (
    _normalize_predicates,
    get_trajectory_tte,
    merge_pred_ttes,
    temporal_auc_from_trajectory_files,
)


def test_normalize_predicates_from_mapping():
    mapping = {
        "A": PlainPredicateConfig("A"),
        "B": OmegaConf.create({"code": "B"}),
    }
    result = _normalize_predicates(mapping)
    assert set(result) == {"A", "B"}
    assert all(isinstance(cfg, PlainPredicateConfig) for cfg in result.values())
    assert result["A"].code == "A"
    assert result["B"].code == "B"


def test_normalize_predicates_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "preds.yaml"
    yaml_path.write_text("predicates:\n  X:\n    code: X\n  Y:\n    code: Y\n")
    result = _normalize_predicates(yaml_path)
    assert set(result) == {"X", "Y"}
    assert all(isinstance(cfg, PlainPredicateConfig) for cfg in result.values())


def test_merge_pred_ttes():
    dt = datetime(2021, 1, 1, tzinfo=UTC)
    df1 = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": [dt, dt],
            "tte/A": [timedelta(days=1), timedelta(days=2)],
        }
    )
    df2 = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "prediction_time": [dt, dt],
            "tte/A": [timedelta(days=3), None],
        }
    )
    merged = merge_pred_ttes([df1, df2])
    assert merged.shape == (2, 3)
    assert merged["tte/A"].to_list() == [
        [timedelta(days=1), timedelta(days=3)],
        [timedelta(days=2), None],
    ]


def test_get_trajectory_tte():
    dt1 = datetime(2022, 1, 1, tzinfo=UTC)
    dt2 = datetime(2022, 1, 3, tzinfo=UTC)
    trajectory_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "prediction_time": [dt1, dt1, dt2, dt2],
            "time": [
                dt1 + timedelta(days=1),
                dt1 + timedelta(days=2),
                dt2 + timedelta(days=1),
                dt2 + timedelta(days=3),
            ],
            "code": ["A", "B", "A", "B"],
        }
    )
    preds = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
    result = get_trajectory_tte(trajectory_df, preds)
    assert result.shape == (2, 4)
    assert result["tte/A"].to_list() == [timedelta(days=1), timedelta(days=1)]
    assert result["tte/B"].to_list() == [timedelta(days=2), timedelta(days=3)]


def test_temporal_auc_from_trajectory_files(monkeypatch, tmp_path: Path):
    dt = datetime(2022, 1, 1, tzinfo=UTC)
    MEDS_df = pl.DataFrame(
        {
            "subject_id": [1],
            "prediction_time": [dt],
            "time": [dt],
            "code": ["A"],
        }
    )

    traj_df = MEDS_df.clone()
    traj_fp = tmp_path / "t.parquet"
    traj_df.write_parquet(traj_fp)

    monkeypatch.setattr(
        "MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC.get_trajectory_tte",
        lambda df, preds: pl.DataFrame(
            {"subject_id": [1], "prediction_time": [dt], "tte/A": [timedelta(days=1)]}
        ),
    )
    monkeypatch.setattr(
        "MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC.merge_pred_ttes",
        lambda dfs: dfs[0],
    )
    monkeypatch.setattr(
        "MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC.get_raw_tte",
        lambda MEDS_df, index_df, preds, **kwargs: pl.DataFrame(
            {"subject_id": [1], "prediction_time": [dt], "tte/A": [timedelta(days=1)]}
        ),
    )
    monkeypatch.setattr(
        "MEDS_trajectory_evaluation.temporal_AUC_evaluation.trajectory_AUC.temporal_aucs",
        lambda true, pred, **kwargs: pl.DataFrame({"duration": [timedelta(days=1)], "AUC/A": [1.0]}),
    )

    result = temporal_auc_from_trajectory_files(
        MEDS_df,
        traj_fp,
        {"A": PlainPredicateConfig("A")},
        duration_grid=[timedelta(days=1)],
    )
    assert result["AUC/A"][0] == 1.0
