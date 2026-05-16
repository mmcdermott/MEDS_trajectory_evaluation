"""End-to-end test for the ``ZSACES_predict`` pipeline.

Stages exercised:
  1. ``label_trajectories`` — invoked once per input trajectory parquet, writes
     ``(subject_id, prediction_time, valid, determinable, label)`` to
     ``output_dir/label_trajectories/<shard>/<traj>.parquet``.
  2. ``aggregate_predictions`` — invoked once per input shard directory, reads every
     trajectory's labels for that shard, joins with ground-truth labels, writes one
     ``meds-evaluation``-compatible parquet at
     ``output_dir/aggregate_predictions/<shard>.parquet``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import polars as pl
from meds import LabelSchema


def _shard_trajectories(
    src_dfs: dict[str, pl.DataFrame],
    root: Path,
    shard: str = "shard_0",
) -> None:
    """Write the sample-trajectory fixtures into a single named shard directory."""
    shard_dir = root / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    for fn, df in src_dfs.items():
        df.write_parquet(shard_dir / fn, use_pyarrow=True)


def _write_ground_truth(src_dfs: dict[str, pl.DataFrame], fp: Path) -> None:
    """Synthesize a ground-truth labels parquet for every (subject, prediction_time).

    The label values themselves are immaterial for the pipeline structure test — we pick all-True so the
    comparison is unambiguous.
    """
    rows = []
    seen: set[tuple[int, object]] = set()
    for df in src_dfs.values():
        for s, t in zip(
            df[LabelSchema.subject_id_name].to_list(),
            df[LabelSchema.prediction_time_name].to_list(),
            strict=True,
        ):
            if (s, t) not in seen:
                seen.add((s, t))
                rows.append({"subject_id": s, "prediction_time": t, "boolean_value": True})
    pl.DataFrame(rows).write_parquet(fp, use_pyarrow=True)


def test_predict_pipeline_end_to_end(
    sample_labeled_trajectories_dfs: dict[str, pl.DataFrame],
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        traj_dir = root / "trajectories"
        labels_fp = root / "ground_truth.parquet"
        out_dir = root / "output"

        _shard_trajectories(sample_labeled_trajectories_dfs, traj_dir, shard="shard_0")
        _write_ground_truth(sample_labeled_trajectories_dfs, labels_fp)

        cmd = [
            "ZSACES_predict",
            f"task.criteria_fp={sample_task_criteria_fp!s}",
            f"task.predicates_fp={sample_predicates_fp!s}",
            f"trajectories_dir={traj_dir!s}",
            f"task_labels_fp={labels_fp!s}",
            f"output_dir={out_dir!s}",
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True)
        err = [f"Stdout: {completed.stdout.decode()}", f"Stderr: {completed.stderr.decode()}"]
        assert completed.returncode == 0, "\n".join([f"exit={completed.returncode}", *err])

        # Stage 1 wrote per-trajectory labels at the same shard path as input.
        labels = sorted((out_dir / "label_trajectories").rglob("*.parquet"))
        assert len(labels) == len(sample_labeled_trajectories_dfs), "\n".join(
            ["per-trajectory label files missing", *err]
        )

        # Stage 2 wrote exactly one predictions parquet for the single input shard.
        predictions_files = sorted((out_dir / "aggregate_predictions").rglob("*.parquet"))
        assert len(predictions_files) == 1, "\n".join(
            [f"expected 1 predictions parquet, got {len(predictions_files)}", *err]
        )
        predictions = pl.read_parquet(predictions_files[0])
        expected_cols = {
            "subject_id",
            "prediction_time",
            "boolean_value",
            "predicted_boolean_probability",
            "n_total",
            "n_valid",
            "n_usable",
        }
        assert set(predictions.columns) == expected_cols, (
            f"unexpected columns: {set(predictions.columns) ^ expected_cols}"
        )

        # Every ground-truth (subject, prediction_time) shows up exactly once.
        ground_truth = pl.read_parquet(labels_fp)
        assert predictions.height == ground_truth.height
        # boolean_value is preserved from ground truth.
        assert predictions["boolean_value"].to_list() == [True] * predictions.height
        # n_total is per-(subject, prediction_time): the count of trajectory files in which
        # that subject appears. Bounded above by the total number of trajectory files; every
        # ground-truth subject is present in at least one trajectory in the fixture.
        n_total = predictions["n_total"]
        assert n_total.min() >= 1
        assert n_total.max() <= len(sample_labeled_trajectories_dfs)


def test_predict_pipeline_is_resumable_per_shard(
    sample_labeled_trajectories_dfs: dict[str, pl.DataFrame],
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
) -> None:
    """Re-running after the predictions parquet is deleted must produce the same output.

    The intermediate per-trajectory labels in ``output_dir/label_trajectories/`` are the cache that makes this
    efficient — neither stage re-does work whose output already exists.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        traj_dir = root / "trajectories"
        labels_fp = root / "ground_truth.parquet"
        out_dir = root / "output"

        _shard_trajectories(sample_labeled_trajectories_dfs, traj_dir, shard="shard_0")
        _write_ground_truth(sample_labeled_trajectories_dfs, labels_fp)

        common = [
            "ZSACES_predict",
            f"task.criteria_fp={sample_task_criteria_fp!s}",
            f"task.predicates_fp={sample_predicates_fp!s}",
            f"trajectories_dir={traj_dir!s}",
            f"task_labels_fp={labels_fp!s}",
            f"output_dir={out_dir!s}",
        ]
        assert subprocess.run(common, check=False, capture_output=True).returncode == 0

        predictions_fp = next((out_dir / "aggregate_predictions").rglob("*.parquet"))
        first = pl.read_parquet(predictions_fp)
        labels_mtimes_before = {
            p: p.stat().st_mtime for p in (out_dir / "label_trajectories").rglob("*.parquet")
        }

        # Knock out the predictions but leave the per-traj labels intact, then re-run.
        predictions_fp.unlink()
        assert subprocess.run(common, check=False, capture_output=True).returncode == 0

        second = pl.read_parquet(predictions_fp)
        assert first.equals(second), "second run produced different predictions"

        # Per-trajectory label files weren't rewritten (mtime preserved).
        for p, mtime in labels_mtimes_before.items():
            assert p.stat().st_mtime == mtime, f"stage 1 re-ran for {p}"
