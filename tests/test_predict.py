import subprocess
import tempfile
from pathlib import Path

import polars as pl
from meds_evaluation.schema import PredictionSchema


def test_aggregate_runs(
    sample_labeled_trajectories_on_disk: Path,
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        labels_dir = Path(tmpdir) / "labels"
        cmd_label = [
            "MTE_label",
            f"task.criteria_fp={sample_task_criteria_fp!s}",
            f"task.predicates_fp={sample_predicates_fp!s}",
            f"output_dir={labels_dir!s}",
            f"trajectories_dir={sample_labeled_trajectories_on_disk!s}",
        ]

        out_label = subprocess.run(cmd_label, shell=False, check=False, capture_output=True)
        label_err_lines = [f"Stdout: {out_label.stdout.decode()}", f"Stderr: {out_label.stderr.decode()}"]
        assert out_label.returncode == 0, "\n".join(
            [f"Expected return code 0; got {out_label.returncode}", *label_err_lines]
        )

        out_fp = Path(tmpdir) / "preds.parquet"
        cmd_agg = [
            "MTE_aggregate",
            f"labels_dir={labels_dir!s}",
            f"output_fp={out_fp!s}",
        ]

        out = subprocess.run(cmd_agg, shell=False, check=False, capture_output=True)

        err_lines = [f"Stdout: {out.stdout.decode()}", f"Stderr: {out.stderr.decode()}"]

        assert out.returncode == 0, "\n".join([f"Expected return code 0; got {out.returncode}", *err_lines])

        assert out_fp.exists(), "\n".join(["Expected output file not found", *err_lines])
        df = pl.read_parquet(out_fp, use_pyarrow=True)
        assert df.height > 0, "\n".join(["Expected output dataframe to have rows", *err_lines])

        PredictionSchema.validate(df.to_arrow())
