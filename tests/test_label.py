import subprocess
import tempfile
from pathlib import Path


def test_labeling_runs(
    sample_labeled_trajectories_on_disk: Path,
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        cmd = [
            "ZSACES_label",
            f"task.criteria_fp={sample_task_criteria_fp!s}",
            f"task.predicates_fp={sample_predicates_fp!s}",
            f"output_dir={out_dir!s}",
            f"trajectories_dir={sample_labeled_trajectories_on_disk!s}",
        ]

        out = subprocess.run(cmd, shell=False, check=False, capture_output=True)

        err_lines = [f"Stdout: {out.stdout.decode()}", f"Stderr: {out.stderr.decode()}"]

        assert out.returncode == 0, "\n".join([f"Expected return code 0; got {out.returncode}", *err_lines])

        out_files = list(out_dir.rglob("*.parquet"))
        assert len(out_files) > 0, "\n".join(["Expected at least one output file; got 0", *err_lines])
