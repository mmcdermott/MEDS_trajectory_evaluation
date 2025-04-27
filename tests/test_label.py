import subprocess
from pathlib import Path


def test_labeling_runs(
    sample_labeled_trajectories_on_disk: Path,
    sample_task_criteria_fp: Path,
    sample_predicates_fp: Path,
):
    cmd = [
        "ZSACES_label",
        f"task.criteria_fp={sample_task_criteria_fp!s}",
        f"task.predicates_fp={sample_predicates_fp!s}",
    ]

    out = subprocess.run(cmd, shell=False, check=False, capture_output=True)
    assert out.returncode == 0
