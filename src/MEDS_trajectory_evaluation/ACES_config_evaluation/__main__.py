"""Hydra-based CLI entry point for end-to-end zero-shot prediction.

``ZSACES_predict`` runs a two-stage pipeline:

  1. ``label_trajectories`` — per-trajectory ACES labeling, writing one parquet of
     ``(subject_id, prediction_time, valid, determinable, label)`` rows per input
     trajectory parquet. Output lives at ``output_dir/label_trajectories/`` and mirrors
     the trajectory dir structure (so ``trajectories_dir/train/0/3.parquet`` produces
     ``output_dir/label_trajectories/train/0/3.parquet``).
  2. ``aggregate_predictions`` — per *input shard*, reads every trajectory's labels for
     the shard, applies the configured ``valid`` / ``determinable`` policy, computes
     the empirical probability per ``(subject_id, prediction_time)``, joins with the
     ground-truth labels, and writes a single ``meds-evaluation``-compatible parquet at
     ``output_dir/aggregate_predictions/<shard>.parquet``.

Both stages preserve ``map_over``-style per-shard resumability: a re-run after a crash
skips trajectory parquets whose label outputs already exist, and skips shards whose
predictions parquets already exist.

Example invocation::

    ZSACES_predict task.criteria_fp=/path/to/criteria.yaml \\
        task.predicates_fp=/path/to/predicates.yaml \\
        trajectories_dir=/path/to/trajectories \\
        task_labels_fp=/path/to/ground_truth_labels.parquet \\
        output_dir=/path/to/output \\
        [policy.invalid=drop|negative|positive] \\
        [policy.indeterminable=drop|negative|positive]
"""

from __future__ import annotations

import logging
import random
from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig, OmegaConf

from .aggregate_predictions import (
    Policy,
    _read_shard_dir,
    aggregate_predictions,
    shard_dir_iterator,
)
from .label_trajectories import label_trajectories
from .task_config import resolve_zero_shot_task_cfg
from .utils import get_in_out_fps, hash_based_seed

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_trajectory_evaluation") / "ACES_config_evaluation" / "configs"

LABELS_SUBDIR = "label_trajectories"
PREDICTIONS_SUBDIR = "aggregate_predictions"


def _run_labeling(cfg: DictConfig, labels_dir: Path) -> None:
    """Stage 1: per-trajectory labeling via map_over.

    Writes one parquet per input trajectory at ``labels_dir/<same/relative/path>``. map_over skips outputs
    that already exist, so re-running after a crash resumes cleanly.
    """
    zero_shot_task_cfg = resolve_zero_shot_task_cfg(cfg.task, cfg.labeler)

    in_out_fps = get_in_out_fps(Path(cfg.trajectories_dir), labels_dir)
    seed = hash_based_seed(cfg.seed, cfg.worker)
    random.seed(seed)
    random.shuffle(in_out_fps)

    map_over(
        in_out_fps,
        partial(label_trajectories, zero_shot_task_cfg=zero_shot_task_cfg),
        read_fn=partial(pl.read_parquet, use_pyarrow=True, glob=False),
        write_fn=partial(pl.DataFrame.write_parquet, use_pyarrow=True),
    )


def _run_aggregation(cfg: DictConfig, labels_dir: Path, predictions_dir: Path) -> None:
    """Stage 2: per-shard aggregation.

    Walks ``labels_dir`` for shard directories (every dir that contains a parquet), reads all per-trajectory
    parquets for that shard, applies the policy, joins with ground-truth labels (filtered to the shard's
    subjects), writes a single ``predictions_dir/<shard>.parquet``. Per-shard resumability via output-presence
    check, mirroring map_over.
    """
    ground_truth = pl.read_parquet(cfg.task_labels_fp)
    invalid_policy: Policy = cfg.policy.invalid
    indeterminable_policy: Policy = cfg.policy.indeterminable

    iter_cfg = OmegaConf.create(
        {"stage_cfg": {"data_input_dir": str(labels_dir), "output_dir": str(predictions_dir)}}
    )
    shards, _ = shard_dir_iterator(iter_cfg)

    for shard in shards:
        if shard.out_fp.exists():
            logger.info("Skipping already-aggregated shard: %s", shard.out_fp)
            continue
        shard_labels = _read_shard_dir(shard.in_fp)
        shard_subjects = shard_labels["subject_id"].unique()
        shard_gt = ground_truth.filter(pl.col("subject_id").is_in(shard_subjects))
        predictions = aggregate_predictions(
            shard_labels,
            shard_gt,
            invalid_policy=invalid_policy,
            indeterminable_policy=indeterminable_policy,
        )
        shard.out_fp.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(shard.out_fp)
        logger.info("Wrote %d predictions to %s", predictions.height, shard.out_fp)


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_predict")
def predict(cfg: DictConfig) -> None:
    """Run the full ZSACES prediction pipeline (label + aggregate).

    Args:
        cfg: Hydra config with the keys documented at the top of this module.
    """
    output_dir = Path(cfg.output_dir)
    labels_dir = output_dir / LABELS_SUBDIR
    predictions_dir = output_dir / PREDICTIONS_SUBDIR

    logger.info("Stage 1: labeling trajectories -> %s", labels_dir)
    _run_labeling(cfg, labels_dir)

    logger.info("Stage 2: aggregating predictions -> %s", predictions_dir)
    _run_aggregation(cfg, labels_dir, predictions_dir)
