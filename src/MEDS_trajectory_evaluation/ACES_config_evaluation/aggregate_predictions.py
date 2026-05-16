"""Aggregate per-trajectory ACES labels into per-shard meds-evaluation predictions.

This module exposes the second stage of the ``ZSACES_predict`` pipeline. Per-trajectory labels for one input
shard live in a directory of parquet files (the output of the ``label_trajectories`` stage). This stage reads
every trajectory's labels for the shard, applies a configurable ``valid`` / ``determinable`` policy, computes
the empirical probability per ``(subject_id, prediction_time)``, joins with the ground-truth labels for the
shard's subjects, and writes a single ``meds-evaluation``-compatible parquet per shard.

The aggregation logic itself is exposed as a pure ``pl.DataFrame``-in/``pl.DataFrame``-out function
``aggregate_predictions`` to keep it doctestable and testable independently of MEDS-transforms wiring. The
Stage registration at the bottom of the module is the only glue between that pure function and the MEDS-
transforms runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from meds import LabelSchema
from MEDS_transforms.mapreduce.shard_iteration import InOutFilePair
from MEDS_transforms.mapreduce.stage import map_stage
from MEDS_transforms.stages.base import Stage
from omegaconf import DictConfig  # noqa: TC002 — runtime config access in stage main_fn

Policy = Literal["drop", "negative", "positive"]

PREDICTIONS_SCHEMA_UPDATES = {
    LabelSchema.subject_id_name: pl.Int64,
    LabelSchema.prediction_time_name: pl.Datetime("us"),
    "boolean_value": pl.Boolean,
    "predicted_boolean_probability": pl.Float64,
    "n_total": pl.UInt32,
    "n_valid": pl.UInt32,
    "n_determinable": pl.UInt32,
    "n_usable": pl.UInt32,
}


def _apply_policy(labels: pl.DataFrame, *, invalid: Policy, indeterminable: Policy) -> pl.DataFrame:
    """Override the per-trajectory ``label`` per the requested policy and drop rows policy excludes.

    Each input row is a per-trajectory label with three relevant fields: ``valid``,
    ``determinable``, ``label``. The policies act on those fields:

      - ``drop``: rows where the relevant flag is ``False`` (or ``null``) are removed
        before aggregation.
      - ``negative`` / ``positive``: rows are *kept*, but their ``label`` is overridden
        to ``False`` / ``True``.

    The ``invalid`` policy fires first (``valid=False``), then ``indeterminable``
    (``valid=True, determinable=False``).

    Examples:
        >>> labels = pl.DataFrame({
        ...     "label": [True, False, True, None, None],
        ...     "valid": [True, True, False, True, True],
        ...     "determinable": [True, True, None, False, False],
        ... })
        >>> _apply_policy(labels, invalid="drop", indeterminable="drop")
        shape: (2, 3)
        ┌───────┬───────┬──────────────┐
        │ label ┆ valid ┆ determinable │
        │ ---   ┆ ---   ┆ ---          │
        │ bool  ┆ bool  ┆ bool         │
        ╞═══════╪═══════╪══════════════╡
        │ true  ┆ true  ┆ true         │
        │ false ┆ true  ┆ true         │
        └───────┴───────┴──────────────┘
        >>> _apply_policy(labels, invalid="negative", indeterminable="negative")
        shape: (5, 3)
        ┌───────┬───────┬──────────────┐
        │ label ┆ valid ┆ determinable │
        │ ---   ┆ ---   ┆ ---          │
        │ bool  ┆ bool  ┆ bool         │
        ╞═══════╪═══════╪══════════════╡
        │ true  ┆ true  ┆ true         │
        │ false ┆ true  ┆ true         │
        │ false ┆ false ┆ null         │
        │ false ┆ true  ┆ false        │
        │ false ┆ true  ┆ false        │
        └───────┴───────┴──────────────┘
    """
    valid = pl.col("valid").fill_null(False)
    determinable = pl.col("determinable").fill_null(False)

    if invalid == "drop":
        labels = labels.filter(valid)
    else:
        labels = labels.with_columns(
            pl.when(valid).then(pl.col("label")).otherwise(pl.lit(invalid == "positive")).alias("label")
        )

    if indeterminable == "drop":
        labels = labels.filter(valid & determinable)
    else:
        labels = labels.with_columns(
            pl.when(~valid | determinable)
            .then(pl.col("label"))
            .otherwise(pl.lit(indeterminable == "positive"))
            .alias("label")
        )

    return labels


def aggregate_predictions(
    labels: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    invalid_policy: Policy = "drop",
    indeterminable_policy: Policy = "drop",
) -> pl.DataFrame:
    """Aggregate per-trajectory labels into per-(subject, prediction_time) predictions.

    Args:
        labels: Concatenation of all per-trajectory label dataframes for a single input
            shard. Each row has ``subject_id``, ``prediction_time``, ``valid``,
            ``determinable``, ``label``. A given ``(subject_id, prediction_time)`` pair
            appears once per generated trajectory.
        ground_truth: Single dataframe (typically filtered to this shard's subjects)
            with ``subject_id``, ``prediction_time``, ``boolean_value`` — the real
            label from running ACES on the actual MEDS data.
        invalid_policy: How to treat trajectories with ``valid=False``. ``"drop"``
            removes them; ``"negative"`` / ``"positive"`` counts them with that label.
        indeterminable_policy: As above but for ``valid=True, determinable=False``.

    Returns:
        A meds-evaluation-compatible dataframe with one row per
        ``(subject_id, prediction_time)`` and the columns:

        - ``subject_id``, ``prediction_time`` (join keys)
        - ``boolean_value`` (from ``ground_truth``)
        - ``predicted_boolean_probability`` (mean of ``label`` over usable trajectories;
          ``null`` if no trajectory survived the policy for this row)
        - ``n_total``: total trajectories per row before any filtering
        - ``n_valid``: trajectories with ``valid=True``
        - ``n_determinable``: trajectories with ``valid=True`` and ``determinable=True``
        - ``n_usable``: trajectories that contributed to the probability

    Examples:
        >>> from datetime import datetime
        >>> labels = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "prediction_time": [datetime(2022, 1, 1)] * 5,
        ...     "valid": [True, True, True, True, False],
        ...     "determinable": [True, True, False, True, None],
        ...     "label": [True, False, None, True, None],
        ... })
        >>> ground_truth = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        ...     "boolean_value": [True, False],
        ... })

    The default ``drop`` policy keeps only ``valid=True, determinable=True`` trajectories.
    Subject 1 has 3 trajectories total (3 valid, 2 determinable) and the determinable
    pair is ``[True, False]`` -> probability 0.5. Subject 2 has 2 trajectories total
    (1 valid + 1 invalid-dropped) and the single usable label is ``True`` -> probability 1.0.

        >>> default = aggregate_predictions(labels, ground_truth)
        >>> default.columns
        ['subject_id', 'prediction_time', 'boolean_value', 'predicted_boolean_probability',
         'n_total', 'n_valid', 'n_usable']
        >>> default.select("subject_id", "predicted_boolean_probability",
        ...                "n_total", "n_valid", "n_usable").to_dicts()
        [{'subject_id': 1, 'predicted_boolean_probability': 0.5,
          'n_total': 3, 'n_valid': 3, 'n_usable': 2},
         {'subject_id': 2, 'predicted_boolean_probability': 1.0,
          'n_total': 2, 'n_valid': 1, 'n_usable': 1}]

    A row whose every trajectory is dropped (e.g. all-invalid) gets a null probability:

        >>> all_invalid = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "prediction_time": [datetime(2022, 1, 1)] * 2,
        ...     "valid": [False, False],
        ...     "determinable": [None, None],
        ...     "label": [None, None],
        ... }, schema_overrides={"label": pl.Boolean, "determinable": pl.Boolean})
        >>> aggregate_predictions(all_invalid, ground_truth).select(
        ...     "subject_id", "predicted_boolean_probability", "n_usable"
        ... ).to_dicts()
        [{'subject_id': 1, 'predicted_boolean_probability': None, 'n_usable': 0},
         {'subject_id': 2, 'predicted_boolean_probability': None, 'n_usable': 0}]

    With ``indeterminable_policy="negative"`` the ``determinable=False`` trajectory for
    subject 1 counts as a negative, so subject 1's probability is ``1/3``:

        >>> aggregate_predictions(labels, ground_truth, indeterminable_policy="negative").select(
        ...     "subject_id", "predicted_boolean_probability", "n_usable"
        ... ).to_dicts()
        [{'subject_id': 1, 'predicted_boolean_probability': 0.3333333333333333, 'n_usable': 3},
         {'subject_id': 2, 'predicted_boolean_probability': 1.0, 'n_usable': 1}]
    """
    ids = [LabelSchema.subject_id_name, LabelSchema.prediction_time_name]
    valid = pl.col("valid").fill_null(False)
    determinable = pl.col("determinable").fill_null(False)

    coverage = labels.group_by(ids, maintain_order=True).agg(
        pl.len().cast(pl.UInt32).alias("n_total"),
        valid.sum().cast(pl.UInt32).alias("n_valid"),
        (valid & determinable).sum().cast(pl.UInt32).alias("n_determinable"),
    )

    after_policy = _apply_policy(labels, invalid=invalid_policy, indeterminable=indeterminable_policy)
    probs = after_policy.group_by(ids, maintain_order=True).agg(
        pl.col("label").mean().alias("predicted_boolean_probability"),
        pl.len().cast(pl.UInt32).alias("n_usable"),
    )

    return (
        ground_truth.select(*ids, "boolean_value")
        .join(coverage, on=ids, how="left")
        .join(probs, on=ids, how="left")
        .with_columns(
            pl.col("n_total").fill_null(0),
            pl.col("n_valid").fill_null(0),
            pl.col("n_determinable").fill_null(0),
            pl.col("n_usable").fill_null(0),
        )
        .select(
            LabelSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            "boolean_value",
            "predicted_boolean_probability",
            "n_total",
            "n_valid",
            "n_usable",
        )
    )


def shard_dir_iterator(cfg: DictConfig) -> tuple[list[InOutFilePair], bool]:
    """Custom shard iterator: yield one entry per *input shard directory*, not per file.

    The labeling stage writes ``<input_dir>/<shard>/<trajectory>.parquet`` (one parquet
    per generated trajectory, nested under the shard path). This stage's input is the
    whole shard's worth of trajectory labels — so each "shard" for map_stage is the
    directory ``<input_dir>/<shard>/`` itself, and the corresponding output is a single
    file ``<output_dir>/<shard>.parquet``.

    Returns:
        A pair of ``(shards, includes_only_train)``. ``shards`` is a list of
        ``InOutFilePair(in_fp, out_fp)`` where ``in_fp`` is a *directory* path and
        ``out_fp`` is a parquet file path. ``includes_only_train`` is always False.
    """
    stage_cfg = cfg.stage_cfg
    input_dir = Path(stage_cfg.data_input_dir)
    output_dir = Path(stage_cfg.output_dir)

    shard_dirs = sorted({fp.parent for fp in input_dir.rglob("*.parquet")})
    pairs = [
        InOutFilePair(in_fp=d, out_fp=output_dir / f"{d.relative_to(input_dir)}.parquet") for d in shard_dirs
    ]
    return pairs, False


def _read_shard_dir(shard_dir: Path) -> pl.DataFrame:
    """Read every parquet under ``shard_dir`` and concat vertically."""
    fps = sorted(shard_dir.glob("*.parquet"))
    return pl.concat([pl.read_parquet(fp) for fp in fps], how="vertical_relaxed")


def aggregate_predictions_stage(cfg: DictConfig) -> None:
    """Stage main: read ground-truth labels once, then map over input shard directories.

    Each shard's per-trajectory labels are aggregated into a per-shard predictions parquet via
    ``aggregate_predictions``, scoped to the subjects appearing in that shard.
    """
    ground_truth = pl.read_parquet(cfg.stage_cfg.task_labels_fp)
    invalid_policy: Policy = cfg.stage_cfg.get("invalid_policy", "drop")
    indeterminable_policy: Policy = cfg.stage_cfg.get("indeterminable_policy", "drop")

    def map_fn(shard_labels: pl.DataFrame) -> pl.DataFrame:
        shard_subjects = shard_labels[LabelSchema.subject_id_name].unique()
        shard_gt = ground_truth.filter(pl.col(LabelSchema.subject_id_name).is_in(shard_subjects))
        return aggregate_predictions(
            shard_labels,
            shard_gt,
            invalid_policy=invalid_policy,
            indeterminable_policy=indeterminable_policy,
        )

    map_stage(
        cfg,
        map_fn=map_fn,
        read_fn=_read_shard_dir,
        shard_iterator_fntr=shard_dir_iterator,
    )


stage = Stage.register(
    main_fn=aggregate_predictions_stage,
    stage_name="aggregate_predictions",
    output_schema_updates=PREDICTIONS_SCHEMA_UPDATES,
    default_config={"invalid_policy": "drop", "indeterminable_policy": "drop"},
    # The output is per-shard predictions parquets (one per input shard), not MEDS-format
    # data and not single-file metadata. From MEDS-transforms' perspective this is closest
    # to a metadata stage (custom output schema, no MEDS-data invariants), so flag it as
    # such to satisfy the Stage constructor's MAP/MAPREDUCE/metadata trichotomy.
    is_metadata=True,
)
