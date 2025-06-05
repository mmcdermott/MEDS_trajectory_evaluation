"""Helpers to compute temporal AUCs from trajectory files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

import polars as pl
from aces.config import PlainPredicateConfig
from meds import LabelSchema
from omegaconf import OmegaConf

from .get_ttes import (
    PREDICATES_T,
    get_raw_tte,
    get_trajectory_tte,
    merge_pred_ttes,
)
from .temporal_AUCS import temporal_aucs


def _normalize_predicates(predicates: PREDICATES_T | str | Path) -> PREDICATES_T:
    """Load and normalize predicate configurations.

    Args:
        predicates: Either a mapping from predicate name to
            :class:`PlainPredicateConfig` or a path to a YAML file with a
            ``predicates`` mapping.

    Returns:
        The dictionary of plain predicate configurations.
    """

    if isinstance(predicates, str | Path):
        cfg = OmegaConf.load(str(predicates))
        predicates = cfg.get("predicates", cfg)

    out: PREDICATES_T = {}
    for name, cfg in predicates.items():
        if isinstance(cfg, PlainPredicateConfig):
            out[name] = cfg
        else:
            out[name] = PlainPredicateConfig(**OmegaConf.to_container(cfg, resolve=True))

    return out


def temporal_auc_from_trajectory_files(
    MEDS_df: pl.DataFrame,
    trajectories: Sequence[str | Path] | Path | str,
    predicates: PREDICATES_T | str | Path,
    *,
    duration_grid: str | int | None | list[timedelta] = 10000,
    AUC_dist_approx: int = -1,
    seed: int = 0,
) -> pl.DataFrame:
    """Compute temporal AUCs over a collection of trajectory files.

    Args:
        MEDS_df: The MEDS dataframe with the reference data.
        trajectories: Iterable of parquet files or a directory containing them.
        predicates: Mapping of predicate names to configs or a YAML file path.
        duration_grid: Duration grid for :func:`temporal_aucs`.
        AUC_dist_approx: Distribution approximation size.
        seed: Random seed for subsampling.

    Returns:
        A dataframe containing the temporal AUCs for each predicate.
    """

    preds = _normalize_predicates(predicates)

    if isinstance(trajectories, str | Path):
        t_root = Path(trajectories)
        t_files = sorted(t_root.rglob("*.parquet")) if t_root.is_dir() else [t_root]
    else:
        t_files = [Path(p) for p in trajectories]

    pred_dfs = []
    index_dfs = []
    for fp in t_files:
        df = pl.read_parquet(fp, use_pyarrow=True)
        pred_dfs.append(get_trajectory_tte(df, preds))
        index_dfs.append(df.select(LabelSchema.subject_id_name, LabelSchema.prediction_time_name).unique())

    merged_pred = merge_pred_ttes(pred_dfs)
    index_df = pl.concat(index_dfs, how="vertical").unique(maintain_order=True)
    true_tte = get_raw_tte(MEDS_df, index_df, preds)

    return temporal_aucs(
        true_tte,
        merged_pred,
        duration_grid=duration_grid,
        AUC_dist_approx=AUC_dist_approx,
        seed=seed,
    )
