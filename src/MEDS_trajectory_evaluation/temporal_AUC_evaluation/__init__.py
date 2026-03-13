"""Temporal AUC evaluation: compute AUROCs across prediction horizons for ACES predicates.

This subpackage extracts time-to-first-event observations from both real MEDS data and generated
trajectories, then computes AUC summaries across configurable duration grids.

Key entry points:
  - :func:`temporal_AUCS.temporal_aucs` — compute AUCs from pre-extracted TTE DataFrames.
  - :func:`trajectory_AUC.temporal_auc_from_trajectory_files` — end-to-end AUC from trajectory files.
  - :func:`get_ttes.get_raw_tte` — extract true time-to-event from MEDS data.
  - :func:`get_ttes.get_trajectory_tte` — extract time-to-event from generated trajectories.
  - :func:`get_ttes.merge_pred_ttes` — merge multiple trajectory TTE DataFrames into list columns.
"""
