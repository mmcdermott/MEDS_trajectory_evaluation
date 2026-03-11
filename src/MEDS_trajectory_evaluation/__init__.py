"""MEDS Trajectory Evaluation: utilities for evaluating autoregressive generated trajectories over MEDS data.

This package provides:
  - A schema for representing generated trajectories (:class:`schema.GeneratedTrajectorySchema`).
  - Temporal AUC evaluation for simple predicate prediction (``temporal_AUC_evaluation``).
  - Full ACES task labeling with configurable relaxations (``ACES_config_evaluation``).
"""
