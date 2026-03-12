"""ACES configuration evaluation: zero-shot labeling of generated trajectories against ACES task configs.

This subpackage converts ACES task configurations into zero-shot labeling configs (with optional
relaxations) and applies them to generated trajectory DataFrames to produce validity, determinability,
and label outcomes.

Key entry points:
  - :func:`task_config.convert_to_zero_shot` — convert an ACES config to a zero-shot config.
  - :func:`label.label_trajectories` — label a DataFrame of trajectories against a zero-shot config.
  - ``ZSACES_label`` CLI — Hydra-based command-line entry point for batch labeling.
"""
