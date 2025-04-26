import polars as pl


def label_for_trajectories(
    trajectories: pl.DataFrame,
    zero_shot_task_cfg,
) -> pl.DataFrame:
    """Takes a dataframe of trajectories and a zero-shot task configuration and returns the labels.

    Args:
        trajectories: A dataframe containing the trajectories to be labeled.
        zero_shot_task_cfg: The zero-shot task configuration to use for labeling.

    Returns:
        A dataframe with the labels each trajectory evaluates to for the given config.
    """

    raise NotImplementedError("This is a placeholder for the actual prediction code.")
