import logging
import random
from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from MEDS_transforms.mapreduce.mapper import map_over
from omegaconf import DictConfig

from .label import label_trajectories
from .task_config import resolve_zero_shot_task_cfg
from .utils import get_in_out_fps, hash_based_seed

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_trajectory_evaluation") / "ACES_config_evaluation" / "configs"


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_label")
def label(cfg: DictConfig):
    # 1. Validate and prepare the config for the zero-shot context
    zero_shot_task_cfg = resolve_zero_shot_task_cfg(cfg.task, cfg.labeler)

    # 2. Iterate through the trajectory files and process each one
    in_out_fps = get_in_out_fps(Path(cfg.trajectories_dir), Path(cfg.output_dir))
    seed = hash_based_seed(cfg.seed, cfg.worker)
    random.seed(seed)
    random.shuffle(in_out_fps)

    map_over(
        in_out_fps,
        partial(label_trajectories, zero_shot_task_cfg=zero_shot_task_cfg),
        read_fn=partial(pl.read_parquet, use_pyarrow=True, glob=False),
        write_fn=partial(pl.DataFrame.write_parquet, use_pyarrow=True),
    )
