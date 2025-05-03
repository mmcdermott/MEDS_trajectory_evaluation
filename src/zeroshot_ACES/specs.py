from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
from meds import code_field, prediction_time_field, subject_id_field, time_field


@dataclass
class TrajectorySpec:
    model_root: Path
    index_name: str
    data_root: Path

    @property
    def index_root(self) -> Path:
        return self.data_root / "index_dataframes"

    @property
    def trajectories_root(self) -> Path:
        return self.model_root / "trajectories" / self.index_name

    @cached_property
    def index_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.index_root / self.index_name, use_pyarrow=True)

    @cached_property
    def subjects_by_split(self) -> dict[str, set[int]]:
        out = {}
        for sp in self.subject_splits.select(pl.col("split")).unique()["split"]:
            out[sp] = set(self.subject_splits.filter(pl.col("split") == sp)[subject_id_field])
        return out

    @cached_property
    def index_df_by_split(self) -> dict[str, pl.DataFrame]:
        out = {}
        for sp, subj in self.subjects_by_split.items():
            out[sp] = self.index_df.filter(pl.col(subject_id_field).is_in(subj))
        return out

    def real_futures(self, split: str) -> pl.DataFrame:
        data_dir = self.data_root / "data"
        data_fp_by_shard = {
            fp.relative_to(data_dir).with_suffix("").as_posix(): fp for fp in data_dir.rglob("*.parquet")
        }

        if any(shard.startswith(f"{split}/") for shard in data_fp_by_shard):
            data_fp_by_shard = {
                shard: fp for shard, fp in data_fp_by_shard.items() if shard.startswith(f"{split}/")
            }
            do_filter = False
        else:
            do_filter = True
            filter_expr = pl.col(subject_id_field).is_in(self.subjects_by_split[split])

        index_df = self.index_df_by_split[split].select(subject_id_field, prediction_time_field).lazy()

        dfs = {}
        for shard, fp in data_fp_by_shard.items():
            df = pl.scan_parquet(fp)
            if do_filter:
                df = df.filter(filter_expr)

            dfs[shard] = (
                df.join(
                    index_df,
                    on=[subject_id_field],
                    how="inner",
                    maintain_order="left",
                    coaliasce=True,
                )
                .filter(pl.col(time_field) >= pl.col(prediction_time_field))
                .collect()
            )

        return pl.concat(dfs.values(), how="vertical")

    @cached_property
    def _raw_trajectories(self) -> dict[str, pl.DataFrame]:
        trajectory_fps = list(self.trajectories_root.rglob("*.parquet"))

        if not trajectory_fps:
            raise FileNotFoundError(f"No trajectory files found in {self.trajectories_root!s}")

        trajectories = {}
        for fp in trajectory_fps:
            n = fp.relative_to(self.trajectories_root).with_suffix("").as_posix()
            trajectories[n] = pl.read_parquet(fp, use_pyarrow=True)
        return trajectories

    @cached_property
    def trajectories(self) -> dict[str, pl.DataFrame]:
        raise NotImplementedError("May not be needed once prediction time is just added in.")

    def __getitem__(self, k: str) -> Path | str | pl.DataFrame | dict[str, pl.DataFrame]:
        if hasattr(self, k):
            return getattr(self, k)

        if k == "index_df":
            return self.index_df

        if k in self._raw_trajectories:
            return self._raw_trajectories[k]

        raise ValueError(f"Key {k} not found in {self.__class__.__name__}.")


SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * 1_000_000


@dataclass
class PlotSpec:
    split: str
    time_span: timedelta = timedelta(days=365 * 5)
    n_bins: int = 100

    @property
    def span_days(self) -> float:
        return (self.time_span / timedelta(seconds=1)) / SECONDS_PER_DAY

    @cached_property
    def time_bins_days(self) -> np.ndarray:
        return np.linspace(start=0, stop=self.span_days, num=self.n_bins)

    @cached_property
    def time_bins_labels(self) -> list[str]:
        labels = ["ERROR"]
        for i in range(self.n_bins - 1):
            left = self.time_bins_days[i]
            right = self.time_bins_days[i + 1]
            labels.append(f"[{left:.1f}d, {right:.1f}d)")
        labels.append(f"[{self.time_bins_days[-1]:.1f}d, +inf)")
        return labels

    @property
    def binned_time_delta_expr(self) -> pl.Expr:
        time_delta = pl.col(time_field) - pl.col(prediction_time_field)
        time_delta_days = time_delta.dt.total_microseconds() / MICROSECONDS_PER_DAY
        return time_delta_days.cut(self.time_bins_days, labels=self.time_bins_labels, left_closed=True)

    def matching_measurements(self, df: pl.DataFrame) -> pl.DataFrame:
        id_fields = [subject_id_field]
        if prediction_time_field in df.collect_schema():
            id_fields.append(prediction_time_field)

        return (
            df.filter(~pl.col(code_field).str.starts_with("TIMELINE//"))
            .group_by(*id_fields, time_field, maintain_order=True)
            .agg(pl.len().alias("n_measurements"))
            .select(
                *id_fields,
                time_field,
                pl.col("n_measurements").cum_sum().over(id_fields).alias("cumulative_measurements"),
            )
        )

    def align_trajectories(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            self.matching_measurements(df)
            .select(
                subject_id_field,
                prediction_time_field,
                self.binned_time_delta_expr.alias("time_delta_bin"),
                "cumulative_measurements",
            )
            .group_by(subject_id_field, prediction_time_field, "time_delta_bin", maintain_order=True)
            .agg(pl.col("cumulative_measurements").max().alias("cumulative_measurements"))
        )

    def pivot_trajectories(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pivot(
            index=[subject_id_field, prediction_time_field],
            columns="time_delta_bin",
            values="cumulative_measurements",
            aggregate_function=None,
            maintain_order=True,
        ).select(subject_id_field, prediction_time_field, *self.time_bins_labels[1:])

    def agg_trajectories(self, trajectory_spec: TrajectorySpec) -> pl.DataFrame:
        rel_keys = sorted(k for k in trajectory_spec.trajectories if k.startswith(f"{self.split}/"))

        if not rel_keys:
            raise ValueError

        aligned_dfs = [self.align_trajectories(trajectory_spec.trajectories[k]) for k in rel_keys]

        id_cols = [subject_id_field, prediction_time_field, "time_delta_bin"]
        meas_dtype = aligned_dfs[0].collect_schema()["cumulative_measurements"]

        cum_meas_list = (
            pl.col("cumulative_measurements")
            .cast(pl.List(meas_dtype))
            .fill_null([])
            .alias("cumulative_measurements")
        )

        def check_nulls(df: pl.DataFrame):
            n_nulls = df.select(pl.col("cumulative_measurements").is_null().sum()).item()
            if n_nulls > 0:
                raise ValueError(f"{n_nulls} Null values found among {df.shape[0]} rows.")

        df = aligned_dfs[0].with_columns(cum_meas_list)
        check_nulls(df)

        for samp_df in aligned_dfs[1:]:
            samp_df = samp_df.with_columns(cum_meas_list)
            check_nulls(samp_df)

            df = df.join(samp_df, on=id_cols, how="full", coalesce=True).select(
                *id_cols,
                pl.concat_list(
                    pl.col("cumulative_measurements").fill_null([]),
                    pl.col("cumulative_measurements_right").fill_null([]),
                ).alias("cumulative_measurements"),
            )
            check_nulls(df)

        return self.pivot_trajectories(df)
