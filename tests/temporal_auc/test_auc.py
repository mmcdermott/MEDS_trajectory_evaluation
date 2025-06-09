import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import df_AUC

from .helpers import _manual_auc


@st.composite
def probability_lists(draw: st.DrawFn, min_size: int = 0, max_size: int = 40) -> list[float]:
    """Generate a list of probabilities."""
    return sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=min_size,
                max_size=max_size,
            )
        )
    )


@st.composite
def probability_dicts(
    draw: st.DrawFn, tasks: list[str], min_size: int = 0, max_size: int = 40
) -> dict[str, list[float]]:
    true_probs = {f"true/{t}": probability_lists(min_size=min_size, max_size=max_size) for t in tasks}
    false_probs = {f"false/{t}": probability_lists(min_size=min_size, max_size=max_size) for t in tasks}

    return draw(st.fixed_dictionaries(true_probs | false_probs))


@st.composite
def _df_inputs(
    draw: st.DrawFn,
    min_n_probs: int = 0,
    max_n_probs: int = 40,
    min_n_rows: int = 1,
    max_n_rows: int = 10,
    min_n_task_cols: int = 1,
    max_n_task_cols: int = 10,
    min_n_id_cols: int = 0,
    max_n_id_cols: int = 5,
) -> tuple[pl.DataFrame, list[str]]:
    n_task_cols = draw(st.integers(min_value=min_n_task_cols, max_value=max_n_task_cols))
    tasks = [f"Task_{i}" for i in range(n_task_cols)]

    df_rows = draw(
        st.lists(
            probability_dicts(tasks, min_size=min_n_probs, max_size=max_n_probs),
            min_size=min_n_rows,
            max_size=max_n_rows,
        )
    )

    n_rows = len(df_rows)

    n_id_cols = draw(st.integers(min_value=min_n_id_cols, max_value=max_n_id_cols))
    id_cols = [f"id_col_{i}" for i in range(n_id_cols)]

    id_col_values = [
        {col: draw(st.integers(min_value=1, max_value=100)) for col in id_cols} for _ in range(n_rows)
    ]

    df_rows = [{**id_cols, **row} for row, id_cols in zip(df_rows, id_col_values, strict=True)]

    df_schema = {
        **{f"true/{task}": pl.List(pl.Float64) for task in tasks},
        **{f"false/{task}": pl.List(pl.Float64) for task in tasks},
        **dict.fromkeys(id_cols, pl.Int64),
    }

    return pl.DataFrame(df_rows, schema=df_schema), tasks, id_cols


@given(_df_inputs())
def test_df_auc_matches_manual(data):
    df, tasks, id_cols = data
    result = df_AUC(df)

    want_columns = set(id_cols) | {f"AUC/{task}" for task in tasks}

    assert set(result.columns) == want_columns, f"Expected columns {want_columns}, got {set(result.columns)}"

    for row_idx, row in enumerate(df.iter_rows(named=True)):
        for task in tasks:
            auc_val = result[f"AUC/{task}"][row_idx]
            expected = _manual_auc(row[f"true/{task}"], row[f"false/{task}"])
            if expected is None:
                assert auc_val is None
            else:
                assert math.isclose(auc_val, expected, rel_tol=1e-9)
                assert 0.0 <= auc_val <= 1.0
