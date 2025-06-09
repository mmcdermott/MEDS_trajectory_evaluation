import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from MEDS_trajectory_evaluation.temporal_AUC_evaluation.temporal_AUCS import df_AUC
from tests.helpers import _manual_auc


@st.composite
def _df_inputs(draw):
    tasks = draw(st.lists(st.sampled_from(["A", "B", "C"]), min_size=1, max_size=2, unique=True))
    n_rows = draw(st.integers(min_value=1, max_value=2))
    rows = []
    for _ in range(n_rows):
        row = {}
        for task in tasks:
            true_vals = sorted(
                draw(
                    st.lists(
                        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                        min_size=0,
                        max_size=3,
                    )
                )
            )
            false_vals = sorted(
                draw(
                    st.lists(
                        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                        min_size=0,
                        max_size=3,
                    )
                )
            )
            row[f"true/{task}"] = true_vals
            row[f"false/{task}"] = false_vals
        rows.append(row)
    return pl.DataFrame(rows), tasks


@given(_df_inputs())
def test_df_auc_matches_manual(data):
    df, tasks = data
    result = df_AUC(df)
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        for task in tasks:
            auc_val = result[f"AUC/{task}"][row_idx]
            expected = _manual_auc(row[f"true/{task}"], row[f"false/{task}"])
            if expected is None:
                assert auc_val is None
            else:
                assert math.isclose(auc_val, expected, rel_tol=1e-9)
                assert 0.0 <= auc_val <= 1.0
