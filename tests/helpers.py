"""Test helper utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from collections.abc import Iterable


def _manual_auc(true_vals: Iterable[float], false_vals: Iterable[float]) -> float | None:
    """Compute AUC via sklearn.

    This uses sklearn's `roc_auc_score` logic to compute an AUC directly.

    Args:
        true_vals: probabilities for samples where the label is `True`.
        false_vals probabilities for samples where the label is `False`.

    Returns:
        AUC score, or `None` if either `true_vals` or `false_vals` is empty.

    Examples:
        >>> _manual_auc([0.9, 0.8], [0.7, 0.6])
        1.0
        >>> _manual_auc([0.9, 0.7], [0.8, 0.6])
        0.75
        >>> _manual_auc([0.9, 0.8], [0.9, 0.8])
        0.5
        >>> _manual_auc([0.8], [0.8, 0.7])
        0.75
        >>> print(_manual_auc([], [0.7, 0.6]))
        None
        >>> print(_manual_auc([0.7, 0.6], []))
        None
        >>> print(_manual_auc([], []))
        None
    """
    y_score = [*true_vals, *false_vals]
    y_true = [1] * len(true_vals) + [0] * len(false_vals)

    if not y_score:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        score = roc_auc_score(y_true, y_score)
        return None if np.isnan(score) else score
