"""Test helper utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def _manual_auc(true_vals: Iterable[float], false_vals: Iterable[float]) -> float | None:
    """Compute AUC via exhaustive pair counting."""
    true_list = list(true_vals)
    false_list = list(false_vals)
    if not true_list or not false_list:
        return None
    pairs = len(true_list) * len(false_list)
    score = 0.0
    for t in true_list:
        for f in false_list:
            if t > f:
                score += 1.0
            elif t == f:
                score += 0.5
    return score / pairs
