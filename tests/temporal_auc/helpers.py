"""Test helper utilities."""

from __future__ import annotations

import dataclasses
import warnings
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st
from meds import DataSchema
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


@st.composite
def probabilities(draw: st.DrawFn) -> float:
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@dataclasses.dataclass(frozen=True)
class MEDSVocabulary:
    codes: list[str]
    code_occurrences_probs: list[float]
    values_probs: list[float]

    @property
    def n_codes(self) -> int:
        return len(self.codes)

    @st.composite
    def rvs(draw, self, size: int = 1) -> list[tuple[str, float]]:
        """Generate a random measurement code and its occurrence probability."""
        rng = draw(st.randoms())
        code_indices = rng.choices(self.n_codes, weights=self.code_occurrences_probs, k=size)

        codes = [self.codes[code_idx] for code_idx in code_indices]

        value_probs = [self.values_probs[code_idx] for code_idx in code_indices]
        include_vals = [rng.random() < vp for vp in value_probs]
        values = [rng.normalvariate() if include_val else None for include_val in include_vals]

        return list(zip(codes, values, strict=False))

    @classmethod
    @st.composite
    def strategy(
        draw,
        cls,
        size: st.SearchStrategy[int] | None = None,
        frac_codes_with_values: st.SearchStrategy[float] | None = None,
        frac_values_always_present: st.SearchStrategy[float] = None,
        values_present_probabilities: st.SearchStrategy[float] = None,
        code_occurrences_probs: st.SearchStrategy[float] = None,
    ) -> MEDSVocabulary:
        """Generate a random vocabulary of task codes."""

        if size is None:
            size = st.integers(min_value=1, max_value=500)
        if frac_codes_with_values is None:
            frac_codes_with_values = probabilities()
        if frac_values_always_present is None:
            frac_values_always_present = probabilities()
        if values_present_probabilities is None:
            values_present_probabilities = probabilities()
        if code_occurrences_probs is None:
            code_occurrences_probs = probabilities()

        n_codes = draw(size)
        codes = [f"Code_{i}" for i in range(n_codes)]

        code_occurrences_probs = draw(st.lists(code_occurrences_probs, min_size=n_codes, max_size=n_codes))

        n_with_values = int(n_codes * draw(frac_codes_with_values))
        n_without_values = n_codes - n_with_values
        without_values_probs = [0.0] * n_without_values

        n_always_present = int(n_with_values * draw(frac_values_always_present))
        always_present_probs = [1.0] * n_always_present

        n_sometimes_present = n_with_values - n_always_present
        sometimes_present_probs = draw(
            st.lists(
                values_present_probabilities,
                min_size=n_sometimes_present,
                max_size=n_sometimes_present,
            )
        )

        values_probs = [*without_values_probs, *always_present_probs, *sometimes_present_probs]
        return cls(codes=codes, code_occurrences_probs=code_occurrences_probs, values_probs=values_probs)


@dataclasses.dataclass(frozen=True)
class MEDSEvent:
    n_measurements: int

    @st.composite
    def measurements(draw, self, vocabulary: MEDSVocabulary, **kwargs) -> list[DataSchema]:
        """Generate a list of measurements for this event."""
        measurement_values = draw(vocabulary.rvs(size=self.n_measurements))
        return [DataSchema(code=code, numeric_value=value, **kwargs) for code, value in measurement_values]

    @classmethod
    @st.composite
    def strategy(draw, cls, n_measurements: st.SearchStrategy[int] | None = None, **kwargs) -> MEDSEvent:
        if n_measurements is None:
            n_measurements = st.integers(min_value=1, max_value=10)

        return MEDSEvent(**kwargs, n_measurements=draw(n_measurements))


@dataclasses.dataclass(frozen=True)
class MEDSSubject:
    n_static_measurements: int
    events: list[MEDSEvent]
    init_time: datetime
    time_between_events: st.SearchStrategy[timedelta]

    @property
    def static_event(self) -> MEDSEvent:
        """Static event with no measurements."""
        return MEDSEvent(n_measurements=self.n_static_measurements)

    @property
    def has_static_measurements(self) -> bool:
        """Whether this subject has static measurements."""
        return self.n_static_measurements > 0

    @property
    def time_delta_st(self) -> st.SearchStrategy[list[timedelta]]:
        n = len(self.events) - 1
        return st.lists(self.time_between_events, min_size=n, max_size=n)

    @st.composite
    def sample(draw, self, **kwargs) -> list[DataSchema]:
        if self.has_static_measurements:
            all_measurements = draw(self.static_event.measurements(time=None, **kwargs))
        else:
            all_measurements = []

        time_deltas = draw(self.time_delta_st)
        time = self.init_time

        for event, td in zip(self.events, [*time_deltas, timedelta(seconds=0)], strict=False):
            all_measurements.extend(draw(event.measurements(time=time, **kwargs)))
            time += td

        return all_measurements

    @classmethod
    @st.composite
    def strategy(
        draw,
        cls,
        n_static_measurements: st.SearchStrategy[int] | None = None,
        events: st.SearchStrategy[list[MEDSEvent]] | None = None,
        init_time: st.SearchStrategy[datetime] | None = None,
        time_between_events: st.SearchStrategy[timedelta] | None = None,
        **kwargs,
    ) -> MEDSSubject:
        if n_static_measurements is None:
            n_static_measurements = st.integers(min_value=0, max_value=10)
        if events is None:
            events = st.lists(MEDSEvent.strategy(), min_size=1, max_size=50)
        if init_time is None:
            init_time = st.datetimes(
                min_value=datetime(1980, 1, 1, tzinfo=UTC),
                max_value=datetime(2023, 1, 1, tzinfo=UTC),
            )
        if time_between_events is None:
            time_between_events = st.timedeltas(
                min_value=timedelta(seconds=1),
                max_value=timedelta(days=365 * 10),  # up to 10 years
            )

        return cls(
            n_static_measurements=draw(n_static_measurements),
            events=draw(events),
            init_time=draw(init_time),
            time_between_events=draw(time_between_events),
            **kwargs,
        )


@dataclasses.dataclass(frozen=True)
class MEDSData:
    subjects: list[MEDSSubject]
    vocabulary: MEDSVocabulary

    @st.composite
    def sample(draw, self) -> list[DataSchema]:
        all_measurements = []
        for i, subject in enumerate(self.subjects):
            all_measurements.extend(draw(subject.sample(subject_id=i, vocabulary=self.vocabulary)))
        return all_measurements

    @classmethod
    @st.composite
    def strategy(
        draw,
        cls,
        subjects: st.SearchStrategy[list[MEDSSubject]] | None = None,
        vocabulary: st.SearchStrategy[MEDSVocabulary] | None = None,
    ) -> MEDSData:
        if subjects is None:
            subjects = st.lists(MEDSSubject.strategy(), min_size=1, max_size=20)
        if vocabulary is None:
            vocabulary = MEDSVocabulary.strategy()

        return cls(subjects=draw(subjects), vocabulary=draw(vocabulary))
