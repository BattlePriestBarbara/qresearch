"""Rolling and expanding normalizers with a provable :math:`[t-L, t-1]` contract (task ``INF-06``).

The hazard (``PROJECT_SPEC.md`` 1.2.1): a z-score computed with a **global** standard deviation
gives the row at :math:`t` information about the volatility at :math:`T`.  The model then appears
to forecast better than it can, and the error is invisible to ordinary unit tests because the code
"works" - only the *timing* is wrong.

Contract enforced here
----------------------
For every processor in this module the statistic transforming date :math:`t` is computed from
data in

.. math::

    \\mathcal{H}_t = [\\,t - L,\\; t - 1\\,]

i.e. the current row is **excluded** by construction (an explicit ``shift(1)`` precedes every
rolling window).  Two consequences follow, and both are tested:

* warm-up rows (fewer than ``min_periods`` observations of history) are ``NaN``, never imputed;
* mutating any value at a date :math:`> t` cannot change the output at :math:`t`
  (see :func:`assert_causal`, the reusable form of the mandated injection test).

Cross-sectional processors (``CS*``) need no window at all: they only use the *same date's*
cross-section, a subset of :math:`\\mathcal{F}_t`.  ``GlobalZScoreNorm`` is provided solely as a
**negative control** - it is the leaky implementation, and the tests REQUIRE demonstrating that
the audit catches it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final

import numpy as np
import pandas as pd

__all__ = [
    "PANEL_PROCESSORS",
    "CSRankNorm",
    "CSRobustZScoreNorm",
    "CSZScoreNorm",
    "CausalityReport",
    "ExpandingTSZScoreNorm",
    "GlobalZScoreNorm",
    "PanelProcessor",
    "RollingTSRobustZScoreNorm",
    "RollingTSZScoreNorm",
    "RollingWindow",
    "assert_causal",
    "build_processor",
]

_MAD_TO_SIGMA: Final[float] = 1.482602218505602  # consistency constant of the MAD under normality


@dataclass(frozen=True)
class RollingWindow:
    """A rolling window specification.

    Attributes
    ----------
    length : int
        Window length :math:`L` in observations.  The window used at date :math:`t` is
        :math:`[t-L, t-1]`, i.e. it *ends the period before* the transformed row.
    min_periods : int
        Minimum number of non-missing observations required for a non-``NaN`` output.  Rows below
        it stay ``NaN`` so the warm-up is explicit rather than silently imputed.
    """

    length: int = 60
    min_periods: int = 20

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}")
        if not 1 <= self.min_periods <= self.length:
            raise ValueError(f"min_periods must be in [1, length], got {self.min_periods} (length={self.length})")


class PanelProcessor(ABC):
    """Base class of every normalizer; transforms a ``(date x instrument)`` frame.

    Implementations MUST be pure: no fitting state, and no reference to rows outside
    :math:`\\mathcal{H}_t` (or outside the same date's cross-section for the ``CS*`` family).
    """

    name: ClassVar[str] = "panel_processor"
    uses_history_window: ClassVar[bool] = True

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed frame, preserving index, columns and ``NaN`` positions."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Alias of :meth:`transform` so processors behave like functions."""
        return self.transform(frame)

    def describe(self) -> dict[str, object]:
        """Return a configuration description suitable for a provenance block."""
        return {"class": type(self).__name__, "name": self.name, "uses_history_window": self.uses_history_window}


def _history(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame shifted by one period: the strict look-back of the current row.

    ``shift(1)`` is the single line that enforces :math:`\\mathcal{H}_t = [t-L, t-1]`: without it,
    a ``rolling(L)`` window would include :math:`t` itself.
    """
    return frame.shift(1)


class RollingTSZScoreNorm(PanelProcessor):
    """Per-instrument rolling z-score over :math:`[t-L, t-1]`.

    Replaces the leaky global ``ZScoreNorm`` of the upstream framework, which estimates one mean
    and one standard deviation over a fit window and applies them to every date.
    """

    name = "rolling_ts_zscore"

    def __init__(self, window: RollingWindow | None = None) -> None:
        self.window = window or RollingWindow()

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        rolling = _history(frame).rolling(window=self.window.length, min_periods=self.window.min_periods)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        return (frame - mean) / std.replace(0.0, np.nan)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "window": self.window.length, "min_periods": self.window.min_periods}


class RollingTSRobustZScoreNorm(PanelProcessor):
    """Per-instrument rolling robust z-score ``(x_t - median) / (MAD * 1.4826)``.

    Median and MAD are computed over :math:`[t-L, t-1]`, so a single extreme observation at
    :math:`t` (or later) cannot move the scale applied on the day it occurs.
    """

    name = "rolling_ts_robust_zscore"

    def __init__(self, window: RollingWindow | None = None, *, clip: float | None = 5.0) -> None:
        self.window = window or RollingWindow()
        self.clip = clip

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        history = _history(frame)
        window, min_periods = self.window.length, self.window.min_periods
        median = history.rolling(window=window, min_periods=min_periods).median()
        mad = (history - median).abs().rolling(window=window, min_periods=min_periods).median()
        standardized = (frame - median) / (mad * _MAD_TO_SIGMA).replace(0.0, np.nan)
        if self.clip is not None:
            standardized = standardized.clip(lower=-self.clip, upper=self.clip)
        return standardized

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "window": self.window.length,
            "min_periods": self.window.min_periods,
            "clip": self.clip,
        }


class ExpandingTSZScoreNorm(PanelProcessor):
    """Per-instrument expanding z-score over :math:`[\\text{start},\\; t-1]`.

    Uses all available history at every date: useful for short samples where a rolling window would
    leave too large a warm-up, and still strictly causal.
    """

    name = "expanding_ts_zscore"

    def __init__(self, *, min_periods: int = 20) -> None:
        if min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {min_periods}")
        self.min_periods = min_periods

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        expanding = _history(frame).expanding(min_periods=self.min_periods)
        return (frame - expanding.mean()) / expanding.std(ddof=1).replace(0.0, np.nan)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "min_periods": self.min_periods}


class CSRankNorm(PanelProcessor):
    """Cross-sectional rank rescaled into ``(0, 1)`` within each date (row-wise).

    Rank is same-period and parameter-free: no statistic is estimated across dates and the
    cross-section at :math:`t` is a subset of :math:`\\mathcal{F}_t`, so it cannot leak.
    """

    name = "cs_rank"
    uses_history_window = False

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        counts = frame.notna().sum(axis=1)
        ranks = frame.rank(axis=1, method="average")
        return ranks.div(counts + 1.0, axis=0)


class CSZScoreNorm(PanelProcessor):
    """Cross-sectional z-score computed within each date (same-period, therefore causal)."""

    name = "cs_zscore"
    uses_history_window = False

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1, ddof=1).replace(0.0, np.nan)
        return frame.sub(mean, axis=0).div(std, axis=0)


class CSRobustZScoreNorm(PanelProcessor):
    """Cross-sectional robust z-score ``(x - median) / (MAD * 1.4826)`` within each date."""

    name = "cs_robust_zscore"
    uses_history_window = False

    def __init__(self, *, clip: float | None = 5.0) -> None:
        self.clip = clip

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        median = frame.median(axis=1)
        mad = frame.sub(median, axis=0).abs().median(axis=1)
        standardized = frame.sub(median, axis=0).div((mad * _MAD_TO_SIGMA).replace(0.0, np.nan), axis=0)
        if self.clip is not None:
            standardized = standardized.clip(lower=-self.clip, upper=self.clip)
        return standardized


class GlobalZScoreNorm(PanelProcessor):
    """Whole-sample z-score - **the leaky implementation, for negative controls only**.

    Every row is standardized with statistics estimated over the entire sample, so the row at
    :math:`t` embeds information from :math:`T`.  Construction is refused unless
    ``allow_leaky=True``; its only supported use is to prove that :func:`assert_causal` - and hence
    the leakage audit - has the power to detect a real defect.
    """

    name = "global_zscore_LEAKY"
    uses_history_window = False

    def __init__(self, *, allow_leaky: bool = False) -> None:
        if not allow_leaky:
            raise ValueError(
                "GlobalZScoreNorm estimates statistics over the full sample and therefore leaks "
                "future information (PROJECT_SPEC.md 1.2.1). Pass allow_leaky=True to build it as a "
                "negative control for the leakage audit."
            )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (frame - frame.mean()) / frame.std(ddof=1)


@dataclass(frozen=True)
class CausalityReport:
    """Outcome of a causality (future-independence) probe on one processor."""

    processor: str
    probe_position: int
    probe_date: object
    max_abs_change: float
    changed_cells: int
    is_causal: bool

    def format(self) -> str:
        """Return a one-line human-readable summary."""
        status = "CAUSAL" if self.is_causal else "LEAKS FUTURE INFORMATION"
        return (
            f"{self.processor}: {status} at probe date {self.probe_date} "
            f"(max|delta|={self.max_abs_change:.6g}, changed cells={self.changed_cells})"
        )


def _nan_aware_max_abs_diff(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, int]:
    """Return ``(max |left - right|, differing cell count)``, treating ``NaN == NaN`` as equal."""
    both_nan = left.isna() & right.isna()
    difference = (left - right).abs().where(~both_nan, 0.0)
    changed = int((difference > 0).to_numpy().sum())
    values = difference.to_numpy()
    maximum = float(np.nanmax(values)) if values.size else 0.0
    return (0.0 if np.isnan(maximum) else maximum), changed


def assert_causal(
    processor: PanelProcessor,
    frame: pd.DataFrame,
    *,
    probe_position: int | None = None,
    extreme: float = 1e6,
    rtol: float = 1e-12,
) -> CausalityReport:
    """Prove that ``processor`` ignores everything after the probe date.

    The probe overwrites **every row strictly after** ``probe_position`` with an extreme value and
    flips half of those rows to ``NaN`` (which also exercises the missing-data path), then recomputes
    the transform.  A causal processor returns the probe row unchanged; a processor standardizing
    with global statistics cannot, because the injected outlier moves the mean and the deviation.

    Parameters
    ----------
    processor : PanelProcessor
        Processor under test.
    frame : pd.DataFrame
        Wide ``(date x instrument)`` panel.
    probe_position : int | None
        Row to probe; defaults to the middle row.
    extreme : float
        Value written into the future rows.
    rtol : float
        Absolute tolerance of the comparison (default ``1e-12``).

    Returns
    -------
    CausalityReport

    Raises
    ------
    AssertionError
        When the probe row changed, i.e. the processor consumed future information.  The message
        names the processor, the date and the magnitude so the defect is actionable.
    """
    if probe_position is None:
        probe_position = max(1, len(frame) // 2)
    if not 0 <= probe_position < len(frame) - 1:
        raise ValueError(f"probe_position must leave at least one future row, got {probe_position}")

    baseline = processor.transform(frame)
    mutated = frame.copy()
    mutated.iloc[probe_position + 1 :] = extreme
    mutated.iloc[probe_position + 1 :: 2] = np.nan

    perturbed = processor.transform(mutated)
    max_change, changed_cells = _nan_aware_max_abs_diff(
        baseline.iloc[[probe_position]], perturbed.iloc[[probe_position]]
    )
    report = CausalityReport(
        processor=type(processor).__name__,
        probe_position=probe_position,
        probe_date=frame.index[probe_position],
        max_abs_change=max_change,
        changed_cells=changed_cells,
        is_causal=max_change <= rtol,
    )
    if not report.is_causal:
        raise AssertionError(
            f"{processor.name} used future information: mutating rows after "
            f"{frame.index[probe_position]:%Y-%m-%d} changed the output at that date by "
            f"{max_change:.6g} ({changed_cells} cells; future rows set to {extreme} and NaN). "
            "Statistics must come from [t-L, t-1] only (PROJECT_SPEC.md 1.2.1, task INF-06)."
        )
    return report


PANEL_PROCESSORS: dict[str, type[PanelProcessor]] = {
    RollingTSZScoreNorm.name: RollingTSZScoreNorm,
    RollingTSRobustZScoreNorm.name: RollingTSRobustZScoreNorm,
    ExpandingTSZScoreNorm.name: ExpandingTSZScoreNorm,
    CSRankNorm.name: CSRankNorm,
    CSZScoreNorm.name: CSZScoreNorm,
    CSRobustZScoreNorm.name: CSRobustZScoreNorm,
    GlobalZScoreNorm.name: GlobalZScoreNorm,
}


def build_processor(spec: dict[str, object]) -> PanelProcessor:
    """Build a processor from a declarative configuration block.

    Parameters
    ----------
    spec : dict[str, object]
        ``{"class": <registry key or class name>, "kwargs": {...}}``, matching the configuration
        style of ``PROJECT_SPEC.md`` 3.5.

    Returns
    -------
    PanelProcessor

    Raises
    ------
    KeyError
        When the class is not registered.  Unknown processors are never silently ignored.
    """
    class_name = str(spec.get("class", ""))
    kwargs = spec.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError(f"kwargs must be a mapping, got {type(kwargs).__name__}")
    registry: dict[str, type[PanelProcessor]] = {
        **PANEL_PROCESSORS,
        **{processor.__name__: processor for processor in PANEL_PROCESSORS.values()},
    }
    if class_name not in registry:
        raise KeyError(f"unknown processor {class_name!r}; registered: {sorted(PANEL_PROCESSORS)}")
    return registry[class_name](**kwargs)
