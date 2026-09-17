"""Information-coefficient primitives (task ``ST-01``, pulled forward as a dependency of ``INF-08``).

This module implements the cross-sectional statistics defined in ``PROJECT_SPEC.md`` 2.2.1:

.. math::

    \\rho_t = \\mathrm{Corr}_S\\!\\left(\\{\\hat{y}_{i,t}\\}_{i \\in \\mathcal{U}_t},
    \\{y_{i,t+1}\\}_{i \\in \\mathcal{U}_t}\\right)

Only per-period statistics live here.  Significance testing (Newey-West HAC, bootstrap,
multiple-testing control) belongs to ``ST-02`` .. ``ST-07`` and is deliberately NOT faked here:
:func:`ic_moments` reports moments and an i.i.d. reference statistic, and says so.  The
permutation-based inference used by ``INF-08`` derives its p-value from an empirical null
distribution and therefore needs no HAC estimator.

Index convention (``PROJECT_SPEC.md`` 3.4.6): inputs are ``pd.Series`` indexed by a ``MultiIndex``
named ``["datetime", "instrument"]``; per-period outputs are indexed by ``datetime`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, overload

import numpy as np
import pandas as pd

__all__ = [
    "IC_METHODS",
    "ICMoments",
    "align_predictions",
    "ic_moments",
    "pearson_ic",
    "rank_ic",
]

IC_METHODS: Final[tuple[str, ...]] = ("spearman", "pearson")
ICMethod = Literal["spearman", "pearson"]

_DATETIME_LEVEL: Final[str] = "datetime"
_INSTRUMENT_LEVEL: Final[str] = "instrument"


@dataclass(frozen=True)
class ICMoments:
    """First and second moments of a per-period IC series."""

    n_periods: int
    mean: float
    std: float
    icir: float
    icir_annualized: float
    t_iid: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return {
            "n_periods": self.n_periods,
            "mean": self.mean,
            "std": self.std,
            "icir": self.icir,
            "icir_annualized": self.icir_annualized,
            "t_iid": self.t_iid,
        }


def _require_multiindex(series: pd.Series, name: str) -> None:
    """Raise ``ValueError`` unless ``series`` carries the mandatory two-level index."""
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError(f"{name} must be indexed by a MultiIndex of (datetime, instrument); got {series.index!r}")
    if not {_DATETIME_LEVEL, _INSTRUMENT_LEVEL}.issubset(set(series.index.names)):
        raise ValueError(
            f"{name} index levels must be named {_DATETIME_LEVEL!r} and {_INSTRUMENT_LEVEL!r}; "
            f"got {series.index.names!r}"
        )


def align_predictions(prediction: pd.Series, label: pd.Series) -> pd.DataFrame:
    """Align a prediction and a label series on their common, fully-observed observations.

    Parameters
    ----------
    prediction : pd.Series
        Model score on ``MultiIndex(datetime, instrument)``.
    label : pd.Series
        Realized forward return on the same index.

    Returns
    -------
    pd.DataFrame
        Frame with columns ``["prediction", "label"]``, sorted lexicographically, holding only
        rows observed in both inputs.

    Notes
    -----
    Rows missing in either input are dropped **pairwise**, which is the correct treatment for an
    information coefficient: the statistic is defined on the intersection of the two supports.
    Sample-level exclusion for training and optimization is a different concern, handled by the
    boolean-mask contract of ``PROJECT_SPEC.md`` 3.4.3 (task ``INF-07``).
    """
    _require_multiindex(prediction, "prediction")
    _require_multiindex(label, "label")
    duplicates = prediction.index[prediction.index.duplicated()].unique()
    if len(duplicates):
        raise ValueError(
            "prediction has duplicate (datetime, instrument) entries, e.g. "
            f"{list(duplicates[:3])}. The panel contract requires a unique index "
            "(PROJECT_SPEC.md 3.4.2); duplicated rows indicate a join or split defect, and "
            "computing a cross-sectional statistic on them would double-count observations."
        )
    label_duplicates = label.index[label.index.duplicated()].unique()
    if len(label_duplicates):
        raise ValueError(f"label has duplicate (datetime, instrument) entries, e.g. {list(label_duplicates[:3])}")
    frame = pd.DataFrame({"prediction": prediction, "label": label}).dropna(how="any")
    return frame.sort_index()


def _cross_sectional_correlation(
    frame: pd.DataFrame,
    *,
    method: ICMethod,
    min_obs: int,
) -> tuple[pd.Series, int]:
    """Return the per-period cross-sectional correlation series and the dropped-period count."""
    if method not in IC_METHODS:
        raise ValueError(f"method must be one of {IC_METHODS}, got {method!r}")
    if min_obs < 3:
        raise ValueError("min_obs must be at least 3 for a correlation to be defined")

    counts = frame.groupby(level=_DATETIME_LEVEL, sort=True).size()
    admissible = counts[counts >= min_obs]
    dropped = int((counts < min_obs).sum())
    if admissible.empty:
        empty = pd.Series(dtype="float64", name="ic")
        empty.index.name = _DATETIME_LEVEL
        return empty, dropped

    mask = frame.index.get_level_values(_DATETIME_LEVEL).isin(admissible.index)
    block = frame.loc[mask]
    if method == "spearman":
        # Spearman is Pearson on within-period ranks.  Ranks are computed cross-sectionally only,
        # so no information crosses period boundaries (PIT-1, PROJECT_SPEC.md 3.6).
        block = block.groupby(level=_DATETIME_LEVEL).rank()

    centred = block - block.groupby(level=_DATETIME_LEVEL).transform("mean")
    numerator = (centred["prediction"] * centred["label"]).groupby(level=_DATETIME_LEVEL).sum()
    pred_ss = (centred["prediction"] ** 2).groupby(level=_DATETIME_LEVEL).sum()
    label_ss = (centred["label"] ** 2).groupby(level=_DATETIME_LEVEL).sum()
    series = (numerator / np.sqrt(pred_ss * label_ss)).replace([np.inf, -np.inf], np.nan).dropna()
    series.name = "ic"
    series.index.name = _DATETIME_LEVEL
    return series.astype("float64"), dropped


@overload
def rank_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: Literal[False] = False,
) -> pd.Series: ...


@overload
def rank_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: Literal[True],
) -> tuple[pd.Series, int]: ...


def rank_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: bool = False,
) -> pd.Series | tuple[pd.Series, int]:
    """Return the per-period Spearman rank IC series (``PROJECT_SPEC.md`` 2.2.1).

    Parameters
    ----------
    prediction : pd.Series
        Model score on ``MultiIndex(datetime, instrument)``; a higher score means a more
        attractive asset (the sign convention of ``PROJECT_SPEC.md`` 3.4.3).
    label : pd.Series
        Realized forward return on the same index.
    min_obs : int
        Minimum admissible cross-section size; periods below it are dropped and counted.
    return_dropped : bool
        Also return the number of dropped periods.

    Returns
    -------
    pd.Series or tuple
        Per-period rank IC indexed by ``datetime``, optionally with the dropped-period count.
    """
    frame = align_predictions(prediction, label)
    series, dropped = _cross_sectional_correlation(frame, method="spearman", min_obs=min_obs)
    return (series, dropped) if return_dropped else series


@overload
def pearson_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: Literal[False] = False,
) -> pd.Series: ...


@overload
def pearson_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: Literal[True],
) -> tuple[pd.Series, int]: ...


def pearson_ic(
    prediction: pd.Series,
    label: pd.Series,
    *,
    min_obs: int = 30,
    return_dropped: bool = False,
) -> pd.Series | tuple[pd.Series, int]:
    """Return the per-period Pearson IC series (magnitude, rather than ordering, of the signal)."""
    frame = align_predictions(prediction, label)
    series, dropped = _cross_sectional_correlation(frame, method="pearson", min_obs=min_obs)
    return (series, dropped) if return_dropped else series


def ic_moments(ic: pd.Series, *, periods_per_year: int = 252) -> ICMoments:
    """Summarize an IC series with moments and an **i.i.d.** reference statistic.

    Parameters
    ----------
    ic : pd.Series
        Per-period IC series (the output of :func:`rank_ic` or :func:`pearson_ic`).
    periods_per_year : int
        Annualization factor for the IC information ratio.

    Returns
    -------
    ICMoments

    Notes
    -----
    ``t_iid`` divides the mean by the i.i.d. standard error and is reported **only** for
    orientation.  It MUST NOT be used for inference: the IC series is serially correlated, so the
    admissible test is the HAC (Newey-West) statistic of ``PROJECT_SPEC.md`` 2.2.2, implemented in
    ``ST-02``.
    """
    values = ic.dropna().astype("float64")
    n_periods = int(values.size)
    if n_periods < 2:
        nan = float("nan")
        return ICMoments(n_periods, nan, nan, nan, nan, nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    # A *numerically* zero dispersion is not a dispersion: a constant series yields std ~ 1e-18 from
    # floating-point noise, which would otherwise turn the ICIR into an astronomically large number
    # (observed: 6.8e15) instead of the undefined value it actually is.  The tolerance is relative to
    # the scale of the series so it behaves the same for percentages and for log-returns.
    scale = float(np.max(np.abs(values))) if n_periods else 0.0
    if std <= 1e-12 * max(scale, 1.0):
        return ICMoments(n_periods, mean, std, float("nan"), float("nan"), float("nan"))
    icir = mean / std
    return ICMoments(
        n_periods=n_periods,
        mean=mean,
        std=std,
        icir=icir,
        icir_annualized=icir * float(np.sqrt(periods_per_year)),
        t_iid=mean / (std / np.sqrt(n_periods)),
    )
