"""Label construction under the strict-causality contract (task ``INF-05``).

``PROJECT_SPEC.md`` 2.1.1 defines the canonical target as a forward return on execution prices:

.. math::

    y_{i,\\,t+1} = \\frac{P^{\\,open}_{i,\\,t+1+h}}{P^{\\,open}_{i,\\,t+1}} - 1

Signal at :math:`t`, execution at the open of :math:`t+\\text{lag}`, measurement over
:math:`[t+\\text{lag},\\; t+\\text{lag}+h]`.  Two consequences are made mechanical here:

* **Availability.** A label exists only if *both* endpoints of its price window exist and the
  instrument is actually tradable at the execution date.  Where it does not exist, the row is
  kept and flagged as unavailable - never silently dropped (that is the survivorship-bias failure
  mode of ``PROJECT_SPEC.md`` 1.2.1 / task ``INF-07``).
* **Overlap.** The label at :math:`t` uses prices up to :math:`t+\\text{lag}+h`, so two labels
  closer than :math:`\\text{lag}+h` periods share information.  :attr:`LabelSpec.purge_radius`
  exposes that distance for the purged splitters (``ST-08``) and the leakage audit (``INF-08``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from qresearch.utils.typing import IntArray

__all__ = [
    "LabelSpec",
    "apply_variant",
    "forward_return",
    "label_availability",
    "purge_positions",
]

_DATETIME_LEVEL: Final[str] = "datetime"
_INSTRUMENT_LEVEL: Final[str] = "instrument"
LabelVariant = Literal["raw", "rank", "zscore"]


@dataclass(frozen=True)
class LabelSpec:
    """Specification of the target variable.

    Attributes
    ----------
    horizon : int
        Measurement horizon :math:`h` in trading days (``PROJECT_SPEC.md`` 2.1.1).
    execution_lag : int
        Delay between the decision date and the execution date.  ``1`` (default) means: signal at
        the close of :math:`t`, execution at the open of :math:`t+1`.
    price_field : str
        Execution price used for the target; ``"open"`` matches the backtest's ``deal_price``.
    variant : {"raw", "rank", "zscore"}
        Output transform: raw return, cross-sectional rank in :math:`(0,1)`, or cross-sectional
        :math:`z`-score.
    winsorize : float | None
        Optional two-sided winsorization quantile applied *cross-sectionally per period* before
        the variant transform.  Cross-sectional winsorization is a same-period operation and
        therefore point-in-time safe; a *time-series* quantile would not be.
    """

    horizon: int = 1
    execution_lag: int = 1
    price_field: str = "open"
    variant: LabelVariant = "raw"
    winsorize: float | None = 0.01

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.execution_lag < 0:
            raise ValueError(f"execution_lag must be >= 0, got {self.execution_lag}")
        if self.variant not in {"raw", "rank", "zscore"}:
            raise ValueError(f"variant must be raw/rank/zscore, got {self.variant!r}")
        if self.winsorize is not None and not 0.0 <= self.winsorize < 0.5:
            raise ValueError(f"winsorize must be in [0, 0.5), got {self.winsorize}")

    @property
    def purge_radius(self) -> int:
        """Number of periods by which two labels can share prices.

        Derivation.  The label at :math:`t` consumes the price window
        :math:`[t + \\text{lag},\\; t + \\text{lag} + h]` (inclusive).  For two decision dates
        :math:`t_1 \\le t_2` those windows intersect iff :math:`t_2 - t_1 \\le h`, because the
        execution lag cancels.  Hence two samples within :attr:`purge_radius` of each other
        share information and MUST NOT be separated across a train/test boundary.

        Notes
        -----
        ``PROJECT_SPEC.md`` 2.2.5 states the purge window literally as
        :math:`[j - h + 1,\\; j + h - 1]`, which corresponds to the *half-open* window
        :math:`[t, t+h)`.  This project uses the inclusive window and therefore purges one extra
        period; that is the conservative direction (it removes information, never adds it).
        The clarification is recorded in ``docs/adr/ADR-004``.
        """
        return self.horizon

    @property
    def window(self) -> tuple[int, int]:
        """Return the inclusive offset window ``(start, end)`` of the label's price span."""
        return self.execution_lag, self.execution_lag + self.horizon


def forward_return(prices: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    """Compute the forward-return target from an execution-price panel.

    Parameters
    ----------
    prices : pd.DataFrame
        Execution prices indexed by date, one column per instrument.  Values for suspended days
        are ``NaN`` and stay ``NaN`` through the calculation.
    spec : LabelSpec
        Target specification.

    Returns
    -------
    pd.Series
        Label on ``MultiIndex(datetime, instrument)``.  Rows that cannot be computed are kept as
        ``NaN`` - they are **not** removed, because removing them would silently truncate the
        panel and bias the universe (the survivorship failure mode of ``PROJECT_SPEC.md`` 1.2.1).
        Use :func:`label_availability` for the corresponding boolean mask.

    Notes
    -----
    The return is computed at date :math:`t` from prices at :math:`t+\\text{lag}` and
    :math:`t+\\text{lag}+h`; that is legitimate because the label is realised *after* the
    decision.  The feature side must never touch these prices (``PIT-4``).
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame indexed by date with one column per instrument")
    entry = prices.shift(-spec.execution_lag)
    exit_price = prices.shift(-(spec.execution_lag + spec.horizon))
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = exit_price / entry - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).astype("float64")
    # The grid is built explicitly rather than with DataFrame.stack: the product index is then
    # guaranteed (date-major, instrument-minor) and independent of pandas' stacking semantics, which
    # are deprecated in 2.1+ (DataFrame.stack(dropna=...) removal).
    index = pd.MultiIndex.from_product([returns.index, returns.columns], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
    series = pd.Series(returns.to_numpy().reshape(-1), index=index, name="label")
    series.name = f"label_{spec.price_field}_h{spec.horizon}_lag{spec.execution_lag}"
    return series


def apply_variant(labels: pd.Series, spec: LabelSpec) -> pd.Series:
    """Apply the cross-sectional transform selected by ``spec.variant``.

    All transforms are **same-period** operations (winsorization quantiles, ranks, moments are
    computed within each date's cross-section), which makes them point-in-time safe by
    construction: no statistic is estimated across dates, so no future observation can influence
    a past row.  A *time-series* quantile would not have this property.
    """
    values = labels.astype("float64")
    grouped = values.groupby(level=_DATETIME_LEVEL)

    if spec.winsorize is not None:
        lower = grouped.transform(lambda column: column.quantile(spec.winsorize))
        upper = grouped.transform(lambda column: column.quantile(1.0 - float(spec.winsorize)))
        values = values.clip(lower=lower, upper=upper)

    if spec.variant == "raw":
        return values.rename("label")
    if spec.variant == "rank":
        rank = values.groupby(level=_DATETIME_LEVEL).rank(method="average")
        count = values.groupby(level=_DATETIME_LEVEL).transform("count")
        return (rank / (count + 1.0)).rename("label")
    if spec.variant == "zscore":
        grouped = values.groupby(level=_DATETIME_LEVEL)
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        return ((values - mean) / std.replace(0.0, np.nan)).rename("label")
    raise ValueError(f"unsupported variant {spec.variant!r}")


def label_availability(
    labels: pd.Series,
    spec: LabelSpec,
    prices: pd.DataFrame,
    *,
    tradability: pd.DataFrame | None = None,
) -> pd.Series:
    """Return the boolean availability mask of a label series (never a filtered subset).

    A label is available when its whole price window exists and, if a tradability panel is
    supplied, the instrument is actually tradable on the execution date.  The mask is returned
    separately from the labels so that downstream code can *exclude* unavailable samples from the
    loss and from the optimizer while keeping them visible to the accounting
    (``PROJECT_SPEC.md`` 3.4.3, task ``INF-07``).

    Notes
    -----
    Execution happens at ``t + execution_lag``, so the tradability check is shifted **forward**:
    a name suspended on the execution date has no usable label even when the decision date itself
    was perfectly tradable.  This is the mechanism that prevents a suspension from silently turning
    into a tradable sample.
    """
    dense = labels.unstack(_INSTRUMENT_LEVEL)
    available = dense.notna()
    window = prices.reindex(index=dense.index, columns=dense.columns)
    tradable_window = (
        window.shift(-spec.execution_lag).notna() & window.shift(-(spec.execution_lag + spec.horizon)).notna()
    )
    available &= tradable_window
    if tradability is not None:
        exec_tradable = (
            tradability.reindex(index=dense.index, columns=dense.columns)
            .astype("boolean")
            .shift(-spec.execution_lag)
            .fillna(False)
            .astype(bool)
        )
        available &= exec_tradable
    index = pd.MultiIndex.from_product([dense.index, dense.columns], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
    series = pd.Series(available.to_numpy().reshape(-1), index=index, name="label_available")
    return series


def purge_positions(center: int, positions: IntArray, spec: LabelSpec) -> IntArray:
    """Return the positions whose label windows overlap that of ``center``.

    Used by the purged splitters (``ST-08``) and by the leakage audit (``INF-08``) whenever
    temporal neighbours must be excluded around an evaluation point.
    """
    array = np.asarray(positions, dtype="int64")
    within = np.abs(array - center) <= spec.purge_radius
    return np.asarray(array[within], dtype="int64")
