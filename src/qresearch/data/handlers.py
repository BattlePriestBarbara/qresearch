"""Sequence tensor assembly with explicit boolean masks (task ``INF-07``).

``PROJECT_SPEC.md`` 3.4.3 specifies the tensor contract; this module implements the part that is
easy to get silently wrong:

===============  ==========================  ===========================================
Object           Shape / dtype               Meaning
===============  ==========================  ===========================================
``X``            ``(N, L, d)`` ``float32``   feature windows, ``[sample, time, channel]``
``y``            ``(N, 1)`` ``float32``      realized label (``NaN`` where unavailable)
``weight``       ``(N, 1)`` ``float32``      sample weight (ones by default)
``mask``         ``(N,)`` ``bool``           sample-level validity
``step_mask``    ``(N, L)`` ``bool``         per-timestep availability inside the window
===============  ==========================  ===========================================

**No global dropna.**  A suspension, a delisting or a missing fundamental produces
``mask[i] = False`` and a neutral fill value in ``X`` - never a removed row.  Dropping such rows
would silently remove assets from the sample, which is exactly the survivorship bias that makes a
backtest unreproducible (``PROJECT_SPEC.md`` 1.2.1, ``PIT-1`` .. ``PIT-8``).  The mask is the only
sanctioned way to exclude a sample, and the loss (``NN-02``) and the optimizer (``PO-03``) MUST
apply it; :func:`masked_mean` is the reference reduction, provided so that both layers agree on
what "excluded" means.

Two distinct notions are kept apart on purpose:

* ``step_mask`` records **data availability** inside the window (``NaN`` in the source panel);
* ``mask`` records **sample validity**: label availability, the caller's availability mask, the
  tradability of the decision date, and optionally a minimum number of valid steps.

A suspension is a *tradability* fact rather than a missing observation - the prints exist, they are
simply not tradable - so it is reported through ``mask`` and, by default, the decision-date step is
neutralised so that stale prices cannot be learned from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from qresearch.utils.typing import Array, BoolArray, FloatArray

__all__ = [
    "MASK_COMPONENTS",
    "PanelBundle",
    "build_sequence_bundle",
    "masked_mean",
]

_DATETIME_LEVEL = "datetime"
_INSTRUMENT_LEVEL = "instrument"
MASK_COMPONENTS = ("label_available", "provided_mask", "tradable_on_decision_date", "enough_valid_steps")


@dataclass(frozen=True)
class PanelBundle:
    """Immutable tensor bundle with masks and provenance metadata."""

    X: FloatArray
    y: FloatArray
    weight: FloatArray
    mask: BoolArray
    step_mask: BoolArray
    index: pd.MultiIndex
    feature_names: tuple[str, ...]
    fill_value: float
    component_masks: Mapping[str, BoolArray]

    @property
    def n_samples(self) -> int:
        """Total number of rows, including invalid ones (nothing was dropped)."""
        return int(self.X.shape[0])

    @property
    def n_valid(self) -> int:
        """Number of rows the loss and the optimizer are allowed to use."""
        return int(self.mask.sum())

    @property
    def coverage(self) -> float:
        """Fraction of rows that are valid."""
        return float(self.mask.mean()) if self.n_samples else float("nan")

    @property
    def n_features(self) -> int:
        """Number of feature channels."""
        return int(self.X.shape[2])

    @property
    def lookback(self) -> int:
        """Sequence length :math:`L`."""
        return int(self.X.shape[1])

    def describe(self) -> dict[str, float | int | str]:
        """Return coverage diagnostics that MUST be logged with every training run."""
        diagnostics: dict[str, float | int | str] = {
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "coverage": round(self.coverage, 6),
            "lookback": self.lookback,
            "n_features": self.n_features,
            "fill_value": self.fill_value,
            "mean_valid_steps": round(float(self.step_mask.sum(axis=1).mean()), 3),
        }
        for name, component in self.component_masks.items():
            diagnostics[f"invalid_by_{name}"] = int((~component).sum())
        return diagnostics

    def select(self, name: str) -> BoolArray:
        """Return one component mask by name (see :data:`MASK_COMPONENTS`)."""
        if name not in self.component_masks:
            raise KeyError(f"unknown mask component {name!r}; available: {sorted(self.component_masks)}")
        return self.component_masks[name]


def _window_array(panel: pd.DataFrame, lookback: int) -> Array:
    """Return ``(T, I, L)`` windows of ``panel``; the window at ``t`` ends on ``t``.

    Warm-up rows are padded with ``NaN``, so the first ``L-1`` dates have explicitly incomplete
    windows rather than a wrapped, forward-filled or zero-filled one.
    """
    values = panel.to_numpy(dtype="float64")
    n_dates, n_instruments = values.shape
    padded = np.vstack([np.full((lookback - 1, n_instruments), np.nan), values])
    windows = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=0)
    assert windows.shape == (n_dates, n_instruments, lookback), windows.shape
    return np.ascontiguousarray(windows)


def _as_grid(series: pd.Series, dates: pd.DatetimeIndex, instruments: pd.Index) -> pd.DataFrame:
    """Align a labelled series to the full ``(dates x instruments)`` grid, filling with ``NaN``.

    Reindexing - rather than dropping - is what keeps the row count equal to the panel size: an
    unavailable value becomes ``NaN`` plus ``mask == False``, never a missing row.
    """
    grid = pd.MultiIndex.from_product([dates, instruments], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
    return series.reindex(grid).unstack(_INSTRUMENT_LEVEL)


def build_sequence_bundle(
    features: Mapping[str, pd.DataFrame],
    labels: pd.Series,
    *,
    lookback: int,
    fill_value: float = 0.0,
    provided_mask: pd.Series | None = None,
    tradability: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    min_valid_steps: int | None = None,
    neutralize_untradable: bool = True,
) -> PanelBundle:
    """Assemble ``(X, y, mask, ...)`` from wide feature panels and a labelled series.

    Parameters
    ----------
    features : Mapping[str, pd.DataFrame]
        Feature name -> wide ``(date x instrument)`` panel; ``NaN`` marks unavailable data.
    labels : pd.Series
        Target on ``MultiIndex(datetime, instrument)`` (typically
        :func:`qresearch.data.labels.forward_return`).
    lookback : int
        Sequence length :math:`L`.
    fill_value : float
        Neutral value written into ``X`` where a feature is unavailable.  The corresponding
        ``step_mask`` entry is ``False``, so the fill never counts as data.
    provided_mask : pd.Series | None
        Optional availability mask (e.g. :func:`qresearch.data.labels.label_availability`).
    tradability : pd.DataFrame | None
        Optional tradability mask at the *decision* date (``INF-04``).
    weights : pd.Series | None
        Optional sample weights; ones by default.
    min_valid_steps : int | None
        When given, a sample additionally needs at least this many valid timesteps in its window.
        ``None`` (default) imposes no step requirement: partial windows are allowed and stay
        visible through ``step_mask``.
    neutralize_untradable : bool
        When ``True`` (default) and a ``tradability`` panel is supplied, the *decision-date* step of
        a suspended, limit-locked or non-member name is replaced by ``fill_value`` and its
        ``step_mask`` entry is cleared.  Rationale: a suspended session is not a valid observation of
        the state - its prices are stale and its volume is zero - so feeding the raw values would let
        the model learn from a non-tradable print.  Set to ``False`` only to study that effect.

    Returns
    -------
    PanelBundle

    Raises
    ------
    ValueError
        On an empty feature mapping, a non-positive ``lookback``, or a label series whose index is
        not the declared ``MultiIndex(datetime, instrument)``.
    """
    if not features:
        raise ValueError("features must contain at least one panel")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not isinstance(labels.index, pd.MultiIndex) or labels.index.nlevels != 2:
        raise ValueError("labels must be indexed by MultiIndex(datetime, instrument)")
    if labels.index.names != [_DATETIME_LEVEL, _INSTRUMENT_LEVEL]:
        raise ValueError(
            f"labels index levels must be named {[_DATETIME_LEVEL, _INSTRUMENT_LEVEL]}, got {labels.index.names}"
        )

    dates = pd.DatetimeIndex(labels.index.get_level_values(_DATETIME_LEVEL).unique()).sort_values()
    instruments = labels.index.get_level_values(_INSTRUMENT_LEVEL).unique().sort_values()

    label_matrix = _as_grid(labels.astype("float64"), dates, instruments).to_numpy(dtype="float64")
    n_dates, n_instruments = label_matrix.shape

    filled_blocks: list[Array] = []
    step_blocks: list[BoolArray] = []
    for name in sorted(features):
        panel = features[name].reindex(index=dates, columns=instruments)
        windows = _window_array(panel, lookback)
        step_blocks.append(np.isfinite(windows))
        filled_blocks.append(np.where(np.isfinite(windows), windows, fill_value))

    stacked = np.stack(filled_blocks, axis=-1)  # (T, I, L, d)
    steps = np.stack(step_blocks, axis=-1)  # (T, I, L, d)
    n_samples = n_dates * n_instruments
    x_tensor = stacked.reshape(n_samples, lookback, -1).astype("float32")
    step_mask = steps.all(axis=-1).reshape(n_samples, lookback)
    label_vector = label_matrix.reshape(n_samples, 1)
    y = label_vector.astype("float32")

    components: dict[str, BoolArray] = {}
    label_available = np.isfinite(label_vector).ravel()
    components["label_available"] = label_available
    mask = label_available

    if provided_mask is not None:
        aligned = _as_grid(provided_mask.astype("float64"), dates, instruments).to_numpy(dtype="float64")
        provided = np.nan_to_num(aligned, nan=0.0).reshape(n_samples) > 0.5
        components["provided_mask"] = provided
        mask = mask & provided
    if tradability is not None:
        tradable_panel = tradability.reindex(index=dates, columns=instruments).fillna(False)
        tradable = tradable_panel.to_numpy(dtype=bool).reshape(n_samples)
        components["tradable_on_decision_date"] = tradable
        mask = mask & tradable
        if neutralize_untradable:
            # A suspended / limit-locked / non-member decision date is not a valid observation: its
            # last step is neutralised and flagged, so the raw (stale) print cannot be learned from.
            untradable = np.flatnonzero(~tradable)
            x_tensor[untradable, -1, :] = np.float32(fill_value)
            step_mask[untradable, -1] = False
    if min_valid_steps is not None:
        if min_valid_steps < 1:
            raise ValueError(f"min_valid_steps must be >= 1, got {min_valid_steps}")
        enough = step_mask.sum(axis=1) >= min_valid_steps
        components["enough_valid_steps"] = enough
        mask = mask & enough

    if weights is not None:
        weight_matrix = _as_grid(weights.astype("float64"), dates, instruments).to_numpy(dtype="float64")
        weight_vector = np.nan_to_num(weight_matrix, nan=0.0).reshape(n_samples, 1)
    else:
        weight_vector = np.ones((n_samples, 1), dtype="float64")

    index = pd.MultiIndex.from_product([dates, instruments], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
    return PanelBundle(
        X=x_tensor,
        y=y,
        weight=weight_vector.astype("float32"),
        mask=mask.astype(bool),
        step_mask=step_mask.astype(bool),
        index=index,
        feature_names=tuple(sorted(features)),
        fill_value=float(fill_value),
        component_masks=MappingProxyType(components),
    )


def masked_mean(values: FloatArray, mask: BoolArray, weights: FloatArray | None = None) -> float:
    """Reduce ``values`` over valid samples only - the reference "excluded" semantics.

    Parameters
    ----------
    values : Array
        Per-sample quantities (e.g. the Student-``t`` NLL of ``NN-02``), shape ``(N,)`` or ``(N, 1)``.
    mask : Array
        Sample mask, shape ``(N,)``.
    weights : Array | None
        Optional sample weights, shape ``(N, 1)``.

    Returns
    -------
    float
        Weighted mean over masked-in samples; ``nan`` when no sample is valid, which is the honest
        answer - returning ``0.0`` would look like a perfect loss.

    Notes
    -----
    ``PROJECT_SPEC.md`` 2.3.3 normalizes the objective by :math:`\\sum m_{i,t} w_{i,t}`, which is
    exactly this reduction.  An unmasked mean would let suspended or untradable rows contribute to
    the gradient.
    """
    flat_values = np.asarray(values, dtype="float64").reshape(-1)
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if flat_values.shape != flat_mask.shape:
        raise ValueError(f"values and mask must have the same length, got {flat_values.shape} and {flat_mask.shape}")
    if weights is None:
        flat_weights = np.ones_like(flat_values)
    else:
        flat_weights = np.asarray(weights, dtype="float64").reshape(-1)
        if flat_weights.shape != flat_values.shape:
            raise ValueError(f"weights must have length {flat_values.shape[0]}, got {flat_weights.shape}")
    effective = flat_weights * flat_mask
    total = float(effective.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.nansum(flat_values * effective) / total)
