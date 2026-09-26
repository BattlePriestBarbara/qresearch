"""Synthetic ground-truth panel generator (task ``INF-11``, ``PROJECT_SPEC.md`` 3.8).

``3.8`` requires a "panel with controllable IC, kurtosis and autocorrelation, plus documented ground
truth", and makes it the shared input of every later statistical test.  This module *is* that
generator; ``tests/conftest.py::known_signal_panel`` exposes it as the fixture the specification
names.

Data-generating process
-----------------------
For instruments :math:`i = 1..N` and dates :math:`t = 1..T`:

.. math::

    x_{i,f,t} &= \\varphi_f x_{i,f,t-1} + \\sqrt{1-\\varphi_f^2}\\,\\eta_{i,f,t},
        \\qquad \\eta \\sim \\mathcal{N}(0, 1)
    s_{i,t}   &= \\frac{\\sum_f w_f x_{i,f,t}}{\\mathrm{sd}_{i,t}\\left(\\sum_f w_f x_{i,f,t}\\right)}
        \\qquad \\text{(cross-sectionally standardised, so } \\sigma_s = 1)
    \\varepsilon_{i,t} &= \\varrho\\, \\varepsilon_{i,t-1} + \\sqrt{1-\\varrho^2}\\,\\xi_{i,t},
        \\qquad \\xi \\sim t_{\\nu}\\ \\text{rescaled to unit variance}
    y_{i,t}   &= \\beta\\, s_{i,t} + \\sigma\\, \\varepsilon_{i,t+1},
        \\qquad \\beta = \\sigma\\, \\frac{\\rho}{\\sqrt{1-\\rho^2}}

The label stored at row :math:`t` is the **forward** return :math:`y_{t+1}`, i.e. the same
point-in-time convention as ``qresearch.data.labels``: one row carries :math:`(x_t, s_t, y_{t+1})`.
Because :math:`s_t` and :math:`\\varepsilon_{t+1}` are independent with unit variance, the
**population correlation between the signal and the label is exactly**

.. math:: \\rho = \\frac{\\beta}{\\sqrt{\\beta^2 + \\sigma^2}} ,

which is the injected, known IC of the request.  :math:`\\beta` is solved for :math:`\\rho`, not
tuned, so the truth is closed-form rather than empirical.

Ground truth reported by :attr:`SyntheticPanel.truth`
-----------------------------------------------------
* ``population_correlation`` - :math:`\\rho` above.  Exact by construction, and a property of the
  *variables*, not of the realised cross-sections.
* ``expected_rank_ic`` / ``expected_pearson_ic`` - the estimand the tests actually compare against:
  :math:`\\mathbb{E}[\\rho_t \\mid s]`, the average per-date cross-sectional statistic **conditional on
  the realised signal**.  It has no closed form here (the noise is Student-:math:`t` **and** AR(1), so
  the usual Gaussian-copula conversion :func:`normal_copula_rank_ic` does not apply: measured on the
  default configuration the conditional mean is ``~0.053`` against a copula value of ``0.0478``).  It is
  therefore estimated by holding the signal fixed and redrawing the noise ``monte_carlo_draws`` times,
  which makes the average over ``draws x n_days`` cross-sections converge to that conditional mean for
  *any* signal shape (linear, nonlinear, unit-root).  The Monte-Carlo standard error is reported next to
  it, so a test can size its tolerance from two known errors instead of a hand-tuned constant: the
  Monte-Carlo error and the panel's own :math:`\\sigma_\\rho/\\sqrt{T}`.
* ``noise_innovation_excess_kurtosis`` - the theoretical excess kurtosis of the injected
  Student-:math:`t_{\\nu}` innovations, ``6/(nu-4)``; this is the fat-tail knob.
* ``measured_noise_excess_kurtosis`` / ``measured_label_excess_kurtosis`` - what the drawn arrays
  actually exhibit.  They sit *below* the innovation value, because the AR(1) filter is a moving
  average of innovations and averages are thinner-tailed than their inputs; the tests assert the
  relation that matters (clearly above the Gaussian 0 with ``nu = 5``, clearly at 0 in the Gaussian
  control), not the innovation value itself.
* ``ic_autocorrelation_lag1`` - the lag-1 autocorrelation of the realised IC series.  The persistent
  idiosyncratic noise makes it positive (measured ``~0.28`` on the default configuration), which is
  exactly the property ``2.2.2`` says the HAC estimator must handle and the i.i.d. standard error
  must not be used for.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from ..utils.typing import FloatArray

__all__ = [
    "FEATURE_KINDS",
    "SIGNAL_KINDS",
    "SyntheticPanel",
    "SyntheticPanelSpec",
    "generate_panel",
    "normal_copula_rank_ic",
    "per_date_correlation",
]

FEATURE_KINDS: Final[tuple[str, ...]] = ("ar1", "random_walk")
SIGNAL_KINDS: Final[tuple[str, ...]] = ("linear", "nonlinear")

MINIMUM_NOISE_DF: Final[float] = 2.5
"""Below this the standardised Student-t has no stable unit-variance rescaling on a finite draw."""


def normal_copula_rank_ic(rho: float) -> float:
    """Return the population Spearman correlation of a bivariate normal pair with Pearson ``rho``.

    .. math:: \\rho_S = \\frac{6}{\\pi} \\arcsin\\!\\left(\\frac{\\rho}{2}\\right)

    Reported for reference: it is the classical conversion between the two ICs, and it is exact for
    jointly Gaussian ``(s, y)``.  With Student-:math:`t` noise the population rank IC differs from it,
    which is why :attr:`SyntheticPanel.truth` carries a simulated value as well.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    return float(6.0 / np.pi * np.arcsin(rho / 2.0))


@dataclass(frozen=True)
class SyntheticPanelSpec:
    """The knobs of the DGP; every default is the configuration the INF-11 tests exercise."""

    n_instruments: int = 100
    n_days: int = 1000
    n_features: int = 3
    feature_kind: str = "ar1"
    feature_ar1_rho: float = 0.95
    signal_kind: str = "linear"
    target_correlation: float = 0.05
    noise_df: float = 5.0
    noise_ar1_rho: float = 0.3
    min_obs: int = 30
    seed: int = 20260926
    monte_carlo_draws: int = 32

    def __post_init__(self) -> None:
        """Reject a specification that cannot produce a valid panel or a valid ground truth."""
        if self.n_instruments < 3:
            raise ValueError(f"n_instruments must be >= 3, got {self.n_instruments}")
        if self.n_days < 3:
            raise ValueError(f"n_days must be >= 3, got {self.n_days}")
        if not 1 <= self.n_features <= 5:
            raise ValueError(f"n_features must lie in [1, 5], got {self.n_features}")
        if self.feature_kind not in FEATURE_KINDS:
            raise ValueError(f"feature_kind must be one of {FEATURE_KINDS}, got {self.feature_kind!r}")
        if not 0.0 <= self.feature_ar1_rho <= 1.0:
            raise ValueError(f"feature_ar1_rho must lie in [0, 1], got {self.feature_ar1_rho}")
        if self.signal_kind not in SIGNAL_KINDS:
            raise ValueError(f"signal_kind must be one of {SIGNAL_KINDS}, got {self.signal_kind!r}")
        if self.signal_kind == "nonlinear" and self.n_features < 3:
            raise ValueError("signal_kind='nonlinear' needs at least 3 features (tanh + cross term + level)")
        if not 0.0 <= self.target_correlation < 1.0:
            raise ValueError(f"target_correlation must lie in [0, 1), got {self.target_correlation}")
        if self.noise_df <= MINIMUM_NOISE_DF:
            raise ValueError(f"noise_df must exceed {MINIMUM_NOISE_DF}, got {self.noise_df}")
        if not 0.0 <= self.noise_ar1_rho < 1.0:
            raise ValueError(f"noise_ar1_rho must lie in [0, 1), got {self.noise_ar1_rho}")
        if self.min_obs < 3:
            raise ValueError(f"min_obs must be >= 3, got {self.min_obs}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.monte_carlo_draws < 0:
            raise ValueError("monte_carlo_draws must be non-negative (0 falls back to the copula value)")

    @property
    def n_observations(self) -> int:
        """Number of rows of the panel."""
        return self.n_instruments * self.n_days

    @property
    def beta(self) -> float:
        """The signal loading that realizes :attr:`target_correlation` with unit noise variance."""
        return self.target_correlation / float(np.sqrt(1.0 - self.target_correlation**2))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "n_instruments": self.n_instruments,
            "n_days": self.n_days,
            "n_features": self.n_features,
            "feature_kind": self.feature_kind,
            "feature_ar1_rho": self.feature_ar1_rho,
            "signal_kind": self.signal_kind,
            "target_correlation": self.target_correlation,
            "noise_df": self.noise_df,
            "noise_ar1_rho": self.noise_ar1_rho,
            "min_obs": self.min_obs,
            "seed": self.seed,
            "beta": self.beta,
        }


@dataclass(frozen=True)
class SyntheticPanel:
    """A generated panel together with the ground truth it was generated from."""

    spec: SyntheticPanelSpec
    features: pd.DataFrame
    signal: pd.Series
    label: pd.Series
    noise: pd.Series
    truth: MappingProxyType[str, object]

    @property
    def ic_series(self) -> pd.Series:
        """Return the per-date Spearman IC series implied by the ground truth (for quick checks).

        This is a convenience for diagnostics; the statistically binding computation is
        :func:`qresearch.stats.ic.rank_ic`, which every test MUST use instead.
        """
        signal_matrix = self.signal.to_numpy(dtype="float64").reshape(self.spec.n_days, self.spec.n_instruments)
        label_matrix = self.label.to_numpy(dtype="float64").reshape(self.spec.n_days, self.spec.n_instruments)
        values = per_date_correlation(signal_matrix, label_matrix, method="spearman")
        dates = self.signal.index.get_level_values("datetime").unique()
        return pd.Series(values, index=dates, name="ic")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary (never the panel itself: it is large by design)."""
        return {
            "spec": self.spec.to_dict(),
            "truth": dict(self.truth),
            "n_observations": int(self.features.shape[0]),
            "columns": list(self.features.columns),
        }


# ---------------------------------------------------------------------------------------
# Vectorised cross-sectional statistics (the primitive the truth is estimated with)
# ---------------------------------------------------------------------------------------
def _rank_rows(values: FloatArray) -> FloatArray:
    """Return the within-row (i.e. within-date) ranks of a ``(n_dates, n_instruments)`` array.

    Ties are broken by position; the DGP is continuous, so a tie has probability zero and the choice
    is irrelevant - it only has to be deterministic.
    """
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.float64)
    rows = np.arange(values.shape[0])
    ranks[rows[:, None], order] = np.arange(values.shape[1], dtype=np.float64)[None, :]
    return ranks


def per_date_correlation(signal: FloatArray, label: FloatArray, *, method: str = "spearman") -> FloatArray:
    """Return the per-date cross-sectional correlation of two ``(n_dates, n_instruments)`` arrays.

    This mirrors the estimand of ``PROJECT_SPEC.md`` 2.2.1 for a complete panel (no mask, no missing
    rows) and is used to estimate population quantities over a large synthetic draw.  It is *not* a
    replacement for :func:`qresearch.stats.ic.rank_ic`, which owns the admissible-set and
    minimum-support rules and is what tests must assert against.

    Parameters
    ----------
    signal, label : FloatArray
        Arrays of the same 2-D shape; rows are dates, columns are instruments.
    method : str
        ``"spearman"`` (ranks) or ``"pearson"`` (levels).

    Returns
    -------
    FloatArray
        One correlation per date, ``nan`` where a cross-section has no dispersion.
    """
    first = np.asarray(signal, dtype="float64")
    second = np.asarray(label, dtype="float64")
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(f"signal and label must share a 2-D shape, got {first.shape} and {second.shape}")
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if method == "spearman":
        first, second = _rank_rows(first), _rank_rows(second)
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt((first * first).sum(axis=1) * (second * second).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0.0, (first * second).sum(axis=1) / denominator, np.nan)


def _lag1_autocorrelation(values: FloatArray) -> float:
    """Return the lag-1 autocorrelation of a one-dimensional series (``nan`` if degenerate)."""
    clean = values[np.isfinite(values)]
    if clean.size < 3:
        return float("nan")
    centred = clean - clean.mean()
    denominator = float((centred * centred).sum())
    if denominator <= 0.0:
        return float("nan")
    return float((centred[1:] * centred[:-1]).sum() / denominator)


def _excess_kurtosis(values: FloatArray) -> float:
    """Return the sample excess kurtosis of an array (``0`` for a Gaussian sample)."""
    flat = np.asarray(values, dtype="float64").ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < 4:
        return float("nan")
    centred = flat - flat.mean()
    variance = float((centred**2).mean())
    if variance <= 0.0:
        return float("nan")
    return float((centred**4).mean() / variance**2 - 3.0)


# ---------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------
DEFAULT_START_DATE: Final[str] = "2000-01-03"
"""A Monday; the panel uses business days so that the gap statistics are realistic."""

BURN_IN_PERIODS: Final[int] = 200
"""Periods discarded so that the AR(1) noise starts from its stationary distribution."""


def _feature_matrix(spec: SyntheticPanelSpec, rng: np.random.Generator) -> FloatArray:
    """Return the ``(n_days, n_instruments, n_features)`` feature panel.

    ``feature_kind="ar1"`` produces a stationary AR(1) with the requested persistence, initialised at
    its stationary distribution; ``feature_kind="random_walk"`` produces cumulative sums (a unit-root
    feature), which is the other process the specification names.
    """
    shape = (spec.n_days, spec.n_instruments, spec.n_features)
    if spec.feature_kind == "random_walk":
        return np.cumsum(rng.standard_normal(shape), axis=0)
    scale = float(np.sqrt(max(1.0 - spec.feature_ar1_rho**2, 0.0)))
    innovations = rng.standard_normal(shape) * scale
    matrix = np.empty(shape, dtype="float64")
    matrix[0] = rng.standard_normal((spec.n_instruments, spec.n_features))
    for step in range(1, spec.n_days):
        matrix[step] = spec.feature_ar1_rho * matrix[step - 1] + innovations[step]
    return matrix


def _standardise_rows(values: FloatArray) -> FloatArray:
    """Return ``values`` centred and scaled within each row (a zero-variance row is left at zero)."""
    centred: FloatArray = values - values.mean(axis=1, keepdims=True)
    scale: FloatArray = centred.std(axis=1, ddof=0, keepdims=True)
    # `np.where` is typed as returning `Any`; the annotation keeps the contract explicit instead of
    # letting `Any` propagate through the module under `warn_return_any`.
    safe_scale: FloatArray = np.where(scale > 0.0, scale, 1.0)
    return centred / safe_scale


def _signal_matrix(spec: SyntheticPanelSpec, features: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Return the cross-sectionally standardised signal for ``linear`` or ``nonlinear`` combinations.

    The weights are drawn once and *not* reported as truth: what is controlled is the correlation
    between the signal and the label, not the loading on any individual feature, so the test cannot be
    coupled to an implementation detail of the combination.
    """
    weights = rng.standard_normal(spec.n_features) / float(np.sqrt(spec.n_features))
    if spec.signal_kind == "linear":
        raw = features @ weights
    else:
        raw = np.tanh(weights[0] * features[:, :, 0]) + weights[1] * features[:, :, 0] * features[:, :, 1]
        if spec.n_features > 2:
            raw = raw + weights[2] * features[:, :, 2]
    return _standardise_rows(raw)


def _noise_matrix(spec: SyntheticPanelSpec, rng: np.random.Generator, *, n_periods: int) -> FloatArray:
    """Return ``(n_periods, n_instruments)`` persistent, fat-tailed, unit-variance noise.

    The innovations are Student-:math:`t_{\\nu}` rescaled to unit variance (the fat-tail knob); the
    per-instrument AR(1) recursion makes the noise persistent, which is what turns the realised IC
    series into a serially correlated one - the property ``2.2.2`` exists to handle.
    """
    innovations = rng.standard_t(spec.noise_df, size=(n_periods + BURN_IN_PERIODS, spec.n_instruments))
    innovations /= float(np.sqrt(spec.noise_df / (spec.noise_df - 2.0)))
    if spec.noise_ar1_rho == 0.0:
        return innovations[BURN_IN_PERIODS:]
    scale = float(np.sqrt(1.0 - spec.noise_ar1_rho**2))
    matrix = np.zeros_like(innovations)
    for step in range(1, innovations.shape[0]):
        matrix[step] = spec.noise_ar1_rho * matrix[step - 1] + scale * innovations[step]
    return matrix[BURN_IN_PERIODS:]


def _build_matrices(spec: SyntheticPanelSpec) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(features, signal, label, noise)`` as arrays; the single entry point of the DGP.

    ``noise`` is the :math:`\\varepsilon_{t+1}` term the label actually consumed, kept so that a test
    of the Student-:math:`t` likelihood can be run against a known noise distribution.
    """
    rng = np.random.default_rng(spec.seed)
    features = _feature_matrix(spec, rng)
    signal = _signal_matrix(spec, features, rng)
    noise = _noise_matrix(spec, rng, n_periods=spec.n_days + 1)[1:]
    label = spec.beta * signal + noise
    return features, signal, label, noise


def _panel_index(spec: SyntheticPanelSpec) -> pd.MultiIndex:
    """Return the ``(datetime, instrument)`` index mandated by ``3.4.6``."""
    dates = pd.bdate_range(DEFAULT_START_DATE, periods=spec.n_days, name="datetime")
    instruments = pd.Index([f"S{position:04d}" for position in range(spec.n_instruments)], name="instrument")
    return pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])


def _draw_noise(spec: SyntheticPanelSpec, *, seed: int) -> FloatArray:
    """Return the label noise of one independent realisation: ``(n_days, n_instruments)``."""
    return _noise_matrix(spec, np.random.default_rng(seed), n_periods=spec.n_days + 1)[1:]


def _conditional_ic(
    spec: SyntheticPanelSpec,
    signal: FloatArray,
    *,
    method: str,
) -> tuple[float, float]:
    """Return ``(mean, standard_error)`` of the per-date IC conditional on the given signal.

    The signal is held fixed and only the noise is redrawn, so the average over
    ``monte_carlo_draws x n_days`` cross-sections estimates :math:`\\mathbb{E}[\\rho_t \\mid s]`.  The
    panel realises a single such draw, which is what makes this the correct reference: a difference
    between the two is sampling error with a known standard error, not a definitional gap.

    The draw seeds differ by a large prime so that they are independent of the panel's own seed and of
    each other, while the whole estimate stays a deterministic function of the spec.
    """
    draws = max(spec.monte_carlo_draws, 1)
    samples = [
        per_date_correlation(
            signal,
            spec.beta * signal + _draw_noise(spec, seed=spec.seed + 7919 * (index + 1)),
            method=method,
        )
        for index in range(draws)
    ]
    stacked = np.concatenate(samples)
    finite = stacked[np.isfinite(stacked)]
    if finite.size < 2:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1) / np.sqrt(finite.size))


def _ground_truth(
    spec: SyntheticPanelSpec,
    signal: FloatArray,
    label: FloatArray,
    noise: FloatArray,
) -> dict[str, object]:
    """Return the documented ground truth of the panel ``(signal, label, noise)`` was built from.

    ``expected_rank_ic`` / ``expected_pearson_ic`` are the only quantities the DGP does not determine in
    closed form once the noise is Student-:math:`t` and AR(1); they are estimated conditionally on the
    realised signal, with the Monte-Carlo standard error reported alongside.
    """
    rank_ic = per_date_correlation(signal, label, method="spearman")
    pearson = per_date_correlation(signal, label, method="pearson")
    innovation_excess_kurtosis = 6.0 / (spec.noise_df - 4.0)
    truth: dict[str, object] = {
        "population_correlation": spec.target_correlation,
        "signal_loading_beta": spec.beta,
        "realized_mean_rank_ic": float(np.nanmean(rank_ic)),
        "realized_mean_pearson_ic": float(np.nanmean(pearson)),
        "realized_rank_ic_standard_deviation": float(np.nanstd(rank_ic, ddof=1)),
        "normal_copula_rank_ic": normal_copula_rank_ic(spec.target_correlation),
        "noise_innovation_excess_kurtosis": innovation_excess_kurtosis,
        "measured_noise_excess_kurtosis": _excess_kurtosis(noise),
        "measured_label_excess_kurtosis": _excess_kurtosis(label),
        "ic_autocorrelation_lag1": _lag1_autocorrelation(rank_ic),
        "n_dates": spec.n_days,
        "n_instruments": spec.n_instruments,
        "n_observations": spec.n_observations,
    }
    if spec.monte_carlo_draws > 0:
        rank_mean, rank_error = _conditional_ic(spec, signal, method="spearman")
        pearson_mean, pearson_error = _conditional_ic(spec, signal, method="pearson")
        truth["expected_rank_ic"] = rank_mean
        truth["expected_rank_ic_standard_error"] = rank_error
        truth["expected_pearson_ic"] = pearson_mean
        truth["expected_pearson_ic_standard_error"] = pearson_error
        truth["expected_ic_monte_carlo"] = {
            "draws": spec.monte_carlo_draws,
            "dates_per_draw": spec.n_days,
            "seed": spec.seed,
            "estimand": "conditional expectation given the realised signal",
        }
    else:
        # Without a Monte-Carlo draw only the Gaussian-copula reference is available; it is exact only
        # for joint-normal noise, so it is flagged as an approximation instead of presented as truth.
        truth["expected_rank_ic"] = normal_copula_rank_ic(spec.target_correlation)
        truth["expected_rank_ic_standard_error"] = float("nan")
        truth["expected_pearson_ic"] = spec.target_correlation
        truth["expected_pearson_ic_standard_error"] = float("nan")
        truth["expected_ic_monte_carlo"] = None
    return truth


def generate_panel(spec: SyntheticPanelSpec | None = None) -> SyntheticPanel:
    """Generate the synthetic panel described by ``spec`` (defaults: ``N=100``, ``T=1000``, IC 0.05).

    Parameters
    ----------
    spec : SyntheticPanelSpec | None
        DGP knobs; ``None`` uses the documented defaults.

    Returns
    -------
    SyntheticPanel
        Features, the oracle signal, the forward label and the ground truth.  Deterministic: the same
        spec always yields bit-identical arrays.

    Examples
    --------
    >>> from qresearch.data.synthetic import generate_panel
    >>> panel = generate_panel()
    >>> (panel.label.index.names, panel.features.shape[0])
    (['datetime', 'instrument'], 100000)
    """
    resolved = SyntheticPanelSpec() if spec is None else spec
    features, signal, label, noise = _build_matrices(resolved)
    index = _panel_index(resolved)
    frame = pd.DataFrame(
        {f"f{position}": features[:, :, position].ravel() for position in range(resolved.n_features)},
        index=index,
    )
    truth = _ground_truth(resolved, signal, label, noise)
    return SyntheticPanel(
        spec=resolved,
        features=frame,
        signal=pd.Series(signal.ravel(), index=index, name="score"),
        label=pd.Series(label.ravel(), index=index, name="label"),
        noise=pd.Series(noise.ravel(), index=index, name="noise"),
        truth=MappingProxyType(truth),
    )
