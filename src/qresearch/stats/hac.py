"""Newey-West (HAC) long-run variance and ``t``-statistic (task ``ST-02``).

This module implements ``PROJECT_SPEC.md`` 2.2.2 verbatim.  Write :math:`u_t = \\rho_t - \\bar{\\rho}`
for the centred IC series and :math:`T` for its length; the sample autocovariances, the
Bartlett-kernel long-run variance, its bandwidth and the test statistic are

.. math::

    \\hat{\\gamma}_0 = \\frac{1}{T}\\sum_{t=1}^{T} u_t^2,
    \\qquad
    \\hat{\\gamma}_{\\ell} = \\frac{1}{T}\\sum_{t=\\ell+1}^{T} u_t\\, u_{t-\\ell},
    \\qquad \\ell = 1, \\dots, L_{NW}

    \\widehat{\\sigma}^{\\,2}_{NW} = \\hat{\\gamma}_0 + 2\\sum_{\\ell=1}^{L_{NW}}
    \\left(1 - \\frac{\\ell}{L_{NW}+1}\\right)\\hat{\\gamma}_{\\ell},
    \\qquad
    L_{NW} = \\left\\lfloor 4\\left(\\frac{T}{100}\\right)^{2/9}\\right\\rfloor

    t_{NW} = \\frac{\\bar{\\rho}}{\\sqrt{\\widehat{\\sigma}^{\\,2}_{NW} / T}},
    \\qquad
    p_{NW} = 2\\left(1 - \\Phi\\!\\left(\\left| t_{NW} \\right|\\right)\\right)

The three small-sample options of 2.2.2 are implemented and **reported** (2.2.2 requires both):

1. :attr:`NeweyWestResult.reference` ``= "student_t"`` uses the Student-:math:`t` reference
   distribution with :math:`\\nu_{df} = T - 1` instead of the normal.
2. :attr:`NeweyWestResult.small_sample_correction` multiplies the standard error by
   :math:`\\sqrt{T/(T - k_{reg})}`; :math:`k_{reg}` counts every estimated parameter *including
   the bandwidth* (2.2.2), i.e. the mean plus the bandwidth when the bandwidth is automatic, and
   the mean alone when the caller fixes it.
3. :attr:`NeweyWestResult.prewhiten` fits an AR(1) to :math:`u_t`, applies the kernel to the
   residual series and re-colours the variance by :math:`1/(1-\\hat{\\rho})^2`
   (Andrews-Monahan 1992).  See the ambiguity note below.

``PROJECT_SPEC.md`` 3.4.6 names the toolkit entry point :func:`newey_west_se`, declared there as
``newey_west_se(x: np.ndarray, bandwidth: int | None = None, prewhite: bool = False, small_sample:
TestType) -> float``.  ``TestType`` is not defined anywhere in the specification; the placeholder is
bound to option 2 above, i.e. ``small_sample`` is the boolean that toggles the
:math:`\\sqrt{T/(T-k_{reg})}` factor.  It cannot denote option 1, because the Student-:math:`t`
reference distribution changes the *p*-value and not the standard error that
:func:`newey_west_se` returns; option 1 stays reachable through :func:`newey_west`'s ``reference``
argument.  See the README provenance note.

Correspondence with ``statsmodels`` (verified numerically on the pinned 0.14.2, see
``tests/test_hac.py``)
---------------------------------------------------------------------------------------------
* ``statsmodels.stats.sandwich_covariance.cov_hac`` is an alias of ``cov_hac_simple``, whose
  ``nlags=None`` default is documented as ``floor[4(T/100)^(2/9)]`` - the same rule as 2.2.2.
* Its ``weights_bartlett(L)`` returns ``1 - arange(L+1)/(L+1)`` - the same kernel as 2.2.2.
* Its ``use_correction=True`` multiplies the covariance by ``T/(T - k_params)`` where
  ``k_params`` is the number of regressors (1 for a constant-only mean regression).  Setting
  :attr:`NeweyWestResult.k_reg` ``= 1`` therefore reproduces it exactly, while the 2.2.2 default
  counts the bandwidth as well.
* ``OLS.fit(cov_type="HAC")`` ignores ``cov_hac``'s automatic bandwidth: it requires an explicit
  ``maxlags`` in ``cov_kwds`` and otherwise raises ``KeyError``.  Its ``p``-values use the
  Student-:math:`t` distribution with ``df = T - 1``, which is option 1 above.
* :func:`newey_west_se` therefore reproduces ``cov_hac(..., use_correction=True)`` to machine
  precision when the caller fixes a bandwidth (2.2.2 then counts one parameter, the same count as
  ``cov_hac``'s constant-only regression; the measured deviation is ``1.7e-15`` relative on the
  pinned stack), and differs from it by exactly :math:`\\sqrt{T/(T-2)}` when the bandwidth is
  automatic, because 2.2.2 counts the estimated bandwidth as a parameter and ``statsmodels`` cannot
  express that convention.

Documented ambiguity (NOT resolved silently)
--------------------------------------------
2.2.2 requires "automatic bandwidth re-estimation" after pre-whitening but does not name the rule.
This module re-applies the 2.2.2 automatic rule to the pre-whitened series, so the numeric effect
of pre-whitening comes from the residual series and the re-colouring factor rather than from a
different bandwidth rule (for the Bartlett kernel the rule depends on :math:`T` only).  Choosing
the Andrews (1991) AR(1) plug-in instead would change :math:`L_{NW}` for strongly autocorrelated
series; that is a specification decision and MUST be recorded in an ADR before it is implemented
here.

Notes
-----
The IC series is serially correlated by construction, so the i.i.d. standard error of
:func:`qresearch.stats.ic.ic_moments` MUST NOT be used for inference.  A pointwise
:math:`t_{NW}` is nevertheless insufficient on its own: 2.2.2 additionally requires the block
bootstrap of 2.2.3 (task ``ST-04``) because the IC series is left-skewed, kurtotic and exhibits
volatility clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "REFERENCE_DISTRIBUTIONS",
    "NeweyWestResult",
    "automatic_bandwidth",
    "bartlett_weights",
    "long_run_variance",
    "newey_west",
    "newey_west_se",
    "nw_t_stat",
    "sample_autocovariances",
]

ReferenceDistribution = Literal["normal", "student_t"]
REFERENCE_DISTRIBUTIONS: Final[tuple[str, ...]] = ("normal", "student_t")

MINIMUM_OBSERVATIONS: Final[int] = 3
"""A variance needs at least two centred observations; three keeps the two-sided test honest."""

AR1_CLIP: Final[float] = 0.99
"""Upper bound on the AR(1) pre-whitening coefficient; :math:`|\\rho| \\to 1` is not invertible."""


@dataclass(frozen=True)
class NeweyWestResult:
    """A frozen record of the HAC estimate and of every option that produced it.

    ``PROJECT_SPEC.md`` 2.2.2 requires the small-sample options to be configurable **and
    reported**, so the effective configuration travels with the numbers instead of being an
    implicit property of the caller.
    """

    n_obs: int
    mean: float
    variance: float
    standard_error: float
    t_stat: float
    p_value: float
    bandwidth: int
    bandwidth_is_automatic: bool
    reference: str
    degrees_of_freedom: float
    small_sample_correction: bool
    k_reg: int
    correction_factor: float
    prewhiten: bool
    ar1_coefficient: float | None

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Return a JSON-serializable representation for the provenance block."""
        return {
            "n_obs": self.n_obs,
            "mean": self.mean,
            "variance": self.variance,
            "standard_error": self.standard_error,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "bandwidth": self.bandwidth,
            "bandwidth_is_automatic": self.bandwidth_is_automatic,
            "reference": self.reference,
            "degrees_of_freedom": self.degrees_of_freedom,
            "small_sample_correction": self.small_sample_correction,
            "k_reg": self.k_reg,
            "correction_factor": self.correction_factor,
            "prewhiten": self.prewhiten,
            "ar1_coefficient": self.ar1_coefficient,
        }

    def format(self) -> str:
        """Return a human-readable report of the estimate and of its effective options."""
        options = [
            f"bandwidth={self.bandwidth}{' (automatic)' if self.bandwidth_is_automatic else ' (fixed)'}",
            f"reference={self.reference}",
            f"small_sample_correction={self.small_sample_correction} (k_reg={self.k_reg}, "
            f"x{self.correction_factor:.6f})",
            f"prewhiten={self.prewhiten}",
        ]
        if self.prewhiten:
            options.append(f"ar1_coefficient={self.ar1_coefficient}")
        lines = [
            f"Newey-West HAC ({self.reference}) on {self.n_obs} periods",
            f"  mean           : {self.mean:.10g}",
            f"  LR variance    : {self.variance:.10g}",
            f"  std. error     : {self.standard_error:.10g}",
            f"  t statistic    : {self.t_stat:.10g}",
            f"  p value        : {self.p_value:.10g} (df={self.degrees_of_freedom})",
            "  options        : " + "; ".join(options),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Primitives - each is one expression of PROJECT_SPEC.md 2.2.2 and is tested in isolation
# ---------------------------------------------------------------------------------------
def automatic_bandwidth(n_obs: int) -> int:
    """Return the Newey-West (1994) automatic bandwidth :math:`L_{NW}`.

    :math:`L_{NW} = \\lfloor 4 (T/100)^{2/9} \\rfloor`, clamped to :math:`T - 1` so that the
    largest admissible lag always has at least one product to average.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    bandwidth = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    return min(bandwidth, n_obs - 1)


def bartlett_weights(bandwidth: int) -> NDArray[np.float64]:
    """Return the Bartlett kernel weights :math:`1 - \\ell/(L_{NW}+1)` for :math:`\\ell = 0..L_{NW}`.

    This is the kernel **as written in 2.2.2** and numerically identical to
    ``statsmodels.stats.sandwich_covariance.weights_bartlett``.  The factor two that multiplies
    every lagged term of the estimator lives in :func:`long_run_variance`, not here, so that each
    function matches its counterpart in the literature one-to-one.

    The weight at :math:`\\ell = 0` is exactly one, which is why the kernel is non-negative
    definite and the resulting long-run variance is guaranteed to be non-negative.
    """
    if bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")
    return 1.0 - np.arange(bandwidth + 1, dtype=np.float64) / (bandwidth + 1.0)


def sample_autocovariances(values: NDArray[np.float64], bandwidth: int) -> NDArray[np.float64]:
    """Return :math:`\\hat{\\gamma}_0, \\dots, \\hat{\\gamma}_{L}`, each divided by ``T``.

    Notes
    -----
    The estimator follows 2.2.2 exactly: the divisor is the **full** sample length :math:`T`, not
    :math:`T - \\ell`.  Using the lag-adjusted divisor would inflate each :math:`\\hat{\\gamma}` by
    :math:`T/(T-\\ell)` and break the numerical agreement with ``statsmodels``.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")
    n_obs = int(values.size)
    if bandwidth > n_obs - 1:
        raise ValueError(
            f"bandwidth {bandwidth} exceeds the {max(n_obs - 1, 0)} admissible lags of a length-{n_obs} series"
        )
    centred = values - values.mean()
    covariances = np.empty(bandwidth + 1, dtype=np.float64)
    covariances[0] = float(centred @ centred) / n_obs
    for lag in range(1, bandwidth + 1):
        covariances[lag] = float(centred[lag:] @ centred[:-lag]) / n_obs
    return covariances


def long_run_variance(values: NDArray[np.float64], bandwidth: int) -> float:
    """Return :math:`\\widehat{\\sigma}^{\\,2}_{NW}` for the centred ``values``.

    Implements 2.2.2 literally as :math:`\\hat{\\gamma}_0 + 2\\sum_{\\ell=1}^{L}
    (1 - \\ell/(L+1))\\hat{\\gamma}_\\ell`: the factor two multiplies **every** lagged term and the
    zero-lag term enters exactly once.  Omitting it (or applying it to the zero lag as well)
    silently rescales the standard error - a defect that a comparison against ``statsmodels``
    catches immediately, which is why that comparison is part of the Definition of Done.
    """
    covariances = sample_autocovariances(values, bandwidth)
    weights = bartlett_weights(bandwidth)
    weights[1:] *= 2.0
    return float(weights @ covariances)


def _ar1_prewhiten(centred: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return the AR(1) residuals of a centred series and the estimated coefficient.

    The regression is :math:`u_t = \\rho u_{t-1} + e_t`, estimated by OLS without a further
    constant (the input is already centred), and :math:`\\rho` is clipped to
    :math:`\\pm` :data:`AR1_CLIP` so that the re-colouring factor stays finite.  The first
    observation is consumed by the lag and is therefore not part of the residual series.
    """
    lagged = centred[:-1]
    current = centred[1:]
    denominator = float(lagged @ lagged)
    rho = float(lagged @ current) / denominator if denominator > 0.0 else 0.0
    rho = float(np.clip(rho, -AR1_CLIP, AR1_CLIP))
    return current - rho * lagged, rho


def _two_sided_p_value(t_stat: float, reference: ReferenceDistribution, degrees_of_freedom: float) -> float:
    """Return :math:`2(1 - F(|t|))` for the normal or the Student-t reference distribution."""
    if not np.isfinite(t_stat):
        return float("nan")
    if reference == "student_t":
        return 2.0 * float(stats.t.sf(abs(t_stat), degrees_of_freedom))
    return 2.0 * float(stats.norm.sf(abs(t_stat)))


def newey_west(
    ic: pd.Series,
    *,
    bandwidth: int | None = None,
    small_sample_correction: bool = True,
    k_reg: int | None = None,
    reference: ReferenceDistribution = "normal",
    prewhiten: bool = False,
    min_obs: int = MINIMUM_OBSERVATIONS,
) -> NeweyWestResult:
    """Estimate the HAC variance of an IC series and test its mean (``PROJECT_SPEC.md`` 2.2.2).

    Parameters
    ----------
    ic : pd.Series
        Per-period IC series, e.g. the output of :func:`qresearch.stats.ic.rank_ic`.  Missing
        values are dropped **before** the lags are formed: a single ``NaN`` would otherwise enter
        every :math:`\\hat{\\gamma}_\\ell` it touches and silently deflate the variance.
    bandwidth : int | None
        ``None`` (default) selects :math:`L_{NW}` by the Newey-West (1994) automatic rule; an
        integer fixes it, which is what a reader of a reported table needs in order to reproduce
        the number.
    small_sample_correction : bool
        Apply the optional :math:`\\sqrt{T/(T - k_{reg})}` factor of 2.2.2 (default ``True``,
        matching ``statsmodels``' ``use_correction=True``).
    k_reg : int | None
        Number of estimated parameters used by that factor.  ``None`` follows 2.2.2 literally:
        the mean **plus the bandwidth when the latter was estimated automatically**.
    reference : {"normal", "student_t"}
        Reference distribution of the two-sided p-value: :math:`\\Phi` of 2.2.2, or the
        Student-:math:`t` small-sample option with :math:`\\nu_{df} = T - 1`.
    prewhiten : bool
        Fit an AR(1), apply the kernel to the residuals and re-colour the variance by
        :math:`1/(1-\\hat{\\rho})^2` (Andrews-Monahan 1992).  See the module docstring for the
        bandwidth-re-estimation ambiguity this leaves open.
    min_obs : int
        Minimum number of usable periods.  Below it the statistics are ``NaN`` (a short series is a
        data condition, not a programming error) while ``n_obs`` still reports what was available.

    Returns
    -------
    NeweyWestResult
        Frozen estimate carrying the statistic, the value and every effective option.

    Raises
    ------
    ValueError
        If ``reference`` is unknown, ``min_obs`` is below three, ``bandwidth`` is negative or
        exceeds the admissible lags, or ``k_reg`` is not a positive integer below the sample size.

    Notes
    -----
    A degenerate series (constant IC, hence :math:`\\hat{\\gamma}_0 = 0`) yields ``NaN`` rather
    than a spuriously infinite :math:`t`: zero dispersion means the standard error is
    *undefined*, not zero.
    """
    if reference not in REFERENCE_DISTRIBUTIONS:
        raise ValueError(f"reference must be one of {REFERENCE_DISTRIBUTIONS}, got {reference!r}")
    if min_obs < MINIMUM_OBSERVATIONS:
        raise ValueError(f"min_obs must be at least {MINIMUM_OBSERVATIONS}, got {min_obs}")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")
    if k_reg is not None and k_reg < 1:
        raise ValueError(f"k_reg must be a positive number of parameters, got {k_reg}")

    values = np.asarray(ic.dropna().astype("float64").to_numpy(dtype="float64"), dtype=np.float64)
    n_obs = int(values.size)
    if n_obs < min_obs:
        return NeweyWestResult(
            n_obs=n_obs,
            mean=float(values.mean()) if n_obs else float("nan"),
            variance=float("nan"),
            standard_error=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            bandwidth=0,
            bandwidth_is_automatic=bandwidth is None,
            reference=reference,
            degrees_of_freedom=float("nan"),
            small_sample_correction=small_sample_correction,
            k_reg=k_reg if k_reg is not None else 1,
            correction_factor=float("nan"),
            prewhiten=prewhiten,
            ar1_coefficient=None,
        )

    automatic = bandwidth is None
    centred = values - values.mean()
    ar1_coefficient: float | None = None
    if prewhiten:
        centred, ar1_coefficient = _ar1_prewhiten(centred)
    effective_n = int(centred.size)
    resolved_bandwidth = automatic_bandwidth(effective_n) if automatic else int(bandwidth)  # type: ignore[arg-type]
    variance = long_run_variance(centred, resolved_bandwidth)
    if ar1_coefficient is not None:
        # Re-colour: the residuals e_t = (1 - rho B) u_t satisfy sigma_e^2 = sigma_u^2 (1 - rho)^2.
        variance /= (1.0 - ar1_coefficient) ** 2

    resolved_k_reg = k_reg if k_reg is not None else 1 + (1 if automatic else 0)
    if resolved_k_reg >= effective_n:
        raise ValueError(f"k_reg={resolved_k_reg} must be smaller than the {effective_n} usable observations")
    correction = float(np.sqrt(effective_n / (effective_n - resolved_k_reg))) if small_sample_correction else 1.0

    standard_error = float(np.sqrt(variance / effective_n)) * correction if variance > 0.0 else float("nan")
    mean = float(values.mean())
    t_stat = mean / standard_error if np.isfinite(standard_error) and standard_error > 0.0 else float("nan")
    degrees_of_freedom = float(effective_n - 1)

    return NeweyWestResult(
        n_obs=effective_n,
        mean=mean,
        variance=float(variance),
        standard_error=float(standard_error),
        t_stat=float(t_stat),
        p_value=_two_sided_p_value(float(t_stat), reference, degrees_of_freedom),
        bandwidth=resolved_bandwidth,
        bandwidth_is_automatic=automatic,
        reference=reference,
        degrees_of_freedom=degrees_of_freedom,
        small_sample_correction=small_sample_correction,
        k_reg=resolved_k_reg,
        correction_factor=correction,
        prewhiten=prewhiten,
        ar1_coefficient=ar1_coefficient,
    )


def newey_west_se(
    x: NDArray[np.float64],
    bandwidth: int | None = None,
    prewhite: bool = False,
    small_sample: bool = True,
) -> float:
    """Return the HAC long-run standard error of the sample mean (``PROJECT_SPEC.md`` 3.4.6).

    This is the estimator of 2.2.2 without the test: the entry point for callers that build their own
    statistic (a two-sided confidence interval, a Diebold-Mariano comparison of two IC series, a
    report table).  The arguments are positional because 3.4.6 declares them that way.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional ``float64`` series, typically an IC series.  ``NaN`` entries are dropped
        before the lags are formed, so a missing period removes one observation instead of
        corrupting every :math:`\\hat{\\gamma}_\\ell` it would otherwise enter.
    bandwidth : int | None
        ``None`` (default) applies the Newey-West (1994) automatic rule of 2.2.2; an integer fixes
        :math:`L_{NW}`.  Pass a fixed value when the number must be reproducible from a table.
    prewhite : bool
        AR(1) pre-whitening with re-colouring (option 3 of 2.2.2, Andrews-Monahan 1992).
    small_sample : bool
        Apply the :math:`\\sqrt{T/(T - k_{reg})}` factor (option 2 of 2.2.2), with :math:`k_{reg}`
        counted as 2.2.2 prescribes: the mean always, plus the bandwidth when the bandwidth was
        selected automatically.

    Returns
    -------
    float
        The HAC standard error of the sample mean, or ``NaN`` for a degenerate series (zero
        dispersion) or one shorter than :data:`MINIMUM_OBSERVATIONS`.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional, or if any option is invalid for :func:`newey_west`.

    Notes
    -----
    3.4.6 declares ``small_sample: TestType`` and never defines ``TestType``; the placeholder is
    bound to the boolean correction factor, because option 1 (the Student-:math:`t` reference)
    changes the p-value rather than the standard error this function returns - use
    :func:`newey_west` for that.  See the module docstring.

    The agreement with ``statsmodels`` is at machine precision for a fixed bandwidth: 2.2.2 then
    counts a single parameter, which is the parameter count of ``cov_hac``'s one-column regression,
    and the measured deviation is ``1.7e-15`` relative on the pinned stack - seven orders of
    magnitude inside the ``1e-8`` the Definition of Done requires.  With the automatic bandwidth the
    two differ by exactly the documented :math:`\\sqrt{T/(T-2)}` factor because 2.2.2 counts the
    estimated bandwidth as a parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from qresearch.stats.hac import newey_west_se
    >>> bool(np.isfinite(newey_west_se(np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2]))))
    True
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {values.shape}")
    result = newey_west(
        pd.Series(values, name="ic", dtype="float64"),
        bandwidth=bandwidth,
        small_sample_correction=small_sample,
        prewhiten=prewhite,
    )
    return float(result.standard_error)


def nw_t_stat(
    ic: pd.Series,
    *,
    bandwidth: int | None = None,
    small_sample_correction: bool = True,
    k_reg: int | None = None,
    reference: ReferenceDistribution = "normal",
    prewhiten: bool = False,
    min_obs: int = MINIMUM_OBSERVATIONS,
) -> tuple[float, float, float]:
    """Return ``(t_stat, p_value, bandwidth)`` for an IC series.

    Convenience entry point for downstream tasks and for report tables, which need exactly these
    three numbers; call :func:`newey_west` directly when the effective options must be inspected,
    logged or embedded in a provenance block.
    """
    result = newey_west(
        ic,
        bandwidth=bandwidth,
        small_sample_correction=small_sample_correction,
        k_reg=k_reg,
        reference=reference,
        prewhiten=prewhiten,
        min_obs=min_obs,
    )
    return result.t_stat, result.p_value, result.bandwidth
