"""Tests for the Newey-West HAC estimator (task ``ST-02``).

The Definition of Done for ``ST-02`` requires the unit tests to match ``statsmodels`` within
``1e-8``, and ``PROJECT_SPEC.md`` 2.2.2 requires the small-sample options to be configurable *and*
reported.  The tests therefore do three things:

1. **Re-derive** every primitive independently of the implementation - the long-run variance is
   expanded from the 2.2.2 formula in the test body, and the bandwidth / kernel / correction
   identities are asserted against closed forms.
2. **Cross-check** the full statistic against ``statsmodels`` at the documented call site
   (``OLS.fit(cov_type="HAC", cov_kwds={"maxlags": L, ...})`` and
   ``sandwich_covariance.cov_hac``) at machine precision, and the p-values against ``scipy``.
3. **Pin the behaviour** that the specification motivates: the HAC standard error must exceed the
   i.i.d. one on an autocorrelated series, a degenerate series must yield ``NaN`` rather than a
   spuriously infinite ``t``, and every effective option must be reported.

Two facts established empirically on the pinned ``statsmodels==0.14.2`` are encoded here, because
a cross-check is only meaningful if the oracle's own conventions are known:

* ``OLS.fit(cov_type="HAC")`` requires an explicit ``maxlags`` (otherwise ``KeyError``) and its
  p-value uses the **normal** distribution (``use_t=False``), i.e. 2.2.2's default reference.
* ``cov_hac`` is an alias of ``cov_hac_simple``: ``nlags=None`` means ``floor[4(T/100)^{2/9}]``,
  ``weights_bartlett(L) = 1 - arange(L+1)/(L+1)`` and ``use_correction=True`` scales the
  covariance by ``T/(T - k_params)`` with ``k_params`` the number of regressors (one).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats import sandwich_covariance as sw

from qresearch.stats.hac import (
    AR1_CLIP,
    MINIMUM_OBSERVATIONS,
    NeweyWestResult,
    automatic_bandwidth,
    bartlett_weights,
    long_run_variance,
    newey_west,
    newey_west_se,
    nw_t_stat,
    sample_autocovariances,
)

REL_TOLERANCE: float = 1e-8
"""The ``statsmodels`` tolerance of the ``ST-02`` Definition of Done."""

TOLERANCE: float = 1e-12
"""Tolerance for comparisons against closed forms and ``scipy`` oracles."""


def ar1_path(n_obs: int, rho: float, *, seed: int, sigma: float = 1.0, burn_in: int = 250) -> np.ndarray:
    """Return a stationary AR(1) path ``u_t = rho * u_{t-1} + e_t`` with the burn-in discarded.

    The path is deterministic under ``seed``, so every assertion below is reproducible; the
    long-run variance of the process, ``sigma**2 / (1 - rho)**2``, is closed-form ground truth.
    """
    rng = np.random.default_rng(seed)
    errors = rng.normal(scale=sigma, size=n_obs + burn_in)
    path = np.zeros(n_obs + burn_in)
    for step in range(1, n_obs + burn_in):
        path[step] = rho * path[step - 1] + errors[step]
    return path[burn_in:]


@pytest.fixture(scope="module")
def autocorrelated_ic() -> pd.Series:
    """Return a length-500 AR(1) IC series with ``rho = 0.3`` and unit innovation variance."""
    return pd.Series(ar1_path(500, 0.3, seed=7), name="ic")


@pytest.mark.unit
@pytest.mark.parametrize("n_obs", [3, 12, 50, 100, 200, 500, 1000, 5000])
def test_automatic_bandwidth_matches_the_specification_rule(n_obs: int) -> None:
    """``L_NW`` must be ``floor(4 (T/100)^(2/9))``, clamped to ``T - 1`` (PROJECT_SPEC.md 2.2.2)."""
    expected = min(int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))), n_obs - 1)
    assert automatic_bandwidth(n_obs) == expected
    assert 0 <= automatic_bandwidth(n_obs) <= n_obs - 1

    with pytest.raises(ValueError, match="n_obs must be positive"):
        automatic_bandwidth(0)


@pytest.mark.unit
def test_automatic_bandwidth_equals_the_statsmodels_default() -> None:
    """``cov_hac(nlags=None)`` documents the same rule, so both must select the same lag."""
    series = pd.Series(ar1_path(240, 0.4, seed=13))
    fit = sm.OLS(series.to_numpy(), np.ones((series.size, 1))).fit()
    implicit = sw.cov_hac(fit, nlags=None, use_correction=True)[0, 0]
    explicit = sw.cov_hac(fit, nlags=automatic_bandwidth(series.size), use_correction=True)[0, 0]
    assert implicit == pytest.approx(explicit, rel=TOLERANCE, abs=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("bandwidth", [0, 1, 4, 9, 25])
def test_bartlett_weights_match_the_statsmodels_kernel(bandwidth: int) -> None:
    """The kernel must be ``1 - l/(L+1)``, identical to ``weights_bartlett``."""
    ours = bartlett_weights(bandwidth)
    assert ours.size == bandwidth + 1
    assert ours[0] == 1.0
    assert np.allclose(ours, sw.weights_bartlett(bandwidth), rtol=0.0, atol=0.0)
    assert np.allclose(ours, 1.0 - np.arange(bandwidth + 1) / (bandwidth + 1.0), rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="bandwidth must be non-negative"):
        bartlett_weights(-1)


@pytest.mark.unit
def test_sample_autocovariances_match_a_direct_sum() -> None:
    """Each ``gamma_l`` must be ``(1/T) * sum_t u_t u_{t-l}`` on the centred series."""
    values = np.asarray(ar1_path(80, 0.5, seed=21), dtype=np.float64)
    centred = values - values.mean()
    covariances = sample_autocovariances(values, 6)

    assert covariances.size == 7
    for lag in range(7):
        expected = sum(centred[t] * centred[t - lag] for t in range(lag, centred.size)) / centred.size
        assert covariances[lag] == pytest.approx(expected, rel=TOLERANCE, abs=0.0)

    with pytest.raises(ValueError, match="one-dimensional"):
        sample_autocovariances(np.ones((4, 4)), 1)
    with pytest.raises(ValueError, match="exceeds the"):
        sample_autocovariances(values, values.size)


@pytest.mark.unit
def test_long_run_variance_equals_the_expanded_specification() -> None:
    """Re-derive 2.2.2 in the test: ``gamma_0 + 2 * sum_l (1 - l/(L+1)) gamma_l``.

    This is the regression guard for the defect this test suite actually caught: applying the
    Bartlett kernel without the factor two on the lagged terms (as ``weights_bartlett`` returns
    them) understates the variance by tens of percent.
    """
    values = np.asarray(ar1_path(300, 0.45, seed=5), dtype=np.float64)
    bandwidth = 6
    covariances = sample_autocovariances(values, bandwidth)
    doubling = 2.0 * sum((1.0 - lag / (bandwidth + 1.0)) * covariances[lag] for lag in range(1, bandwidth + 1))
    expected = covariances[0] + doubling
    assert long_run_variance(values, bandwidth) == pytest.approx(expected, rel=TOLERANCE, abs=0.0)

    un_doubled = covariances[0] + doubling / 2.0
    assert long_run_variance(values, bandwidth) != pytest.approx(un_doubled, rel=1e-6, abs=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("n_obs", [60, 200, 1000])
def test_newey_west_matches_statsmodels_within_1e_8(n_obs: int) -> None:
    """The ``ST-02`` Definition of Done: agree with ``statsmodels`` to ``1e-8``.

    ``k_reg=1`` is passed because ``statsmodels``' ``use_correction`` counts only the regressors
    of the regression (the constant), whereas 2.2.2 counts the bandwidth as an estimated parameter
    as well - see :func:`test_small_sample_correction_identities`.
    """
    values = ar1_path(n_obs, 0.4, seed=n_obs)
    bandwidth = automatic_bandwidth(n_obs)
    fit = sm.OLS(values, np.ones((n_obs, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": bandwidth, "use_correction": True, "kernel": "bartlett"}
    )
    ours = newey_west(pd.Series(values), k_reg=1)

    assert ours.bandwidth == bandwidth
    assert ours.bandwidth_is_automatic is True
    assert ours.standard_error == pytest.approx(float(fit.bse[0]), rel=REL_TOLERANCE, abs=0.0)
    assert ours.t_stat == pytest.approx(float(fit.tvalues[0]), rel=REL_TOLERANCE, abs=0.0)
    assert ours.p_value == pytest.approx(float(fit.pvalues[0]), rel=REL_TOLERANCE, abs=0.0)


@pytest.mark.unit
def test_newey_west_matches_the_public_cov_hac_function() -> None:
    """``sandwich_covariance.cov_hac`` (the function the task names) must agree as well."""
    n_obs = 320
    values = ar1_path(n_obs, 0.35, seed=99)
    bandwidth = 7
    fit = sm.OLS(values, np.ones((n_obs, 1))).fit()
    direct = float(sw.cov_hac(fit, nlags=bandwidth, use_correction=True)[0, 0])

    ours = newey_west(pd.Series(values), bandwidth=bandwidth, k_reg=1)
    assert ours.standard_error**2 == pytest.approx(direct, rel=REL_TOLERANCE, abs=0.0)

    uncorrected = float(sw.cov_hac(fit, nlags=bandwidth, use_correction=False)[0, 0])
    raw = newey_west(pd.Series(values), bandwidth=bandwidth, small_sample_correction=False)
    assert raw.standard_error**2 == pytest.approx(uncorrected, rel=REL_TOLERANCE, abs=0.0)


@pytest.mark.unit
def test_p_value_reference_distributions_match_scipy(autocorrelated_ic: pd.Series) -> None:
    """``normal`` must be ``2(1-Phi)`` of 2.2.2; ``student_t`` must use ``nu_df = T - 1``."""
    normal = newey_west(autocorrelated_ic, reference="normal")
    student = newey_west(autocorrelated_ic, reference="student_t")

    assert normal.p_value == pytest.approx(2.0 * float(stats.norm.sf(abs(normal.t_stat))), rel=TOLERANCE, abs=0.0)
    assert student.p_value == pytest.approx(
        2.0 * float(stats.t.sf(abs(student.t_stat), student.n_obs - 1)), rel=TOLERANCE, abs=0.0
    )
    assert student.degrees_of_freedom == student.n_obs - 1
    assert student.p_value > normal.p_value, "thicker tails must yield a larger two-sided p-value"


@pytest.mark.unit
def test_hac_standard_error_exceeds_the_iid_standard_error(autocorrelated_ic: pd.Series) -> None:
    """The rationale of 2.2.2: on a positively autocorrelated IC series the i.i.d. error is biased down."""
    result = newey_west(autocorrelated_ic, small_sample_correction=False)
    iid_error = float(autocorrelated_ic.std(ddof=1) / np.sqrt(autocorrelated_ic.size))

    assert result.bandwidth > 0
    assert result.standard_error > iid_error
    assert result.standard_error / iid_error > 1.1, "the observed inflation on this path is about 1.39"


@pytest.mark.unit
def test_small_sample_correction_identities(autocorrelated_ic: pd.Series) -> None:
    """The correction must be ``sqrt(T/(T - k_reg))`` and ``k_reg`` must follow 2.2.2."""
    n_obs = autocorrelated_ic.size
    raw = newey_west(autocorrelated_ic, small_sample_correction=False)
    one = newey_west(autocorrelated_ic, k_reg=1)
    two = newey_west(autocorrelated_ic, k_reg=2)

    assert raw.correction_factor == 1.0
    assert one.correction_factor == pytest.approx(float(np.sqrt(n_obs / (n_obs - 1))), rel=TOLERANCE)
    assert one.standard_error / raw.standard_error == pytest.approx(float(np.sqrt(n_obs / (n_obs - 1))), rel=TOLERANCE)
    assert two.standard_error / one.standard_error == pytest.approx(
        float(np.sqrt((n_obs - 1) / (n_obs - 2))), rel=TOLERANCE
    )
    assert (one.k_reg, two.k_reg) == (1, 2)

    automatic = newey_west(autocorrelated_ic)
    assert automatic.bandwidth_is_automatic is True
    assert automatic.k_reg == 2, "2.2.2 counts the mean and the automatically selected bandwidth"

    fixed = newey_west(autocorrelated_ic, bandwidth=5)
    assert fixed.bandwidth_is_automatic is False
    assert fixed.bandwidth == 5
    assert fixed.k_reg == 1, "a fixed bandwidth is not an estimated parameter"


@pytest.mark.unit
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.8])
def test_prewhitening_recovers_the_ar1_coefficient(rho: float) -> None:
    """The pre-whitening step must estimate the process it claims to undo."""
    series = pd.Series(ar1_path(2000, rho, seed=17))
    result = newey_west(series, prewhiten=True)

    assert result.prewhiten is True
    assert result.ar1_coefficient is not None
    assert abs(result.ar1_coefficient - rho) <= 0.05
    assert abs(result.ar1_coefficient) <= AR1_CLIP


@pytest.mark.unit
def test_prewhitening_helps_in_the_strongly_autocorrelated_regime() -> None:
    """Ground truth: the long-run variance of an AR(1) is ``sigma_e^2/(1-rho)^2``.

    Pre-whitening is a bias-reduction device for the regime the automatic bandwidth cannot reach:
    at ``rho = 0.8`` the rule gives ``L_NW = 9``, far short of the lags that carry the dependence.
    The test is deliberately asymmetric, and its second half documents the *absence* of a claim:
    for ``rho = 0.5`` the plain Bartlett estimator is already within a fraction of a percent, and
    pre-whitening then costs about 16 % of accuracy.  No general superiority is asserted.
    """
    strong = pd.Series(ar1_path(4000, 0.8, seed=11, burn_in=500))
    truth_strong = 1.0 / (1.0 - 0.8) ** 2
    plain_strong = newey_west(strong, small_sample_correction=False)
    pre_strong = newey_west(strong, small_sample_correction=False, prewhiten=True)
    assert abs(pre_strong.variance - truth_strong) < abs(plain_strong.variance - truth_strong)

    mild = pd.Series(ar1_path(4000, 0.5, seed=11, burn_in=500))
    truth_mild = 1.0 / (1.0 - 0.5) ** 2
    plain_mild = newey_west(mild, small_sample_correction=False)
    pre_mild = newey_west(mild, small_sample_correction=False, prewhiten=True)
    assert abs(plain_mild.variance - truth_mild) < 0.05 * truth_mild
    assert abs(pre_mild.variance - truth_mild) < 0.25 * truth_mild


@pytest.mark.unit
def test_missing_values_are_dropped_before_the_lags_are_formed(autocorrelated_ic: pd.Series) -> None:
    """A ``NaN`` must be removed, not propagated into every autocovariance it enters."""
    with_gaps = autocorrelated_ic.copy()
    with_gaps.iloc[[3, 7, 40]] = np.nan
    retained = np.asarray(with_gaps.dropna().to_numpy(dtype="float64"), dtype=np.float64)

    result = newey_west(with_gaps)
    assert result.n_obs == retained.size == autocorrelated_ic.size - 3
    assert result.variance == pytest.approx(
        long_run_variance(retained, automatic_bandwidth(retained.size)), rel=TOLERANCE, abs=0.0
    )


@pytest.mark.unit
def test_degenerate_inputs_return_nan_instead_of_infinities() -> None:
    """Zero dispersion means the standard error is undefined, not zero (2.2.2 rationale)."""
    constant = newey_west(pd.Series([0.05] * 100))
    assert constant.variance == 0.0
    assert np.isnan(constant.standard_error)
    assert np.isnan(constant.t_stat)
    assert np.isnan(constant.p_value)
    assert constant.mean == pytest.approx(0.05)

    short = newey_west(pd.Series([0.1, 0.2]))
    assert short.n_obs == 2
    assert short.bandwidth == 0
    assert np.isnan(short.t_stat)
    assert "n_obs" in short.to_dict()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference": "normalish"}, "reference must be one of"),
        ({"min_obs": MINIMUM_OBSERVATIONS - 1}, "min_obs must be at least"),
        ({"bandwidth": -1}, "bandwidth must be non-negative"),
        ({"k_reg": 0}, "k_reg must be a positive"),
        ({"k_reg": 10_000}, "must be smaller than the"),
        ({"bandwidth": 10_000}, "exceeds the"),
    ],
)
def test_invalid_arguments_are_rejected(autocorrelated_ic: pd.Series, kwargs: dict[str, object], message: str) -> None:
    """Contract violations raise ``ValueError`` with a diagnostic (never a silent fallback)."""
    with pytest.raises(ValueError, match=message):
        newey_west(autocorrelated_ic, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_nw_t_stat_returns_the_documented_triple(autocorrelated_ic: pd.Series) -> None:
    """``(t_stat, p_value, bandwidth)`` is the interface downstream report tables rely on."""
    triple = nw_t_stat(autocorrelated_ic)
    assert isinstance(triple, tuple)
    assert len(triple) == 3

    t_stat, p_value, bandwidth = triple
    result = newey_west(autocorrelated_ic)
    assert (t_stat, p_value, bandwidth) == (result.t_stat, result.p_value, result.bandwidth)
    assert bandwidth == automatic_bandwidth(autocorrelated_ic.size)
    assert 0.0 <= p_value <= 1.0


@pytest.mark.unit
def test_result_is_frozen_and_reports_every_effective_option(autocorrelated_ic: pd.Series) -> None:
    """2.2.2 requires the small-sample options to be reported, not merely applied."""
    result: NeweyWestResult = newey_west(autocorrelated_ic, prewhiten=True, reference="student_t")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_obs = 1  # type: ignore[misc]  # frozen dataclass

    assert set(result.to_dict()) == {
        "n_obs",
        "mean",
        "variance",
        "standard_error",
        "t_stat",
        "p_value",
        "bandwidth",
        "bandwidth_is_automatic",
        "reference",
        "degrees_of_freedom",
        "small_sample_correction",
        "k_reg",
        "correction_factor",
        "prewhiten",
        "ar1_coefficient",
    }
    text = result.format()
    assert "bandwidth=" in text
    assert "reference=student_t" in text
    assert f"k_reg={result.k_reg}" in text
    assert "prewhiten=True" in text
    assert result.to_dict()["ar1_coefficient"] is not None


@pytest.mark.unit
@pytest.mark.parametrize("bandwidth", [0, 1, 4, 12])
def test_newey_west_se_reproduces_statsmodels_within_1e_8(bandwidth: int) -> None:
    """``PROJECT_SPEC.md`` 3.4.6: the array entry point must match ``statsmodels`` to ``1e-8``.

    A **fixed** bandwidth is what makes the comparison direct: 2.2.2 then counts a single estimated
    parameter, the same count as ``cov_hac``'s constant-only regression, so the two agree to machine
    precision.  The square is compared because that is the quantity ``cov_hac`` returns, and the
    quantity every other cross-check in this file compares.
    """
    n_obs = 500
    values = ar1_path(n_obs, 0.35, seed=23)
    fit = sm.OLS(values, np.ones((n_obs, 1))).fit()

    corrected = float(sw.cov_hac(fit, nlags=bandwidth, use_correction=True)[0, 0])
    uncorrected = float(sw.cov_hac(fit, nlags=bandwidth, use_correction=False)[0, 0])
    assert newey_west_se(values, bandwidth=bandwidth) ** 2 == pytest.approx(corrected, rel=REL_TOLERANCE, abs=0.0)
    assert newey_west_se(values, bandwidth=bandwidth, small_sample=False) ** 2 == pytest.approx(
        uncorrected, rel=REL_TOLERANCE, abs=0.0
    )


@pytest.mark.unit
def test_newey_west_se_rescales_the_oracle_when_the_bandwidth_is_automatic() -> None:
    """With ``bandwidth=None`` 2.2.2 counts the bandwidth, so the oracle needs ``T/(T-2)``.

    ``cov_hac(use_correction=True)`` cannot express that convention, hence the explicit rescaling of
    its ``use_correction=False`` value: the assertion then pins both the oracle and the
    ``k_reg = 2`` decision of 2.2.2, instead of pinning one of them by accident.
    """
    n_obs = 400
    values = ar1_path(n_obs, 0.45, seed=31)
    fit = sm.OLS(values, np.ones((n_obs, 1))).fit()
    bandwidth = automatic_bandwidth(n_obs)
    uncorrected = float(sw.cov_hac(fit, nlags=bandwidth, use_correction=False)[0, 0])

    assert bandwidth == 5
    assert newey_west_se(values) ** 2 == pytest.approx(uncorrected * n_obs / (n_obs - 2), rel=REL_TOLERANCE, abs=0.0)


@pytest.mark.unit
def test_newey_west_se_is_the_standard_error_of_the_documented_result(autocorrelated_ic: pd.Series) -> None:
    """Option for option, the array entry point must equal :func:`newey_west`'s ``standard_error``."""
    values = autocorrelated_ic.to_numpy(dtype="float64")

    assert newey_west_se(values) == newey_west(autocorrelated_ic).standard_error
    assert newey_west_se(values, prewhite=True) == newey_west(autocorrelated_ic, prewhiten=True).standard_error
    assert newey_west_se(values, 5) == newey_west(autocorrelated_ic, bandwidth=5).standard_error
    assert (
        newey_west_se(values, 5, False, False)
        == newey_west(autocorrelated_ic, bandwidth=5, small_sample_correction=False).standard_error
    )


@pytest.mark.unit
def test_newey_west_se_handles_gaps_and_rejects_non_vectors(autocorrelated_ic: pd.Series) -> None:
    """``NaN`` periods are dropped and the shape contract is enforced (3.4.6)."""
    values = autocorrelated_ic.to_numpy(dtype="float64")
    gapped = values.copy()
    gapped[[3, 7, 40]] = np.nan

    assert newey_west_se(gapped) == newey_west_se(np.delete(values, [3, 7, 40]))
    assert np.isnan(newey_west_se(values[:2]))
    with pytest.raises(ValueError, match="one-dimensional"):
        newey_west_se(values.reshape(25, -1))
