"""Synthetic ground-truth panel tests (task ``INF-11``, ``PROJECT_SPEC.md`` 3.8).

``3.8`` asks for "a panel with controllable IC, kurtosis and autocorrelation, plus documented ground
truth", and section 4 makes this generator the shared input of every later statistical test.  These
tests therefore check two different things:

1. **structure** - shapes, index, determinism, the cross-sectional standardisation, the fat tails; and
2. **truth** - the IC computed by the *production* estimator (``ST-01``: :func:`qresearch.stats.ic.rank_ic`,
   :func:`~qresearch.stats.ic.pearson_ic`) agrees with the injected population parameter, and the
   ``ST-02`` Newey-West statistic is consistent with a signal that is genuinely present.

Every tolerance is derived from two reported errors - the panel's own :math:`\\sigma_\\rho/\\sqrt{T}` and
the Monte-Carlo standard error of the conditional expectation - so a failure means the estimator and the
truth disagree beyond their stated precision, not that a hand-tuned constant drifted.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qresearch.data.synthetic import (
    SyntheticPanel,
    SyntheticPanelSpec,
    generate_panel,
    normal_copula_rank_ic,
    per_date_correlation,
)
from qresearch.stats.hac import newey_west
from qresearch.stats.ic import ic_moments, pearson_ic, rank_ic

TOLERANCE_SIGMA = 4.0
"""Tolerance in combined standard errors: the panel error plus the Monte-Carlo error."""

FAST_MONTE_CARLO_DRAWS = 8
"""Draws for the auxiliary panels; the session fixture keeps the full default precision."""


def _combined_error(ic_series: pd.Series, truth: SyntheticPanel) -> float:
    """Return the combined standard error of the mean IC: panel sampling + Monte-Carlo."""
    panel_error = float(ic_series.std(ddof=1) / np.sqrt(ic_series.size))
    monte_carlo_error = float(truth.truth["expected_rank_ic_standard_error"])
    return float(np.hypot(panel_error, monte_carlo_error))


@pytest.mark.unit
def test_panel_structure_follows_the_contract(known_signal_panel: SyntheticPanel) -> None:
    """The index, shape and dtypes are the ones ``3.4.6`` mandates for panel inputs."""
    panel = known_signal_panel
    assert list(panel.label.index.names) == ["datetime", "instrument"]
    assert panel.features.shape == (panel.spec.n_observations, panel.spec.n_features)
    assert list(panel.features.columns) == [f"f{index}" for index in range(panel.spec.n_features)]
    for series in (panel.signal, panel.label, panel.noise):
        assert series.index.equals(panel.features.index)
        assert series.dtype == np.float64
        assert series.notna().all()
    assert panel.features.notna().all().all()
    assert len(panel.signal.index.get_level_values("instrument").unique()) == panel.spec.n_instruments


@pytest.mark.unit
def test_panel_is_deterministic_and_seed_sensitive(synthetic_panel_spec: SyntheticPanelSpec) -> None:
    """Reproducibility is a requirement, not a convenience: the same spec must be bit-identical."""
    first = generate_panel(replace(synthetic_panel_spec, monte_carlo_draws=2))
    second = generate_panel(replace(synthetic_panel_spec, monte_carlo_draws=2))
    assert first.label.equals(second.label)
    assert first.signal.equals(second.signal)
    assert first.truth["expected_rank_ic"] == second.truth["expected_rank_ic"]

    other_seed = generate_panel(replace(synthetic_panel_spec, seed=synthetic_panel_spec.seed + 1, monte_carlo_draws=2))
    assert not other_seed.label.equals(first.label)


@pytest.mark.unit
def test_signal_is_standardised_within_every_cross_section(known_signal_panel: SyntheticPanel) -> None:
    """``sigma_s = 1`` per date is what makes the closed-form correlation exact."""
    by_date = known_signal_panel.signal.groupby(level="datetime")
    assert float(by_date.mean().abs().max()) < 1e-12
    assert float((by_date.std(ddof=0) - 1.0).abs().max()) < 1e-10


@pytest.mark.unit
def test_label_is_the_injected_signal_plus_noise(known_signal_panel: SyntheticPanel) -> None:
    """The label must be literally ``beta * signal + noise``: the DGP is inspectable, not implied."""
    panel = known_signal_panel
    reconstructed = panel.spec.beta * panel.signal + panel.noise
    assert np.allclose(reconstructed.to_numpy(), panel.label.to_numpy(), rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_rank_ic_matches_the_injected_truth(known_signal_panel: SyntheticPanel) -> None:
    """The ``ST-01`` estimator must reproduce the population parameter the DGP injected."""
    panel = known_signal_panel
    ic = rank_ic(panel.signal, panel.label)
    expected = float(panel.truth["expected_rank_ic"])
    assert ic.mean() == pytest.approx(expected, abs=TOLERANCE_SIGMA * _combined_error(ic, panel))
    assert panel.truth["population_correlation"] == panel.spec.target_correlation
    assert expected == pytest.approx(panel.spec.target_correlation, abs=0.01)


@pytest.mark.unit
def test_pearson_ic_matches_the_injected_truth(known_signal_panel: SyntheticPanel) -> None:
    """The Pearson IC is reported alongside (2.2.1); it must agree with the same truth."""
    panel = known_signal_panel
    ic = pearson_ic(panel.signal, panel.label)
    expected = float(panel.truth["expected_pearson_ic"])
    panel_error = float(ic.std(ddof=1) / np.sqrt(ic.size))
    monte_carlo_error = float(panel.truth["expected_pearson_ic_standard_error"])
    assert ic.mean() == pytest.approx(expected, abs=TOLERANCE_SIGMA * np.hypot(panel_error, monte_carlo_error))


@pytest.mark.unit
def test_injected_alpha_is_detectable_by_the_newey_west_statistic(known_signal_panel: SyntheticPanel) -> None:
    """An injected IC of 0.05 over 1000 decisions must be significant - under the HAC error, not the i.i.d. one."""
    panel = known_signal_panel
    ic = rank_ic(panel.signal, panel.label)
    result = newey_west(ic)
    moments = ic_moments(ic)
    iid_error = moments.std / np.sqrt(moments.n_periods)

    assert result.t_stat > 4.0
    assert result.p_value < 1e-4
    assert result.standard_error > iid_error, "the IC series is autocorrelated: the i.i.d. error understates it"
    assert float(panel.truth["ic_autocorrelation_lag1"]) > 0.1, "the DGP must make the IC series autocorrelated"


@pytest.mark.unit
def test_vectorised_ic_agrees_with_the_production_estimator(known_signal_panel: SyntheticPanel) -> None:
    """The generator's fast primitive and ``ST-01``'s estimator must compute the same series."""
    panel = known_signal_panel
    production = rank_ic(panel.signal, panel.label)
    vectorised = panel.ic_series
    assert np.allclose(production.to_numpy(), vectorised.reindex(production.index).to_numpy(), rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_null_panel_is_indistinguishable_from_no_signal(synthetic_panel_spec: SyntheticPanelSpec) -> None:
    """The other half of a size study: with ``rho = 0`` the IC must not be significant."""
    null_spec = replace(synthetic_panel_spec, target_correlation=0.0, monte_carlo_draws=FAST_MONTE_CARLO_DRAWS)
    panel = generate_panel(null_spec)
    ic = rank_ic(panel.signal, panel.label)
    result = newey_west(ic)

    assert abs(ic.mean()) <= TOLERANCE_SIGMA * _combined_error(ic, panel)
    assert abs(result.t_stat) < 3.0
    assert result.p_value > 0.05
    assert abs(float(panel.truth["expected_rank_ic"])) < 0.01


@pytest.mark.unit
def test_fat_tails_are_injected_and_controllable(synthetic_panel_spec: SyntheticPanelSpec) -> None:
    """The Student-t knob must actually thicken the tails, and the Gaussian control must be thin."""
    heavy = generate_panel(replace(synthetic_panel_spec, monte_carlo_draws=0))
    gaussian = generate_panel(replace(synthetic_panel_spec, noise_df=1.0e6, monte_carlo_draws=0))

    assert float(heavy.truth["measured_noise_excess_kurtosis"]) > 1.0
    assert float(heavy.truth["measured_label_excess_kurtosis"]) > 1.0
    assert float(gaussian.truth["measured_noise_excess_kurtosis"]) < 0.2
    assert float(gaussian.truth["measured_label_excess_kurtosis"]) < 0.2
    assert float(gaussian.truth["noise_innovation_excess_kurtosis"]) < 1e-4

    nu = synthetic_panel_spec.noise_df
    assert float(heavy.truth["noise_innovation_excess_kurtosis"]) == pytest.approx(6.0 / (nu - 4.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    "variant",
    [
        {"feature_kind": "random_walk"},
        {"signal_kind": "nonlinear"},
        {"noise_ar1_rho": 0.0},
        {"noise_df": 30.0},
        {"target_correlation": 0.02},
    ],
)
def test_alternative_configurations_keep_a_valid_truth(
    synthetic_panel_spec: SyntheticPanelSpec, variant: dict[str, object]
) -> None:
    """The knobs must produce panels whose estimator still agrees with the documented truth."""
    spec = replace(synthetic_panel_spec, monte_carlo_draws=FAST_MONTE_CARLO_DRAWS, **variant)
    panel = generate_panel(spec)
    ic = rank_ic(panel.signal, panel.label)
    expected = float(panel.truth["expected_rank_ic"])
    assert ic.mean() == pytest.approx(expected, abs=TOLERANCE_SIGMA * _combined_error(ic, panel))
    assert float(panel.truth["population_correlation"]) == spec.target_correlation


@pytest.mark.unit
def test_specification_rejects_unusable_knobs() -> None:
    """An impossible DGP must fail loudly at construction time."""
    for kwargs in (
        {"n_instruments": 2},
        {"n_days": 2},
        {"n_features": 0},
        {"feature_kind": "ou"},
        {"feature_ar1_rho": 1.5},
        {"signal_kind": "spline"},
        {"signal_kind": "nonlinear", "n_features": 2},
        {"target_correlation": 1.0},
        {"target_correlation": -0.1},
        {"noise_df": 2.0},
        {"noise_ar1_rho": 1.0},
        {"min_obs": 1},
        {"seed": -1},
        {"monte_carlo_draws": -1},
    ):
        with pytest.raises(ValueError):
            SyntheticPanelSpec(**kwargs)


@pytest.mark.unit
def test_normal_copula_conversion_is_the_closed_form() -> None:
    """The reference conversion must match ``(6/pi) arcsin(rho/2)``, including its known values."""
    assert normal_copula_rank_ic(0.05) == pytest.approx(6.0 / np.pi * np.arcsin(0.025))
    assert normal_copula_rank_ic(0.05) == pytest.approx(0.047751457918883584)
    assert normal_copula_rank_ic(0.0) == 0.0
    for bad in (-1.0, 1.0, 2.0):
        with pytest.raises(ValueError):
            normal_copula_rank_ic(bad)


@pytest.mark.unit
def test_per_date_correlation_matches_scipy_on_the_same_cross_sections() -> None:
    """The vectorised primitive is cross-checked against ``scipy`` on identical data."""
    from scipy import stats

    rng = np.random.default_rng(11)
    signal = rng.standard_normal((7, 25))
    label = 0.4 * signal + rng.standard_normal((7, 25))
    for method, reference in (("spearman", stats.spearmanr), ("pearson", stats.pearsonr)):
        computed = per_date_correlation(signal, label, method=method)
        expected = np.array([reference(signal[row], label[row]).statistic for row in range(signal.shape[0])])
        assert np.allclose(computed, expected, rtol=0.0, atol=1e-12)

    with pytest.raises(ValueError, match="2-D shape"):
        per_date_correlation(signal, label[:, :5])
    with pytest.raises(ValueError, match="method"):
        per_date_correlation(signal, label, method="kendall")


@pytest.mark.unit
def test_panel_summary_is_serializable(known_signal_panel: SyntheticPanel) -> None:
    """The summary is what a report or a tracker entry quotes; it must be JSON-friendly."""
    summary = known_signal_panel.to_dict()
    assert summary["n_observations"] == known_signal_panel.spec.n_observations
    assert summary["columns"] == ["f0", "f1", "f2"]
    assert "expected_rank_ic" in summary["truth"]
    import json

    assert json.loads(json.dumps(summary, default=str))["spec"]["seed"] == known_signal_panel.spec.seed
