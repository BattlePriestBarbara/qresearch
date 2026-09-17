"""Mask and alignment tests for the tensor assembler (task ``INF-07``).

The requirement under test is negative in form: **nothing may be dropped**.  A suspended name, a
delisted name or a missing fundamental must appear in the bundle with ``mask == False`` and a neutral
fill value, so that the sample is *excluded from the loss* without disappearing from the panel.  The
tests therefore assert both halves:

* the invalid rows are still present, and their component masks say why;
* the neutral fill cannot be mistaken for data (``step_mask`` is ``False`` exactly there);
* the reference reduction :func:`masked_mean` excludes them, and reports ``nan`` - not ``0.0`` - when
  nothing is valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qresearch.data.handlers import build_sequence_bundle, masked_mean
from qresearch.data.labels import LabelSpec, forward_return, label_availability
from qresearch.data.universe import TradabilityRules, build_tradability_mask

N_DATES = 60
LOOKBACK = 5
SUSPENSION_SLICE = slice(21, 28)
LIMIT_LOCK_INDEX = 40
MISSING_CELL_DATE = 35
FILL_VALUE = -123.0
INSTRUMENTS = ["A", "B", "C"]
FEATURE_NAMES = ("f_beta", "f_close", "f_volume")


@pytest.fixture()
def market() -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Synthetic OHLCV panels with a suspension, a limit-locked day and a missing feature cell."""
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 + rng.normal(scale=0.4, size=(N_DATES, len(INSTRUMENTS))).cumsum(axis=0),
        index=dates,
        columns=INSTRUMENTS,
    )
    volume = pd.DataFrame(1.0e6, index=dates, columns=INSTRUMENTS)
    # a suspension: no volume for seven sessions, starting one day after a tradable date
    volume.loc[dates[SUSPENSION_SLICE], "B"] = 0.0
    # a limit-locked session: |move| >= 9.5% on the close-to-close basis
    close.loc[dates[LIMIT_LOCK_INDEX], "C"] = close.loc[dates[LIMIT_LOCK_INDEX - 1], "C"] * 1.099
    # a pure data gap in a feature that does NOT drive tradability
    beta = pd.DataFrame(rng.normal(size=(N_DATES, len(INSTRUMENTS))), index=dates, columns=INSTRUMENTS)
    beta.loc[dates[MISSING_CELL_DATE], "A"] = np.nan
    open_ = close.shift(1).bfill()
    panels = {"open": open_, "close": close, "volume": volume, "beta": beta}
    return dates, panels


@pytest.fixture()
def label_spec() -> LabelSpec:
    """Canonical target: signal at t, execution at the open of t+1, one-day measurement."""
    return LabelSpec(horizon=1, execution_lag=1, winsorize=None)


@pytest.fixture()
def bundle_and_inputs(market, label_spec):
    """Assemble the bundle together with the inputs the assertions need."""
    dates, panels = market
    labels = forward_return(panels["open"], label_spec)
    tradability = build_tradability_mask(
        panels, dates, rules=TradabilityRules(min_listed_days=5, require_volume=True, limit_threshold=0.095)
    )
    availability = label_availability(labels, label_spec, panels["open"], tradability=tradability)
    features = {"f_close": panels["close"], "f_volume": np.log1p(panels["volume"]), "f_beta": panels["beta"]}
    bundle = build_sequence_bundle(
        features,
        labels,
        lookback=LOOKBACK,
        fill_value=FILL_VALUE,
        provided_mask=availability,
        tradability=tradability,
    )
    return bundle, labels, availability, tradability, dates, panels


@pytest.mark.unit
def test_no_rows_are_dropped(bundle_and_inputs) -> None:
    """Every (date, instrument) pair survives; unavailable labels keep their row."""
    bundle, labels, _, _, dates, _ = bundle_and_inputs
    assert bundle.n_samples == len(dates) * len(INSTRUMENTS) == len(labels)
    assert np.isnan(bundle.y).any(), "the fixture must contain unavailable labels for this test to bite"
    assert bundle.n_valid < bundle.n_samples, "some rows must be invalid, otherwise nothing is tested"


@pytest.mark.unit
def test_tensor_shapes_and_dtypes(bundle_and_inputs) -> None:
    """The declared contract of ``PROJECT_SPEC.md`` 3.4.3 holds exactly."""
    bundle, _, _, _, dates, _ = bundle_and_inputs
    n_samples = len(dates) * len(INSTRUMENTS)
    assert bundle.X.shape == (n_samples, LOOKBACK, len(FEATURE_NAMES))
    assert bundle.X.dtype == np.float32
    assert bundle.y.shape == (n_samples, 1) and bundle.y.dtype == np.float32
    assert bundle.weight.shape == (n_samples, 1) and bundle.weight.dtype == np.float32
    assert bundle.mask.shape == (n_samples,) and bundle.mask.dtype == np.bool_
    assert bundle.step_mask.shape == (n_samples, LOOKBACK) and bundle.step_mask.dtype == np.bool_
    assert bundle.index.names == ["datetime", "instrument"]
    assert bundle.feature_names == FEATURE_NAMES


@pytest.mark.unit
def test_suspended_rows_are_masked_and_neutrally_filled(bundle_and_inputs) -> None:
    """A suspension invalidates the sample and neutralises its decision-date step."""
    bundle, _, _, tradability, dates, _ = bundle_and_inputs
    suspended_date = dates[SUSPENSION_SLICE.start + 1]
    assert not tradability.loc[suspended_date, "B"]
    row = bundle.index.get_loc((suspended_date, "B"))

    assert not bundle.mask[row], "a suspended name must not be usable as a sample"
    assert not bundle.step_mask[row, -1], "the suspended decision-date step must be flagged"
    np.testing.assert_array_equal(bundle.X[row, -1], np.full(len(FEATURE_NAMES), np.float32(FILL_VALUE)))
    assert not bundle.select("tradable_on_decision_date")[row]


@pytest.mark.unit
def test_missing_feature_cell_flags_the_step_but_keeps_the_row(bundle_and_inputs) -> None:
    """A data gap is a flagged step, never a dropped row."""
    bundle, _, _, tradability, dates, _ = bundle_and_inputs
    missing_date = dates[MISSING_CELL_DATE]
    assert tradability.loc[missing_date, "A"], "the gap is in a feature, not in tradability"
    row = bundle.index.get_loc((missing_date, "A"))

    assert bundle.n_samples == len(dates) * len(INSTRUMENTS), "the row must still be present"
    assert not bundle.step_mask[row, -1], "the missing cell must be flagged at the last step"
    beta_index = bundle.feature_names.index("f_beta")
    assert bundle.X[row, -1, beta_index] == pytest.approx(FILL_VALUE, rel=1e-6)
    close_index = bundle.feature_names.index("f_close")
    assert bundle.X[row, -1, close_index] == pytest.approx(
        float(bundle_and_inputs[5]["close"].loc[missing_date, "A"]), rel=1e-6
    )


@pytest.mark.unit
def test_window_alignment_matches_the_source_panel(bundle_and_inputs) -> None:
    """The last step of a window equals the contemporaneous panel value, for every channel."""
    bundle, _, _, tradability, dates, panels = bundle_and_inputs
    date = dates[30]
    close_channel = bundle.feature_names.index("f_close")
    volume_channel = bundle.feature_names.index("f_volume")
    for instrument in INSTRUMENTS:
        assert tradability.loc[date, instrument], "the fixture must have all names tradable on this date"
        row = bundle.index.get_loc((date, instrument))
        assert bundle.step_mask[row, -1]
        assert bundle.X[row, -1, close_channel] == pytest.approx(float(panels["close"].loc[date, instrument]), rel=1e-6)
        expected_volume = float(np.log1p(panels["volume"].loc[date, instrument]))
        assert bundle.X[row, -1, volume_channel] == pytest.approx(expected_volume, rel=1e-6)
        assert bundle.index[row] == (date, instrument)


@pytest.mark.unit
def test_warmup_steps_are_flagged(bundle_and_inputs) -> None:
    """A window that starts before the sample has flagged padded steps, not imputed ones."""
    _, labels, _, _, dates, panels = bundle_and_inputs
    features = {"f_close": panels["close"]}
    plain = build_sequence_bundle(features, labels, lookback=LOOKBACK)

    first_row = plain.index.get_loc((dates[0], "A"))
    assert not plain.step_mask[first_row, : LOOKBACK - 1].any(), "padded steps must be flagged"
    assert plain.step_mask[first_row, LOOKBACK - 1], "the single real step must be valid"
    np.testing.assert_array_equal(
        plain.X[first_row, : LOOKBACK - 1, 0], np.zeros(LOOKBACK - 1, dtype="float32")
    ), "the fill value must be the neutral default here"

    second_row = plain.index.get_loc((dates[1], "A"))
    assert plain.step_mask[second_row, LOOKBACK - 2 :].all()
    assert not plain.step_mask[second_row, : LOOKBACK - 2].any()


@pytest.mark.unit
def test_untradable_row_has_its_decision_step_neutralised(bundle_and_inputs) -> None:
    """The default neutralisation clears the decision-date step of a non-tradable name.

    The fixture's seasoning rule makes the first dates non-tradable, so the raw (stale) close would
    otherwise be visible to the model even though the sample is masked out.
    """
    bundle, _, _, tradability, dates, _ = bundle_and_inputs
    untradable_row = bundle.index.get_loc((dates[0], "A"))
    assert not tradability.loc[dates[0], "A"]
    assert not bundle.mask[untradable_row]
    assert not bundle.step_mask[untradable_row, -1], "the decision step must be cleared"
    assert bundle.X[untradable_row, -1, bundle.feature_names.index("f_close")] == pytest.approx(FILL_VALUE, rel=1e-6)


@pytest.mark.unit
def test_min_valid_steps_excludes_incomplete_windows(bundle_and_inputs) -> None:
    """``min_valid_steps`` turns short windows into masked-out samples."""
    _, labels, _, _, dates, panels = bundle_and_inputs
    features = {"f_volume": np.log1p(panels["volume"])}
    reference = build_sequence_bundle(features, labels, lookback=LOOKBACK)
    strict = build_sequence_bundle(features, labels, lookback=LOOKBACK, min_valid_steps=LOOKBACK)

    assert strict.mask.sum() < reference.mask.sum(), "requiring full windows must exclude the warm-up"
    assert np.array_equal(strict.mask, strict.select("label_available") & strict.select("enough_valid_steps"))
    assert not strict.mask[strict.index.get_loc((dates[0], "A"))]


@pytest.mark.unit
def test_suspension_on_the_execution_date_invalidates_the_label(bundle_and_inputs) -> None:
    """A name suspended at the execution date has no tradable label, even if ``t`` itself is fine."""
    bundle, _, availability, tradability, dates, _ = bundle_and_inputs
    decision_date = dates[SUSPENSION_SLICE.start - 1]
    execution_date = dates[SUSPENSION_SLICE.start]
    assert tradability.loc[decision_date, "B"], "the decision date itself is tradable"
    assert not tradability.loc[execution_date, "B"], "the execution date is the suspended one"
    assert not availability.loc[
        (decision_date, "B")
    ], "the label must be unavailable because execution at t+1 is impossible"
    row = bundle.index.get_loc((decision_date, "B"))
    assert not bundle.mask[row]
    assert not bundle.select("provided_mask")[row]


@pytest.mark.unit
def test_diagnostics_report_every_mask_component(bundle_and_inputs) -> None:
    """Coverage and per-component invalid counts are exposed for the run log."""
    bundle, _, _, _, _, _ = bundle_and_inputs
    diagnostics = bundle.describe()
    assert diagnostics["n_valid"] == int(bundle.mask.sum())
    assert 0.0 < float(diagnostics["coverage"]) <= 1.0
    for component in ("label_available", "provided_mask", "tradable_on_decision_date"):
        assert f"invalid_by_{component}" in diagnostics
    with pytest.raises(KeyError, match="unknown mask component"):
        bundle.select("not_a_component")


@pytest.mark.unit
def test_masked_mean_excludes_invalid_rows_and_reports_nan_when_empty(bundle_and_inputs) -> None:
    """The reference reduction is the contract the loss and the optimizer must follow."""
    bundle, _, _, _, _, _ = bundle_and_inputs
    per_sample = np.abs(bundle.y.astype("float64"))

    masked = masked_mean(per_sample, bundle.mask)
    unmasked = float(np.nanmean(np.nan_to_num(per_sample, nan=0.0)))
    assert masked != pytest.approx(unmasked), "the mask must change the reduction, or it is decorative"
    assert masked == pytest.approx(float(per_sample[bundle.mask].mean()), rel=1e-12)

    assert np.isnan(
        masked_mean(per_sample, np.zeros_like(bundle.mask))
    ), "with no valid sample the honest answer is nan, not 0.0"

    doubled = masked_mean(per_sample, bundle.mask, weights=2.0 * bundle.weight.astype("float64"))
    assert doubled == pytest.approx(masked, rel=1e-6), "a constant weight must not change the mean"

    with pytest.raises(ValueError, match="same length"):
        masked_mean(per_sample, bundle.mask[:-1])
