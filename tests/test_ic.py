"""Tests for the IC primitives (task ``ST-01``).

The Definition of Done for ``ST-01`` requires a numerical cross-check against an independent
implementation wherever one exists, so every per-period statistic is compared with
``scipy.stats.spearmanr`` / ``pearsonr`` on the same cross-sections.  The interface guards (minimum
cross-section size, pairwise NaN handling, duplicate-index rejection) are tested too, because a
silently mis-aligned panel is the failure mode that produces plausible but wrong IC numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qresearch.stats.ic import ICMoments, align_predictions, ic_moments, pearson_ic, rank_ic

N_DATES = 40
N_INSTRUMENTS = 40
SEED = 20260915


@pytest.fixture()
def panel() -> tuple[pd.Series, pd.Series]:
    """Return a prediction and a label series with a known positive rank relationship."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    instruments = [f"S{index:03d}" for index in range(N_INSTRUMENTS)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    truth = rng.normal(size=(N_DATES, N_INSTRUMENTS))
    signal = truth + rng.normal(scale=1.0, size=(N_DATES, N_INSTRUMENTS))
    labels = pd.Series(truth.reshape(-1), index=index, name="label")
    predictions = pd.Series(signal.reshape(-1), index=index, name="score")
    return predictions, labels


@pytest.mark.unit
def test_rank_ic_matches_scipy_per_date(panel) -> None:
    """The per-period Spearman IC equals ``scipy.stats.spearmanr`` on the same cross-sections."""
    predictions, labels = panel
    ours = rank_ic(predictions, labels, min_obs=5)
    assert len(ours) == N_DATES

    for date in ours.index[:12]:
        expected = stats.spearmanr(predictions.loc[date], labels.loc[date]).statistic
        assert ours.loc[date] == pytest.approx(float(expected), abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_pearson_ic_matches_scipy_per_date(panel) -> None:
    """The per-period Pearson IC equals ``scipy.stats.pearsonr`` on the same cross-sections."""
    predictions, labels = panel
    ours = pearson_ic(predictions, labels, min_obs=5)

    for date in ours.index[:12]:
        expected = stats.pearsonr(predictions.loc[date], labels.loc[date]).statistic
        assert ours.loc[date] == pytest.approx(float(expected), abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_perfect_and_reversed_signals_have_unit_ic(panel) -> None:
    """The sign convention is pinned: a perfect forecast gives +1, a reversed one gives -1."""
    _, labels = panel
    assert rank_ic(labels.copy(), labels, min_obs=5).mean() == pytest.approx(1.0)

    reversed_scores = -labels.astype("float64")
    assert rank_ic(reversed_scores, labels, min_obs=5).mean() == pytest.approx(-1.0)


@pytest.mark.unit
def test_min_obs_drops_and_counts_periods(panel) -> None:
    """Thin cross-sections are dropped and the count is reported, never silently absorbed."""
    predictions, labels = panel
    series, dropped = rank_ic(predictions, labels, min_obs=N_INSTRUMENTS // 2 + 1, return_dropped=True)
    assert isinstance(series, pd.Series)
    assert dropped == 0

    series, dropped = rank_ic(predictions, labels, min_obs=N_INSTRUMENTS + 1, return_dropped=True)
    assert series.empty
    assert dropped == N_DATES

    with pytest.raises(ValueError, match="at least 3"):
        rank_ic(predictions, labels, min_obs=2)


@pytest.mark.unit
def test_alignment_drops_only_pairwise_missing_rows(panel) -> None:
    """Rows missing in either input are removed pairwise; the rest are untouched."""
    predictions, labels = panel
    predictions = predictions.copy()
    labels = labels.copy()
    first_date = labels.index.get_level_values("datetime")[0]
    predictions.loc[(first_date, "S000")] = np.nan

    aligned = align_predictions(predictions, labels)
    assert len(aligned) == N_DATES * N_INSTRUMENTS - 1
    assert (first_date, "S000") not in aligned.index
    assert aligned.index.equals(aligned.sort_index().index), "the frame must be sorted"

    series = rank_ic(predictions, labels, min_obs=5)
    assert len(series) == N_DATES, "one missing cell must not remove a whole period"


@pytest.mark.unit
def test_duplicate_index_is_rejected_with_an_actionable_message(panel) -> None:
    """A duplicated panel is a join/split defect, not something to average over silently."""
    predictions, labels = panel
    doubled = pd.concat([labels, labels])
    with pytest.raises(ValueError, match="duplicate"):
        rank_ic(predictions, doubled, min_obs=5)

    doubled_predictions = pd.concat([predictions, predictions])
    with pytest.raises(ValueError, match="duplicate"):
        rank_ic(doubled_predictions, labels, min_obs=5)


@pytest.mark.unit
def test_mismatched_grids_are_aligned_by_index_not_by_position(panel) -> None:
    """Rows present in only one input are excluded, so input order cannot silently matter."""
    predictions, labels = panel
    subset_dates = labels.index.get_level_values("datetime").unique()[:10]
    subset = predictions.loc[predictions.index.get_level_values("datetime").isin(subset_dates)]

    series = rank_ic(subset, labels, min_obs=5)
    assert len(series) == 10
    assert sorted(series.index) == sorted(subset_dates)


@pytest.mark.unit
def test_ic_moments_reports_annualized_ratio(panel) -> None:
    """The moments block exposes the mean, the dispersion and the annualized ICIR."""
    predictions, labels = panel
    series = rank_ic(predictions, labels, min_obs=5)
    moments = ic_moments(series)
    assert isinstance(moments, ICMoments)
    assert moments.n_periods == N_DATES
    assert moments.mean == pytest.approx(float(series.mean()), rel=1e-12)
    assert moments.std == pytest.approx(float(series.std(ddof=1)), rel=1e-12)
    assert moments.icir == pytest.approx(moments.mean / moments.std, rel=1e-12)
    assert moments.icir_annualized == pytest.approx(moments.icir * np.sqrt(252), rel=1e-12)
    assert moments.t_iid == pytest.approx(moments.mean / (moments.std / np.sqrt(N_DATES)), rel=1e-12)
    assert set(moments.to_dict()) >= {"n_periods", "mean", "std", "icir", "icir_annualized", "t_iid"}


@pytest.mark.unit
def test_ic_moments_handles_degenerate_input() -> None:
    """A constant or single-observation series yields ``nan`` rather than a fabricated number."""
    constant = pd.Series([0.05] * 10, dtype="float64")
    moments = ic_moments(constant)
    assert moments.n_periods == 10
    assert np.isnan(moments.icir)
    assert np.isnan(moments.t_iid)

    single = ic_moments(pd.Series([0.05]))
    assert single.n_periods == 1
    assert np.isnan(single.mean)
