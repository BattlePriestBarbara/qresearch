"""Point-in-time stress test for the announcement lag (task ``INF-04``).

The scenario is the mentor-mandated one: a company's Q1 numbers describe the period ending
**2024-03-31** but are only announced on **2024-04-25**.  Nothing derived from them may enter
:math:`\\mathcal{F}_t` before that date, and everything derived from them must enter on it.

Both directions are tested, because a test that cannot fail proves nothing:

* the correct pipeline is invisible through 2024-04-24 and visible from 2024-04-25;
* the deliberately leaky pipeline (``mode="report_date"``) *is* visible earlier and is caught by
  :func:`~qresearch.data.universe.assert_point_in_time` - the negative control demonstrating that
  the check has power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qresearch.data.universe import (
    AnnouncementLagSpec,
    assert_point_in_time,
    first_visible_date,
    to_panel,
    to_point_in_time,
)

REPORT_DATE = pd.Timestamp("2024-03-31")
ANNOUNCE_DATE = pd.Timestamp("2024-04-25")
INVISIBLE_THROUGH = pd.Timestamp("2024-04-24")
ROE_VALUE = 42.0
INSTRUMENT = "AAA"
SPEC = AnnouncementLagSpec(value_cols=("roe",))


@pytest.fixture()
def calendar() -> pd.DatetimeIndex:
    """Trading days spanning the report period and its announcement."""
    return pd.bdate_range("2024-03-01", "2024-05-31")


@pytest.fixture()
def fundamentals() -> pd.DataFrame:
    """One fundamental observation: Q1 produced on 31 March, announced on 25 April."""
    return pd.DataFrame(
        {
            "instrument": [INSTRUMENT],
            "report_date": [REPORT_DATE],
            "announce_date": [ANNOUNCE_DATE],
            "roe": [ROE_VALUE],
        }
    )


@pytest.mark.unit
def test_value_is_invisible_before_the_announcement(calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame) -> None:
    """Before 2024-04-25 the fundamental must be absent from every date's information set."""
    published = to_point_in_time(fundamentals, calendar, SPEC)
    panel = to_panel(published, calendar, [INSTRUMENT], SPEC.value_cols)["roe"]

    before = panel.loc[panel.index <= INVISIBLE_THROUGH, INSTRUMENT]
    assert before.isna().all(), (
        "the Q1 value leaked into F_t before it was announced; offending dates: "
        f"{list(before.dropna().index.strftime('%Y-%m-%d'))[:5]}"
    )
    assert ANNOUNCE_DATE not in before.index


@pytest.mark.unit
def test_value_is_visible_from_the_announcement_date(calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame) -> None:
    """On and after 2024-04-25 the value enters the information set, unchanged."""
    published = to_point_in_time(fundamentals, calendar, SPEC)
    panel = to_panel(published, calendar, [INSTRUMENT], SPEC.value_cols)["roe"]

    after = panel.loc[panel.index >= ANNOUNCE_DATE, INSTRUMENT]
    assert not after.isna().any(), f"value missing after the announcement on {list(after[after.isna()].index)}"
    assert np.allclose(after.to_numpy(), ROE_VALUE)
    assert len(after) > 20, "the sample must span a useful range after the announcement"


@pytest.mark.unit
def test_first_visible_date_is_the_first_trading_day_on_or_after_announcement(
    calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame
) -> None:
    """The value becomes knowable on the first trading day at or after the announcement."""
    assert ANNOUNCE_DATE in calendar
    assert first_visible_date(fundamentals, calendar, SPEC, instrument=INSTRUMENT) == ANNOUNCE_DATE


@pytest.mark.unit
def test_correct_pipeline_passes_the_point_in_time_verifier(
    calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame
) -> None:
    """The independent verifier confirms every published row was knowable at its own date."""
    published = to_point_in_time(fundamentals, calendar, SPEC)
    assert_point_in_time(published, fundamentals, SPEC)


@pytest.mark.unit
def test_leaky_report_date_pipeline_is_visible_early_and_is_caught(
    calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame
) -> None:
    """Negative control: indexing by report date leaks, and the verifier detects it.

    This is the test's own power check.  ``mode="report_date"`` publishes the Q1 value from the
    period end (31 March) instead of the announcement (25 April), so the leaky panel is populated
    almost a month early - and :func:`assert_point_in_time` rejects it.
    """
    leaky = to_point_in_time(fundamentals, calendar, SPEC, mode="report_date", allow_leaky=True)
    leaky_panel = to_panel(leaky, calendar, [INSTRUMENT], SPEC.value_cols)["roe"]

    early = leaky_panel.loc[(leaky_panel.index > REPORT_DATE) & (leaky_panel.index < ANNOUNCE_DATE), INSTRUMENT]
    assert early.notna().any(), "the negative control must actually leak, otherwise it proves nothing"

    with pytest.raises(AssertionError, match="published before it was knowable"):
        assert_point_in_time(leaky, fundamentals, SPEC)


@pytest.mark.unit
def test_leaky_mode_requires_an_explicit_flag(calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame) -> None:
    """A deliberate look-ahead cannot be constructed by accident."""
    with pytest.raises(ValueError, match="deliberate look-ahead"):
        to_point_in_time(fundamentals, calendar, SPEC, mode="report_date")


@pytest.mark.unit
def test_negative_announcement_lag_is_rejected(calendar: pd.DatetimeIndex) -> None:
    """An announcement dated before its report period end is inconsistent source data."""
    inconsistent = pd.DataFrame(
        {
            "instrument": [INSTRUMENT],
            "report_date": [ANNOUNCE_DATE],
            "announce_date": [REPORT_DATE],
            "roe": [ROE_VALUE],
        }
    )
    with pytest.raises(ValueError, match="before its report period end"):
        to_point_in_time(inconsistent, calendar, SPEC)


@pytest.mark.unit
def test_staleness_expiry_stops_publication(calendar: pd.DatetimeIndex, fundamentals: pd.DataFrame) -> None:
    """A finite staleness horizon retires the value after the configured number of periods."""
    published = to_point_in_time(fundamentals, calendar, SPEC, max_staleness=5)
    panel = to_panel(published, calendar, [INSTRUMENT], SPEC.value_cols)["roe"]
    visible = panel[INSTRUMENT].dropna()
    assert len(visible) == 5
    assert visible.index[0] == ANNOUNCE_DATE
    assert visible.index[-1] < calendar[-1]


@pytest.mark.unit
def test_model_predictions_before_the_announcement_cannot_depend_on_the_value(
    calendar: pd.DatetimeIndex,
) -> None:
    """Model-level statement of the requirement, with its own negative control.

    Two pipelines are built - one that knows the fundamentals and one where the rows do not exist -
    and a model is fitted on data up to 2024-04-24.  Under the correct (announcement-based) pipeline
    the Q1 value is invisible through that date, so the in-sample predictions must be identical.

    The control uses the *same* data under the leaky (report-date) pipeline, where the feature takes
    two different values inside the training window and therefore changes the fit.  Two details
    matter and are the reason this test is written this way:

    * the control MUST be time-varying - a single constant leaked value is absorbed by the intercept
      and a linear model could not detect it, which would make the control vacuous;
    * the assertion is on *in-sample* fitted values before the announcement, i.e. exactly the
      "model prediction" statement the mentor asked for.
    """
    two_periods = pd.DataFrame(
        {
            "instrument": [INSTRUMENT, INSTRUMENT],
            "report_date": [pd.Timestamp("2023-12-31"), REPORT_DATE],
            "announce_date": [pd.Timestamp("2024-01-20"), ANNOUNCE_DATE],
            "roe": [7.0, ROE_VALUE],
        }
    )
    published = to_point_in_time(two_periods, calendar, SPEC)
    panel = to_panel(published, calendar, [INSTRUMENT], SPEC.value_cols)["roe"]
    empty = to_panel(to_point_in_time(two_periods.iloc[0:0], calendar, SPEC), calendar, [INSTRUMENT], SPEC.value_cols)[
        "roe"
    ]
    leaky = to_panel(
        to_point_in_time(two_periods, calendar, SPEC, mode="report_date", allow_leaky=True),
        calendar,
        [INSTRUMENT],
        SPEC.value_cols,
    )["roe"]

    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=(len(calendar), 1)), index=calendar, columns=[INSTRUMENT])
    # The target is generated by the *leaked* step function plus a little noise, i.e. it is exactly
    # what a pipeline that peeks at the report date could fit.  A correct pipeline must not be able
    # to fit it, because the only knowable value inside the training window is the constant Q4 one.
    step = leaky[INSTRUMENT].to_numpy()
    target = step + rng.normal(scale=1e-3, size=len(calendar))
    cutoff = INVISIBLE_THROUGH

    def fit_and_predict(feature: pd.DataFrame) -> np.ndarray:
        design = np.nan_to_num(np.hstack([noise.to_numpy(), feature.to_numpy()]), nan=0.0)
        augmented = np.hstack([np.ones((len(calendar), 1)), design])
        train = calendar <= cutoff
        coefficients = np.linalg.lstsq(augmented[train], target[train], rcond=None)[0]
        return augmented[train] @ coefficients

    # Inside the training window the PIT feature is constant (only the Q4 value is knowable) and
    # therefore cannot move the fit, while the leaky feature reproduces the target's own step.
    assert panel.loc[calendar <= cutoff, INSTRUMENT].nunique() == 1
    assert leaky.loc[calendar <= cutoff, INSTRUMENT].nunique() == 2

    pit_predictions = fit_and_predict(panel)
    blind_predictions = fit_and_predict(empty)
    leaky_predictions = fit_and_predict(leaky)

    assert np.allclose(pit_predictions, blind_predictions), (
        "the fundamental changed in-sample predictions before its announcement, so it was visible "
        "through the report date instead of the announcement date"
    )
    assert np.abs(pit_predictions - target[: len(pit_predictions)]).mean() > 1.0, (
        "the correct pipeline must NOT be able to fit the leaked step, otherwise this test is "
        "measuring model capacity rather than leakage"
    )
    assert (
        np.abs(leaky_predictions - blind_predictions).max() > 1e-3
    ), "the negative control must differ, otherwise this test cannot detect a leak"
    assert np.abs(leaky_predictions - target[: len(leaky_predictions)]).mean() < 1.0
    assert empty[INSTRUMENT].isna().all()
