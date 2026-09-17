"""Causality tests for the normalizers (task ``INF-06``).

Requirement under test: the statistic transforming date :math:`t` may use only
:math:`[t-L,\\; t-1]`.  Each processor is therefore probed two ways:

1. by :func:`~qresearch.data.processors.assert_causal`, which mutates **every** row after the probe
   date and demands the probe row be unchanged;
2. by the explicit injection experiment - a single extreme outlier at :math:`t+1` - which must leave
   the output at :math:`t` *bit-identical* while demonstrably changing the output at :math:`t+1`.

The second assertion keeps the first meaningful: without it, a processor that ignored its input
entirely would also "pass".  The leaky ``GlobalZScoreNorm`` is the negative control and MUST fail
both probes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qresearch.data.processors import (
    PANEL_PROCESSORS,
    CSRankNorm,
    CSRobustZScoreNorm,
    CSZScoreNorm,
    ExpandingTSZScoreNorm,
    GlobalZScoreNorm,
    PanelProcessor,
    RollingTSRobustZScoreNorm,
    RollingTSZScoreNorm,
    RollingWindow,
    assert_causal,
    build_processor,
)

N_DATES = 120
N_INSTRUMENTS = 6
WINDOW = RollingWindow(length=20, min_periods=10)
PROBE = 60
OUTLIER = 1e12


def _panel() -> pd.DataFrame:
    """Return a reproducible random panel with a realistic scale."""
    rng = np.random.default_rng(12345)
    return pd.DataFrame(
        rng.normal(loc=1.0, scale=0.2, size=(N_DATES, N_INSTRUMENTS)),
        index=pd.bdate_range("2020-01-01", periods=N_DATES, name="datetime"),
        columns=[f"S{index:02d}" for index in range(N_INSTRUMENTS)],
    )


CAUSAL_PROCESSORS: list[PanelProcessor] = [
    RollingTSZScoreNorm(WINDOW),
    RollingTSRobustZScoreNorm(WINDOW),
    ExpandingTSZScoreNorm(min_periods=10),
    CSRankNorm(),
    CSZScoreNorm(),
    CSRobustZScoreNorm(),
]

ROLLING_PROCESSORS: list[PanelProcessor] = [
    RollingTSZScoreNorm(WINDOW),
    RollingTSRobustZScoreNorm(WINDOW),
    ExpandingTSZScoreNorm(min_periods=10),
]


@pytest.mark.unit
@pytest.mark.parametrize("processor", CAUSAL_PROCESSORS, ids=lambda processor: processor.name)
def test_probe_reports_the_processor_is_causal(processor: PanelProcessor) -> None:
    """Mutating every future row must not change the probe row for any causal processor."""
    report = assert_causal(processor, _panel(), probe_position=PROBE)
    assert report.is_causal
    assert report.max_abs_change == 0.0
    assert "CAUSAL" in report.format()


@pytest.mark.unit
@pytest.mark.parametrize("processor", ROLLING_PROCESSORS, ids=lambda processor: processor.name)
def test_future_outlier_leaves_the_past_untouched(processor: PanelProcessor) -> None:
    """A single extreme value at ``t+1`` must leave the standardized value at ``t`` identical.

    This is the mandated injection experiment.  The second half asserts that the outlier *does*
    change the row at ``t+1``: a processor that ignored its input would otherwise pass vacuously.
    """
    frame = _panel()
    baseline = processor.transform(frame)

    mutated = frame.copy()
    mutated.iloc[PROBE + 1, 0] = OUTLIER
    perturbed = processor.transform(mutated)

    np.testing.assert_array_equal(
        baseline.iloc[PROBE].to_numpy(),
        perturbed.iloc[PROBE].to_numpy(),
        err_msg="an extreme value at t+1 changed the standardized row at t",
    )
    assert not np.allclose(
        baseline.iloc[PROBE + 1].to_numpy(), perturbed.iloc[PROBE + 1].to_numpy()
    ), "the injected outlier had no effect at all, so this test cannot detect a change"


@pytest.mark.unit
def test_rolling_window_excludes_the_current_row_by_construction() -> None:
    """Warm-up rows are ``NaN``: the window holds exactly ``[t-L, t-1]`` observations."""
    processor = RollingTSZScoreNorm(RollingWindow(length=20, min_periods=20))
    transformed = processor.transform(_panel())

    assert transformed.iloc[:20].isna().all(axis=None), "the warm-up must be explicit, not imputed"
    assert transformed.iloc[20].notna().all(), "the first full window must produce a value"


@pytest.mark.unit
def test_cross_sectional_processors_depend_only_on_their_own_row() -> None:
    """A ``CS*`` transform of one date equals that date's transform inside the full panel."""
    frame = _panel()
    for processor in (CSRankNorm(), CSZScoreNorm(), CSRobustZScoreNorm()):
        full = processor.transform(frame)
        single = processor.transform(frame.iloc[[PROBE]])
        assert_frame_equal(full.iloc[[PROBE]], single, check_exact=True)


@pytest.mark.unit
def test_leaky_processor_is_refused_without_the_explicit_flag() -> None:
    """The whole-sample z-score cannot be built accidentally."""
    with pytest.raises(ValueError, match="leaks future information"):
        GlobalZScoreNorm()


@pytest.mark.unit
def test_leaky_processor_is_caught_by_the_causality_probe() -> None:
    """Negative control: the audit detects a processor that uses the full sample.

    Both probes must fire - the generic one and the single-outlier injection - otherwise the
    verification above would be vacuous.
    """
    leaky = GlobalZScoreNorm(allow_leaky=True)

    with pytest.raises(AssertionError, match="used future information"):
        assert_causal(leaky, _panel(), probe_position=PROBE)

    frame = _panel()
    baseline = leaky.transform(frame)
    mutated = frame.copy()
    mutated.iloc[PROBE + 1, 0] = OUTLIER
    perturbed = leaky.transform(mutated)
    assert not np.allclose(
        baseline.iloc[PROBE].to_numpy(), perturbed.iloc[PROBE].to_numpy()
    ), "a global z-score must react to a future outlier, which is exactly why it is forbidden"


@pytest.mark.unit
def test_processor_registry_builds_from_configuration() -> None:
    """Processors are selectable declaratively, and unknown names are refused."""
    processor = build_processor(
        {"class": "rolling_ts_zscore", "kwargs": {"window": RollingWindow(length=10, min_periods=5)}}
    )
    assert isinstance(processor, RollingTSZScoreNorm)
    assert processor.window.length == 10
    assert isinstance(build_processor({"class": "CSRankNorm"}), CSRankNorm)

    with pytest.raises(KeyError, match="unknown processor"):
        build_processor({"class": "not_a_processor"})


@pytest.mark.unit
def test_every_registered_processor_is_audited() -> None:
    """Each non-leaky registry entry passes the causality probe; the leaky one is named as such."""
    for key, processor_class in PANEL_PROCESSORS.items():
        if "LEAKY" in key:
            assert processor_class is GlobalZScoreNorm
            continue
        assert_causal(processor_class(), _panel(), probe_position=PROBE)
