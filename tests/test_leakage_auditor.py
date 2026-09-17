"""Tests for the permutation-based leakage audit (task ``INF-08``).

Four properties are established, and the negative controls matter as much as the positive ones:

1. a **correct** pipeline is not flagged, even when the model is genuinely skillful - this is the
   regression test for the failure mode of comparing the real-label statistic against the shuffle
   null, which would have flagged every good model;
2. a split that puts the same sample (or its duplicate) on both sides of the boundary **is**
   flagged, because the model can then memorise time-permuted labels;
3. duplicating rows is not itself a defect: with purged folds the auditor stays clean, so the test
   isolates the *split* rather than punishing duplicate data;
4. the audit is reproducible from its seed, and the complementary prefix-invariance instrument
   catches what label shuffling cannot (a feature that embeds future information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qresearch.data.audit import (
    AuditConfig,
    DataLeakageDetectedError,
    LeakageAuditor,
    assert_no_leakage,
    audit_prefix_invariance,
    permute_labels_by_date,
)
from qresearch.data.handlers import build_sequence_bundle

N_DATES = 120
N_INSTRUMENTS = 25
N_PERMUTATIONS = 99
SEED = 4242


@pytest.fixture()
def panel() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Return the dates and an informed feature panel."""
    dates = pd.bdate_range("2018-01-01", periods=N_DATES)
    rng = np.random.default_rng(11)
    features = pd.DataFrame(
        rng.normal(size=(N_DATES, N_INSTRUMENTS)),
        index=dates,
        columns=[f"S{index:02d}" for index in range(N_INSTRUMENTS)],
    )
    return dates, features


@pytest.fixture()
def labelled(panel) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return ``(design, labels, row_index_as_frame)`` for a genuinely predictable target."""
    dates, features = panel
    rng = np.random.default_rng(12)
    market = rng.normal(size=N_DATES)  # a date-level component
    label_panel = 0.6 * features.to_numpy() + market[:, None]
    label_panel += rng.normal(scale=0.5, size=label_panel.shape)

    index = pd.MultiIndex.from_product([dates, features.columns], names=["datetime", "instrument"])
    labels = pd.Series(label_panel.reshape(-1), index=index, name="label")

    noise = rng.normal(size=(N_DATES, N_INSTRUMENTS)).reshape(-1, 1)
    trend = np.repeat(np.arange(N_DATES) / N_DATES, N_INSTRUMENTS).reshape(-1, 1)
    design = np.hstack([features.to_numpy().reshape(-1, 1), noise, trend])
    return design, labels, trend


def _config(**overrides: object) -> AuditConfig:
    """Return an audit configuration tuned for a fast, deterministic test run."""
    defaults: dict[str, object] = {
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "n_folds": 3,
        "min_obs": 10,
    }
    defaults.update(overrides)
    return AuditConfig(**defaults)  # type: ignore[arg-type]


def _duplicate_samples(design: np.ndarray, labels: pd.Series) -> tuple[np.ndarray, pd.Series]:
    """Append a copy of every sample under a distinct instrument suffix, keeping rows aligned.

    This reproduces the classic contamination defect - the same *feature vector* appearing on both
    sides of the train/evaluate boundary - while keeping the ``(datetime, instrument)`` index unique,
    as the panel contract requires.  A split that respects time (purged folds) keeps both copies in
    the same fold, so the duplication is harmless there; a row-wise split separates them and the
    model can memorise the label of the copy it saw.

    Returns
    -------
    tuple[np.ndarray, pd.Series]
        A design matrix and label series whose rows correspond **one-to-one and in order**; the
        auditor derives no alignment information from the index, so a sorted label series with an
        unsorted design would silently pair the wrong feature vector with the wrong target.
    """
    copies = labels.copy()
    copies.index = pd.MultiIndex.from_arrays(
        [
            labels.index.get_level_values("datetime"),
            labels.index.get_level_values("instrument") + "#dup",
        ],
        names=["datetime", "instrument"],
    )
    combined = pd.concat([labels, copies]).sort_index()

    stacked = np.vstack([design, design])
    base_positions = {key: position for position, key in enumerate(labels.index)}
    positions: list[int] = []
    for date, instrument in combined.index:
        is_copy = instrument.endswith("#dup")
        base = (date, instrument[: -len("#dup")] if is_copy else instrument)
        positions.append((len(labels) if is_copy else 0) + base_positions[base])
    return stacked[positions], combined


@pytest.mark.statistical
def test_skillful_pipeline_is_not_flagged(labelled) -> None:
    """A correct pipeline stays clean even though the model has real predictive power.

    This is the regression test for the invalid construction (real-label statistic versus shuffle
    null): there, a genuinely good model was indistinguishable from a leaking one.  Here the
    observed IC must be clearly positive while the shuffled-label IC stays centred on zero.
    """
    design, labels, _ = labelled
    result = LeakageAuditor(_config(fold_scheme="purged")).audit(design, labels, horizon=1, raise_on_leakage=False)

    assert result.observed_ic > 0.15, "the fixture must contain a real signal for this test to bite"
    assert result.is_clean, f"a purged pipeline with real signal must not be flagged: {result.format()}"
    assert result.p_value > 0.05
    assert abs(result.shuffle_mean_ic) < 0.05, "shuffled labels must be unpredictable under purging"
    assert result.purge_radius == 1


def _memorising_lookup(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """A deliberately high-capacity audit model: exact-match lookup with a mean fallback.

    A linear model cannot reproduce a *random relabelling* of feature vectors, so duplicate-row
    contamination is invisible to it.  A lookup table can: when a test row's identical copy is in the
    training set, it returns that copy's label - which, under a date permutation, is precisely the
    test row's own permuted label.  This is the model class that gives the audit power against
    memorisation, and it is used to demonstrate that power.
    """
    table = {row.tobytes(): float(label) for row, label in zip(train_x, train_y, strict=True)}
    fallback = float(np.mean(train_y))
    return np.array([table.get(row.tobytes(), fallback) for row in test_x], dtype="float64")


@pytest.mark.statistical
def test_memorisation_leak_is_detected_by_a_capacity_enabled_model(labelled) -> None:
    """Negative control: duplicate rows plus a row-wise split IS flagged.

    The same sample appears on both sides of the boundary, so a model with lookup capacity can
    reproduce its (time-permuted) label out of sample.  The auditor MUST reject that pipeline - this
    is the demonstration that the test has power.
    """
    design, labels, _ = labelled
    duplicated_design, duplicated_labels = _duplicate_samples(design, labels)

    auditor = LeakageAuditor(_config(fold_scheme="overlapping"))
    with pytest.raises(DataLeakageDetectedError, match="TIME-PERMUTED labels"):
        auditor.audit(duplicated_design, duplicated_labels, horizon=1, model=_memorising_lookup)

    result = auditor.audit(
        duplicated_design, duplicated_labels, horizon=1, model=_memorising_lookup, raise_on_leakage=False
    )
    assert result.verdict == "leakage_detected"
    assert result.shuffle_mean_ic > 0.05, "memorisation must produce a positive shuffled IC"
    assert result.p_value < 0.05
    assert "LEAKAGE_DETECTED" in result.format()


@pytest.mark.statistical
def test_capacity_enabled_model_stays_clean_under_purged_folds(labelled) -> None:
    """The same model and data pass once the split respects time, so the split is the defect."""
    design, labels, _ = labelled
    duplicated_design, duplicated_labels = _duplicate_samples(design, labels)

    result = LeakageAuditor(_config(fold_scheme="purged")).audit(
        duplicated_design, duplicated_labels, horizon=1, model=_memorising_lookup, raise_on_leakage=False
    )
    assert result.is_clean, (
        "with purged folds the duplicates share a fold, so lookup capacity finds nothing to copy: " f"{result.format()}"
    )


@pytest.mark.statistical
def test_linear_audit_model_does_not_detect_a_memorisation_only_leak(labelled) -> None:
    """Documented sensitivity limit: the default linear model is blind to memorisation leaks.

    A random relabelling of continuous features is not linearly reproducible, so a ridge audit model
    cannot exploit duplicate-row contamination and reports a clean verdict.  The test records that
    limit explicitly, and the paired test above shows the same data being flagged once the audit
    model has lookup capacity.  Conclusion for practice: run the audit with a model at least as
    expressive as the production model (``NN-10``), never with a weaker surrogate.
    """
    design, labels, _ = labelled
    duplicated_design, duplicated_labels = _duplicate_samples(design, labels)

    result = LeakageAuditor(_config(fold_scheme="overlapping")).audit(
        duplicated_design, duplicated_labels, horizon=1, raise_on_leakage=False
    )
    assert result.is_clean, (
        "if this ever starts firing, the linear audit model has gained capacity and the note above " "must be revised"
    )


@pytest.mark.statistical
def test_audit_is_reproducible_from_its_seed(labelled) -> None:
    """The same seed and data give bit-identical statistics, as required for a tracked study."""
    design, labels, _ = labelled
    first = LeakageAuditor(_config()).audit(design, labels, horizon=1, raise_on_leakage=False)
    second = LeakageAuditor(_config()).audit(design, labels, horizon=1, raise_on_leakage=False)

    assert first.p_value == second.p_value
    assert first.shuffle_mean_ic == second.shuffle_mean_ic
    assert first.shuffle_t_statistic == second.shuffle_t_statistic
    assert first.to_dict() == second.to_dict()

    other_seed = LeakageAuditor(_config(seed=SEED + 1)).audit(design, labels, horizon=1, raise_on_leakage=False)
    assert other_seed.verdict == first.verdict, "a different seed must not flip the verdict"


@pytest.mark.unit
def test_contrived_design_and_label_shapes_are_rejected(labelled) -> None:
    """Interface misuse fails loudly instead of producing a meaningless verdict."""
    design, labels, _ = labelled
    auditor = LeakageAuditor(_config(n_permutations=5))

    with pytest.raises(ValueError, match="rows but labels"):
        auditor.audit(design[:-1], labels)
    with pytest.raises(ValueError, match="must be indexed by MultiIndex"):
        auditor.audit(design, pd.Series(labels.to_numpy()))
    with pytest.raises(ValueError, match="must be 2-D"):
        auditor.audit(design.reshape(-1), labels)
    with pytest.raises(ValueError, match="alpha must be in"):
        AuditConfig(alpha=1.5)
    with pytest.raises(ValueError, match="fold_scheme"):
        AuditConfig(fold_scheme="random")  # type: ignore[arg-type]


@pytest.mark.unit
def test_permutation_preserves_every_cross_section(labelled) -> None:
    """The shuffle moves whole dates, so the collection of cross-sections is preserved exactly.

    Each date after the permutation carries the values of *some* original date - not necessarily its
    own - so the property to assert is that the multiset of per-date value-multisets is unchanged,
    and that the mapping between old and new cross-sections is a bijection.
    """
    _, labels, _ = labelled
    rng = np.random.default_rng(0)
    permuted = permute_labels_by_date(labels, rng)

    assert permuted.index.equals(labels.index), "the index must be untouched"
    assert sorted(permuted.to_numpy()) == sorted(labels.to_numpy()), "the values must be a permutation"

    original = labels.unstack("instrument")
    shuffled = permuted.unstack("instrument")
    original_signature = sorted(tuple(sorted(row.dropna())) for _, row in original.iterrows())
    shuffled_signature = sorted(tuple(sorted(row.dropna())) for _, row in shuffled.iterrows())
    assert (
        shuffled_signature == original_signature
    ), "each date must still hold exactly the values of one original cross-section"
    assert original.index.equals(shuffled.index)
    assert original.notna().sum(axis=1).equals(shuffled.notna().sum(axis=1)), "support is preserved"

    with pytest.raises(ValueError, match="MultiIndex"):
        permute_labels_by_date(pd.Series([1.0, 2.0]), rng)


@pytest.mark.unit
def test_flatten_bundle_produces_a_row_aligned_design() -> None:
    """The bridge from a ``PanelBundle`` exposes the last timestep as a flat design matrix."""
    dates = pd.bdate_range("2020-01-01", periods=6)
    panel = pd.DataFrame(np.arange(12, dtype="float64").reshape(6, 2), index=dates, columns=["A", "B"])
    panel.iloc[3, 0] = np.nan
    label_panel = panel.shift(-1) / panel - 1.0
    index = pd.MultiIndex.from_product([dates, panel.columns], names=["datetime", "instrument"])
    labels = pd.Series(label_panel.to_numpy().reshape(-1), index=index)

    bundle = build_sequence_bundle({"f": panel}, labels, lookback=3, fill_value=0.0)
    design = LeakageAuditor.flatten_bundle(bundle)
    assert design.shape == (bundle.n_samples, 1)
    assert np.isfinite(design).all()
    row = bundle.index.get_loc((dates[3], "A"))
    assert design[row, 0] == 0.0, "the missing cell must be filled with the bundle's neutral value"

    with pytest.raises(TypeError, match="expects a PanelBundle"):
        LeakageAuditor.flatten_bundle(design)


@pytest.mark.statistical
def test_assert_no_leakage_raises_for_a_leaking_split_and_passes_for_a_clean_one(labelled) -> None:
    """The promotion gate: clean passes, leaking raises, and the message is actionable."""
    design, labels, _ = labelled
    clean = assert_no_leakage(design, labels, config=_config(fold_scheme="purged"))
    assert clean.is_clean

    duplicated_design, duplicated_labels = _duplicate_samples(design, labels)
    with pytest.raises(DataLeakageDetectedError) as error_info:
        assert_no_leakage(
            duplicated_design,
            duplicated_labels,
            config=_config(fold_scheme="overlapping"),
            model=_memorising_lookup,
        )
    message = str(error_info.value)
    assert "date overlap between folds" in message
    assert "INF-08" in message


@pytest.mark.statistical
def test_prefix_invariance_catches_what_shuffling_cannot(panel) -> None:
    """Complementary instrument: a feature that peeks at the future is caught here.

    A pipeline standardized with **full-sample** statistics is invisible to the label-shuffling test
    (shuffling breaks the feature-label alignment as well) but cannot survive a prefix replay: adding
    future rows changes its own history.  The causal counterpart must pass.
    """
    _, features = panel

    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        """Deliberately leaky: statistics estimated over the whole sample."""
        return (frame - frame.mean()) / frame.std()

    with pytest.raises(AssertionError, match="prefix-invariance violated"):
        audit_prefix_invariance(leaky, features)

    def rolling(frame: pd.DataFrame) -> pd.DataFrame:
        """Causal counterpart: the window ends one period before the transformed row."""
        history = frame.shift(1).rolling(20, min_periods=5)
        return (frame - history.mean()) / history.std().replace(0.0, np.nan)

    audit_prefix_invariance(rolling, features)

    with pytest.raises(ValueError, match="keep_ratio"):
        audit_prefix_invariance(rolling, features, keep_ratio=1.0)
