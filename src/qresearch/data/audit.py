"""Permutation-based leakage audit (task ``INF-08``) - the statistical guard of the data pipeline.

Conventional unit tests verify that code *does what it says*; they cannot verify that a pipeline
only knows what it *should* know.  This module supplies the missing instrument: a **randomization
test** over the whole training-and-evaluation path.

Procedure
---------
1. **Break the temporal dependence, keep the cross-section.**  Permute the label panel across
   *dates*: ``y_perm(perm[i], j) = y(i, j)``.  The distribution inside every date is untouched - so
   cross-sectional regularity survives - while the temporal alignment between features and labels is
   destroyed.
2. **Run the real pipeline** on the permuted labels: same split, same model fit, same out-of-sample
   statistic :math:`s_b` (the mean rank IC), repeated for :math:`B` independent permutations.
3. **Test the shuffled-label performance against zero.**  Under a pipeline that only knows what it
   should know, permuted labels are unpredictable, so :math:`\\mathbb{E}[s_b] = 0` no matter how
   expressive the model is.  A pipeline that can *memorise* the permuted labels - because the split
   lets it see them, or because a feature or a statistic carries row identity across the
   train/evaluate boundary - produces :math:`\\hat{s} = \\frac{1}{B}\\sum_b s_b` systematically
   positive.  The test is the one-sided t-test

   .. math::

       t = \\frac{\\hat{s}}{\\widehat{se}(s_b)/\\sqrt{B}}, \\qquad
       \\hat{p} = 1 - F_{t_{B-1}}(t)

   and ``p < alpha`` raises :class:`DataLeakageDetectedError`.

Why not compare the *real-label* statistic with the permutation null?
--------------------------------------------------------------------
That construction looks natural and is wrong: a genuinely predictive model scores high on the real
labels, and a memorising pipeline scores high on the permuted labels too, so the two distributions
move together and the test loses its power.  The real-label IC is therefore reported as a
**diagnostic** (``observed_ic``) and is deliberately excluded from the decision.

What this test does and does not catch
--------------------------------------
It catches any defect that lets row identity (and therefore the permuted label) travel across the
train/evaluate boundary: overlapping folds, duplicated samples, features that identify a row, and
statistics fitted on the full sample.

It does **not** catch a feature that embeds the label of *its own* row, because shuffling breaks that
alignment as well.  That failure mode is covered by the complementary prefix-invariance check
:func:`audit_prefix_invariance`, which recomputes a feature pipeline on a truncated calendar and
requires the observable history to be unchanged.  Both instruments are required by ``INF-08``.

Reproducibility
---------------
Every permutation draws from ``np.random.SeedSequence(entropy=seed, spawn_key=(b,))``, so results
depend only on the seed and the data - never on execution order or parallelism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from qresearch.stats.ic import rank_ic
from qresearch.utils.typing import Array, BoolArray, IntArray

__all__ = [
    "AuditConfig",
    "DataLeakageDetectedError",
    "LeakageAuditor",
    "PermutationAuditResult",
    "assert_no_leakage",
    "permute_labels_by_date",
]

_DATETIME_LEVEL: Final[str] = "datetime"
_INSTRUMENT_LEVEL: Final[str] = "instrument"
FoldScheme = Literal["purged", "overlapping"]


class DataLeakageDetectedError(RuntimeError):
    """Raised when the randomization test rejects the no-predictive-ability null hypothesis.

    Inherits from :class:`RuntimeError`: a leak is an environment-level defect that MUST abort a
    study, not a data error a caller may reasonably catch and continue past.
    """


@dataclass(frozen=True)
class AuditConfig:
    """Configuration of the leakage audit.

    Attributes
    ----------
    n_permutations : int
        Number of random date permutations :math:`B`.  The smallest attainable p-value is
        :math:`1/(B+1)`, so ``B >= 199`` is required to reach ``alpha = 0.05``.
    alpha : float
        Significance level; a smaller p-value triggers :class:`DataLeakageDetectedError`.
    seed : int
        Master seed; all randomness derives from it, making the audit reproducible.
    fold_scheme : {"purged", "overlapping"}
        ``"purged"`` builds expanding folds separated by ``purge_radius`` dates (admissible).
        ``"overlapping"`` performs a plain K-fold **over rows**, so the same date lands in training
        and evaluation; it exists only as the negative control that proves the audit has power.
    n_folds : int
        Number of evaluation folds.
    purge_radius : int | None
        Dates removed on each side of a test fold; ``None`` uses the label horizon passed to
        :meth:`LeakageAuditor.audit`.
    min_obs : int
        Minimum cross-section size for a date to contribute an IC observation.
    ridge_alpha : float
        L2 penalty of the built-in linear model, keeping the audit cheap enough to run ``B`` times.
    """

    n_permutations: int = 200
    alpha: float = 0.05
    seed: int = 20260915
    fold_scheme: FoldScheme = "purged"
    n_folds: int = 3
    purge_radius: int | None = None
    min_obs: int = 20
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {self.n_permutations}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.fold_scheme not in {"purged", "overlapping"}:
            raise ValueError(f"fold_scheme must be purged/overlapping, got {self.fold_scheme!r}")
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {self.n_folds}")
        if self.purge_radius is not None and self.purge_radius < 0:
            raise ValueError(f"purge_radius must be >= 0, got {self.purge_radius}")


@dataclass(frozen=True)
class PermutationAuditResult:
    """Outcome of the randomization test."""

    n_permutations: int
    alpha: float
    seed: int
    fold_scheme: str
    n_folds: int
    purge_radius: int
    observed_ic: float
    shuffle_mean_ic: float
    shuffle_std_ic: float
    shuffle_t_statistic: float
    shuffle_q05: float
    shuffle_q95: float
    p_value: float
    n_test_observations: int
    n_features: int
    verdict: Literal["clean", "leakage_detected"]

    @property
    def is_clean(self) -> bool:
        """Whether the audit found no evidence of leakage."""
        return self.verdict == "clean"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for the provenance block."""
        return {
            "n_permutations": self.n_permutations,
            "alpha": self.alpha,
            "seed": self.seed,
            "fold_scheme": self.fold_scheme,
            "n_folds": self.n_folds,
            "purge_radius": self.purge_radius,
            "observed_ic": self.observed_ic,
            "shuffle_mean_ic": self.shuffle_mean_ic,
            "shuffle_std_ic": self.shuffle_std_ic,
            "shuffle_t_statistic": self.shuffle_t_statistic,
            "shuffle_q05": self.shuffle_q05,
            "shuffle_q95": self.shuffle_q95,
            "p_value": self.p_value,
            "n_test_observations": self.n_test_observations,
            "n_features": self.n_features,
            "verdict": self.verdict,
        }

    def format(self) -> str:
        """Return a human-readable multi-line summary."""
        return (
            f"leakage audit [{self.verdict.upper()}] scheme={self.fold_scheme} "
            f"folds={self.n_folds} purge={self.purge_radius}\n"
            f"  permutations      : {self.n_permutations} (seed={self.seed}, alpha={self.alpha})\n"
            f"  observed IC       : {self.observed_ic:+.4f} (diagnostic only, not the test)\n"
            f"  shuffled-label IC : mean {self.shuffle_mean_ic:+.4f}, sd {self.shuffle_std_ic:.4f}, "
            f"q05 {self.shuffle_q05:+.4f}, q95 {self.shuffle_q95:+.4f}\n"
            f"  one-sided t-test  : t = {self.shuffle_t_statistic:+.3f}, p = {self.p_value:.4f} "
            f"over {self.n_test_observations} scored rows"
        )


def permute_labels_by_date(labels: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Return ``labels`` with its **dates permuted**, keeping every cross-section intact.

    Parameters
    ----------
    labels : pd.Series
        Label series on ``MultiIndex(datetime, instrument)``.
    rng : np.random.Generator
        Source of randomness; pass a seeded generator for reproducibility.

    Returns
    -------
    pd.Series
        Same index as ``labels``; the value at ``(perm[i], j)`` equals the original value at
        ``(i, j)``.  The set of values inside every date is unchanged while any temporal relation to
        the features is destroyed.
    """
    if not isinstance(labels.index, pd.MultiIndex):
        raise ValueError("labels must be indexed by MultiIndex(datetime, instrument)")
    dates = pd.DatetimeIndex(labels.index.get_level_values(_DATETIME_LEVEL).unique()).sort_values()
    permutation = rng.permutation(len(dates))
    mapping = {date: dates[permutation[position]] for position, date in enumerate(dates)}
    permuted_dates = labels.index.get_level_values(_DATETIME_LEVEL).map(mapping.get)
    permuted = labels.copy()
    permuted.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(permuted_dates), labels.index.get_level_values(_INSTRUMENT_LEVEL)],
        names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL],
    )
    return permuted.sort_index()


def _row_date_positions(labels: pd.Series) -> tuple[pd.DatetimeIndex, IntArray]:
    """Return the sorted unique dates and the position of every row's date within them."""
    row_dates = labels.index.get_level_values(_DATETIME_LEVEL)
    dates = pd.DatetimeIndex(row_dates.unique()).sort_values()
    positions = dates.get_indexer(row_dates)
    if (positions < 0).any():  # pragma: no cover - get_indexer over own level cannot fail
        raise ValueError("failed to map label rows onto the calendar")
    return dates, positions


def _build_folds(
    n_dates: int,
    row_positions: IntArray,
    config: AuditConfig,
    purge_radius: int,
    rng: np.random.Generator,
) -> list[tuple[BoolArray, BoolArray]]:
    """Return the ``(train_mask, test_mask)`` row masks of every evaluation fold.

    ``fold_scheme="purged"`` yields chronological expanding folds: fold :math:`k` trains on every
    date before the test block minus ``purge_radius`` dates (the label windows of those neighbours
    overlap the test labels, see ``LabelSpec.purge_radius``) and evaluates on the next block.

    ``fold_scheme="overlapping"`` assigns rows to folds at random, so the same date appears in
    training and evaluation.  It is the negative control: a pipeline built this way MUST be
    rejected by the audit.
    """
    folds: list[tuple[BoolArray, BoolArray]] = []
    if config.fold_scheme == "overlapping":
        assignment = rng.integers(0, config.n_folds, size=len(row_positions))
        for fold in range(config.n_folds):
            test_mask = assignment == fold
            folds.append((~test_mask, test_mask))
        return folds

    boundaries = np.linspace(0, n_dates, config.n_folds + 2).astype(int)
    for fold in range(1, config.n_folds + 1):
        test_lo, test_hi = int(boundaries[fold]), int(boundaries[fold + 1])
        train_hi = max(0, test_lo - purge_radius)
        train_mask = row_positions < train_hi
        test_mask = (row_positions >= test_lo) & (row_positions < test_hi)
        folds.append((train_mask, test_mask))
    return folds


def _ridge_fit_predict(train_x: Array, train_y: Array, test_x: Array, alpha: float) -> Array:
    """Fit an intercept-plus-ridge model in closed form and predict ``test_x``.

    A linear model is used on purpose: the audit must run ``B`` times, and the property under test
    is the *timing* of the pipeline, not the capacity of the model.  A neural network can be plugged
    in through the ``model`` argument of :meth:`LeakageAuditor.audit`.
    """
    design = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # do not penalize the intercept
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.hstack([np.ones((test_x.shape[0], 1)), test_x]) @ coefficients


def _mean_ic_statistic(
    predictions: Array,
    labels: pd.Series,
    index: pd.MultiIndex,
    min_obs: int,
) -> tuple[float, int]:
    """Return ``(mean out-of-sample rank IC, number of scored rows)``.

    The statistic of the randomization test is the mean out-of-sample rank IC over every evaluated
    date: a pipeline that can memorise permuted labels beats chance, which moves this mean away
    from zero.

    Notes
    -----
    When a model produces predictions with **no cross-sectional dispersion** - a constant, which is
    what an exact-match lookup yields when it finds nothing to copy - the rank correlation is
    mathematically undefined.  Such a predictor carries no cross-sectional information, so its IC is
    recorded as ``0.0`` rather than dropped: dropping it would bias the shuffle distribution by
    removing exactly the "no information" replicates.  ``NaN`` is returned only when no row could be
    scored at all, which the caller treats as a fatal condition.
    """
    scores = pd.Series(predictions, index=index, name="score").dropna()
    if scores.empty:
        return float("nan"), 0
    ic = rank_ic(scores, labels, min_obs=min_obs)
    if ic.empty:
        return 0.0, int(scores.shape[0])
    return float(ic.mean()), int(scores.shape[0])


class LeakageAuditor:
    """Randomization test for temporal leakage in a training-and-evaluation pipeline.

    See the module docstring for the statistical argument.  In short: permute the labels across
    dates, re-run the *same* pipeline, and require that the out-of-sample mean rank IC on permuted
    labels is consistent with the permutation null.  A pipeline scoring above that null is
    transferring information across the train/evaluate boundary and MUST be fixed before any result
    it produces is reported.
    """

    def __init__(self, config: AuditConfig | None = None) -> None:
        self.config = config or AuditConfig()

    @staticmethod
    def flatten_bundle(bundle: object) -> Array:
        """Return a flat ``(N, d)`` design matrix from a :class:`PanelBundle`'s last timestep.

        Convenience bridge for auditing a sequence pipeline with the cheap linear model: the most
        recent row of each window carries the contemporaneous information, which is where a timing
        defect shows up first.  ``NaN`` cells are replaced by the bundle's neutral fill value,
        because the audit tests timing, not the treatment of missing values.
        """
        # Imported here rather than at module scope: handlers imports this module indirectly through
        # the package, so a top-level import would create a cycle.
        from qresearch.data.handlers import PanelBundle  # pylint: disable=import-outside-toplevel

        if not isinstance(bundle, PanelBundle):
            raise TypeError(f"flatten_bundle expects a PanelBundle, got {type(bundle).__name__}")
        return np.nan_to_num(bundle.X[:, -1, :], nan=bundle.fill_value).astype("float64")

    def audit(
        self,
        design: Array,
        labels: pd.Series,
        *,
        horizon: int = 1,
        model: Callable[[Array, Array, Array], Array] | None = None,
        raise_on_leakage: bool = True,
    ) -> PermutationAuditResult:
        """Run the randomization test.

        Parameters
        ----------
        design : Array
            Row-aligned design matrix ``(N, d)``; the row order MUST match ``labels``.
        labels : pd.Series
            Target on ``MultiIndex(datetime, instrument)``, length ``N``.
        horizon : int
            Label horizon, used as the default purge radius (``PROJECT_SPEC.md`` 2.2.5).
        model : Callable | None
            Optional ``(train_x, train_y, test_x) -> test_pred`` callable, so a neural network can
            replace the built-in ridge model.  The model MUST be deterministic given its inputs:
            randomness inside it would inflate the null and destroy the test's power.
        raise_on_leakage : bool
            When ``True`` (default) a detected leak raises :class:`DataLeakageDetectedError`.

        Returns
        -------
        PermutationAuditResult

        Raises
        ------
        DataLeakageDetectedError
            When ``p_value < alpha`` and ``raise_on_leakage`` is set.
        ValueError
            On shape mismatch or a label index that is not a two-level MultiIndex.
        RuntimeError
            When no permutation produced a finite statistic: the audit cannot conclude anything, and
            a "clean" verdict would then be a false reassurance.
        """
        config = self.config
        matrix = np.asarray(design, dtype="float64")
        if matrix.ndim != 2:
            raise ValueError(f"design must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] != len(labels):
            raise ValueError(f"design has {matrix.shape[0]} rows but labels has {len(labels)} entries")
        if not isinstance(labels.index, pd.MultiIndex):
            raise ValueError("labels must be indexed by MultiIndex(datetime, instrument)")

        purge_radius = config.purge_radius if config.purge_radius is not None else max(0, int(horizon))

        def predict(train_x: Array, train_y: Array, test_x: Array) -> Array:
            if model is not None:
                return model(train_x, train_y, test_x)
            return _ridge_fit_predict(train_x, train_y, test_x, config.ridge_alpha)

        observed_predictions = self._predict_all(matrix, labels, purge_radius, predict)
        observed_ic, n_observations = _mean_ic_statistic(observed_predictions, labels, labels.index, config.min_obs)

        shuffle_statistics: list[float] = []
        for draw in range(config.n_permutations):
            rng = np.random.default_rng(np.random.SeedSequence(entropy=config.seed, spawn_key=(draw + 1,)))
            permuted = permute_labels_by_date(labels, rng)
            permuted_predictions = self._predict_all(matrix, permuted, purge_radius, predict)
            statistic, _ = _mean_ic_statistic(permuted_predictions, permuted, permuted.index, config.min_obs)
            if np.isfinite(statistic):
                shuffle_statistics.append(statistic)
        if len(shuffle_statistics) < 2:
            raise RuntimeError(
                "fewer than two finite shuffle statistics were produced, so no verdict is possible "
                "(reporting 'clean' here would be a false reassurance)"
            )

        shuffled = np.asarray(shuffle_statistics, dtype="float64")
        shuffle_mean = float(shuffled.mean())
        shuffle_std = float(shuffled.std(ddof=1))
        standard_error = shuffle_std / float(np.sqrt(shuffled.size))
        if standard_error > 0.0:
            t_statistic = shuffle_mean / standard_error
            p_value = float(1.0 - student_t.cdf(t_statistic, df=shuffled.size - 1))
        else:
            # A degenerate null (identical replicates).  Zero variance with a positive mean is
            # perfectly reproducible performance on permuted labels, which is the strongest possible
            # evidence of memorisation; a non-positive mean is clean by definition.
            t_statistic = float("inf") if shuffle_mean > 0.0 else 0.0
            p_value = 0.0 if shuffle_mean > 0.0 else 1.0

        verdict: Literal["clean", "leakage_detected"] = "leakage_detected" if p_value < config.alpha else "clean"

        result = PermutationAuditResult(
            n_permutations=config.n_permutations,
            alpha=config.alpha,
            seed=config.seed,
            fold_scheme=config.fold_scheme,
            n_folds=config.n_folds,
            purge_radius=purge_radius,
            observed_ic=observed_ic,
            shuffle_mean_ic=shuffle_mean,
            shuffle_std_ic=shuffle_std,
            shuffle_t_statistic=t_statistic,
            shuffle_q05=float(np.quantile(shuffled, 0.05)),
            shuffle_q95=float(np.quantile(shuffled, 0.95)),
            p_value=p_value,
            n_test_observations=n_observations,
            n_features=int(matrix.shape[1]),
            verdict=verdict,
        )
        if verdict == "leakage_detected" and raise_on_leakage:
            raise DataLeakageDetectedError(
                "the model predicts TIME-PERMUTED labels out of sample, which is only possible if the "
                "pipeline lets information cross the train/evaluate boundary.\n"
                f"{result.format()}\n"
                "Check, in this order: (1) date overlap between folds - the purge radius must be at "
                "least the label horizon; (2) processors or statistics fitted on the full sample; "
                "(3) rolling windows that include the current row; (4) row-identifying features "
                "(calendar trends, ids) combined with overlapping folds. "
                "PROJECT_SPEC.md 3.6 (PIT-1..PIT-8), task INF-08."
            )
        return result

    def _predict_all(
        self,
        design: Array,
        target: pd.Series,
        purge_radius: int,
        predict: Callable[[Array, Array, Array], Array],
    ) -> Array:
        """Return out-of-sample predictions for every evaluated row of one pipeline run."""
        dates, positions = _row_date_positions(target)
        fold_rng = np.random.default_rng(np.random.SeedSequence(entropy=self.config.seed, spawn_key=(0,)))
        folds = _build_folds(len(dates), positions, self.config, purge_radius, fold_rng)
        values = target.to_numpy(dtype="float64")
        predictions = np.full(len(target), np.nan, dtype="float64")
        for train_mask, test_mask in folds:
            usable_train = train_mask & np.isfinite(values)
            usable_test = test_mask & np.isfinite(values)
            if int(usable_test.sum()) == 0 or int(usable_train.sum()) < 10:
                continue
            predictions[usable_test] = predict(design[usable_train], values[usable_train], design[usable_test])
        return predictions


def assert_no_leakage(
    design: Array,
    labels: pd.Series,
    *,
    config: AuditConfig | None = None,
    horizon: int = 1,
    model: Callable[[Array, Array, Array], Array] | None = None,
) -> PermutationAuditResult:
    """Run the leakage audit and raise :class:`DataLeakageDetectedError` when a leak is detected.

    This is the form the training and reporting pipelines MUST call before a study is promoted: a
    result produced by a pipeline that fails this test is not reportable
    (``PROJECT_SPEC.md`` 1.2.2, 6.1).
    """
    return LeakageAuditor(config).audit(design, labels, horizon=horizon, model=model, raise_on_leakage=True)


def audit_prefix_invariance(
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    frame: pd.DataFrame,
    *,
    keep_ratio: float = 0.8,
    future_value: float = 1e9,
    atol: float = 0.0,
) -> None:
    """Assert that a feature pipeline's history does not depend on future rows.

    The complementary instrument to the label-shuffling test: a feature that embeds its own row's
    label (or any other future information) survives label shuffling but never survives this check.

    The pipeline is run twice: once on the calendar truncated to its first ``keep_ratio`` rows, and
    once on a calendar extended by extreme future rows.  Every row of the truncated run must be
    reproduced exactly by the extended run, because those rows precede the extension.

    Parameters
    ----------
    transform : Callable[[pd.DataFrame], pd.DataFrame]
        Feature pipeline taking a wide ``(date x instrument)`` frame and returning one.
    frame : pd.DataFrame
        Input panel; the last ``1 - keep_ratio`` of its rows play the role of "the future".
    keep_ratio : float
        Fraction of rows that must be reproducible without the future.
    future_value : float
        Extreme value written into the present rows before the extension, so a pipeline that peeks
        at them is detected.
    atol : float
        Absolute tolerance; ``0.0`` (default) requires bit-identical reproduction.

    Raises
    ------
    AssertionError
        Naming the first offending date and the magnitude of the difference.
    """
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1), got {keep_ratio}")
    cutoff = int(len(frame) * keep_ratio)
    if cutoff < 2 or cutoff >= len(frame):
        raise ValueError(f"keep_ratio={keep_ratio} leaves no room to split a frame of length {len(frame)}")

    prefix = frame.iloc[:cutoff]
    extended = frame.copy()
    extended.iloc[:cutoff] = future_value
    extended.iloc[::2] = np.nan

    prefix_result = transform(prefix)
    extended_result = transform(extended).reindex(prefix_result.index)

    if prefix_result.shape != extended_result.shape:
        raise AssertionError(
            f"pippeline changed its output shape when fed a longer calendar: "
            f"{prefix_result.shape} vs {extended_result.shape}"
        )
    both_nan = prefix_result.isna() & extended_result.isna()
    difference = (prefix_result - extended_result).abs().where(~both_nan, 0.0)
    offending = difference.max(axis=1)
    worst = float(offending.max()) if len(offending) else 0.0
    if worst > atol or (offending > atol).any():
        first_date = offending[offending > atol].index[0]
        raise AssertionError(
            f"prefix-invariance violated: recomputing the pipeline with future rows present changed "
            f"the output at {first_date:%Y-%m-%d} by {worst:.6g} (atol={atol}). A feature must be "
            "F_t-measurable (PROJECT_SPEC.md 2.1, task INF-08)."
        )
