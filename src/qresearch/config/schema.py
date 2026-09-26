"""Typed configuration schema (task ``INF-10``, ``PROJECT_SPEC.md`` 3.5).

``3.5`` makes two demands that shape this module:

1. *"All runtime behavior MUST be driven by declarative YAML ... never by hard-coded constants in
   library code.  Constants MAY exist only as documented defaults inside a dataclass in
   ``qresearch.config.schema``."*  Every default of this project therefore lives here, in a frozen
   dataclass, with the reason for the value in its docstring.
2. *"The config loader MUST reject unknown keys, missing segments, non-overlapping date ranges,
   ``fit_end_time`` later than the ``train`` end, and any ``*_cost`` equal to zero in a workflow
   that produces reported numbers."*  Structural invariants (types, ranges, enum membership,
   positivity) are enforced in ``__post_init__`` below, so an invalid object cannot be constructed
   at all; the *policy* rules that need the whole document (unknown keys, zero costs, the
   multiprocessing opt-in) live in :mod:`qresearch.config.loader`.

Why ``dataclasses`` and not ``pydantic``: ``3.5`` names a dataclass, and ``pydantic`` is not part of
the frozen dependency closure of ``PROJECT_SPEC.md`` 3.1 / ADR-003 - introducing it would create
exactly the drift the lock files exist to prevent.  The guarantees this project needs
(``mypy --strict`` plus explicit unknown-key rejection) do not require a validation framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from types import MappingProxyType
from typing import Final

from .errors import ConfigError
from .paths import DATA_DIR_ENV_VAR

__all__ = [
    "ENCODERS",
    "LOSSES",
    "REBALANCE_FREQUENCIES",
    "BacktestConfig",
    "DataHandlerConfig",
    "ExchangeCosts",
    "ModelConfig",
    "ModelHyperparameters",
    "ProcessorSpec",
    "QlibInitConfig",
    "SegmentSpec",
    "StrategyConfig",
    "StudyConfig",
]

ENCODERS: Final[tuple[str, ...]] = ("gru", "lstm")
LOSSES: Final[tuple[str, ...]] = ("student_t", "gaussian")
REBALANCE_FREQUENCIES: Final[tuple[str, ...]] = ("daily", "weekly", "monthly")
"""Enumerated configuration values; a misspelling is rejected by name instead of silently ignored."""


def _parse_date(value: str, *, field_name: str) -> date:
    """Return ``value`` as an ISO date, raising :class:`ConfigError` when it is not one."""
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be an ISO date string (YYYY-MM-DD), got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ConfigError(f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}") from error


def _frozen(payload: Mapping[str, object]) -> MappingProxyType[str, object]:
    """Return an immutable view of a free-form ``kwargs`` mapping."""
    return MappingProxyType(dict(payload))


@dataclass(frozen=True)
class ProcessorSpec:
    """One Qlib processor declaration (``init_instance_by_config``-compatible).

    The YAML key is ``class``, which is a Python keyword, hence the ``class_name`` attribute and the
    explicit translation in :meth:`from_mapping`.
    """

    class_name: str
    module_path: str | None = None
    kwargs: MappingProxyType[str, object] = field(default_factory=lambda: _frozen({}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, where: str) -> ProcessorSpec:
        """Build a spec from a YAML mapping, translating ``class`` into ``class_name``."""
        unknown = set(payload) - {"class", "class_name", "module_path", "kwargs"}
        if unknown:
            raise ConfigError(f"{where} has unknown key(s) {sorted(unknown)}; allowed: class, module_path, kwargs")
        class_name = payload.get("class", payload.get("class_name"))
        if not isinstance(class_name, str) or not class_name:
            raise ConfigError(f"{where} must declare a non-empty 'class'")
        module_path = payload.get("module_path")
        if module_path is not None and not isinstance(module_path, str):
            raise ConfigError(f"{where}.module_path must be a string, got {module_path!r}")
        raw_kwargs = payload.get("kwargs") or {}
        if not isinstance(raw_kwargs, dict):
            raise ConfigError(f"{where}.kwargs must be a mapping, got {raw_kwargs!r}")
        return cls(class_name=class_name, module_path=module_path, kwargs=_frozen(raw_kwargs))

    def to_dict(self) -> dict[str, object]:
        """Return the Qlib-compatible mapping (``class`` key restored) for ``resolved_config.yaml``."""
        payload: dict[str, object] = {"class": self.class_name}
        if self.module_path is not None:
            payload["module_path"] = self.module_path
        if self.kwargs:
            payload["kwargs"] = dict(self.kwargs)
        return payload


@dataclass(frozen=True)
class DataHandlerConfig:
    """``AlphaHandlerLP`` data window and processor chain (``PROJECT_SPEC.md`` 3.4.1 / 3.5)."""

    start_time: str
    end_time: str
    fit_start_time: str
    fit_end_time: str
    instruments: str
    infer_processors: tuple[ProcessorSpec, ...] = ()
    learn_processors: tuple[ProcessorSpec, ...] = ()
    freq: str = "day"
    class_name: str = "AlphaHandlerLP"
    module_path: str = "qresearch.data.handlers"

    def __post_init__(self) -> None:
        """Reject an unordered, empty or self-contradictory data window before anything consumes it."""
        start = _parse_date(self.start_time, field_name="data_handler.start_time")
        end = _parse_date(self.end_time, field_name="data_handler.end_time")
        fit_start = _parse_date(self.fit_start_time, field_name="data_handler.fit_start_time")
        fit_end = _parse_date(self.fit_end_time, field_name="data_handler.fit_end_time")
        if end < start:
            raise ConfigError(f"data_handler.end_time {self.end_time} precedes start_time {self.start_time}")
        if fit_end < fit_start:
            raise ConfigError(
                f"data_handler.fit_end_time {self.fit_end_time} precedes fit_start_time {self.fit_start_time}"
            )
        if not self.instruments:
            raise ConfigError("data_handler.instruments must name a market (for example 'csi300')")
        if fit_end > end:
            raise ConfigError(
                f"data_handler.fit_end_time {self.fit_end_time} lies outside the data window (ends "
                f"{self.end_time}); a processor cannot be fitted on data it never sees (PIT-2)"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the handler block as a JSON/YAML-serializable mapping."""
        return {
            "class": self.class_name,
            "module_path": self.module_path,
            "kwargs": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "fit_start_time": self.fit_start_time,
                "fit_end_time": self.fit_end_time,
                "instruments": self.instruments,
                "freq": self.freq,
                "infer_processors": [spec.to_dict() for spec in self.infer_processors],
                "learn_processors": [spec.to_dict() for spec in self.learn_processors],
            },
        }


@dataclass(frozen=True)
class SegmentSpec:
    """``train`` / ``valid`` / ``test`` boundaries (``PROJECT_SPEC.md`` 3.5, purge: ADR-004).

    Each segment is an inclusive ``(start, end)`` ISO-date pair.  The segments MUST be strictly
    increasing and MUST NOT overlap: an overlap silently trains on evaluation data, which is the
    defect the whole point-in-time apparatus exists to prevent.
    """

    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]
    purge_radius: int = 0

    def __post_init__(self) -> None:
        """Validate ordering, non-overlap and the purge radius."""
        named: tuple[tuple[str, tuple[str, str]], ...] = (
            ("train", self.train),
            ("valid", self.valid),
            ("test", self.test),
        )
        bounds: list[tuple[str, date, date]] = []
        for name, pair in named:
            if len(pair) != 2:
                raise ConfigError(f"segments.{name} must be a (start, end) pair, got {pair!r}")
            start = _parse_date(pair[0], field_name=f"segments.{name}[0]")
            end = _parse_date(pair[1], field_name=f"segments.{name}[1]")
            if end < start:
                raise ConfigError(f"segments.{name} end {pair[1]} precedes start {pair[0]}")
            bounds.append((name, start, end))
        for (name, _, previous_end), (next_name, next_start, _) in pairwise(bounds):
            if next_start <= previous_end:
                raise ConfigError(
                    f"segments.{next_name} starts {next_start} on or before the end of segments.{name} "
                    f"({previous_end}): the segments MUST NOT overlap"
                )
        if self.purge_radius < 0:
            raise ConfigError(f"segments.purge_radius must be >= 0, got {self.purge_radius}")

    def to_dict(self) -> dict[str, object]:
        """Return the segment block as a JSON/YAML-serializable mapping."""
        return {
            "train": list(self.train),
            "valid": list(self.valid),
            "test": list(self.test),
            "purge_radius": self.purge_radius,
        }


@dataclass(frozen=True)
class ModelHyperparameters:
    """Neural-arm hyper-parameters; the defaults are the verified configuration of ``3.5``.

    ``nu_min = 2.05`` is the lower bound of the Student-:math:`t` degrees of freedom required by
    ``3.4.3`` (``nu = nu_min + softplus(log_nu)``); a value at or below ``2`` would leave the
    likelihood's variance undefined.
    """

    d_feat: int = 158
    seq_len: int = 20
    hidden_size: int = 128
    num_layers: int = 2
    encoder: str = "gru"
    loss: str = "student_t"
    lambda_rank: float = 0.3
    n_epochs: int = 100
    lr: float = 1.0e-3
    batch_size: int = 1024
    early_stop: int = 20
    seed: int = 42
    nu_min: float = 2.05

    def __post_init__(self) -> None:
        """Reject hyper-parameters that are structurally unusable."""
        if self.encoder not in ENCODERS:
            raise ConfigError(f"model.encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if self.loss not in LOSSES:
            raise ConfigError(f"model.loss must be one of {LOSSES}, got {self.loss!r}")
        if self.nu_min <= 2.0:
            raise ConfigError(f"model.nu_min must exceed 2.0 (Student-t variance), got {self.nu_min}")
        for name, value in (
            ("d_feat", self.d_feat),
            ("seq_len", self.seq_len),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("n_epochs", self.n_epochs),
            ("batch_size", self.batch_size),
            ("early_stop", self.early_stop),
        ):
            if value <= 0:
                raise ConfigError(f"model.{name} must be positive, got {value}")
        if self.lr <= 0.0:
            raise ConfigError(f"model.lr must be positive, got {self.lr}")
        if self.lambda_rank < 0.0:
            raise ConfigError(f"model.lambda_rank must be >= 0, got {self.lambda_rank}")
        if self.seed < 0:
            raise ConfigError(f"model.seed must be >= 0, got {self.seed}")

    def to_dict(self) -> dict[str, object]:
        """Return the hyper-parameter block as a JSON/YAML-serializable mapping."""
        return {
            "d_feat": self.d_feat,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "encoder": self.encoder,
            "loss": self.loss,
            "lambda_rank": self.lambda_rank,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "early_stop": self.early_stop,
            "seed": self.seed,
            "nu_min": self.nu_min,
        }


@dataclass(frozen=True)
class ModelConfig:
    """Model class reference plus its :class:`ModelHyperparameters`."""

    class_name: str = "StudentTSeqModel"
    module_path: str = "qresearch.models.qlib_adapter"
    params: ModelHyperparameters = field(default_factory=ModelHyperparameters)

    def to_dict(self) -> dict[str, object]:
        """Return the model block as a JSON/YAML-serializable mapping."""
        return {"class": self.class_name, "module_path": self.module_path, "kwargs": self.params.to_dict()}


@dataclass(frozen=True)
class ExchangeCosts:
    """Transaction-cost parameters mirroring the Qlib ``Exchange`` semantics (``3.4.5``).

    The defaults are the values ``PROJECT_SPEC.md`` 1.2 fixes for this project (``open_cost =
    0.0015``, ``close_cost = 0.0025``, ``min_cost = 5.0``, ``trade_unit = 100``).  A study that
    declares different values MUST record an ADR, and the loader refuses an *exactly zero* cost in a
    workflow that produces reported numbers - a zero cost turns the reported net performance into a
    gross performance claim, which ``6.5`` lists as a prohibited practice.
    """

    open_cost: float = 0.0015
    close_cost: float = 0.0025
    min_cost: float = 5.0
    trade_unit: int = 100
    limit_threshold: float = 0.095
    deal_price: str = "close"

    def __post_init__(self) -> None:
        """Reject negative costs, a wrong lot size and an unusable price-limit band."""
        if self.trade_unit != 100:
            raise ConfigError(f"exchange.trade_unit MUST be 100 shares (3.4.4), got {self.trade_unit}")
        for name, value in (("open_cost", self.open_cost), ("close_cost", self.close_cost)):
            if value < 0.0:
                raise ConfigError(f"exchange.{name} must be >= 0, got {value}")
        if self.min_cost < 0.0:
            raise ConfigError(f"exchange.min_cost must be >= 0, got {self.min_cost}")
        if not 0.0 < self.limit_threshold < 1.0:
            raise ConfigError(f"exchange.limit_threshold must lie in (0, 1), got {self.limit_threshold}")
        if self.deal_price not in {"close", "open"}:
            raise ConfigError(f"exchange.deal_price must be 'close' or 'open', got {self.deal_price!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the ``exchange_kwargs`` mapping of ``3.4.5``."""
        return {
            "open_cost": self.open_cost,
            "close_cost": self.close_cost,
            "min_cost": self.min_cost,
            "trade_unit": self.trade_unit,
            "limit_threshold": self.limit_threshold,
            "deal_price": self.deal_price,
        }


@dataclass(frozen=True)
class StrategyConfig:
    """Allocation strategy parameters (``3.4.4``; ``TopK`` baseline and cost-aware MV arm)."""

    class_name: str = "CostAwareMVStrategy"
    module_path: str = "qresearch.portfolio.strategies"
    topk: int = 50
    n_drop: int = 5
    risk_degree: float = 0.95
    rebalance: str = "daily"

    def __post_init__(self) -> None:
        """Reject an unusable portfolio size, risk budget or cadence."""
        if self.topk <= 0:
            raise ConfigError(f"strategy.topk must be positive, got {self.topk}")
        if self.n_drop < 0:
            raise ConfigError(f"strategy.n_drop must be >= 0, got {self.n_drop}")
        if self.n_drop >= self.topk:
            raise ConfigError(f"strategy.n_drop ({self.n_drop}) must be smaller than topk ({self.topk})")
        if not 0.0 < self.risk_degree <= 1.0:
            raise ConfigError(f"strategy.risk_degree must lie in (0, 1], got {self.risk_degree}")
        if self.rebalance not in REBALANCE_FREQUENCIES:
            raise ConfigError(f"strategy.rebalance must be one of {REBALANCE_FREQUENCIES}, got {self.rebalance!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the strategy block as a serializable mapping."""
        return {
            "class": self.class_name,
            "module_path": self.module_path,
            "kwargs": {
                "topk": self.topk,
                "n_drop": self.n_drop,
                "risk_degree": self.risk_degree,
                "rebalance": self.rebalance,
            },
        }


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest window, account and cost model (``3.4.5``)."""

    start_time: str
    end_time: str
    benchmark: str = "SH000300"
    account: float = 1.0e8
    executor: str = "SimulatorExecutor"
    executor_module_path: str = "qlib.backtest.executor"
    time_per_step: str = "day"
    exchange: ExchangeCosts = field(default_factory=ExchangeCosts)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def __post_init__(self) -> None:
        """Reject an unordered window, an empty benchmark or a non-daily step."""
        start = _parse_date(self.start_time, field_name="backtest.start_time")
        end = _parse_date(self.end_time, field_name="backtest.end_time")
        if end <= start:
            raise ConfigError(f"backtest.end_time {self.end_time} must follow start_time {self.start_time}")
        if not self.benchmark:
            raise ConfigError("backtest.benchmark must name an index; the run MUST fail loudly without one")
        if self.account <= 0.0:
            raise ConfigError(f"backtest.account must be positive, got {self.account}")
        if self.time_per_step != "day":
            raise ConfigError(
                f"backtest.time_per_step MUST be 'day' (3.4.5); got {self.time_per_step!r}. "
                "A different cadence is realized by the strategy returning an empty TradeDecisionWO "
                "on non-rebalance steps, not by the executor."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the backtest block as a serializable mapping."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "benchmark": self.benchmark,
            "account": self.account,
            "executor": {
                "class": self.executor,
                "module_path": self.executor_module_path,
                "kwargs": {"time_per_step": self.time_per_step, "generate_portfolio_metrics": True},
            },
            "exchange_kwargs": self.exchange.to_dict(),
            "strategy": self.strategy.to_dict(),
        }


@dataclass(frozen=True)
class QlibInitConfig:
    """The ``qlib.init`` block (``3.5``).

    ``provider_uri`` is *not* a literal in any tracked file: :mod:`qresearch.config.loader` fills it
    from :func:`qresearch.config.paths.data_dir` (``$QLIB_DATA_DIR`` with the documented fallback),
    which is why the dataclass only requires a non-empty string.

    ``kernels`` MUST stay ``1`` unless a study opts in explicitly: ``5.3`` records that Windows +
    non-ASCII paths + the torch ``DataLoader`` spawn do not mix, and that determinism suffers.
    """

    provider_uri: str
    region: str = "cn"
    kernels: int = 1
    expression_cache: str | None = None
    dataset_cache: str | None = None
    mlflow_uri: str | None = None
    default_exp_name: str = "alpha_research"
    allow_multiprocessing: bool = False

    def __post_init__(self) -> None:
        """Reject a missing provider, a bad region and un-opted-in multiprocessing."""
        if not self.provider_uri:
            raise ConfigError(
                "qlib_init.provider_uri is empty; it is resolved from "
                f"qresearch.config.paths.data_dir() ({DATA_DIR_ENV_VAR} or the documented fallback)"
            )
        if self.region not in {"cn", "us"}:
            raise ConfigError(f"qlib_init.region must be 'cn' or 'us', got {self.region!r}")
        if self.kernels < 1:
            raise ConfigError(f"qlib_init.kernels must be >= 1, got {self.kernels}")
        if self.kernels != 1 and not self.allow_multiprocessing:
            raise ConfigError(
                f"qlib_init.kernels={self.kernels} requires allow_multiprocessing: true "
                "(PROJECT_SPEC.md 5.3: the Windows spawn and non-ASCII paths do not mix, and "
                "determinism is a requirement of the reported statistics)"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the whole ``qlib_init`` block of ``3.5``, caches and experiment manager included."""
        return {
            "provider_uri": self.provider_uri,
            "region": self.region,
            "kernels": self.kernels,
            "expression_cache": self.expression_cache,
            "dataset_cache": self.dataset_cache,
            "exp_manager": {
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": self.mlflow_uri, "default_exp_name": self.default_exp_name},
            },
        }


@dataclass(frozen=True)
class StudyConfig:
    """The fully resolved study document: the root of every ``resolved_config.yaml``.

    The ``__post_init__`` checks are the *cross-block* rules of ``3.5`` that a single block cannot
    see on its own:

    * ``fit_end_time`` MUST NOT be later than the end of the ``train`` segment (otherwise a
      processor is fitted on evaluation data, violating PIT-2);
    * every segment MUST lie inside the handler's data window;
    * the backtest window MUST lie inside the ``test`` segment - ``1.4`` makes the acceptance
      criteria binding "on the ``test`` segment", so a reported number computed elsewhere is not a
      criterion result.
    """

    name: str
    qlib_init: QlibInitConfig
    data_handler: DataHandlerConfig
    segments: SegmentSpec
    backtest: BacktestConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        """Enforce the cross-block date rules and the identity of the document."""
        if not self.name:
            raise ConfigError("study.name must be non-empty; it names the experiment record")
        if self.seed < 0:
            raise ConfigError(f"study.seed must be >= 0, got {self.seed}")
        train_start = _parse_date(self.segments.train[0], field_name="segments.train")
        train_end = _parse_date(self.segments.train[1], field_name="segments.train")
        fit_end = _parse_date(self.data_handler.fit_end_time, field_name="data_handler.fit_end_time")
        if fit_end > train_end:
            raise ConfigError(
                f"data_handler.fit_end_time {self.data_handler.fit_end_time} is later than the end of the "
                f"train segment ({self.segments.train[1]}): processors MUST be fitted on train only (PIT-2)"
            )
        window_start = _parse_date(self.data_handler.start_time, field_name="data_handler.start_time")
        window_end = _parse_date(self.data_handler.end_time, field_name="data_handler.end_time")
        if window_start > train_start:
            raise ConfigError(
                f"data_handler.start_time {self.data_handler.start_time} starts after the train segment: "
                "the handler window MUST cover every segment"
            )
        for segment_name, pair in (("valid", self.segments.valid), ("test", self.segments.test)):
            segment_end = _parse_date(pair[1], field_name=f"segments.{segment_name}")
            if segment_end > window_end:
                raise ConfigError(
                    f"the {segment_name} segment ends {pair[1]}, beyond data_handler.end_time "
                    f"{self.data_handler.end_time}: the handler window MUST cover every segment"
                )
        self._assert_backtest_inside_test()

    def _assert_backtest_inside_test(self) -> None:
        """Require the backtest window to be a sub-window of the ``test`` segment (``1.4``)."""
        backtest_start = _parse_date(self.backtest.start_time, field_name="backtest.start_time")
        backtest_end = _parse_date(self.backtest.end_time, field_name="backtest.end_time")
        test_start = _parse_date(self.segments.test[0], field_name="segments.test")
        test_end = _parse_date(self.segments.test[1], field_name="segments.test")
        if backtest_start < test_start or backtest_end > test_end:
            raise ConfigError(
                f"the backtest window {self.backtest.start_time}..{self.backtest.end_time} is not contained "
                f"in the test segment {self.segments.test[0]}..{self.segments.test[1]}: the acceptance "
                "criteria of 1.4 are binding on the test segment only"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete resolved document, ready to be dumped as ``resolved_config.yaml``."""
        return {
            "study": {"name": self.name, "seed": self.seed},
            "qlib_init": self.qlib_init.to_dict(),
            "data_handler": self.data_handler.to_dict(),
            "segments": self.segments.to_dict(),
            "model": self.model.to_dict(),
            "backtest": self.backtest.to_dict(),
        }
