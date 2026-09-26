"""Declarative configuration loading and validation (task ``INF-10``, ``PROJECT_SPEC.md`` 3.5).

The loader turns a YAML document into a frozen :class:`~qresearch.config.schema.StudyConfig` and
enforces the *policy* rules of ``3.5`` that need the whole document:

* **unknown keys are rejected** at every level (a typo in a YAML key would otherwise silently take
  the dataclass default, which is how a study runs with the wrong hyper-parameters);
* **zero transaction costs are rejected** unless the caller passes ``allow_zero_cost=True``; a
  reported net performance computed with a zero cost is a gross claim, which ``6.5`` prohibits;
* **only path variables may be substituted.**  A ``${VAR}`` placeholder is expanded from the
  documented path variables of :mod:`qresearch.config.paths` and nothing else, because ``3.5``
  limits environment overrides to *paths*.  ``${VAR:-fallback}`` supplies an explicit default;
* **every honoured override is recorded** and embedded in ``resolved_config.yaml`` under
  ``environment_overrides`` (``3.5``: overrides MUST be logged when applied).

The fully resolved document is dumped next to the artifacts and hashed with SHA-256
(:func:`config_hash`), which is the ``config_hash`` field ``3.5`` and ``3.7`` require.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, get_type_hints

import yaml

from ..utils.io import hash_mapping, write_yaml
from .errors import ConfigError
from .paths import (
    APPLIED_OVERRIDES_SECTION,
    ARTIFACT_ROOT_ENV_VAR,
    DATA_DIR_ENV_VAR,
    PROVIDER_URI_ENV_VAR,
    applied_overrides,
    artifact_root,
    data_dir,
    describe_overrides,
)
from .schema import (
    BacktestConfig,
    DataHandlerConfig,
    ExchangeCosts,
    ModelConfig,
    ModelHyperparameters,
    ProcessorSpec,
    QlibInitConfig,
    SegmentSpec,
    StrategyConfig,
    StudyConfig,
)

__all__ = [
    "ALLOWED_PLACEHOLDERS",
    "PLACEHOLDER_RE",
    "REQUIRED_BLOCKS",
    "TOP_LEVEL_BLOCKS",
    "build_study_config",
    "config_hash",
    "dump_resolved_config",
    "expand_placeholders",
    "load_study_config",
    "load_yaml_document",
    "resolved_document",
]

PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
"""Matches ``${NAME}`` and ``${NAME:-fallback}``."""

ALLOWED_PLACEHOLDERS: Final[tuple[str, ...]] = (DATA_DIR_ENV_VAR, PROVIDER_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR)
"""The only names a configuration file may substitute (``3.5``: overrides are limited to paths)."""

TOP_LEVEL_BLOCKS: Final[tuple[str, ...]] = (
    "study",
    "qlib_init",
    "data_handler",
    "segments",
    "model",
    "backtest",
)
"""Permitted top-level blocks of a study document."""

ARTIFACTS_SHORTHAND: Final[str] = "artifacts"
"""Accepted ``${ARTIFACT_ROOT:-artifacts}`` fallback, meaning ``<project_root>/artifacts``."""


def expand_placeholders(text: str, *, source: str) -> str:
    """Substitute the documented path variables in ``text``.

    Raises
    ------
    ConfigError
        If the text refers to a variable that is not one of :data:`ALLOWED_PLACEHOLDERS`; any other
        substitution would be a behaviour switch hidden in a config file, which ``3.5`` forbids.
    """

    def resolve(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in ALLOWED_PLACEHOLDERS:
            raise ConfigError(
                f"{source}: ${{{name}}} is not an allowed placeholder. Only path variables may be "
                f"substituted from the environment: {', '.join(ALLOWED_PLACEHOLDERS)}. A non-path "
                "override would let the environment change a study silently (PROJECT_SPEC.md 3.5)."
            )
        configured = (os.environ.get(name) or "").strip()
        resolved = artifact_root() if name == ARTIFACT_ROOT_ENV_VAR else data_dir()
        if configured:
            return as_posix(resolved)
        inline_default = match.group("default")
        if inline_default is not None:
            return _resolve_inline_default(name, inline_default)
        return as_posix(resolved)

    return PLACEHOLDER_RE.sub(resolve, text)


def as_posix(path: Path) -> str:
    """Return ``path`` in POSIX form.

    A substituted path lands inside a YAML scalar, where a Windows separator would either be an
    escape sequence (a doubled separator in front of a letter such as ``U`` opens an eight-digit
    ``\\U`` escape and the document stops parsing) or a literal backslash.  Forward slashes are
    valid in a YAML scalar *and* in Qlib's own path handling (``Path(...).resolve()`` normalizes
    them on Windows), so one substitution rule works on every platform.
    """
    return Path(path).as_posix()


def _resolve_inline_default(name: str, default: str) -> str:
    """Resolve an inline ``${VAR:-fallback}`` value.

    Two spellings are accepted: the literal ``artifacts`` for :data:`ARTIFACT_ROOT_ENV_VAR`, and an
    absolute path for any of the variables.  Anything else is rejected instead of guessed, because a
    relative default depends on the working directory of whoever launches the process.
    """
    candidate = default.strip()
    if name == ARTIFACT_ROOT_ENV_VAR and candidate == ARTIFACTS_SHORTHAND:
        return as_posix(artifact_root())
    if candidate and Path(candidate).is_absolute():
        return as_posix(Path(candidate))
    raise ConfigError(
        f"${{{name}:-{default}}} is not a supported fallback: use an absolute path"
        + (f" or the literal {ARTIFACTS_SHORTHAND!r}" if name == ARTIFACT_ROOT_ENV_VAR else "")
    )


def load_yaml_document(path: Path) -> dict[str, Any]:
    """Read a YAML file, expand the permitted placeholders and require a top-level mapping."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    text = expand_placeholders(source.read_text(encoding="utf-8"), source=str(source))
    loaded: object = _safe_load(text, source=str(source))
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source}: the document must be a mapping of blocks, got {type(loaded).__name__}")
    return {str(key): value for key, value in loaded.items()}


def _safe_load(text: str, *, source: str) -> object:
    """Parse YAML with the safe loader, converting a syntax error into :class:`ConfigError`."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{source}: invalid YAML: {error}") from error


# ---------------------------------------------------------------------------------------
# Block-level helpers: strict key sets and explicit type coercion
# ---------------------------------------------------------------------------------------
def _as_mapping(payload: object, *, where: str) -> dict[str, Any]:
    """Return ``payload`` as a string-keyed dict, raising :class:`ConfigError` otherwise."""
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(payload).__name__}")
    return {str(key): value for key, value in payload.items()}


def _reject_unknown(payload: Mapping[str, Any], allowed: tuple[str, ...] | set[str], *, where: str) -> None:
    """Raise :class:`ConfigError` listing every key of ``payload`` that is not in ``allowed``."""
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _as_str(value: object, *, where: str) -> str:
    """Return ``value`` as a non-empty string, raising :class:`ConfigError` otherwise."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _as_pair(value: object, *, where: str) -> tuple[str, str]:
    """Return a ``(start, end)`` ISO-date pair from a two-element list."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{where} must be a two-element [start, end] list, got {value!r}")
    first, second = value
    return _as_date_text(first, where=f"{where}[0]"), _as_date_text(second, where=f"{where}[1]")


def _as_date_text(value: object, *, where: str) -> str:
    """Return an ISO date string, accepting the ``datetime.date`` that YAML parses a bare date into.

    ``start_time: 2008-01-01`` is parsed by YAML into :class:`datetime.date`, not into a string; both
    spellings are legal configuration and MUST be accepted, so the conversion happens here instead of
    being duplicated in every caller.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    raise ConfigError(f"{where} must be an ISO date (YYYY-MM-DD), got {value!r}")


def _coerce_scalars(target: type, payload: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Type-check and normalize the scalar fields of a dataclass.

    A YAML scalar has one of four shapes (``int``/``float``/``str``/``bool``), and a mismatch here is
    the most common configuration defect (``seq_len: "20"``).  Rejecting it by name *and* type turns
    a silent ``TypeError`` deep inside a training loop into a one-line diagnosis.
    """
    hints = get_type_hints(target)
    _reject_unknown(payload, {item.name for item in fields(target)}, where=where)
    coerced: dict[str, Any] = {}
    for name, value in payload.items():
        expected = hints[name]
        if expected is bool and not isinstance(value, bool):
            raise ConfigError(f"{where}.{name} must be bool, got {value!r}")
        if expected is int and not (isinstance(value, int) and not isinstance(value, bool)):
            raise ConfigError(f"{where}.{name} must be int, got {value!r}")
        if expected is float and (isinstance(value, bool) or not isinstance(value, (int, float, str))):
            raise ConfigError(f"{where}.{name} must be a number, got {value!r}")
        if expected is float and isinstance(value, str):
            # YAML 1.1 (PyYAML) requires a *signed* exponent for a float, so `1.0e-3` is a float
            # while `1.0e8` is the string "1.0e8".  Rejecting the latter would make a
            # specification-conformant document unloadable, so a numeric string is accepted here
            # and normalized; anything that does not parse stays an error.
            try:
                value = float(value)
            except ValueError as error:
                raise ConfigError(f"{where}.{name} must be a number, got {value!r}") from error
        if expected is float:
            value = float(value)
        if expected is str and not isinstance(value, str):
            raise ConfigError(f"{where}.{name} must be str, got {value!r}")
        coerced[name] = value
    return coerced


def _processor_specs(raw: object, *, where: str) -> tuple[ProcessorSpec, ...]:
    """Return the processor chain described by ``raw`` (``None`` means an empty chain)."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"{where} must be a list of processors, got {type(raw).__name__}")
    return tuple(
        ProcessorSpec.from_mapping(_as_mapping(item, where=f"{where}[{index}]"), where=f"{where}[{index}]")
        for index, item in enumerate(raw)
    )


def _handler_config(block: Mapping[str, Any], *, where: str) -> DataHandlerConfig:
    """Build the :class:`DataHandlerConfig` of a ``data_handler`` block."""
    _reject_unknown(block, ("class", "module_path", "kwargs"), where=where)
    kwargs = _as_mapping(block.get("kwargs") or {}, where=f"{where}.kwargs")
    required = ("start_time", "end_time", "fit_start_time", "fit_end_time", "instruments")
    missing = [name for name in required if name not in kwargs]
    if missing:
        raise ConfigError(f"{where}.kwargs is missing the required key(s) {missing}")
    _reject_unknown(kwargs, (*required, "freq", "infer_processors", "learn_processors"), where=f"{where}.kwargs")
    return DataHandlerConfig(
        start_time=_as_date_text(kwargs["start_time"], where=f"{where}.kwargs.start_time"),
        end_time=_as_date_text(kwargs["end_time"], where=f"{where}.kwargs.end_time"),
        fit_start_time=_as_date_text(kwargs["fit_start_time"], where=f"{where}.kwargs.fit_start_time"),
        fit_end_time=_as_date_text(kwargs["fit_end_time"], where=f"{where}.kwargs.fit_end_time"),
        instruments=_as_str(kwargs["instruments"], where=f"{where}.kwargs.instruments"),
        freq=str(kwargs.get("freq", "day")),
        infer_processors=_processor_specs(kwargs.get("infer_processors"), where=f"{where}.kwargs.infer_processors"),
        learn_processors=_processor_specs(kwargs.get("learn_processors"), where=f"{where}.kwargs.learn_processors"),
        class_name=str(block.get("class", "AlphaHandlerLP")),
        module_path=str(block.get("module_path", "qresearch.data.handlers")),
    )


def _segment_spec(block: Mapping[str, Any], *, where: str) -> SegmentSpec:
    """Build the :class:`SegmentSpec`, rejecting a document with a missing segment."""
    _reject_unknown(block, ("train", "valid", "test", "purge_radius"), where=where)
    missing = [name for name in ("train", "valid", "test") if name not in block]
    if missing:
        raise ConfigError(f"{where} is missing the segment(s) {missing}; train/valid/test are all REQUIRED (3.5)")
    purge = block.get("purge_radius", 0)
    if isinstance(purge, bool) or not isinstance(purge, int):
        raise ConfigError(f"{where}.purge_radius must be int, got {purge!r}")
    return SegmentSpec(
        train=_as_pair(block["train"], where=f"{where}.train"),
        valid=_as_pair(block["valid"], where=f"{where}.valid"),
        test=_as_pair(block["test"], where=f"{where}.test"),
        purge_radius=purge,
    )


def _qlib_init_config(block: Mapping[str, Any], *, provider_uri: str | None, where: str) -> QlibInitConfig:
    """Build the :class:`QlibInitConfig`, resolving the provider path through ADR-005."""
    _reject_unknown(
        block,
        (
            "provider_uri",
            "region",
            "kernels",
            "expression_cache",
            "dataset_cache",
            "exp_manager",
            "allow_multiprocessing",
        ),
        where=where,
    )
    manager = _as_mapping(block.get("exp_manager") or {}, where=f"{where}.exp_manager")
    _reject_unknown(manager, ("class", "module_path", "kwargs"), where=f"{where}.exp_manager")
    manager_class = manager.get("class")
    if manager_class not in (None, "MLflowExpManager"):
        raise ConfigError(
            f"{where}.exp_manager.class must be 'MLflowExpManager' (3.7 requires the Qlib recorder backend), "
            f"got {manager_class!r}"
        )
    manager_kwargs = _as_mapping(manager.get("kwargs") or {}, where=f"{where}.exp_manager.kwargs")
    _reject_unknown(manager_kwargs, ("uri", "default_exp_name"), where=f"{where}.exp_manager.kwargs")
    configured_provider = provider_uri if provider_uri is not None else block.get("provider_uri")
    provider = str(configured_provider) if configured_provider is not None else str(data_dir())
    if not Path(provider).is_absolute():
        raise ConfigError(
            f"{where}.provider_uri must be absolute after substitution, got {provider!r}; "
            f"use the {DATA_DIR_ENV_VAR} placeholder instead of a relative literal"
        )
    mlflow_uri = manager_kwargs.get("uri")
    if mlflow_uri is None:
        mlflow_uri = f"file:{(artifact_root() / 'mlruns').as_posix()}"
    caches: dict[str, str | None] = {}
    for cache_name in ("expression_cache", "dataset_cache"):
        cache_value = block.get(cache_name)
        if cache_value is not None and not isinstance(cache_value, str):
            raise ConfigError(f"{where}.{cache_name} must be a directory path or null, got {cache_value!r}")
        caches[cache_name] = cache_value
    return QlibInitConfig(
        provider_uri=provider,
        region=str(block.get("region", "cn")),
        kernels=int(block.get("kernels", 1)),
        expression_cache=caches["expression_cache"],
        dataset_cache=caches["dataset_cache"],
        mlflow_uri=str(mlflow_uri),
        default_exp_name=str(manager_kwargs.get("default_exp_name", "alpha_research")),
        allow_multiprocessing=bool(block.get("allow_multiprocessing", False)),
    )


def _model_config(block: Mapping[str, Any], *, where: str) -> ModelConfig:
    """Build the :class:`ModelConfig`, rejecting an unknown hyper-parameter by name."""
    _reject_unknown(block, ("class", "module_path", "kwargs"), where=where)
    kwargs = _as_mapping(block.get("kwargs") or {}, where=f"{where}.kwargs")
    params = replace(ModelHyperparameters(), **_coerce_scalars(ModelHyperparameters, kwargs, where=f"{where}.kwargs"))
    return ModelConfig(
        class_name=str(block.get("class", "StudentTSeqModel")),
        module_path=str(block.get("module_path", "qresearch.models.qlib_adapter")),
        params=params,
    )


def _backtest_config(block: Mapping[str, Any], *, allow_zero_cost: bool, where: str) -> BacktestConfig:
    """Build the :class:`BacktestConfig`, applying the zero-cost policy of ``3.5``."""
    _reject_unknown(
        block,
        ("start_time", "end_time", "benchmark", "account", "exchange_kwargs", "strategy", "executor"),
        where=where,
    )
    for required in ("start_time", "end_time"):
        if required not in block:
            raise ConfigError(f"{where} is missing the required key {required!r}")
    exchange_block = _as_mapping(block.get("exchange_kwargs") or {}, where=f"{where}.exchange_kwargs")
    exchange_values = _coerce_scalars(ExchangeCosts, exchange_block, where=f"{where}.exchange_kwargs")
    defaults = ExchangeCosts()
    if not allow_zero_cost:
        for cost_name in ("open_cost", "close_cost", "min_cost"):
            effective = exchange_values.get(cost_name, getattr(defaults, cost_name))
            if effective == 0:
                raise ConfigError(
                    f"{where}.exchange_kwargs.{cost_name} is zero. A reported number computed at a zero "
                    "cost is a gross result, which PROJECT_SPEC.md 6.5 prohibits; pass "
                    "allow_zero_cost=True only to study a costless variant explicitly."
                )
    exchange = replace(defaults, **exchange_values)

    strategy_block = _as_mapping(block.get("strategy") or {}, where=f"{where}.strategy")
    _reject_unknown(strategy_block, ("class", "module_path", "kwargs"), where=f"{where}.strategy")
    strategy_kwargs = _as_mapping(strategy_block.get("kwargs") or {}, where=f"{where}.strategy.kwargs")
    strategy = replace(
        StrategyConfig(),
        **_coerce_scalars(StrategyConfig, strategy_kwargs, where=f"{where}.strategy.kwargs"),
        class_name=str(strategy_block.get("class", "CostAwareMVStrategy")),
        module_path=str(strategy_block.get("module_path", "qresearch.portfolio.strategies")),
    )

    executor_block = _as_mapping(block.get("executor") or {}, where=f"{where}.executor")
    _reject_unknown(executor_block, ("class", "module_path", "kwargs"), where=f"{where}.executor")
    executor_kwargs = _as_mapping(executor_block.get("kwargs") or {}, where=f"{where}.executor.kwargs")
    _reject_unknown(executor_kwargs, ("time_per_step", "generate_portfolio_metrics"), where=f"{where}.executor.kwargs")
    account = block.get("account", 1.0e8)
    if isinstance(account, bool) or not isinstance(account, (int, float, str)):
        raise ConfigError(f"{where}.account must be a number, got {account!r}")
    try:
        account_value = float(account)
    except ValueError as error:
        raise ConfigError(f"{where}.account must be a number, got {account!r}") from error
    return BacktestConfig(
        start_time=_as_date_text(block["start_time"], where=f"{where}.start_time"),
        end_time=_as_date_text(block["end_time"], where=f"{where}.end_time"),
        benchmark=str(block.get("benchmark", "SH000300")),
        account=account_value,
        executor=str(executor_block.get("class", "SimulatorExecutor")),
        executor_module_path=str(executor_block.get("module_path", "qlib.backtest.executor")),
        time_per_step=str(executor_kwargs.get("time_per_step", "day")),
        exchange=exchange,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------
REQUIRED_BLOCKS: Final[tuple[str, ...]] = ("qlib_init", "data_handler", "segments", "model", "backtest")
"""Blocks a study document MUST declare; ``study`` is optional (name and seed have defaults)."""


def build_study_config(
    document: Mapping[str, Any],
    *,
    source: str = "<mapping>",
    provider_uri: str | None = None,
    allow_zero_cost: bool = False,
) -> StudyConfig:
    """Validate a study document and return the frozen :class:`StudyConfig`.

    Parameters
    ----------
    document : Mapping[str, Any]
        Parsed (and placeholder-expanded) blocks.
    source : str
        Label used in error messages; the file name when the document came from disk.
    provider_uri : str | None
        Explicit override of ``qlib_init.provider_uri`` (CLI flag); wins over the document, which
        wins over :func:`qresearch.config.paths.data_dir`.
    allow_zero_cost : bool
        Opt out of the zero-cost rejection **explicitly**; see :func:`_backtest_config`.

    Returns
    -------
    StudyConfig
        The validated configuration.

    Raises
    ------
    ConfigError
        On an unknown key, a missing block, a type mismatch, an inconsistent date range or a zero
        transaction cost.
    """
    _reject_unknown(document, TOP_LEVEL_BLOCKS, where=source)
    missing = [name for name in REQUIRED_BLOCKS if name not in document]
    if missing:
        raise ConfigError(f"{source} is missing the required block(s) {missing}; required: {list(REQUIRED_BLOCKS)}")
    study_block = _as_mapping(document.get("study") or {}, where=f"{source}.study")
    _reject_unknown(study_block, ("name", "seed"), where=f"{source}.study")
    seed = study_block.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"{source}.study.seed must be int, got {seed!r}")
    name = _as_str(study_block.get("name", "unnamed_study"), where=f"{source}.study.name")
    qlib_init_block = _as_mapping(document["qlib_init"], where=f"{source}.qlib_init")
    handler_block = _as_mapping(document["data_handler"], where=f"{source}.data_handler")
    segment_block = _as_mapping(document["segments"], where=f"{source}.segments")
    backtest_block = _as_mapping(document["backtest"], where=f"{source}.backtest")
    model_block = _as_mapping(document["model"], where=f"{source}.model")
    return StudyConfig(
        name=name,
        qlib_init=_qlib_init_config(qlib_init_block, provider_uri=provider_uri, where=f"{source}.qlib_init"),
        data_handler=_handler_config(handler_block, where=f"{source}.data_handler"),
        segments=_segment_spec(segment_block, where=f"{source}.segments"),
        backtest=_backtest_config(backtest_block, allow_zero_cost=allow_zero_cost, where=f"{source}.backtest"),
        model=_model_config(model_block, where=f"{source}.model"),
        seed=seed,
    )


def load_study_config(
    path: Path,
    *,
    provider_uri: str | None = None,
    allow_zero_cost: bool = False,
) -> StudyConfig:
    """Load, expand and validate a study document from disk."""
    return build_study_config(
        load_yaml_document(path),
        source=str(path),
        provider_uri=provider_uri,
        allow_zero_cost=allow_zero_cost,
    )


def resolved_document(config: StudyConfig) -> dict[str, Any]:
    """Return the fully resolved document, including the applied environment overrides (``3.5``)."""
    document = config.to_dict()
    document[APPLIED_OVERRIDES_SECTION] = [item.to_dict() for item in applied_overrides()]
    return document


def config_hash(config: StudyConfig | Mapping[str, Any]) -> str:
    """Return the SHA-256 ``config_hash`` of a configuration (``3.5``, ``3.7``)."""
    document = resolved_document(config) if isinstance(config, StudyConfig) else dict(config)
    return hash_mapping(document)


def dump_resolved_config(
    config: StudyConfig,
    directory: Path,
    *,
    filename: str = "resolved_config.yaml",
) -> Path:
    """Write ``resolved_config.yaml`` into ``directory`` and return its path.

    ``3.5`` requires the fully resolved configuration to be dumped next to every artifact, and
    ``3.7`` requires it to be logged together with the configuration hash; the caller therefore
    keeps :func:`config_hash` at hand for the provenance block.
    """
    return write_yaml(Path(directory) / filename, resolved_document(config))


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``qresearch-config``): materialize a template or validate a document.

    ``--expand`` exists because Qlib itself performs only ``~`` expansion (``Path.expanduser``), so a
    ``${...}`` template MUST be materialized before ``qrun`` sees it.  ``--check`` validates a study
    document without touching the data store.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a configuration error.
    """
    parser = argparse.ArgumentParser(
        prog="qresearch-config",
        description="Validate a study configuration or materialize a ${...} template (task INF-10).",
    )
    parser.add_argument("--expand", type=Path, help="print the template with the path placeholders substituted")
    parser.add_argument("--check", type=Path, help="validate a study document and print its config_hash")
    parser.add_argument("--out", type=Path, default=None, help="write the --expand result to this file")
    parser.add_argument("--provider-uri", type=Path, default=None, help="override qlib_init.provider_uri")
    args = parser.parse_args(argv)
    if args.expand is None and args.check is None:
        parser.error("pass --expand FILE or --check FILE")

    try:
        if args.expand is not None:
            source = Path(args.expand)
            if not source.is_file():
                raise ConfigError(f"configuration template not found: {source}")
            expanded = expand_placeholders(source.read_text(encoding="utf-8"), source=str(source))
            if args.out is None:
                print(expanded, end="" if expanded.endswith("\n") else "\n")
            else:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(expanded, encoding="utf-8")
                print(f"materialized {source} -> {args.out}")
        if args.check is not None:
            provider = None if args.provider_uri is None else str(args.provider_uri)
            config = load_study_config(Path(args.check), provider_uri=provider)
            print(f"{args.check}: VALID")
            print(f"  study      : {config.name} (seed={config.seed})")
            print(f"  provider   : {config.qlib_init.provider_uri}")
            print(f"  config_hash: {config_hash(config)}")
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(describe_overrides())
    return 0
