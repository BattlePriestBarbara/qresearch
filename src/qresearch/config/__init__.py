"""Typed configuration schema, path resolution and loaders (task ``INF-10``, ``PROJECT_SPEC.md`` 3.5).

``3.5`` requires configuration to be declarative YAML consumed through
``qlib.utils.init_instance_by_config``; library code MUST NOT hard-code study parameters.  This
package implements the three pieces of that requirement:

* :mod:`qresearch.config.paths` - the ``ADR-005`` resolution policy for every location (no literal
  path may appear in a tracked file);
* :mod:`qresearch.config.schema` - the frozen dataclasses that own every documented default;
* :mod:`qresearch.config.loader` - YAML loading, placeholder expansion, strict validation and the
  resolved-configuration dump with its ``config_hash``.

Nothing in this package imports Qlib: a configuration must be checkable without a data store.
"""

from __future__ import annotations

from .errors import ConfigError
from .loader import (
    ALLOWED_PLACEHOLDERS,
    PLACEHOLDER_RE,
    REQUIRED_BLOCKS,
    TOP_LEVEL_BLOCKS,
    build_study_config,
    config_hash,
    dump_resolved_config,
    expand_placeholders,
    load_study_config,
    load_yaml_document,
    resolved_document,
)
from .paths import (
    APPLIED_OVERRIDES_SECTION,
    ARTIFACT_ROOT_ENV_VAR,
    DATA_DIR_ENV_VAR,
    DEFAULT_DATA_DIR_PARTS,
    INTERPRETER_ENV_VAR,
    PROVIDER_URI_ENV_VAR,
    Override,
    applied_overrides,
    artifact_root,
    cache_root,
    candidate_interpreters,
    clear_applied_overrides,
    data_dir,
    describe_overrides,
    interpreter,
    project_root,
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
    "APPLIED_OVERRIDES_SECTION",
    "ARTIFACT_ROOT_ENV_VAR",
    "DATA_DIR_ENV_VAR",
    "DEFAULT_DATA_DIR_PARTS",
    "INTERPRETER_ENV_VAR",
    "PLACEHOLDER_RE",
    "PROVIDER_URI_ENV_VAR",
    "REQUIRED_BLOCKS",
    "TOP_LEVEL_BLOCKS",
    "BacktestConfig",
    "ConfigError",
    "DataHandlerConfig",
    "ExchangeCosts",
    "ModelConfig",
    "ModelHyperparameters",
    "Override",
    "ProcessorSpec",
    "QlibInitConfig",
    "SegmentSpec",
    "StrategyConfig",
    "StudyConfig",
    "applied_overrides",
    "artifact_root",
    "build_study_config",
    "cache_root",
    "candidate_interpreters",
    "clear_applied_overrides",
    "config_hash",
    "data_dir",
    "describe_overrides",
    "dump_resolved_config",
    "expand_placeholders",
    "interpreter",
    "load_study_config",
    "load_yaml_document",
    "project_root",
    "resolved_document",
]
