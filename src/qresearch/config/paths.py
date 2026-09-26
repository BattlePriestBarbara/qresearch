"""Resolution of every filesystem location the library touches (``ADR-005``, tasks ``INF-09``/``INF-10``).

This module is the **single** place where a directory is decided.  Nothing else in
``src/qresearch``, ``scripts/`` or ``tests/`` may contain a drive letter, a UNC path or a home
shortcut: that rule is machine-checked by ``scripts/prepublish_audit.py`` in the pre-push gate, so a
new literal is a failed push rather than a latent portability defect.

Resolution table (``ADR-005``)
------------------------------
====================== ================================== ==================================
Location               Environment override               Fallback
====================== ================================== ==================================
read-only market data  ``QLIB_DATA_DIR``                  ``<home>/.qlib/qlib_data/cn_data``
read-only market data  ``QLIB_PROVIDER_URI`` (alias)      -- (legacy name of ``3.5``)
generated artifacts    ``ARTIFACT_ROOT``                  ``<project_root>/artifacts``
Qlib caches            -- (derived)                       ``<artifact_root>/qlib_cache``
pinned interpreter     ``QRESEARCH_PYTHON``               ``%CONDA_PREFIX%/python[.exe]``,
                                                          else ``sys.executable``
====================== ================================== ==================================

Policy
------
1. **Declarative, never literal.**  A path is an input, not a constant.
2. **Only paths are overridable.**  No other environment variable may change behaviour;
   ``PROJECT_SPEC.md`` 3.5 limits overrides to paths and this module is where that is enforced.
3. **No silent guessing.**  A malformed override (empty value, relative value, a path that is a
   file) raises :class:`~qresearch.config.errors.ConfigError` with the offending value and the
   remediation.  A *missing* value falls back to the documented default above.
4. **Overrides are logged.**  Every honoured environment override is recorded and can be printed
   with :func:`describe_overrides`; the config loader embeds it in ``resolved_config.yaml``.
5. **Strictness is not weakened.**  ``QRESEARCH_PYTHON`` selects *which* interpreter the contract is
   verified against; it cannot disable the verification (the exact patch version, the pinned
   float-critical distributions, the forbidden-interpreter markers and the absence of a bypass
   variable are all unchanged - see ``qresearch.env``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .errors import ConfigError

__all__ = [
    "APPLIED_OVERRIDES_SECTION",
    "ARTIFACTS_DIR_NAME",
    "ARTIFACT_ROOT_ENV_VAR",
    "CACHE_DIR_NAME",
    "CONDA_PREFIX_ENV_VAR",
    "CONDA_PYTHON_RELATIVE",
    "DATA_DIR_DEFAULT_SOURCE",
    "DATA_DIR_ENV_VAR",
    "DEFAULT_DATA_DIR_PARTS",
    "INTERPRETER_ENV_VAR",
    "PROVIDER_URI_ENV_VAR",
    "Override",
    "applied_overrides",
    "artifact_root",
    "cache_root",
    "candidate_interpreters",
    "clear_applied_overrides",
    "data_dir",
    "describe_overrides",
    "interpreter",
    "project_root",
]

# ---------------------------------------------------------------------------------------
# Documented knobs.  Names are normative: PROJECT_SPEC.md 3.5 lists them.
# ---------------------------------------------------------------------------------------
DATA_DIR_ENV_VAR: Final[str] = "QLIB_DATA_DIR"
PROVIDER_URI_ENV_VAR: Final[str] = "QLIB_PROVIDER_URI"
ARTIFACT_ROOT_ENV_VAR: Final[str] = "ARTIFACT_ROOT"
INTERPRETER_ENV_VAR: Final[str] = "QRESEARCH_PYTHON"
CONDA_PREFIX_ENV_VAR: Final[str] = "CONDA_PREFIX"

DEFAULT_DATA_DIR_PARTS: Final[tuple[str, ...]] = (".qlib", "qlib_data", "cn_data")
"""Fallback market-data location, relative to the user's home directory.

Built with :func:`pathlib.Path.home` rather than a ``~`` literal so that the source contains no
home-directory shortcut (which the pre-publish audit treats as a personal path).
"""

CACHE_DIR_NAME: Final[str] = "qlib_cache"
ARTIFACTS_DIR_NAME: Final[str] = "artifacts"

APPLIED_OVERRIDES_SECTION: Final[str] = "environment_overrides"
"""Key under which :func:`describe_overrides` output lands in ``resolved_config.yaml``."""

_MARKER_FILE: Final[str] = "pyproject.toml"


@dataclass(frozen=True)
class Override:
    """One environment override that was honoured, recorded so it is never silent."""

    name: str
    value: str
    role: str
    source: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation for the provenance block."""
        return {"name": self.name, "value": self.value, "role": self.role, "source": self.source}


_RECORDED: dict[str, Override] = {}


def _remember(name: str, value: Path, *, role: str, source: str) -> None:
    """Record an honoured override; the last resolution of a name wins."""
    _RECORDED[name] = Override(name=name, value=str(value), role=role, source=source)


def applied_overrides() -> tuple[Override, ...]:
    """Return every environment override honoured in this process, sorted by variable name."""
    return tuple(_RECORDED[name] for name in sorted(_RECORDED))


def clear_applied_overrides() -> None:
    """Forget the recorded overrides (test helper; production code never needs this)."""
    _RECORDED.clear()


def describe_overrides() -> str:
    """Return a human-readable, log-ready description of the honoured overrides."""
    recorded = applied_overrides()
    if not recorded:
        return "applied environment overrides: none (all locations use the documented defaults)"
    lines = ["applied environment overrides:"]
    lines.extend(f"  {item.name} ({item.role}) -> {item.value} [{item.source}]" for item in recorded)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Validation helpers - the three rules that turn a string into a usable directory
# ---------------------------------------------------------------------------------------
def _override_value(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the stripped value of ``name``, or ``None`` when the variable is unset.

    An empty or whitespace-only value raises :class:`ConfigError`: it is a configuration mistake,
    and silently substituting the default would hide exactly the kind of accident (a truncated
    ``export``) this module exists to catch.
    """
    lookup = os.environ if environ is None else environ
    raw = lookup.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        raise ConfigError(
            f"{name} is set but empty.\n"
            "  Remediation: unset it to use the documented default, or give it an absolute directory."
        )
    return value


def _as_absolute(value: str | Path, *, source: str) -> Path:
    """Return ``value`` as a path, rejecting a relative one with an actionable message."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigError(
            f"{source} must be an absolute path, got {value!r}.\n"
            "  A relative value depends on the working directory of whoever launches the process, "
            "so it is rejected rather than resolved against an implicit base.\n"
            "  Remediation: set it to an absolute directory, or unset it to use the default."
        )
    return path


def _validated_directory(path: Path, *, source: str, require_exists: bool) -> Path:
    """Reject a path that is a file or a dangling location; return it unchanged otherwise."""
    if path.exists() and not path.is_dir():
        raise ConfigError(f"{source} points at a file, not a directory: {path}")
    if require_exists and not path.is_dir():
        raise ConfigError(
            f"{source} does not exist or is not a directory: {path}\n"
            "  Remediation: point the variable at the existing directory, or create it."
        )
    return path


# ---------------------------------------------------------------------------------------
# Project root - discovered, never declared
# ---------------------------------------------------------------------------------------
def _discover_project_root(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` containing ``pyproject.toml``, else ``None``."""
    for candidate in (start, *start.parents):
        if (candidate / _MARKER_FILE).is_file():
            return candidate
    return None


def project_root() -> Path:
    """Return the project root directory (``src/`` layout: ``<root>/src/qresearch``).

    The root is discovered by walking up from this file until the packaging marker appears, so the
    checkout can live anywhere.  A non-editable install (no marker above the installed package)
    raises :class:`ConfigError` instead of guessing, because writing artifacts into
    ``site-packages`` is never the intent.
    """
    found = _discover_project_root(Path(__file__).resolve().parent)
    if found is not None:
        return found
    raise ConfigError(
        f"cannot locate the project root: no {_MARKER_FILE} above {Path(__file__).resolve()}\n"
        f"  Remediation: run from a source checkout (installed editable, per ADR-003), or set "
        f"{ARTIFACT_ROOT_ENV_VAR} to an absolute directory for generated artifacts."
    )


# ---------------------------------------------------------------------------------------
# Data, artifacts and caches
# ---------------------------------------------------------------------------------------
DATA_DIR_DEFAULT_SOURCE: Final[str] = "default (<home>/.qlib/qlib_data/cn_data)"
"""Source label used when no override is configured for the market data."""


def data_dir(explicit: str | Path | None = None, *, require_exists: bool = False) -> Path:
    """Return the read-only market-data root (the Qlib binary store).

    Resolution order: ``explicit``, then ``QLIB_DATA_DIR``, then ``QLIB_PROVIDER_URI`` (the legacy
    name of ``PROJECT_SPEC.md`` 3.5, kept as an accepted alias so an existing shell profile keeps
    working), then ``<home>/.qlib/qlib_data/cn_data`` - Qlib's own documented default, not a guess.

    Parameters
    ----------
    explicit : str | Path | None
        Highest-priority value, used by CLI flags and by tests.
    require_exists : bool
        When ``True`` a non-existent directory raises :class:`ConfigError`.  Used by consumers that
        are about to open the store, never by the import-time environment check: a fresh clone must
        be importable before the market data has been downloaded.

    Returns
    -------
    Path
        The resolved, validated directory (not necessarily existing, unless required).
    """
    if explicit is not None:
        return _validated_directory(
            _as_absolute(explicit, source="data_dir(explicit)"),
            source="data_dir(explicit)",
            require_exists=require_exists,
        )
    for name in (DATA_DIR_ENV_VAR, PROVIDER_URI_ENV_VAR):
        configured = _override_value(name)
        if configured is None:
            continue
        path = _validated_directory(_as_absolute(configured, source=name), source=name, require_exists=require_exists)
        _remember(name, path, role="market data (read-only)", source="environment")
        return path
    return _validated_directory(
        Path.home().joinpath(*DEFAULT_DATA_DIR_PARTS),
        source=DATA_DIR_DEFAULT_SOURCE,
        require_exists=require_exists,
    )


def artifact_root(explicit: str | Path | None = None, *, require_exists: bool = False) -> Path:
    """Return the writable artifact root (MLflow store, caches, reports, weights, tables).

    Resolution order: ``explicit``, then ``ARTIFACT_ROOT``, then ``<project_root>/artifacts``.
    Everything written by a run lives here, and ``artifacts/`` is git-ignored as a whole.
    """
    if explicit is not None:
        return _validated_directory(
            _as_absolute(explicit, source="artifact_root(explicit)"),
            source="artifact_root(explicit)",
            require_exists=require_exists,
        )
    configured = _override_value(ARTIFACT_ROOT_ENV_VAR)
    if configured is not None:
        path = _validated_directory(
            _as_absolute(configured, source=ARTIFACT_ROOT_ENV_VAR),
            source=ARTIFACT_ROOT_ENV_VAR,
            require_exists=require_exists,
        )
        _remember(ARTIFACT_ROOT_ENV_VAR, path, role="generated artifacts", source="environment")
        return path
    return _validated_directory(
        project_root() / ARTIFACTS_DIR_NAME,
        source="default (<project_root>/artifacts)",
        require_exists=require_exists,
    )


def cache_root(explicit: str | Path | None = None, *, require_exists: bool = False) -> Path:
    """Return the Qlib cache root, which MUST stay outside the read-only provider.

    Resolution order: ``explicit``, then ``<artifact_root>/qlib_cache``.  There is deliberately no
    dedicated variable: a second writable root could be pointed back inside the market data, which
    is the one write the read-only contract forbids (``ADR-005``, consequence 3).
    """
    if explicit is not None:
        return _validated_directory(
            _as_absolute(explicit, source="cache_root(explicit)"),
            source="cache_root(explicit)",
            require_exists=require_exists,
        )
    return _validated_directory(
        artifact_root() / CACHE_DIR_NAME,
        source="derived (<artifact_root>/qlib_cache)",
        require_exists=require_exists,
    )


# ---------------------------------------------------------------------------------------
# The pinned interpreter: selecting a location must not weaken the contract
# ---------------------------------------------------------------------------------------
CONDA_PYTHON_RELATIVE: Final[tuple[str, ...]] = ("python.exe",) if os.name == "nt" else ("bin", "python")
"""Path of the interpreter inside an activated conda prefix, per platform."""


def candidate_interpreters(environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return the interpreter candidates of the pinned contract, in priority order.

    Priority: ``$QRESEARCH_PYTHON``, then ``$CONDA_PREFIX`` joined with
    :data:`CONDA_PYTHON_RELATIVE` (the environment created by ``environment.yml``), then the running
    interpreter (:data:`sys.executable`).

    The function is pure - it inspects a mapping and touches no filesystem - so
    ``scripts/hook_runner.py`` can mirror it in stdlib-only code and a test can assert that the two
    implementations agree candidate for candidate.
    """
    lookup = os.environ if environ is None else environ
    candidates: list[Path] = []
    explicit = (lookup.get(INTERPRETER_ENV_VAR) or "").strip()
    if explicit:
        candidates.append(Path(explicit))
    prefix = (lookup.get(CONDA_PREFIX_ENV_VAR) or "").strip()
    if prefix:
        candidates.append(Path(prefix).joinpath(*CONDA_PYTHON_RELATIVE))
    running = Path(sys.executable)
    if running not in candidates:
        candidates.append(running)
    return tuple(candidates)


def interpreter(explicit: str | Path | None = None) -> Path:
    """Return the interpreter that the environment contract is verified against.

    The **location** of the interpreter is declarative (``ADR-005``); the **strictness** around it
    is not.  ``qresearch.env.verify_environment`` still rejects a wrong patch version, a drifted
    float-critical distribution and a forbidden interpreter build, and
    ``tests/test_env_contract.py`` still proves that no variable can disable those checks.  An
    override therefore selects *which* interpreter must satisfy the contract - it cannot relax it.

    Raises
    ------
    ConfigError
        If ``explicit`` or ``$QRESEARCH_PYTHON`` names something that is not an existing file.
    """
    if explicit is not None:
        return _validated_executable(
            _as_absolute(explicit, source="interpreter(explicit)"), source="interpreter(explicit)"
        )
    configured = _override_value(INTERPRETER_ENV_VAR)
    if configured is not None:
        path = _validated_executable(_as_absolute(configured, source=INTERPRETER_ENV_VAR), source=INTERPRETER_ENV_VAR)
        _remember(INTERPRETER_ENV_VAR, path, role="pinned interpreter", source="environment")
        return path
    prefix = _override_value(CONDA_PREFIX_ENV_VAR)
    if prefix is not None:
        candidate = _as_absolute(prefix, source=CONDA_PREFIX_ENV_VAR).joinpath(*CONDA_PYTHON_RELATIVE)
        if candidate.is_file():
            _remember(CONDA_PREFIX_ENV_VAR, candidate, role="pinned interpreter", source="environment")
            return candidate
    return Path(sys.executable)


def _validated_executable(path: Path, *, source: str) -> Path:
    """Reject an interpreter path that does not exist, with the remediation in the message."""
    if not path.is_file():
        raise ConfigError(
            f"{source} does not point at an existing interpreter: {path}\n"
            f"  Remediation: set {INTERPRETER_ENV_VAR} to the interpreter of the activated "
            "qresearch environment (see environment.yml / ADR-003), or unset it to fall back to "
            "CONDA_PREFIX and then to the running interpreter."
        )
    return path
