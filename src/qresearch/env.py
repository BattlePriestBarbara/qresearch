"""Environment contract enforcement and read-only data-source guards (task ``INF-01``).

This module implements three non-negotiable engineering rules from ``PROJECT_SPEC.md``:

1. **Exact dependency locking.**  The runtime MUST be CPython ``3.12.7`` running the pinned
   interpreter, and every float-critical distribution MUST match its full patch version.
   Any mismatch raises :class:`EnvironmentContractError` (a :class:`RuntimeError`) and the
   process MUST terminate; silent degradation and continue-after-warning are forbidden.
2. **Import-time enforcement.**  :mod:`qresearch` calls :func:`verify_environment` while being
   imported, so the check necessarily precedes any ``qlib.init`` call in user code.
3. **Read-only data source.**  The Qlib binary store located by
   :func:`qresearch.config.paths.data_dir` is a read-only input.  All Qlib caches are redirected to
   a writable directory outside it by :func:`build_qlib_init_kwargs`, and
   :class:`ProviderWriteGuard` verifies by fingerprint that nothing under the provider tree changed
   during a computation.

Path policy (``ADR-005``): no location is written into this file.  The provider uri, the cache
root and the expected interpreter are resolved from the documented environment overrides with the
documented fallbacks, and every honoured override is logged (see
:func:`qresearch.config.paths.describe_overrides`).  Only the *location* is configurable; the
strictness of the contract below is not.

Rationale for strictness (``PROJECT_SPEC.md`` 1.2.2): a version drift in ``numpy``/``scipy``/
``torch`` changes the low-order bits of the IC series, which propagates into the Newey-West
long-run variance and can move a ``p``-value across a significance threshold.  A pipeline that
silently runs on the wrong binaries produces invalid inference.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .config import paths
from .config.errors import ConfigError

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_PROVIDER_URI",
    "EXPECTED_INTERPRETER",
    "POLLUTION_MARKERS",
    "PYTHON_VERSION_REQUIRED",
    "REQUIRED_DISTRIBUTIONS",
    "EnvironmentContractError",
    "EnvironmentReport",
    "ProviderWriteGuard",
    "ReadOnlyViolationError",
    "TreeFingerprint",
    "assert_provider_isolation",
    "build_qlib_init_kwargs",
    "fingerprint_tree",
    "main",
    "verify_environment",
]

# ---------------------------------------------------------------------------------------
# The pinned contract.  Single source of truth for INF-01; mirrored by requirements.txt,
# environment.yml and .mypy.ini so that a drift in any of them is detectable by inspection.
# ---------------------------------------------------------------------------------------
PYTHON_VERSION_REQUIRED: Final[str] = "3.12.7"
# The pinned interpreter is DECLARATIVE (ADR-005): $QRESEARCH_PYTHON, else the interpreter of the
# activated conda prefix, else the running interpreter.  Only the *location* is configurable; the
# strictness below (exact patch version, distribution pins, forbidden markers, no bypass variable)
# is not, and an override that names a non-existent interpreter raises ConfigError instead of being
# ignored.
EXPECTED_INTERPRETER: Final[Path] = paths.interpreter()
FORBIDDEN_INTERPRETER_MARKERS: Final[tuple[str, ...]] = (
    "msys64",
    "mingw64",
    "windowsapps",
    "microsoft\\windowsapps",
)
# Canonical locations, resolved by ADR-005: $QLIB_DATA_DIR (legacy alias $QLIB_PROVIDER_URI) with
# the <home>/.qlib/qlib_data/cn_data fallback, and $ARTIFACT_ROOT with the <project_root>/artifacts
# fallback.  Nothing here names a drive, a user or a checkout.
DEFAULT_PROVIDER_URI: Final[Path] = paths.data_dir()
DEFAULT_CACHE_ROOT: Final[Path] = paths.cache_root()
POLLUTION_MARKERS: Final[tuple[str, ...]] = (
    "qlib_cache",
    "expression_cache",
    "dataset_cache",
    "__pycache__",
)


@dataclass(frozen=True)
class DistributionRequirement:
    """A single pinned distribution and whether its version affects floating-point results."""

    name: str
    version: str
    float_critical: bool = False

    @property
    def requirement(self) -> str:
        """Return the PEP 508 style requirement string, e.g. ``numpy==1.26.4``."""
        return f"{self.name}=={self.version}"


REQUIRED_DISTRIBUTIONS: Final[tuple[DistributionRequirement, ...]] = (
    DistributionRequirement("pyqlib", "0.9.7"),
    DistributionRequirement("numpy", "1.26.4", float_critical=True),
    DistributionRequirement("scipy", "1.13.1", float_critical=True),
    DistributionRequirement("pandas", "2.2.2", float_critical=True),
    DistributionRequirement("statsmodels", "0.14.2"),
    DistributionRequirement("torch", "2.8.0", float_critical=True),
    DistributionRequirement("cvxpy", "1.9.1", float_critical=True),
    DistributionRequirement("mlflow", "3.2.0"),
    DistributionRequirement("lightgbm", "4.6.0", float_critical=True),
    DistributionRequirement("matplotlib", "3.8.4"),
    DistributionRequirement("scikit-learn", "1.5.1", float_critical=True),
    DistributionRequirement("PyYAML", "6.0.1"),
    DistributionRequirement("pyarrow", "21.0.0"),
)


class EnvironmentContractError(ConfigError):
    """Raised when the runtime environment violates the pinned contract (INF-01).

    Inherits from :class:`~qresearch.config.errors.ConfigError` and therefore from
    :class:`RuntimeError`, so an uncaught violation still terminates the process with a non-zero
    exit status instead of being swallowed by ``except Exception`` handlers written for data
    errors, while callers may catch the configuration family as one type.
    """


class ReadOnlyViolationError(ConfigError):
    """Raised when the read-only market-data source may have been written to (INF-01)."""


@dataclass(frozen=True)
class EnvironmentReport:
    """A successful environment verification, suitable for logging into an artifact."""

    interpreter: str
    python_version: str
    distributions: Mapping[str, str]
    provider_uri: str
    cache_root: str
    pollution: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation (used by the provenance block)."""
        return {
            "interpreter": self.interpreter,
            "python_version": self.python_version,
            "distributions": dict(self.distributions),
            "provider_uri": self.provider_uri,
            "cache_root": self.cache_root,
            "pollution": list(self.pollution),
        }

    def format(self) -> str:
        """Return a human-readable multi-line description of the verified contract."""
        lines = [
            "qresearch environment contract: SATISFIED",
            f"  interpreter     : {self.interpreter}",
            f"  python          : {self.python_version} (required exactly {PYTHON_VERSION_REQUIRED})",
            f"  provider_uri    : {self.provider_uri} (read-only)",
            f"  cache_root      : {self.cache_root} (writable, outside the provider)",
            "  pinned distributions:",
        ]
        lines.extend(f"    - {name}=={version}" for name, version in sorted(self.distributions.items()))
        if self.pollution:
            lines.append("  WARNING - unexpected cache-like entries inside the provider:")
            lines.extend(f"    ! {entry}" for entry in self.pollution)
        return "\n".join(lines)


def _installed_version(distribution: str) -> str | None:
    """Return the installed version of ``distribution`` or ``None`` when it is absent."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _detect_pollution(provider_uri: Path) -> tuple[str, ...]:
    """Return cache-like entries found directly inside the provider tree.

    The provider is a read-only input; a cache directory inside it is evidence that some
    earlier process wrote into it and must be reported (never silently ignored).
    """
    if not provider_uri.is_dir():
        return ()
    found: list[str] = []
    for entry in sorted(provider_uri.iterdir()):
        lowered = entry.name.lower()
        if any(marker in lowered for marker in POLLUTION_MARKERS):
            found.append(str(entry))
    return tuple(found)


def verify_environment(
    *,
    require_exact_python: bool = True,
    require_pinned_interpreter: bool = True,
    require_exact_distributions: bool = True,
    distributions: Iterable[DistributionRequirement] = REQUIRED_DISTRIBUTIONS,
    expected_interpreter: Path | None = EXPECTED_INTERPRETER,
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> EnvironmentReport:
    """Verify the pinned runtime contract and raise on any violation.

    Parameters
    ----------
    require_exact_python : bool
        Require ``platform.python_version() == PYTHON_VERSION_REQUIRED`` (default ``True``).
    require_pinned_interpreter : bool
        Require ``sys.executable`` to be the pinned interpreter and to match none of
        :data:`FORBIDDEN_INTERPRETER_MARKERS` (default ``True``).
    require_exact_distributions : bool
        Require every distribution in ``distributions`` at its exact patch version.
    distributions : Iterable[DistributionRequirement]
        The contract to enforce; defaults to :data:`REQUIRED_DISTRIBUTIONS`.
    expected_interpreter : Path | None
        The interpreter the project is pinned to; ``None`` disables the equality check while
        still forbidding the markers above.
    provider_uri : Path
        Read-only market-data root, recorded in the report and scanned for pollution.
    cache_root : Path
        Writable cache root, recorded in the report.

    Returns
    -------
    EnvironmentReport
        A frozen record of the verified contract.

    Raises
    ------
    EnvironmentContractError
        If any part of the contract is violated.  The error message enumerates every
        violation, so a single run diagnoses the whole environment rather than one defect.

    Notes
    -----
    There is deliberately **no** environment-variable bypass.  A bypass would convert a
    hard reproducibility guarantee into a warning, which PROJECT_SPEC.md 1.2.2 forbids.
    """
    problems: list[str] = []

    actual_python = platform.python_version()
    if require_exact_python and actual_python != PYTHON_VERSION_REQUIRED:
        problems.append(f"python: interpreter reports {actual_python}, required exactly {PYTHON_VERSION_REQUIRED}")

    executable = Path(sys.executable)
    executable_text = str(executable).lower()
    for marker in FORBIDDEN_INTERPRETER_MARKERS:
        if marker in executable_text:
            problems.append(
                f"interpreter: {executable} matches forbidden marker {marker!r}; "
                "this interpreter does not carry the pinned dependency set"
            )
            break
    if require_pinned_interpreter and expected_interpreter is not None and executable != expected_interpreter:
        problems.append(f"interpreter: sys.executable={executable}, required {expected_interpreter}")

    resolved: dict[str, str] = {}
    if require_exact_distributions:
        for requirement in distributions:
            found = _installed_version(requirement.name)
            if found is None:
                problems.append(f"{requirement.name}: NOT INSTALLED, required {requirement.requirement}")
                continue
            resolved[requirement.name] = found
            if found != requirement.version:
                tag = " [FLOAT-CRITICAL]" if requirement.float_critical else ""
                problems.append(f"{requirement.name}: found {found}, required {requirement.requirement}{tag}")

    if problems:
        detail = "\n".join(f"  - {item}" for item in problems)
        remediation = (
            "Remediation:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate qresearch\n"
            "  python -m pip install -e . --no-deps\n"
            "  python scripts/check_env.py\n"
            "Do NOT bypass this check: version drift changes floating-point results and "
            "invalidates statistical inference (PROJECT_SPEC.md 1.2.2, 3.1, 6.1)."
        )
        raise EnvironmentContractError(f"Environment contract violated (task INF-01).\n{detail}\n{remediation}")

    return EnvironmentReport(
        interpreter=str(executable),
        python_version=actual_python,
        distributions=MappingProxyType(dict(resolved)),
        provider_uri=str(provider_uri),
        cache_root=str(cache_root),
        pollution=_detect_pollution(provider_uri),
    )


# ---------------------------------------------------------------------------------------
# Read-only guarantee for the market-data provider
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TreeFingerprint:
    """A cheap structural fingerprint of a directory tree.

    Hashing every file of a multi-gigabyte binary provider on every run would be prohibitive,
    so the fingerprint uses four scale-invariant aggregates: file count, directory count,
    total size in bytes and the newest modification timestamp.  Any write - creation,
    deletion, truncation or in-place rewrite - changes at least one of them.
    """

    root: str
    n_files: int
    n_dirs: int
    total_bytes: int
    newest_mtime_ns: int

    def describe_delta(self, other: TreeFingerprint) -> str:
        """Return a field-by-field description of the difference to ``other``."""
        fields = ("n_files", "n_dirs", "total_bytes", "newest_mtime_ns")
        changes = [
            f"{name}: {getattr(self, name)} -> {getattr(other, name)}"
            for name in fields
            if getattr(self, name) != getattr(other, name)
        ]
        return "; ".join(changes) if changes else "no structural difference detected"


def fingerprint_tree(root: Path) -> TreeFingerprint:
    """Compute a :class:`TreeFingerprint` for ``root`` without writing anything.

    Raises
    ------
    ReadOnlyViolationError
        If ``root`` does not exist or is not a directory, because a missing provider means no
        read-only guarantee can be established.
    """
    if not root.is_dir():
        raise ReadOnlyViolationError(f"provider path {root} does not exist or is not a directory")

    n_files = 0
    n_dirs = 0
    total_bytes = 0
    newest_mtime_ns = 0
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError:  # pragma: no cover - platform dependent
                    continue
                if entry.is_dir(follow_symlinks=False):
                    n_dirs += 1
                    stack.append(Path(entry.path))
                else:
                    n_files += 1
                    total_bytes += stat.st_size
                newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return TreeFingerprint(
        root=str(root),
        n_files=n_files,
        n_dirs=n_dirs,
        total_bytes=total_bytes,
        newest_mtime_ns=newest_mtime_ns,
    )


class ProviderWriteGuard:
    """Context manager that asserts the read-only provider was not modified.

    Example
    -------
    .. code-block:: python

        with ProviderWriteGuard(provider_uri) as before:
            run_research_step()          # any code that must only read market data
        # reaching this line proves the provider tree is structurally unchanged

    The guard is intentionally cheap and is suitable for wrapping every data-loading phase
    of a research run (PROJECT_SPEC.md 3.6, PIT-1).
    """

    def __init__(self, provider_uri: Path = DEFAULT_PROVIDER_URI) -> None:
        self.provider_uri = Path(provider_uri)
        self._before: TreeFingerprint | None = None
        self._after: TreeFingerprint | None = None

    @property
    def before(self) -> TreeFingerprint | None:
        """Fingerprint captured on entering the context, if any."""
        return self._before

    @property
    def after(self) -> TreeFingerprint | None:
        """Fingerprint captured by the most recent :meth:`check`, if any."""
        return self._after

    def __enter__(self) -> TreeFingerprint:
        self._before = fingerprint_tree(self.provider_uri)
        return self._before

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.check()

    def check(self) -> TreeFingerprint:
        """Re-fingerprint and raise :class:`ReadOnlyViolationError` if anything changed.

        Returns
        -------
        TreeFingerprint
            The fingerprint taken by this call.
        """
        if self._before is None:
            raise ReadOnlyViolationError("ProviderWriteGuard.check() called before entering the context")
        self._after = fingerprint_tree(self.provider_uri)
        if self._after != self._before:
            detail = self._before.describe_delta(self._after)
            raise ReadOnlyViolationError(
                f"the read-only provider {self.provider_uri} changed during execution ({detail}).\n"
                "Nothing may write into the market-data provider: redirect every Qlib cache via "
                "qresearch.env.build_qlib_init_kwargs() and never point expression_cache or "
                "dataset_cache at a path inside provider_uri."
            )
        return self._after


# ---------------------------------------------------------------------------------------
# Isolation checks and the canonical (cache-free) Qlib initialization kwargs
# ---------------------------------------------------------------------------------------
def _is_within(candidate: Path, root: Path, *, allow_root: bool) -> bool:
    """Return whether ``candidate`` lies in ``root``, optionally allowing equality.

    A path is trivially relative to itself, so the distinction matters: ``provider_uri`` is
    legitimately *equal* to the provider root and must be accepted, while a cache path equal to
    the provider root is a violation (it would write into read-only data).
    """
    if candidate == root:
        return allow_root
    return candidate.is_relative_to(root)


def _assert_no_path_under(value: object, forbidden: Path, *, path: str = "kwargs") -> None:
    """Recursively assert that no string in ``value`` points *inside* ``forbidden``.

    Equality with ``forbidden`` is permitted, because ``provider_uri`` must be allowed to name
    the provider root itself; only strict descendants indicate a writable location that has been
    placed inside the read-only tree.
    """
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute() and _is_within(candidate.resolve(), forbidden, allow_root=False):
            raise ReadOnlyViolationError(f"{path} points inside the read-only provider: {value}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_path_under(item, forbidden, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _assert_no_path_under(item, forbidden, path=f"{path}[{index}]")


def assert_provider_isolation(
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> None:
    """Assert that the provider is a real, external, unpolluted, read-only data root.

    Raises
    ------
    ReadOnlyViolationError
        If the provider is missing, if it lies inside the repository, if the cache root lies
        inside it, or if cache-like entries already exist inside it.
    """
    provider = Path(provider_uri).resolve()
    cache = Path(cache_root).resolve()
    repo_root = paths.project_root()

    if not provider.is_dir():
        raise ReadOnlyViolationError(f"provider_uri {provider} does not exist or is not a directory")
    if _is_within(cache, provider, allow_root=True):
        raise ReadOnlyViolationError(
            f"cache_root {cache} lies inside the read-only provider {provider}; "
            "redirect caches to artifacts/qlib_cache"
        )
    if provider.is_relative_to(repo_root):
        raise ReadOnlyViolationError(
            f"provider_uri {provider} lies inside the repository {repo_root}; "
            "market data must live outside the working tree and never be committed"
        )
    pollution = _detect_pollution(provider)
    if pollution:
        entries = "\n".join(f"  - {item}" for item in pollution)
        raise ReadOnlyViolationError(
            f"the read-only provider {provider} contains cache-like entries:\n{entries}\n"
            "A previous run wrote into the data source. Remove these entries and re-verify "
            "the provider before trusting any result produced after them."
        )


def build_qlib_init_kwargs(
    provider_uri: Path = DEFAULT_PROVIDER_URI,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    region: str = "cn",
    kernels: int = 1,
    enable_cache: bool = False,
) -> dict[str, object]:
    """Return the canonical ``qlib.init`` keyword arguments for this project.

    The returned mapping is verified to contain no path inside ``provider_uri`` and to route
    every writable location (Qlib caches, MLflow tracking store) to ``cache_root``, which defaults
    to ``<artifact_root>/qlib_cache`` (git-ignored, ``ADR-005``).

    Parameters
    ----------
    provider_uri : Path
        Read-only market-data root; defaults to
        :func:`qresearch.config.paths.data_dir` (``$QLIB_DATA_DIR``, else the Qlib default store).
    cache_root : Path
        Writable cache root; MUST NOT be inside ``provider_uri``.
    region : str
        Qlib market region, ``"cn"`` for the China A-share store.
    kernels : int
        Qlib multiprocessing kernel count; ``1`` avoids the Windows spawn and non-ASCII path
        failure mode and keeps runs deterministic (PROJECT_SPEC.md 5.3).
    enable_cache : bool
        When ``True``, enable the Qlib expression and dataset caches, both pointed at
        ``cache_root``.  Disabled by default because a cache is a second source of truth; a
        study that enables it MUST record the cache key in its provenance block.

    Returns
    -------
    dict[str, object]
        Keyword arguments ready to be passed to ``qlib.init(**kwargs)``.
    """
    report = verify_environment(provider_uri=Path(provider_uri), cache_root=Path(cache_root))
    provider = Path(provider_uri).resolve()
    cache = Path(cache_root).resolve()
    assert_provider_isolation(provider, cache)

    cache.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"file:{(cache.parent / 'mlruns').as_posix()}"

    kwargs: dict[str, object] = {
        "provider_uri": str(provider),
        "region": region,
        "kernels": kernels,
        "expression_cache": None,
        "dataset_cache": None,
        "exp_manager": {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": mlflow_uri, "default_exp_name": "alpha_research"},
        },
    }
    if enable_cache:
        expression_cache = cache / "expression_cache"
        dataset_cache = cache / "dataset_cache"
        expression_cache.mkdir(parents=True, exist_ok=True)
        dataset_cache.mkdir(parents=True, exist_ok=True)
        kwargs["expression_cache"] = str(expression_cache)
        kwargs["dataset_cache"] = str(dataset_cache)

    _assert_no_path_under(kwargs, provider)
    assert report.provider_uri == str(provider)
    return kwargs


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``qresearch-check-env``) verifying the environment contract.

    Returns
    -------
    int
        ``0`` on success, ``1`` on an environment-contract violation, ``2`` on a read-only
        provider violation and ``3`` when the provider contains cache-like pollution.
    """
    parser = argparse.ArgumentParser(
        prog="qresearch-check-env",
        description="Verify the pinned qresearch runtime contract (PROJECT_SPEC.md task INF-01).",
    )
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    try:
        report = verify_environment(provider_uri=args.provider_uri, cache_root=args.cache_root)
        assert_provider_isolation(args.provider_uri, args.cache_root)
    except EnvironmentContractError as error:
        print(str(error), file=sys.stderr)
        return 1
    except ReadOnlyViolationError as error:
        print(f"Read-only data-source violation:\n  {error}", file=sys.stderr)
        return 2

    if args.json:
        payload = report.to_dict()
        payload["environment_overrides"] = [item.to_dict() for item in paths.applied_overrides()]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.format())
        print(paths.describe_overrides())

    if report.pollution:
        print("FAIL: the provider contains cache-like entries; see the report above.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
