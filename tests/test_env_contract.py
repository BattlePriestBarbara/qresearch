"""Environment-contract tests for task ``INF-01``.

These tests prove the three mentor-mandated properties:

1. the contract is verified automatically at package import time;
2. a violation raises :class:`RuntimeError` (never a warning-then-continue);
3. the error message diagnoses *every* violating component at once, and offers remediation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import qresearch
from qresearch import env
from qresearch.config import paths
from qresearch.config.errors import ConfigError

MSYS_INTERPRETER = str(Path("D:") / "msys64" / "mingw64" / "bin" / "python.exe")
"""A forbidden interpreter build (MSYS2), assembled from parts so no literal path is tracked."""

FOREIGN_INTERPRETER = str(Path("C:") / "Python310" / "python.exe")
"""An interpreter that is not the pinned one, for the aggregated-diagnosis test."""


@pytest.mark.unit
def test_import_time_enforcement_is_active() -> None:
    """Importing qresearch must have verified the contract already."""
    report = qresearch.ENVIRONMENT_REPORT
    assert isinstance(report, env.EnvironmentReport)
    assert report.python_version == env.PYTHON_VERSION_REQUIRED
    assert Path(report.interpreter) == env.EXPECTED_INTERPRETER
    assert report.distributions["numpy"] == "1.26.4"
    assert report.distributions["torch"] == "2.8.0"
    assert report.distributions["pyqlib"] == "0.9.7"


@pytest.mark.unit
def test_canonical_paths_are_exposed() -> None:
    """The package exposes the pinned provider and cache roots."""
    assert qresearch.PROVIDER_URI == env.DEFAULT_PROVIDER_URI
    assert qresearch.CACHE_ROOT == env.DEFAULT_CACHE_ROOT
    assert env.DEFAULT_CACHE_ROOT != env.DEFAULT_PROVIDER_URI


@pytest.mark.unit
def test_verify_environment_returns_frozen_report() -> None:
    """A successful verification returns an immutable report with a provenance block."""
    report = env.verify_environment()
    payload = report.to_dict()
    assert set(payload) == {"interpreter", "python_version", "distributions", "provider_uri", "cache_root", "pollution"}
    assert "SATISFIED" in report.format()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.python_version = "3.11.0"  # type: ignore[misc]  # frozen dataclass


@pytest.mark.unit
def test_contract_error_is_a_runtime_error() -> None:
    """The violation type must be a RuntimeError so it cannot be silently swallowed."""
    assert issubclass(env.EnvironmentContractError, RuntimeError)
    assert issubclass(env.ReadOnlyViolationError, RuntimeError)


@pytest.mark.unit
def test_wrong_python_version_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong interpreter patch level must abort, not degrade to a warning."""
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.12.6")
    with pytest.raises(env.EnvironmentContractError) as error_info:
        env.verify_environment()
    message = str(error_info.value)
    assert "3.12.6" in message
    assert env.PYTHON_VERSION_REQUIRED in message
    assert "Remediation" in message


@pytest.mark.unit
def test_wrong_distribution_version_raises_and_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drifted float-critical distribution must be reported as such."""
    real_version = env._installed_version  # deliberate internal probe of the version lookup

    def fake_version(distribution: str) -> str | None:
        if distribution == "numpy":
            return "2.5.3"
        if distribution == "torch":
            return None
        return real_version(distribution)

    monkeypatch.setattr(env, "_installed_version", fake_version)
    with pytest.raises(env.EnvironmentContractError) as error_info:
        env.verify_environment()
    message = str(error_info.value)
    assert "numpy: found 2.5.3, required numpy==1.26.4 [FLOAT-CRITICAL]" in message
    assert "torch: NOT INSTALLED, required torch==2.8.0" in message


@pytest.mark.unit
def test_forbidden_interpreter_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MSYS2 interpreter on PATH must never be accepted."""
    # The two interpreters used here are assembled from parts instead of spelled out: a literal
    # machine path must not appear in a tracked file (ADR-005), and the assertions are about the
    # *diagnosis*, not about the machine.
    monkeypatch.setattr(env.sys, "executable", MSYS_INTERPRETER)
    with pytest.raises(env.EnvironmentContractError) as error_info:
        env.verify_environment(require_pinned_interpreter=False, require_exact_distributions=False)
    assert "msys64" in str(error_info.value)


@pytest.mark.unit
def test_multiple_violations_are_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    """One run must diagnose the whole environment, not one defect at a time."""
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.10.13")
    monkeypatch.setattr(env.sys, "executable", FOREIGN_INTERPRETER)
    with pytest.raises(env.EnvironmentContractError) as error_info:
        env.verify_environment(require_exact_distributions=False)
    message = str(error_info.value)
    assert message.count("  - ") >= 2
    assert "python: interpreter reports 3.10.13" in message
    assert f"interpreter: sys.executable={FOREIGN_INTERPRETER}" in message


@pytest.mark.unit
def test_configuration_errors_share_one_family() -> None:
    """A path/configuration failure is a ``ConfigError``; the env and provider errors are members."""
    assert issubclass(env.EnvironmentContractError, ConfigError)
    assert issubclass(env.ReadOnlyViolationError, ConfigError)
    assert issubclass(ConfigError, RuntimeError)


@pytest.mark.unit
def test_contract_constants_come_from_the_resolver() -> None:
    """ADR-005: the contract's locations are resolved from configuration, never written down."""
    assert paths.interpreter() == env.EXPECTED_INTERPRETER
    assert paths.data_dir() == env.DEFAULT_PROVIDER_URI
    assert paths.cache_root() == env.DEFAULT_CACHE_ROOT
    assert paths.data_dir() == qresearch.PROVIDER_URI


@pytest.mark.unit
def test_no_bypass_environment_variable_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting an arbitrary opt-out variable must not weaken the check.

    PROJECT_SPEC.md forbids silent degradation, so no environment variable may disable the
    contract; this test pins that design decision against future "convenience" patches.
    """
    for name in ("QRESEARCH_SKIP_ENV_CHECK", "QRESEARCH_ALLOW_ENV_MISMATCH", "SKIP_ENV_CHECK"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.11.9")
    with pytest.raises(env.EnvironmentContractError):
        env.verify_environment()


@pytest.mark.unit
def test_console_entry_point_is_declared(repo_root: Path) -> None:
    """The env check must be reachable both as a script and as a console entry point."""
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'qresearch-check-env = "qresearch.env:main"' in pyproject
    assert (repo_root / "scripts" / "check_env.py").is_file()
