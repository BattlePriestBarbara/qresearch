"""The portable pre-commit hook runner (``scripts/hook_runner.py``, ``ADR-005``).

The runner is the piece that lets the gate stay strict without naming a machine: these tests pin the
contract that makes it safe - it agrees with :mod:`qresearch.config.paths` candidate for candidate, it
refuses to substitute an interpreter that the operator asked for and that does not exist, and it
propagates the tool's own exit status so a failing hook stays failing.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from qresearch.config import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "hook_runner.py"


def _load_runner() -> ModuleType:
    """Import ``scripts/hook_runner.py`` without requiring ``scripts/`` to be a package."""
    spec = importlib.util.spec_from_file_location("hook_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER_MODULE = _load_runner()


@pytest.mark.unit
@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"QRESEARCH_PYTHON": str(Path("C:") / "tools" / "python.exe")},
        {"CONDA_PREFIX": str(Path("C:") / "envs" / "qresearch")},
        {"QRESEARCH_PYTHON": str(Path("C:") / "tools" / "python.exe"), "CONDA_PREFIX": str(Path("C:") / "envs" / "q")},
    ],
)
def test_runner_mirrors_the_library_resolution(environ: dict[str, str]) -> None:
    """The stdlib-only mirror must stay identical to the library, or the gate silently diverges."""
    assert RUNNER_MODULE.candidate_interpreters(environ) == tuple(
        str(candidate) for candidate in paths.candidate_interpreters(environ)
    )


@pytest.mark.unit
def test_runner_never_substitutes_a_missing_override(tmp_path: Path) -> None:
    """An authoritative override that does not exist stops the gate; it is not replaced by another."""
    missing = str(tmp_path / "missing" / "python.exe")
    assert RUNNER_MODULE.resolve_interpreter({paths.INTERPRETER_ENV_VAR: missing}) is None


@pytest.mark.unit
def test_runner_resolves_the_running_interpreter_by_default() -> None:
    """With nothing configured the running interpreter is used, which is the pinned one in this env."""
    assert RUNNER_MODULE.resolve_interpreter({}) == sys.executable


@pytest.mark.unit
def test_runner_requires_an_argument_and_reports_the_missing_interpreter(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both failure modes exit with the documented code and print an actionable message."""
    assert RUNNER_MODULE.main([]) == RUNNER_MODULE.EXIT_NO_INTERPRETER
    assert "usage:" in capsys.readouterr().err

    monkeypatch.setenv(paths.INTERPRETER_ENV_VAR, str(tmp_path / "missing" / "python.exe"))
    assert RUNNER_MODULE.main(["-c", "print('unreachable')"]) == RUNNER_MODULE.EXIT_NO_INTERPRETER
    assert "no usable interpreter" in capsys.readouterr().err


@pytest.mark.unit
def test_runner_propagates_the_tool_exit_status() -> None:
    """A failing tool must fail the hook; a passing one must not mask it either."""
    assert RUNNER_MODULE.main(["-c", "raise SystemExit(3)"]) == 3
    assert RUNNER_MODULE.main(["-c", "raise SystemExit(0)"]) == 0


@pytest.mark.integration
def test_runner_is_usable_as_a_pre_commit_entry() -> None:
    """The documented entry form (bare python + this script) must work end to end."""
    completed = subprocess.run(
        [sys.executable, str(RUNNER), RUNNER_MODULE.SHOW_INTERPRETER_FLAG],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).is_file()

    failed = subprocess.run(
        [sys.executable, str(RUNNER), "-c", "raise SystemExit(3)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 3
