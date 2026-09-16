"""Repository structure, src-layout and gate-configuration tests (task ``INF-02``).

These tests make the mentor-mandated layout decisions executable rather than aspirational:
src-layout (so tests import the installed package), a strict pre-commit gate, and an ignore
file that keeps data, weights, experiment state and notebook outputs out of Git.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import qresearch

SUBPACKAGES = (
    "qresearch.config",
    "qresearch.data",
    "qresearch.features",
    "qresearch.stats",
    "qresearch.models",
    "qresearch.portfolio",
    "qresearch.evaluation",
    "qresearch.utils",
)


@pytest.mark.unit
def test_package_resolves_through_src_layout(repo_root: Path) -> None:
    """`import qresearch` must resolve to the installed package under src/."""
    package_file = Path(qresearch.__file__).resolve()
    assert package_file.parent.name == "qresearch"
    assert package_file.parent.parent.name == "src"
    assert package_file.is_relative_to(repo_root / "src")


@pytest.mark.unit
def test_no_flat_package_directory_exists(repo_root: Path) -> None:
    """A flat top-level package directory would shadow the installed distribution."""
    assert not (repo_root / "qresearch").exists()


@pytest.mark.unit
def test_py_typed_marker_is_packaged(repo_root: Path) -> None:
    """PEP 561 marker: downstream users type-check against our annotations."""
    marker = repo_root / "src" / "qresearch" / "py.typed"
    assert marker.is_file()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "py.typed" in pyproject["tool"]["setuptools"]["package-data"]["qresearch"]


@pytest.mark.unit
@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_layer_subpackages_are_importable(module_name: str) -> None:
    """Every architecture layer declared in PROJECT_SPEC.md 3.2 must exist and import."""
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} must document its layer responsibility"


@pytest.mark.unit
def test_pyproject_declares_src_layout(repo_root: Path) -> None:
    """Packaging must discover packages only under src/."""
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert pyproject["project"]["requires-python"] == "==3.12.*"


@pytest.mark.unit
def test_precommit_gate_declares_required_hooks(repo_root: Path) -> None:
    """The gate must contain the mentor-mandated tools, pinned to the project interpreter."""
    config = (repo_root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for hook in ("- id: black", "- id: isort", "- id: ruff", "- id: flake8", "- id: mypy", "- id: nbstripout"):
        assert hook in config, f"pre-commit gate is missing {hook}"
    assert "language: system" in config
    # mypy must run in the pinned environment, not in a bare hook venv, or it cannot see
    # the qlib/torch stubs and would have to be weakened with ignore-missing-imports.
    # Forward slashes are mandatory: pre-commit's POSIX shlex eats backslashes in `entry`.
    assert "D:/Anaconda3/python.exe -m mypy" in config
    assert "\\Anaconda3\\python.exe -m" not in config, "backslashes break pre-commit hook entries"
    assert "stages: [manual]" in config, "pylint/pytest must stay in the slow gate"


@pytest.mark.unit
def test_gitignore_excludes_forbidden_categories(repo_root: Path) -> None:
    """Data, caches, weights, tracker state and notebook outputs must not be committable."""
    rules = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "__pycache__/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".qlib/",
        "artifacts/qlib_cache/",
        "mlruns/",
        "wandb/",
        "tensorboard/",
        "*.pth",
        "*.pt",
        "*.pkl",
        "*.safetensors",
        ".ipynb_checkpoints/",
    ):
        assert pattern in rules, f".gitignore is missing {pattern}"
    # exceptions keep the directory skeletons under version control
    assert "!artifacts/.gitkeep" in rules
    assert "!data/.gitkeep" in rules


@pytest.mark.unit
def test_mypy_strictness_is_enabled(repo_root: Path) -> None:
    """Strict typing is the only defence against silent tensor/Index shape errors."""
    config = (repo_root / ".mypy.ini").read_text(encoding="utf-8")
    active_lines = [line.split("#", 1)[0] for line in config.splitlines()]
    active = "\n".join(active_lines)
    assert "strict = True" in active
    assert "disallow_untyped_defs = True" in active
    assert "disallow_untyped_calls = True" in active
    assert "files = src/qresearch" in active
    # No stub-based override may be active: installing pandas-stubs/scipy-stubs was observed to
    # upgrade the float-critical numpy pin (ADR-002), so only ignore_missing_imports overrides
    # are permitted, and only for the libraries that ship no PEP 561 marker.
    assert "stubs" not in active.replace("ignore_missing_imports", "")
    assert "[mypy-pandas.*]" in active
    assert "ignore_missing_imports = False" in active
