"""Dependency-pinning tests for task ``INF-01``.

Rationale (``PROJECT_SPEC.md`` 1.2.2, 3.1): a patch-level drift in ``numpy``/``scipy``/``torch``
perturbs the low-order bits of computed statistics and can move a Newey-West ``p``-value across
a significance threshold.  These tests make the pin policy mechanical: ranges, wildcards and
unpinned names are rejected, and the runtime contract, ``requirements.txt`` and
``environment.yml`` must agree with each other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from qresearch.env import REQUIRED_DISTRIBUTIONS

# Exactly `name==1.2.3` (optionally with a pre/post/local suffix); nothing else is allowed.
_PIN_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\d+(?:\.\d+)*[A-Za-z0-9.+!-]*)$")


def _requirement_lines(path: Path) -> list[str]:
    """Return the active requirement lines of a requirements file."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return lines


def _tolerant_lines(path: Path) -> list[str]:
    """Return active lines of a lock file, keeping ``#``-comments out but tolerating hashes."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0].strip().rstrip("\\").strip()
        if stripped and not stripped.startswith("--"):
            lines.append(stripped)
    return lines


def _lock_pins(path: Path) -> dict[str, str]:
    """Return ``{canonical_name: version}`` for the pinned lines of a lock file."""
    pins: dict[str, str] = {}
    for line in _tolerant_lines(path):
        match = _PIN_PATTERN.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
    return pins


def _environment_yml(repo_root: Path) -> dict[str, object]:
    """Load ``environment.yml``."""
    payload = yaml.safe_load((repo_root / "environment.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _conda_pins(repo_root: Path) -> dict[str, str]:
    """Return the conda-managed pins (excluding python/pip tooling) as ``{name: spec}``."""
    pins: dict[str, str] = {}
    for entry in _environment_yml(repo_root).get("dependencies", []):
        if not isinstance(entry, str):
            continue
        name = entry.split("=", 1)[0].strip().lower()
        if name in {"python", "pip"}:
            continue
        pins[name] = entry.split("=", 1)[1].strip()
    return pins


def _pip_block(repo_root: Path) -> list[str]:
    """Return the ``pip:`` sub-block entries of ``environment.yml``."""
    for entry in _environment_yml(repo_root).get("dependencies", []):
        if isinstance(entry, dict) and "pip" in entry:
            return [str(item).strip() for item in entry["pip"]]
    raise AssertionError("environment.yml declares no pip sub-block")


@pytest.mark.unit
@pytest.mark.parametrize("filename", ["requirements.txt", "requirements-dev.txt"])
def test_every_dependency_is_patch_pinned(repo_root: Path, filename: str) -> None:
    """No ranges, no wildcards, no bare names: only full patch pins."""
    lines = _requirement_lines(repo_root / filename)
    assert lines, f"{filename} must not be empty"
    for line in lines:
        assert not line.startswith("-"), f"{filename}: nested requirement files are not allowed ({line})"
        assert _PIN_PATTERN.match(line), f"{filename}: '{line}' is not a full patch pin"
        for forbidden in ("~=", ">=", "<=", "!=", "*", ";"):
            assert forbidden not in line, f"{filename}: '{line}' contains a range or marker"


@pytest.mark.unit
def test_runtime_contract_matches_requirements_txt(repo_root: Path) -> None:
    """The code-level contract and the installable requirement list must be identical."""
    requirement_lines = _requirement_lines(repo_root / "requirements.txt")
    matches = [match for match in map(_PIN_PATTERN.match, requirement_lines) if match]
    pinned = {match.group("name").lower(): match.group("version") for match in matches}
    for requirement in REQUIRED_DISTRIBUTIONS:
        key = requirement.name.lower()
        assert key in pinned, f"{requirement.name} is enforced at runtime but missing from requirements.txt"
        assert pinned[key] == requirement.version, (
            f"{requirement.name}: requirements.txt pins {pinned[key]} but the runtime contract "
            f"enforces {requirement.version}"
        )


@pytest.mark.unit
def test_environment_yml_is_a_hybrid_spec(repo_root: Path) -> None:
    """The conda spec must pin the interpreter and install the *pip lock*, not loose pins."""
    content = (repo_root / "environment.yml").read_text(encoding="utf-8")
    assert "python=3.12.7" in content
    assert "PyPI wheels" in content, "the conda/PyPI divergence rationale must stay documented"
    assert "ADR-003" in content, "the hybrid strategy must stay traceable to its ADR"

    pip_entries = [entry.replace("-r", "").strip() for entry in _pip_block(repo_root)]
    assert "requirements.lock.pip.txt" in pip_entries, "the pip block must install the pip lock"
    for forbidden in ("requirements.lock.txt", "requirements.txt", "requirements-dev.txt"):
        assert forbidden not in pip_entries, (
            f"the pip block must install the resolved lock, not {forbidden}: an environment "
            "created from loose pins is not reproducible"
        )


@pytest.mark.unit
def test_hybrid_split_reconstructs_the_closure_exactly(repo_root: Path) -> None:
    """The pip set plus the conda set must equal the closure, with no package managed twice.

    This is the mechanical expression of ADR-003: if a package is moved between managers in
    ``environment.yml`` without regenerating the locks, this test fails.
    """
    full = _lock_pins(repo_root / "requirements.lock.txt")
    pip_lock = _lock_pins(repo_root / "requirements.lock.pip.txt")
    hash_lock = _lock_pins(repo_root / "requirements.lock.hashes.txt")
    conda = _conda_pins(repo_root)

    assert conda, "environment.yml must delegate at least one package to conda"
    assert set(pip_lock) == set(hash_lock), "the pip lock and the digest lock must cover the same set"
    assert set(pip_lock) | set(conda) == set(
        full
    ), "the pip-managed set plus the conda-managed set must reconstruct the full closure"
    assert not set(pip_lock) & set(conda), "a package must not be managed by both pip and conda"
    assert not set(conda) & set(hash_lock), "conda-managed packages must not be hash-pinned by pip"


@pytest.mark.unit
def test_conda_managed_packages_are_pinned_with_build_strings(repo_root: Path) -> None:
    """Native extensions must be pinned as ``version=build`` so the *binary* is reproducible."""
    conda = _conda_pins(repo_root)
    native = {name: spec for name, spec in conda.items() if "=" in spec}
    assert {"psutil", "pywin32", "pywinpty"} <= set(native), "the wheel-less extensions must stay delegated"
    for name, spec in native.items():
        version, _, build = spec.partition("=")
        assert version, f"{name} must carry an exact version"
        assert build, f"{name} must carry a build string (version=build), found {spec!r}"


@pytest.mark.unit
def test_sdist_only_pure_python_package_is_pip_managed(repo_root: Path) -> None:
    """``gym`` ships no wheel but needs no compiler, so pip keeps it with a pinned digest.

    conda-forge only reaches gym 0.26.1, so delegating it to conda would silently downgrade a
    dependency of the verified environment (ADR-003).
    """
    pip_lock = _lock_pins(repo_root / "requirements.lock.pip.txt")
    assert pip_lock.get("gym") == "0.26.2"
    hashes = (repo_root / "requirements.lock.hashes.txt").read_text(encoding="utf-8")
    assert "gym==0.26.2 \\" in hashes, "the sdist-only package must still be digest-pinned"
    assert "sdist-only, pure Python" in hashes, "the sdist case must stay documented in the lock"


@pytest.mark.unit
def test_float_critical_packages_are_marked() -> None:
    """The packages whose versions change floating-point output must be declared as such."""
    critical = {requirement.name.lower() for requirement in REQUIRED_DISTRIBUTIONS if requirement.float_critical}
    assert {"numpy", "scipy", "pandas", "torch", "cvxpy", "lightgbm", "scikit-learn"} <= critical


@pytest.mark.unit
def test_lock_generator_exists_and_uses_hashes(repo_root: Path) -> None:
    """A digest-pinned closure must be reproducible from a committed script."""
    script = repo_root / "scripts" / "lock_requirements.py"
    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert "--require-hashes" in source
    assert "pypi.org/pypi" in source, "digests must come from the PyPI JSON API"
    assert (
        "Requires-Dist walk" in source or "Requires-Dist" in source
    ), "the closure method must stay documented (offline metadata walk, not a fresh resolution)"
    assert "NON_WHEEL_CANDIDATES" in source, "distributions without wheels must stay declared"
    assert (
        "CATEGORY_CONDA" in source and "CATEGORY_SDIST_PURE_PYTHON" in source
    ), "the hybrid classification (ADR-003) must stay implemented in the generator"
    assert "abi3" in source, "stable-ABI wheels must be accepted, or the split is wrong"


@pytest.mark.unit
def test_generated_version_lock_is_complete_and_exact(repo_root: Path) -> None:
    """The version lock must pin the whole closure, exactly, with provenance notes."""
    lock = repo_root / "requirements.lock.txt"
    if not lock.is_file():
        pytest.skip("requirements.lock.txt has not been generated yet")
    body = [line.strip() for line in lock.read_text(encoding="utf-8").splitlines()]
    pins = [line for line in body if _PIN_PATTERN.match(line)]
    assert len(pins) >= 150, "the closure must contain the full transitive set"
    pinned = {match.group("name").lower(): match.group("version") for match in map(_PIN_PATTERN.match, pins) if match}
    for name, version in (("numpy", "1.26.4"), ("torch", "2.8.0"), ("pyqlib", "0.9.7"), ("scipy", "1.13.1")):
        assert pinned.get(name) == version, f"{name} must be pinned to {version} in the lock"
    comments = "\n".join(line for line in body if line.startswith("#"))
    assert "gym" in comments, "the pyqlib gym dependency must be documented with its provenance"
    assert "NO PYPI WHEEL" in comments or "conda:" in comments


@pytest.mark.unit
def test_generated_hash_lock_is_pip_require_hashes_format(repo_root: Path) -> None:
    """The digest lock must be a valid ``--require-hashes`` file (validated against pip)."""
    lock = repo_root / "requirements.lock.hashes.txt"
    if not lock.is_file():
        pytest.skip("requirements.lock.hashes.txt has not been generated yet")
    lines = [line.rstrip() for line in lock.read_text(encoding="utf-8").splitlines()]
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    assert body[0] == "--require-hashes"

    pinned = [line for line in body if line.endswith("\\")]
    hashes = [line.strip() for line in body if line.strip().startswith("--hash=")]
    assert len(pinned) >= 150, "the digest lock must cover the installable closure"
    assert len(hashes) >= len(pinned), "every pinned distribution needs at least one digest"
    assert all(line.startswith("--hash=sha256:") for line in hashes), "only SHA-256 digests are accepted"

    # Digest blocks must be indented continuation lines of the preceding pinned requirement.
    continuation = [line for line in body if line.startswith("    --hash=")]
    assert len(continuation) == len(hashes), "digest lines must be indented continuation lines"
    assert body[-1].startswith("    --hash=") or body[-1].endswith("\\") or body[-1].startswith("#")
    assert not any(line.startswith(("psutil==", "pywin32==", "pywinpty==")) for line in body), (
        "conda-managed native extensions have no compatible wheel and must not enter the "
        "--require-hashes body; they are installed by conda instead (ADR-003)"
    )
    assert "NOT hash-pinned here: managed by conda" in "\n".join(lines)


@pytest.mark.unit
def test_no_stub_package_can_drift_the_float_core(repo_root: Path) -> None:
    """Stub packages that force a numpy major upgrade must stay excluded and documented."""
    dev_requirements = (repo_root / "requirements-dev.txt").read_text(encoding="utf-8")
    active = " ".join(_requirement_lines(repo_root / "requirements-dev.txt"))
    assert "pandas-stubs" not in active
    assert "scipy-stubs" not in active
    assert (
        "pandas-stubs" in dev_requirements and "numpy" in dev_requirements
    ), "the rejection of drift-inducing stubs must remain documented"
