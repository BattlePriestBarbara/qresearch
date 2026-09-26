"""Path and environment-variable resolution (``ADR-005``, task ``INF-10``).

Every assertion here is a clause of the policy the ADR fixes: declarative overrides, documented
fallbacks, no silent guessing, and a machine-checked ban on literal local paths in tracked files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from qresearch.config import paths
from qresearch.config.errors import ConfigError

AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepublish_audit.py"
SCANNED_ROOTS = ("src", "scripts", "tests", "configs")
# Files inside SCANNED_ROOTS whose *job* is to spell machine paths out: they carry the gate's own
# exemption marker, exactly as the gate exempts them.  Pinned here so that a production file cannot
# silently gain the marker and disable the rule for itself.
EXPECTED_EXEMPT_IN_SCANNED_ROOTS = frozenset({"scripts/prepublish_audit.py"})


def _load_audit_module() -> ModuleType:
    """Load ``scripts/prepublish_audit.py`` so the tests scan with the *gate's* patterns."""
    spec = importlib.util.spec_from_file_location("prepublish_audit_for_paths", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


@pytest.fixture(autouse=True)
def clean_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the documented defaults, with no override recorded."""
    for name in (
        paths.DATA_DIR_ENV_VAR,
        paths.PROVIDER_URI_ENV_VAR,
        paths.ARTIFACT_ROOT_ENV_VAR,
        paths.INTERPRETER_ENV_VAR,
        paths.CONDA_PREFIX_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    paths.clear_applied_overrides()


@pytest.mark.unit
def test_project_root_is_discovered_from_the_packaging_marker(repo_root: Path) -> None:
    """The root is found by walking up to ``pyproject.toml``; it is never written down."""
    discovered = paths.project_root()
    assert discovered == repo_root
    assert (discovered / "pyproject.toml").is_file()
    assert paths._discover_project_root(discovered / "src" / "qresearch") == repo_root


@pytest.mark.unit
def test_data_dir_falls_back_to_the_documented_default() -> None:
    """With nothing configured, the Qlib default store is used - and no override is recorded."""
    assert paths.data_dir() == Path.home() / ".qlib" / "qlib_data" / "cn_data"
    assert paths.applied_overrides() == ()
    assert "none" in paths.describe_overrides()


@pytest.mark.unit
def test_data_dir_honours_and_records_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A configured directory is used, and the fact that it was honoured is never silent."""
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, str(tmp_path))
    assert paths.data_dir() == tmp_path
    overrides = paths.applied_overrides()
    assert [item.name for item in overrides] == [paths.DATA_DIR_ENV_VAR]
    assert overrides[0].to_dict()["source"] == "environment"
    assert paths.DATA_DIR_ENV_VAR in paths.describe_overrides()


@pytest.mark.unit
def test_legacy_provider_uri_alias_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``QLIB_PROVIDER_URI`` (spec 3.5) keeps working so an existing profile is not broken."""
    monkeypatch.setenv(paths.PROVIDER_URI_ENV_VAR, str(tmp_path))
    assert paths.data_dir() == tmp_path
    assert paths.applied_overrides()[0].name == paths.PROVIDER_URI_ENV_VAR


@pytest.mark.unit
def test_data_dir_precedence_prefers_the_primary_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When both names are set, the primary one wins deterministically."""
    primary = tmp_path / "primary"
    legacy = tmp_path / "legacy"
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, str(primary))
    monkeypatch.setenv(paths.PROVIDER_URI_ENV_VAR, str(legacy))
    assert paths.data_dir() == primary


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", ["relative/cn_data", ".", "cn_data"])
def test_relative_override_raises_config_error(monkeypatch: pytest.MonkeyPatch, bad_value: str) -> None:
    """A relative value depends on the caller's working directory, so it is refused."""
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, bad_value)
    with pytest.raises(ConfigError, match="must be an absolute path"):
        paths.data_dir()


@pytest.mark.unit
def test_empty_override_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value is a mistake, not a request for the default."""
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, "   ")
    with pytest.raises(ConfigError, match="set but empty"):
        paths.data_dir()


@pytest.mark.unit
def test_override_pointing_at_a_file_raises_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A file where a directory is required is diagnosed immediately."""
    target = tmp_path / "cn_data.zip"
    target.write_bytes(b"not a directory")
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, str(target))
    with pytest.raises(ConfigError, match="points at a file"):
        paths.data_dir()


@pytest.mark.unit
def test_missing_directory_fails_only_when_the_consumer_requires_it(tmp_path: Path) -> None:
    """Resolution never requires existence; the consumer that opens the store does."""
    missing = tmp_path / "not-downloaded-yet"
    assert paths.data_dir(missing) == missing
    with pytest.raises(ConfigError, match="does not exist"):
        paths.data_dir(missing, require_exists=True)


@pytest.mark.unit
def test_artifact_root_defaults_to_the_project_and_honours_the_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_root: Path
) -> None:
    """Artifacts land under ``<project_root>/artifacts`` unless ``ARTIFACT_ROOT`` says otherwise."""
    assert paths.artifact_root() == repo_root / "artifacts"
    monkeypatch.setenv(paths.ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    assert paths.artifact_root() == tmp_path
    assert paths.applied_overrides()[0].name == paths.ARTIFACT_ROOT_ENV_VAR


@pytest.mark.unit
def test_cache_root_is_derived_from_the_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The cache is a child of the artifact root, so it can never be aimed inside the provider."""
    monkeypatch.setenv(paths.ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    assert paths.cache_root() == tmp_path / "qlib_cache"
    explicit = tmp_path / "elsewhere"
    assert paths.cache_root(explicit) == explicit
    assert paths.cache_root() != paths.data_dir()


@pytest.mark.unit
def test_interpreter_resolution_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The interpreter location is declarative; the strictness that consumes it is not."""
    assert paths.interpreter() == Path(sys.executable)

    monkeypatch.setenv(paths.INTERPRETER_ENV_VAR, sys.executable)
    assert paths.interpreter() == Path(sys.executable)

    prefix = tmp_path / "envs" / "qresearch"
    conda_python = prefix / Path(*paths.CONDA_PYTHON_RELATIVE)
    conda_python.parent.mkdir(parents=True, exist_ok=True)
    conda_python.write_bytes(b"fake interpreter")
    monkeypatch.delenv(paths.INTERPRETER_ENV_VAR)
    monkeypatch.setenv(paths.CONDA_PREFIX_ENV_VAR, str(prefix))
    assert paths.interpreter() == conda_python
    assert paths.applied_overrides()[0].name == paths.CONDA_PREFIX_ENV_VAR


@pytest.mark.unit
def test_interpreter_override_must_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A wrong ``QRESEARCH_PYTHON`` is a configuration error, not a silent fallback."""
    monkeypatch.setenv(paths.INTERPRETER_ENV_VAR, str(tmp_path / "missing" / "python.exe"))
    with pytest.raises(ConfigError, match="existing interpreter"):
        paths.interpreter()


@pytest.mark.unit
def test_candidate_interpreters_priority_is_pure() -> None:
    """Candidate selection inspects a mapping only, so the stdlib-only hook runner can mirror it."""
    foreign = str(Path("C:") / "tools" / "python.exe")
    prefix = str(Path("C:") / "envs" / "qresearch")
    environ = {paths.INTERPRETER_ENV_VAR: foreign, paths.CONDA_PREFIX_ENV_VAR: prefix}
    candidates = paths.candidate_interpreters(environ)
    assert str(candidates[0]) == foreign
    assert candidates[1] == Path(prefix).joinpath(*paths.CONDA_PYTHON_RELATIVE)
    assert candidates[-1] == Path(sys.executable)
    assert paths.candidate_interpreters({}) == (Path(sys.executable),)


@pytest.mark.unit
def test_clear_applied_overrides_resets_the_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The record is per-process and inspectable, which is what "logged when applied" requires."""
    monkeypatch.setenv(paths.ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    paths.artifact_root()
    assert len(paths.applied_overrides()) == 1
    paths.clear_applied_overrides()
    assert paths.applied_overrides() == ()


@pytest.mark.unit
def test_tracked_sources_contain_no_local_absolute_path(repo_root: Path) -> None:
    """ADR-005 rule 1, checked with the gate's own patterns: no machine path in a loaded file.

    The gate exempts a file that carries its own exemption marker (``AUDIT.ALLOW_MARKER``), so the
    rule is mirrored here instead of approximated - otherwise the gate's own detector, which has to
    spell its patterns out, would read as a violation.  The exempt set is asserted, so that a
    production file cannot quietly acquire the marker and switch the rule off for itself.
    """
    exempt: set[str] = set()
    offenders: list[str] = []
    for relative in AUDIT.tracked_files(repo_root):
        normalised = relative.replace("\\", "/")
        if not normalised.startswith(SCANNED_ROOTS):
            continue
        text = (repo_root / relative).read_text(encoding="utf-8", errors="replace")
        if AUDIT.ALLOW_MARKER in text:
            exempt.add(normalised)
            continue
        for name, line_number, line in AUDIT.match_patterns(text, AUDIT.PATH_PATTERNS):
            offenders.append(f"{relative}:{line_number} [{name}] {line.strip()[:80]}")
    assert exempt == EXPECTED_EXEMPT_IN_SCANNED_ROOTS, "the path-scan exemption changed"
    assert not offenders, "local absolute paths must not appear in loaded files:\n" + "\n".join(offenders)
