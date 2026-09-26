"""Tests for the pre-publish hygiene gate (``scripts/prepublish_audit.py``).

The gate is the artefact that decides whether this repository may be pushed to a public remote,
so it is tested like production code rather than trusted because "it printed PASS":

* the detector table is validated against its synthetic positives and negatives (unit);
* the allow marker is proven to exempt *paths* while never exempting a secret (integration, in a
  throwaway git repository);
* the end-state gate — "this revision is publishable" — is marked ``slow`` on purpose: it is red
  by design while the findings recorded in ``docs/audits/PRE-PUBLISH-AUDIT.md`` are unfixed, and
  the default pre-commit gate runs ``-m "not slow"``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "prepublish_audit.py"


def _load_audit_module() -> ModuleType:
    """Import ``scripts/prepublish_audit.py`` without requiring ``scripts/`` to be a package.

    The module is registered in :data:`sys.modules` before execution because
    :func:`dataclasses.dataclass` resolves annotations through the owning module, which is
    absent for a spec-loaded module otherwise.
    """
    spec = importlib.util.spec_from_file_location("prepublish_audit", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None, "the audit script must be importable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _git(repo: Path, *args: str) -> None:
    """Run a git command inside the scratch repository, failing the test on error."""
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> Path:
    """Create a throwaway git repository (its index is populated by the test that uses it)."""
    repo = tmp_path / "scratch"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


@pytest.mark.unit
def test_exit_codes_are_distinct_bits() -> None:
    """The gate must be usable in a script: distinct, bitwise-combinable exit codes."""
    assert AUDIT.EXIT_CLEAN == 0
    combined = AUDIT.EXIT_ABSOLUTE_PATH | AUDIT.EXIT_DATA_ISOLATION | AUDIT.EXIT_SECRET
    assert combined == 7, "the three finding classes must occupy three distinct bits"
    assert AUDIT.EXIT_AUDIT_FAILURE not in {0, combined}


@pytest.mark.unit
def test_detector_selftest_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """The gate must prove its own detectors before it is trusted with a verdict."""
    assert AUDIT.selftest() == AUDIT.EXIT_CLEAN
    assert "selftest passed" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "detector", "expected", "sample"),
    AUDIT._selftest_cases(),  # the case table is the detector contract
    ids=lambda value: str(value)[:24],
)
def test_detectors_match_expectations(label: str, detector: str, expected: bool, sample: str) -> None:
    """Every sample must be detected (or ignored) exactly as declared."""
    if detector == "paths":
        detected = bool(AUDIT.match_patterns(sample, AUDIT.PATH_PATTERNS))
    else:
        detected = bool(AUDIT.match_secrets(sample))
    assert detected is expected, f"{label}: expected detection={expected}, got {detected} for {sample!r}"


@pytest.mark.unit
def test_load_bearing_classification() -> None:
    """Code and configuration are load-bearing; documentation is not."""
    assert AUDIT.is_load_bearing("src/qresearch/env.py")
    assert AUDIT.is_load_bearing("scripts/prepublish_audit.py")
    assert AUDIT.is_load_bearing(".pre-commit-config.yaml")
    assert AUDIT.is_load_bearing(".gitignore")
    assert not AUDIT.is_load_bearing("README.md")
    assert not AUDIT.is_load_bearing("docs/audits/PRE-PUBLISH-AUDIT.md")
    assert not AUDIT.is_load_bearing("data/.gitkeep")


@pytest.mark.unit
def test_audit_report_is_present_and_self_exempting() -> None:
    """The audit must leave a reviewable artefact, and it must declare its own exemption."""
    report = REPO_ROOT / "docs" / "audits" / "PRE-PUBLISH-AUDIT.md"
    assert report.is_file(), "the pre-publish review report must be committed with the fixes"
    assert AUDIT.ALLOW_MARKER in report.read_text(encoding="utf-8")


@pytest.mark.integration
def test_allow_marker_exempts_paths_but_never_secrets(scratch_repo: Path) -> None:
    """The marker is an auditable exemption for path documentation - not for credentials.

    Samples are assembled from fragments so that this test file itself stays clean for the gate
    it exercises (a test that trips the scanner it tests is worse than useless).
    """
    sep = chr(92)
    drive = "D" + ":" + sep + "private" + sep + "notes"
    secret_line = "pass" + "word = " + '"' + "not-a-real-credential" + '"'
    (scratch_repo / "docs").mkdir()
    marked = scratch_repo / "docs" / "report.md"
    marked.write_text(f"{AUDIT.ALLOW_MARKER}\n{drive}\n{secret_line}\n", encoding="utf-8")
    (scratch_repo / "configs").mkdir()
    unmarked = scratch_repo / "configs" / "bad.yaml"
    unmarked.write_text("provider_uri: " + '"' + drive + '"' + "\n", encoding="utf-8")
    _git(scratch_repo, "add", "-A")

    assert AUDIT.allowlisted_files(scratch_repo) == ["docs/report.md"]

    path_findings = AUDIT.scan_absolute_paths(scratch_repo)
    assert [finding.path for finding in path_findings] == ["configs/bad.yaml"], "only the unmarked file may be reported"
    assert path_findings[0].severity == AUDIT.SEVERITY_BLOCKER
    assert AUDIT.exit_code(path_findings) == AUDIT.EXIT_ABSOLUTE_PATH

    secret_findings = AUDIT.scan_secrets(scratch_repo)
    assert any(
        "hardcoded-credential-assignment" in finding.detail for finding in secret_findings
    ), "the allow marker must never exempt a credential"
    assert all(finding.severity == AUDIT.SEVERITY_BLOCKER for finding in secret_findings)
    assert AUDIT.exit_code(secret_findings) == AUDIT.EXIT_SECRET


@pytest.mark.integration
def test_data_isolation_rules_catch_a_committed_data_store(scratch_repo: Path) -> None:
    """A tracked binary market file must be a blocker even when no ignore rule mentions it."""
    data_dir = scratch_repo / "cn_data" / "features" / "sh600000"
    data_dir.mkdir(parents=True)
    (data_dir / "close.day.bin").write_bytes(b"qresearch-fixture")
    (scratch_repo / "weights.pth").write_bytes(b"qresearch-fixture")
    (scratch_repo / "run.log").write_text("epoch 1\n", encoding="utf-8")
    _git(scratch_repo, "add", "-A")

    findings = AUDIT.scan_data_isolation(scratch_repo)
    blockers = {finding.path for finding in findings if finding.severity == AUDIT.SEVERITY_BLOCKER}
    assert {"cn_data/features/sh600000/close.day.bin", "weights.pth", "run.log"} <= blockers
    assert AUDIT.exit_code(findings) & AUDIT.EXIT_DATA_ISOLATION


@pytest.mark.integration
def test_ignore_gaps_are_reported_as_warnings(scratch_repo: Path) -> None:
    """Hardening gaps warn without blocking: the committed state is what the gate must stop."""
    mandatory = "\n".join(AUDIT.MANDATORY_IGNORE_PATTERNS)
    (scratch_repo / ".gitignore").write_text(mandatory + "\n", encoding="utf-8")
    _git(scratch_repo, "add", "-A")

    findings = AUDIT.scan_data_isolation(scratch_repo)
    gaps = [finding for finding in findings if finding.path in {probe for probe, _ in AUDIT.PROBE_PATHS}]
    assert gaps, "an uncovered leak path must be reported"
    assert all(finding.severity == AUDIT.SEVERITY_WARNING for finding in gaps)
    assert AUDIT.exit_code(findings) == AUDIT.EXIT_CLEAN, "a warning must not block a normal run"
    assert AUDIT.exit_code(findings, strict=True) & AUDIT.EXIT_DATA_ISOLATION, "--strict must promote it"


@pytest.mark.slow
@pytest.mark.integration
def test_repository_is_publish_ready() -> None:
    """End-state gate: the checked-out revision must be clean under ``--strict``.

    Marked ``slow`` deliberately: it stays red until the findings in
    ``docs/audits/PRE-PUBLISH-AUDIT.md`` are fixed, and the default gate runs ``-m "not slow"``.
    """
    findings = AUDIT.run_audit(REPO_ROOT)
    blockers = [finding for finding in findings if finding.severity == AUDIT.SEVERITY_BLOCKER]
    rendered = "\n".join(finding.render() for finding in blockers)
    assert not blockers, f"{len(blockers)} blocker(s) still block publication:\n{rendered}"
    assert AUDIT.exit_code(findings, strict=True) == AUDIT.EXIT_CLEAN, "--strict must be clean too"
