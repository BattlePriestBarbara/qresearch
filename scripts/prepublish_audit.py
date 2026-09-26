"""Pre-publish hygiene gate: absolute paths, data isolation and secret leakage.

This script is the machine-checkable half of the pre-publish review.  It answers the three
questions that must all be answered "yes" before this repository reaches a public remote:

1. **No local absolute paths.**  Every file the toolchain loads - ``src/``, ``scripts/``,
   ``tests/``, ``configs/`` and the root configuration files (``environment.yml``,
   ``pyproject.toml``, ``.pre-commit-config.yaml``, ``.gitignore``, the lock files) - must be free
   of drive-letter, UNC and home-directory paths.  Paths must be relative or come from an
   environment variable.  Documentation may *describe* a machine layout, but that description is
   a warning, not a dependency.
2. **Data stays out of Git.**  A Qlib binary store, model weights, experiment state and log
   files must never be committable, no tracked file may exceed the cap enforced by the
   ``check-added-large-files`` hook, and the repository footprint must stay small.
3. **No secrets.**  API keys, tokens, passwords, private keys and credential-bearing URLs must
   not appear in a tracked file - nor in ``.git/config``, which is *not* tracked and is therefore
   the classic undetected leak (a token embedded in a remote URL).  Commit author identities are
   reported as a privacy item.

Usage
-----
.. code-block:: powershell

    python scripts/prepublish_audit.py                # report; non-zero on any blocker
    python scripts/prepublish_audit.py --strict       # warnings fail as well
    python scripts/prepublish_audit.py --selftest     # validate the detectors themselves

Exit codes (bitwise OR-able, mirroring the distinct-code style of ``scripts/check_env.py``)
------------------------------------------------------------------------------------------
0 - clean
1 - absolute path in a machine-loaded file
2 - data-isolation or repository-size violation
4 - secret, credential or identity finding

Design notes
------------
*   Only **tracked** files are scanned (``git ls-files``).  What cannot be pushed is out of
    scope; what *can* be pushed is what matters.
*   A file may exempt itself by containing the marker ``prepublish-audit:allow``.  The exemption
    is always reported, never silent: the audit report that *documents* the violations it found
    is the intended user of the marker, and a marker added to hide a real secret stays visible
    in the output.
*   ``docs/audits/`` is documentation by construction, so a method that is correct in code is
    still found there as a warning.  Run with ``--strict`` to make documentation warnings fail
    the gate as well.

prepublish-audit:allow
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

EXIT_CLEAN: Final[int] = 0
EXIT_ABSOLUTE_PATH: Final[int] = 1
EXIT_DATA_ISOLATION: Final[int] = 2
EXIT_SECRET: Final[int] = 4
EXIT_AUDIT_FAILURE: Final[int] = 8

CHECK_PATHS: Final[str] = "absolute-paths"
CHECK_DATA: Final[str] = "data-isolation"
CHECK_SECRETS: Final[str] = "secrets"
CHECK_GIT: Final[str] = "git-metadata"

SEVERITY_BLOCKER: Final[str] = "BLOCKER"
SEVERITY_WARNING: Final[str] = "WARNING"

ALLOW_MARKER: Final[str] = "prepublish-audit:allow"

MAX_TRACKED_FILE_BYTES: Final[int] = 2 * 1024 * 1024  # the hook cap: check-added-large-files --maxkb=2048
REVIEW_REPOSITORY_BYTES: Final[int] = 8 * 1024 * 1024
MAX_REPOSITORY_BYTES: Final[int] = 32 * 1024 * 1024
MAX_EXAMPLES_PER_FILE: Final[int] = 3
MAX_DETAIL_CHARS: Final[int] = 78


# Files whose contents are parsed or executed by the toolchain: a path literal here is a
# portability defect, not a documentation style issue.
LOAD_BEARING_PREFIXES: Final[tuple[str, ...]] = ("src/", "scripts/", "tests/", "configs/")
LOAD_BEARING_FILES: Final[tuple[str, ...]] = (
    ".flake8",
    ".gitignore",
    ".mypy.ini",
    ".pre-commit-config.yaml",
    ".pylintrc",
    "environment.yml",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.lock.hashes.txt",
    "requirements.lock.pip.txt",
    "requirements.lock.txt",
    "requirements.txt",
)

# .gitignore rules without which data or weights become committable (mirrors
# tests/test_repo_structure.py::test_gitignore_excludes_forbidden_categories).
MANDATORY_IGNORE_PATTERNS: Final[tuple[str, ...]] = (
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
    "*.bin",
    "*.safetensors",
    ".ipynb_checkpoints/",
    "!artifacts/.gitkeep",
    "!data/.gitkeep",
)

# File types that must never be reachable from a commit: binary market data, model weights,
# serialised experiment state and generated derived datasets.
FORBIDDEN_SUFFIXES: Final[tuple[str, ...]] = (
    ".bin",
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".hdf5",
    ".onnx",
    ".npz",
    ".npy",
    ".safetensors",
    ".model",
    ".weights",
    ".log",
    ".parquet",
    ".feather",
    ".arrow",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".zip",
    ".7z",
    ".tar",
    ".tgz",
    ".gz",
    ".pem",
    ".key",
    ".pfx",
    ".p12",
)

# Paths a study might plausibly produce in the working tree, with the .gitignore pattern that
# would make the working tree itself safe (not merely its committed state).  These are hardening
# gaps reported as warnings: the committed state is checked by the rules above.
PROBE_PATHS: Final[tuple[tuple[str, str], ...]] = (
    ("cn_data/instruments/all.txt", "cn_data/"),
    ("calendars/day.txt", "calendars/"),
    ("predictions.parquet", "*.parquet (or keep derived tables under artifacts/)"),
    ("results.csv", "*.csv (or keep derived tables under artifacts/)"),
    ("features.feather", "*.feather"),
    ("readings.arrow", "*.arrow"),
    ("study.db", "*.db, *.sqlite3"),
    ("run.zarr/.zgroup", "*.zarr/"),
    ("configs/local.yaml", "configs/local*.yaml (per-machine overrides)"),
)

DATA_LOOKALIKE_DIRS: Final[tuple[str, ...]] = (
    "calendars",
    "features",
    "instruments",
    "cn_data",
    "qlib_data",
)

_WIN_SEP: Final[str] = chr(92)

# UNC paths have the shape ``\\host\share``: exactly two separators, then one.  The strict shape
# is what keeps LaTeX maths out of the report - the ``\\not\\subseteq`` sequences in this
# project's docstrings use doubled separators throughout and are therefore never paths.  The
# trade-off is deliberate: a real UNC path matters, a false positive on every formula is worse.
_UNC_SEP: Final[str] = re.escape(_WIN_SEP)
_UNC_CHARS: Final[str] = r"[A-Za-z0-9_.$-]{2,}"
UNC_PATH_RE: Final[re.Pattern[str]] = re.compile(rf"(?:{_UNC_SEP}){{2}}{_UNC_CHARS}(?:{_UNC_SEP}){{1}}{_UNC_CHARS}")

# Path patterns.  The drive-letter form requires a non-alphanumeric predecessor and rejects a
# doubled separator, so URL schemes such as ``https://`` are not mistaken for a drive.
PATH_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "windows-drive-path",
        re.compile(rf"(?:^|[^A-Za-z0-9])[A-Za-z]:[{re.escape(_WIN_SEP)}/](?!/)"),
    ),
    ("windows-unc-path", UNC_PATH_RE),
    (
        "posix-home-path",
        re.compile(
            r"(?:^|[\s\"'(=])(?:" + "|".join(re.escape(part) for part in ("/home/", "/Users/", "/root/", "~/")) + ")"
        ),
    ),
    (
        "home-environment-variable",
        re.compile(r"%USERPROFILE%|%APPDATA%|\$HOME\b", re.IGNORECASE),
    ),
)

SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("private-key-header", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    (
        "cloud-credential-prefix",
        re.compile(
            r"\b(?:A[KS]IA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}"
            r"|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
    ),
    ("credential-bearing-url", re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s:@]+@")),
)

SENSITIVE_ASSIGNMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|access[_-]?key|secret[_-]?key|client[_-]?secret|secret|token|password|passwd"
    r"|private[_-]?key)\b\s*[:=]\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
)

# Values that are obviously not credentials: environment lookups, placeholders, prose.
PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(?:<.*>|\$\{?[A-Za-z0-9_]+\}?|%[A-Za-z0-9_]+%|[A-Z][A-Z0-9_]{2,}"
    r"|.*(?:redact|placeholder|example|dummy|fake|changeme|todo|none|null|xxx).*)$"
)

EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
# Domains whose addresses carry no personal information: documentation placeholders, the
# publisher noreply identity and localhost.
EXAMPLE_DOMAINS: Final[tuple[str, ...]] = (
    "users.noreply.github.com",
    "noreply.github.com",
    "example.com",
    "example.org",
    "example.net",
    "localhost",
)

MINIMUM_SECRET_LENGTH: Final[int] = 8


class AuditError(RuntimeError):
    """Raised when the audit itself cannot be completed (for example, git is unavailable)."""


@dataclass(frozen=True)
class Finding:
    """A single audit result, rendered as one line of the report."""

    check: str
    severity: str
    path: str
    line: int | None
    detail: str

    def render(self) -> str:
        """Return the one-line rendering used by the console report."""
        location = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"  {self.severity:<7} {location:<52} {_truncate(self.detail, MAX_DETAIL_CHARS)}"


# ---------------------------------------------------------------------------------------
# Pure helpers - testable without git, a repository, or the network
# ---------------------------------------------------------------------------------------
def _truncate(text: str, limit: int) -> str:
    """Return ``text`` flattened to a single line and shortened to ``limit`` characters."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 3] + "..."


def is_load_bearing(relative_path: str) -> bool:
    """Return whether ``relative_path`` is parsed or executed (as opposed to documentation)."""
    posix = relative_path.replace(chr(92), "/")
    return posix in LOAD_BEARING_FILES or posix.startswith(LOAD_BEARING_PREFIXES)


def match_patterns(text: str, patterns: Sequence[tuple[str, re.Pattern[str]]]) -> list[tuple[str, int, str]]:
    """Return ``(rule_name, line_number, line)`` for every line matching one of ``patterns``."""
    matches: list[tuple[str, int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in patterns:
            if pattern.search(line):
                matches.append((name, line_number, line))
                break
    return matches


def match_secrets(text: str) -> list[tuple[str, int, str]]:
    """Return ``(rule_name, line_number, line)`` for every secret-like construct in ``text``.

    Three rule families are applied: well-known credential formats, credential-bearing URLs, and
    assignments of a sensitive name to a quoted literal that is not recognisably a placeholder or
    an environment lookup.  Personal e-mail addresses are reported as well, because publishing a
    commit or a file that carries a private address is a desensitisation defect, not a secret.
    """
    matches = list(match_patterns(text, SECRET_PATTERNS))
    for line_number, line in enumerate(text.splitlines(), start=1):
        assignment = SENSITIVE_ASSIGNMENT_RE.search(line)
        if assignment is not None:
            value = assignment.group("value").strip()
            if len(value) >= MINIMUM_SECRET_LENGTH and PLACEHOLDER_RE.match(value) is None:
                matches.append(("hardcoded-credential-assignment", line_number, line))
        for address in EMAIL_RE.findall(line):
            if not address.lower().endswith(EXAMPLE_DOMAINS):
                matches.append(("personal-identifier-email", line_number, line))
                break
    return sorted(set(matches), key=lambda match: match[1])


def redact_email(address: str) -> str:
    """Return ``address`` with the local part masked, keeping the domain for triage."""
    local, _, domain = address.partition("@")
    prefix = local[:3] if len(local) > 3 else local[:1]
    return f"{prefix}{'*' * 3}@{domain}"


# ---------------------------------------------------------------------------------------
# Git-backed access (read-only: this audit never writes to the repository or the index)
# ---------------------------------------------------------------------------------------
def _run_raw_git(*args: str, root: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Run a git command inside ``root`` without raising, returning the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _run_git(*args: str, root: Path = REPO_ROOT) -> str:
    """Run a git command inside ``root`` and return stdout, raising on failure."""
    result = _run_raw_git(*args, root=root)
    if result.returncode != 0:
        raise AuditError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: {_truncate(result.stderr, 200)}"
        )
    return result.stdout


def tracked_files(root: Path = REPO_ROOT) -> list[str]:
    """Return every tracked path, POSIX-normalised, in git's own order."""
    listing = _run_git("ls-files", root=root)
    return [line.strip().replace(chr(92), "/") for line in listing.splitlines() if line.strip()]


def is_ignored(relative_path: str, root: Path = REPO_ROOT) -> bool:
    """Return whether ``.gitignore`` excludes ``relative_path`` (via ``git check-ignore``)."""
    return _run_raw_git("check-ignore", "-q", relative_path, root=root).returncode == 0


def _read_text(path: Path) -> str | None:
    """Return the decoded text of ``path``, or ``None`` when it is missing, binary or unreadable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def allowlisted_files(root: Path = REPO_ROOT) -> list[str]:
    """Return the tracked files that carry :data:`ALLOW_MARKER` and are therefore exempt."""
    exempt: list[str] = []
    for relative in tracked_files(root):
        text = _read_text(root / relative)
        if text is not None and ALLOW_MARKER in text:
            exempt.append(relative)
    return exempt


# ---------------------------------------------------------------------------------------
# Check 1 - local absolute paths in machine-loaded files
# ---------------------------------------------------------------------------------------
def scan_absolute_paths(root: Path = REPO_ROOT) -> list[Finding]:
    """Report every local absolute path found in a machine-loaded tracked file.

    One finding is emitted per offending line, so a file with twelve hits yields twelve
    entries; the renderer collapses them per file.  Documentation paths are warnings, because
    a README may legitimately describe where the market data lives on the author's machine -
    the defect is when *code* depends on that location.
    """
    findings: list[Finding] = []
    for relative in tracked_files(root):
        text = _read_text(root / relative)
        if text is None or ALLOW_MARKER in text:
            continue
        hits = match_patterns(text, PATH_PATTERNS)
        if not hits:
            continue
        severity = SEVERITY_BLOCKER if is_load_bearing(relative) else SEVERITY_WARNING
        first_hit_per_line: dict[int, tuple[str, str]] = {}
        for name, line_number, line in hits:
            first_hit_per_line.setdefault(line_number, (name, line))
        for line_number, (name, line) in sorted(first_hit_per_line.items()):
            findings.append(Finding(CHECK_PATHS, severity, relative, line_number, f"{name}: {line.strip()}"))
    return findings


# ---------------------------------------------------------------------------------------
# Check 2 - data isolation and repository footprint
# ---------------------------------------------------------------------------------------
_SIZE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bytes|KiB|MiB|GiB)$", re.IGNORECASE
)
_UNIT_FACTORS: Final[dict[str, int]] = {"bytes": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3}


def _parse_size(text: str) -> int:
    """Parse a ``git count-objects -vH`` size field such as ``389.97 KiB`` into bytes."""
    match = _SIZE_RE.match(text.strip())
    if match is None:
        raise AuditError(f"cannot parse the git size field {text!r}")
    return int(float(match.group("value")) * _UNIT_FACTORS[match.group("unit").lower()])


def _tracked_but_ignored(root: Path) -> list[Finding]:
    """Report files that are in the index although ``.gitignore`` matches them."""
    listing = _run_git("ls-files", "-i", "-c", "--exclude-standard", root=root)
    return [
        Finding(CHECK_DATA, SEVERITY_BLOCKER, line.strip(), None, "in the index although .gitignore matches it")
        for line in listing.splitlines()
        if line.strip()
    ]


def _forbidden_tracked_files(root: Path) -> list[Finding]:
    """Report tracked files whose extension marks them as data, weights or serialised state."""
    findings: list[Finding] = []
    for relative in tracked_files(root):
        suffix = Path(relative).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(Finding(CHECK_DATA, SEVERITY_BLOCKER, relative, None, f"{suffix} is data/weights/state"))
    return findings


def _oversized_tracked_files(root: Path) -> list[Finding]:
    """Report tracked files above the commit-size cap enforced by the pre-commit hook."""
    findings: list[Finding] = []
    for relative in tracked_files(root):
        try:
            size = (root / relative).stat().st_size
        except OSError:
            continue
        if size > MAX_TRACKED_FILE_BYTES:
            cap_mib = MAX_TRACKED_FILE_BYTES // 1024 // 1024
            findings.append(
                Finding(CHECK_DATA, SEVERITY_BLOCKER, relative, None, f"{size / 1024 / 1024:.1f} MB > {cap_mib} MB cap")
            )
    return findings


def _missing_ignore_patterns(root: Path) -> list[Finding]:
    """Report .gitignore rules whose removal would make data or weights committable."""
    declared = set((_read_text(root / ".gitignore") or "").split())
    return [
        Finding(
            CHECK_DATA,
            SEVERITY_BLOCKER,
            ".gitignore",
            None,
            f"missing rule {pattern!r}; its category becomes committable",
        )
        for pattern in MANDATORY_IGNORE_PATTERNS
        if pattern not in declared
    ]


def _ignore_gaps(root: Path) -> list[Finding]:
    """Report working-tree locations a study could create that .gitignore does not yet cover."""
    findings: list[Finding] = []
    for probe, suggestion in PROBE_PATHS:
        if not is_ignored(probe, root):
            findings.append(Finding(CHECK_DATA, SEVERITY_WARNING, probe, None, f"not ignored; consider {suggestion}"))
    return findings


def _repository_footprint(root: Path) -> list[Finding]:
    """Report the size of the git object database against the publish budget."""
    fields = dict(
        line.split(":", 1) for line in _run_git("count-objects", "-vH", root=root).splitlines() if ":" in line
    )
    loose = _parse_size(fields.get("size", "0 bytes"))
    packed = _parse_size(fields.get("size-pack", "0 bytes"))
    total = loose + packed
    detail = f"object database {total / 1024 / 1024:.2f} MB (loose {loose / 1024 / 1024:.2f} MB)"
    if total > MAX_REPOSITORY_BYTES:
        return [
            Finding(
                CHECK_DATA,
                SEVERITY_BLOCKER,
                "git objects",
                None,
                f"{detail} exceeds the {MAX_REPOSITORY_BYTES // 1024 // 1024} MB budget",
            )
        ]
    if total > REVIEW_REPOSITORY_BYTES:
        return [
            Finding(CHECK_DATA, SEVERITY_WARNING, "git objects", None, f"{detail} - approaching the publish budget")
        ]
    return []


def _lookalike_directories(root: Path) -> list[Finding]:
    """Report market-data look-alike directories present inside the working tree."""
    return [
        Finding(
            CHECK_DATA,
            SEVERITY_WARNING,
            f"{name}/",
            None,
            "store-like directory inside the repository; market data belongs outside it",
        )
        for name in DATA_LOOKALIKE_DIRS
        if (root / name).is_dir()
    ]


def _unreachable_objects(root: Path) -> list[Finding]:
    """Report unreachable objects, which a mirror push or an archive would still carry."""
    listing = _run_git("fsck", "--no-progress", "--unreachable", "--no-dangling", root=root)
    count = len([line for line in listing.splitlines() if line.startswith("unreachable ")])
    if not count:
        return []
    return [
        Finding(
            CHECK_DATA,
            SEVERITY_WARNING,
            "git objects",
            None,
            f"{count} unreachable object(s); `git gc --prune=now` before mirroring",
        )
    ]


def scan_data_isolation(root: Path = REPO_ROOT) -> list[Finding]:
    """Report every way data, weights or repository bloat could reach a public remote."""
    findings: list[Finding] = []
    for check in (
        _tracked_but_ignored,
        _forbidden_tracked_files,
        _oversized_tracked_files,
        _missing_ignore_patterns,
        _ignore_gaps,
        _repository_footprint,
        _lookalike_directories,
        _unreachable_objects,
    ):
        findings.extend(check(root))
    return findings


# ---------------------------------------------------------------------------------------
# Check 3 - secrets, credentials and identity
# ---------------------------------------------------------------------------------------
def scan_secrets(root: Path = REPO_ROOT) -> list[Finding]:
    """Report secret-like content in tracked files.

    The ``prepublish-audit:allow`` marker exempts a file from the *path* scan, never from this
    one: a credential is a blocker even inside a file that documents path violations.  A personal
    e-mail address is a warning, because it is a desensitisation defect rather than a secret.
    """
    findings: list[Finding] = []
    for relative in tracked_files(root):
        text = _read_text(root / relative)
        if text is None:
            continue
        for rule, line_number, line in match_secrets(text):
            severity = SEVERITY_WARNING if rule == "personal-identifier-email" else SEVERITY_BLOCKER
            findings.append(Finding(CHECK_SECRETS, severity, relative, line_number, f"{rule}: {line.strip()}"))
    return findings


def _local_state_refs(root: Path = REPO_ROOT) -> list[str]:
    """Return refs that are neither a branch, a tag nor a remote-tracking ref."""
    listing = _run_git("for-each-ref", "--format=%(refname)", root=root)
    keep = ("refs/heads/", "refs/tags/", "refs/remotes/")
    return [line.strip() for line in listing.splitlines() if line.strip() and not line.strip().startswith(keep)]


def _commit_identities(root: Path = REPO_ROOT) -> dict[str, int]:
    """Return the number of commits per author e-mail address, across every ref."""
    counts: dict[str, int] = {}
    for line in _run_git("log", "--all", "--format=%ae", root=root).splitlines():
        address = line.strip().lower()
        if address:
            counts[address] = counts.get(address, 0) + 1
    return counts


def scan_git_metadata(root: Path = REPO_ROOT) -> list[Finding]:
    """Report leaks that live in ``.git`` rather than in the working tree.

    ``.git/config`` is not tracked, so a personal access token embedded in a remote URL never
    shows up in a source scan.  Commit identities are reported because they are published with
    the history and are the usual source of an unintended personal e-mail address; local-state
    refs are reported because a mirror push or an archive carries them.
    """
    findings: list[Finding] = []
    config = _read_text(root / ".git" / "config")
    if config is not None:
        for rule, line_number, line in match_secrets(config):
            findings.append(Finding(CHECK_GIT, SEVERITY_BLOCKER, ".git/config", line_number, f"{rule}: {line.strip()}"))
    remotes = [" ".join(line.split()) for line in _run_git("remote", "-v", root=root).splitlines() if line.strip()]
    for remote in remotes:
        findings.append(
            Finding(CHECK_GIT, SEVERITY_WARNING, remote.split()[0], None, f"push target {remote.split()[1]}")
        )
    for address, count in sorted(_commit_identities(root).items()):
        findings.append(
            Finding(
                CHECK_GIT,
                SEVERITY_WARNING,
                "git history",
                None,
                f"{count} commit(s) by {redact_email(address)}; published with the history",
            )
        )
    for ref in _local_state_refs(root):
        findings.append(
            Finding(CHECK_GIT, SEVERITY_WARNING, ref, None, "local-state ref; a mirror push or an archive carries it")
        )
    return findings


def run_audit(root: Path = REPO_ROOT) -> list[Finding]:
    """Return every finding produced by the three checks."""
    return [
        *scan_absolute_paths(root),
        *scan_data_isolation(root),
        *scan_secrets(root),
        *scan_git_metadata(root),
    ]


def exit_code(findings: Sequence[Finding], *, strict: bool = False) -> int:
    """Return the bitwise exit code implied by ``findings`` (warnings count only with ``strict``)."""
    code = EXIT_CLEAN
    for finding in findings:
        if finding.severity != SEVERITY_BLOCKER and not strict:
            continue
        if finding.check == CHECK_PATHS:
            code |= EXIT_ABSOLUTE_PATH
        elif finding.check == CHECK_DATA:
            code |= EXIT_DATA_ISOLATION
        else:
            code |= EXIT_SECRET
    return code


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def _render_section(title: str, findings: Sequence[Finding], *, verbose: bool) -> None:
    """Print one report section, collapsing repeated hits per file unless ``verbose``."""
    blockers = [finding for finding in findings if finding.severity == SEVERITY_BLOCKER]
    warnings = [finding for finding in findings if finding.severity == SEVERITY_WARNING]
    status = "FAIL" if blockers else ("REVIEW" if warnings else "PASS")
    print(f"\n[{title}] {status} - {len(blockers)} blocker(s), {len(warnings)} warning(s)")
    shown: dict[str, int] = {}
    for finding in [*blockers, *warnings]:
        shown[finding.path] = shown.get(finding.path, 0) + 1
        if verbose or shown[finding.path] <= MAX_EXAMPLES_PER_FILE:
            print(finding.render())
    if not verbose:
        for path, count in shown.items():
            if count > MAX_EXAMPLES_PER_FILE:
                print(f"  ... {count - MAX_EXAMPLES_PER_FILE} more finding(s) in {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit, print the report and return the bitwise exit code."""
    parser = argparse.ArgumentParser(
        prog="prepublish-audit",
        description="Pre-publish hygiene gate: absolute paths, data isolation, secrets, git metadata.",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository to audit (default: this checkout)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--verbose", action="store_true", help="print every finding instead of three per file")
    parser.add_argument("--selftest", action="store_true", help="validate the detectors on synthetic samples")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        findings = run_audit(args.root)
        exempt = allowlisted_files(args.root)
        head = _run_git("log", "-1", "--format=%h %d", root=args.root).strip() or "(no commit yet)"
        tracked = len(tracked_files(args.root))
    except AuditError as error:
        print(f"FATAL: the audit could not be completed: {error}", file=sys.stderr)
        return EXIT_AUDIT_FAILURE

    print("qresearch pre-publish audit")
    print(f"  repository  : {args.root}")
    print(f"  revision    : {head}")
    print(f"  tracked     : {tracked} file(s)")
    print(f"  allow-listed: {', '.join(exempt) if exempt else 'none'}")

    _render_section(
        "1/3 absolute paths in machine-loaded files",
        [f for f in findings if f.check == CHECK_PATHS],
        verbose=args.verbose,
    )
    _render_section(
        "2/3 data isolation and repository size", [f for f in findings if f.check == CHECK_DATA], verbose=args.verbose
    )
    secret_findings = [f for f in findings if f.check in {CHECK_SECRETS, CHECK_GIT}]
    _render_section("3/3 secrets, credentials and identity", secret_findings, verbose=args.verbose)

    code = exit_code(findings, strict=args.strict)
    blockers = len([finding for finding in findings if finding.severity == SEVERITY_BLOCKER])
    print()
    if code == EXIT_CLEAN:
        print(f"VERDICT: SAFE TO PUBLISH - {len(findings)} warning(s), 0 blockers")
    else:
        mode = "with warnings promoted by --strict" if blockers == 0 else f"{blockers} blocker(s)"
        print(f"VERDICT: NOT SAFE TO PUBLISH - {mode} (exit code {code})")
    return code


# ---------------------------------------------------------------------------------------
# Detector self-test: proves the gate itself works before it is trusted
# ---------------------------------------------------------------------------------------
def _selftest_cases() -> list[tuple[str, str, bool, str]]:
    """Return ``(label, detector, expected_detection, sample)`` cases for :func:`selftest`.

    Every sample is assembled from fragments at run time, so this file stores no
    credential-shaped or path-shaped literal of its own and cannot poison its own scan.
    """
    sep = _WIN_SEP
    return [
        ("windows drive path", "paths", True, "D" + ":" + sep + "workspace" + sep + "market_data"),
        ("windows drive path, forward slashes", "paths", True, "D" + ":" + "/" + "workspace/data"),
        ("windows UNC path", "paths", True, sep * 2 + "fileserver" + sep + "share" + sep + "data.bin"),
        (
            "latex maths is not a UNC path",
            "paths",
            False,
            sep * 2 + "not" + sep * 2 + "subseteq" + sep * 2 + "mathcal{F}_t",
        ),
        (
            "escaped doubled literal is out of scope",
            "paths",
            False,
            sep * 2 + "Anaconda3" + sep * 2 + "python.exe",
        ),
        ("posix home path", "paths", True, "/home/" + "researcher/data"),
        ("HOME environment variable", "paths", True, "$" + "HOME/data"),
        ("USERPROFILE environment variable", "paths", True, "%" + "USERPROFILE%" + sep + "data"),
        ("https URL is not a drive path", "paths", False, "https" + "://" + "pypi.org/pypi/pyqlib/0.9.7/json"),
        ("env-var provider root is portable", "paths", False, "provider_uri: ${QRESEARCH_DATA_ROOT}/cn_data"),
        ("cloud credential prefix", "secrets", True, "AKIA" + "A" * 16),
        ("private key header", "secrets", True, "-----BEGIN " + "RSA " + "PRIVATE KEY-----"),
        (
            "credential-bearing URL",
            "secrets",
            True,
            "git" + "://" + "ci-user" + ":" + "s3cr3t-passphrase@x.example.net/r",
        ),
        ("hardcoded password assignment", "secrets", True, "password = " + '"' + "hunter2-not-a-real-secret" + '"'),
        ("environment lookup is not a secret", "secrets", False, 'api_key = os.environ["QRESEARCH' + '_API_KEY"]'),
        ("placeholder token is not a secret", "secrets", False, "token: " + '"${GITHUB_TOKEN}"'),
        ("publisher noreply address", "secrets", False, "noreply" + "@" + "users.noreply.github.com"),
    ]


def selftest() -> int:
    """Validate every detector against its positive sample and a benign counter-sample."""
    failures: list[str] = []
    for label, detector, expected, sample in _selftest_cases():
        detected = bool(match_patterns(sample, PATH_PATTERNS)) if detector == "paths" else bool(match_secrets(sample))
        verdict = "ok  " if detected == expected else "FAIL"
        expected_text = "detect" if expected else "ignore"
        print(f"  {verdict} {detector:<8} {label:<42} expected {expected_text}")
        if detected != expected:
            failures.append(label)
    print()
    if failures:
        print(f"selftest FAILED: {len(failures)} case(s) misbehaved: {', '.join(failures)}")
        return EXIT_AUDIT_FAILURE
    print(f"selftest passed: {len(_selftest_cases())} cases, detectors fire and stay silent as designed")
    return EXIT_CLEAN


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point
    raise SystemExit(main())
