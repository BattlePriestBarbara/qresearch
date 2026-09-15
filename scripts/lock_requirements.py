"""Generate the dependency lock artefacts for the verified environment (task ``INF-01``).

Why this is not a plain ``pip-compile``
---------------------------------------
``pyqlib==0.9.7`` requires ``gym``, and ``gym 0.26.2`` publishes **no wheel for CPython 3.12**
on PyPI (sdist only), so a naive ``pip install --dry-run --only-binary=:all:`` resolution of the
full closure fails with ``No matching distribution found for gym``.  The verified environment
nevertheless contains ``gym`` (provided by conda with an exact build string).

The lock is therefore computed **offline from the metadata of the verified environment**
(an ``importlib.metadata.requires`` walk), which is the stronger statement anyway: it records the
closure that actually produced the verified numerical results, not the one a resolver would pick
today.  Digests are then fetched from the PyPI JSON API, which is lightweight and yields hashes
for every file of each release (the convention ``pip-compile --generate-hashes`` follows).

Artefacts
---------
``requirements.lock.txt``
    Exact version closure: ``name==version`` for every distribution reachable from
    ``requirements.txt`` plus ``requirements-dev.txt``.  Offline and deterministic.
``requirements.lock.hashes.txt``
    The same closure with ``--hash=sha256:...`` digests, usable as
    ``pip install --require-hashes -r requirements.lock.hashes.txt``.
    Distributions with no PyPI wheel for this interpreter are listed in a comment block with
    their conda build string instead of being silently dropped.

Usage
-----
.. code-block:: powershell

    D:\\Anaconda3\\python.exe scripts/lock_requirements.py            # version lock (offline)
    D:\\Anaconda3\\python.exe scripts/lock_requirements.py --hashes   # + digest lock (network)
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
REQ_FILES = (REPO_ROOT / "requirements.txt", REPO_ROOT / "requirements-dev.txt")
VERSION_LOCK = REPO_ROOT / "requirements.lock.txt"
HASH_LOCK = REPO_ROOT / "requirements.lock.hashes.txt"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"

# Distributions that MUST be present in the closure but cannot be satisfied from a PyPI wheel on
# this interpreter. They are recorded with their provenance instead of being silently dropped.
NON_WHEEL_CANDIDATES = ("gym", "gym-notices")


def _canonical(name: str) -> str:
    """Return the PEP 503 canonical form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _direct_requirements(paths: Iterable[Path]) -> dict[str, str]:
    """Parse the exact pinned direct requirements into ``{canonical_name: version}``."""
    direct: dict[str, str] = {}
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            match = re.match(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$", line)
            if not match:
                raise SystemExit(f"{path.name}: '{line}' is not a full patch pin")
            direct[_canonical(match.group("name"))] = match.group("version")
    return direct


def _active_requirements(distribution: str) -> list[str]:
    """Return the distribution names required by ``distribution`` under this interpreter.

    Environment markers are evaluated with ``packaging`` (standards-compliant) and optional
    ``extra``-gated requirements are skipped, because the project installs no extras.
    """
    names: list[str] = []
    for raw in importlib_metadata.requires(distribution) or []:
        try:
            requirement = Requirement(raw)
        except Exception:  # pragma: no cover - malformed metadata in a third-party package
            continue
        if requirement.marker is not None:
            try:
                if not requirement.marker.evaluate({"extra": ""}):
                    continue
            except Exception:  # pragma: no cover - unevaluable marker
                continue
        names.append(requirement.name)
    return names


def _closure(direct: dict[str, str]) -> dict[str, str]:
    """Walk ``Requires-Dist`` transitively and return ``{canonical_name: installed_version}``."""
    resolved: dict[str, str] = {}
    queue: list[str] = list(direct)
    for name in queue:
        resolved.setdefault(name, "")
    while queue:
        name = queue.pop()
        try:
            resolved[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            raise SystemExit(
                f"distribution '{name}' is required but not installed; "
                "run `D:\\Anaconda3\\python.exe scripts/check_env.py` to diagnose the environment"
            ) from None
        for child in _active_requirements(name):
            key = _canonical(child)
            if key not in resolved:
                resolved[key] = ""
                queue.append(key)
    return {name: version for name, version in resolved.items() if version}


def _installer(distribution: str) -> str:
    """Return the tool that installed ``distribution`` ('pip', 'conda', or 'unknown')."""
    try:
        raw = importlib_metadata.distribution(distribution).read_text("INSTALLER")
    except (importlib_metadata.PackageNotFoundError, FileNotFoundError, OSError):
        return "unknown"
    return (raw or "unknown").strip().lower() or "unknown"


def _provenance_notes(closure: dict[str, str], conda_records: dict[str, str]) -> dict[str, str]:
    """Return human-readable provenance notes for the distributions that need one.

    Two cases are reported:

    * **non-wheel candidates** (``gym``, ``gym-notices``): required by ``pyqlib`` but sdist-only on
      PyPI for CPython 3.12, so they cannot enter the digest lock.  Full detail is emitted.
    * **conda-provided distributions**: installed from a conda channel rather than a PyPI wheel, so
      their build string is part of the reproducible identity.  Terse form is emitted.
    """
    candidates = {_canonical(name) for name in NON_WHEEL_CANDIDATES}
    notes: dict[str, str] = {}
    for name in sorted(closure):
        installer = _installer(name)
        record = conda_records.get(name, "")
        if name in candidates:
            parts = [f"NO PYPI WHEEL for this interpreter: installer={installer}"]
            if record:
                parts.append(record)
            notes[name] = " ".join(parts)
        elif installer != "pip" and record:
            channel = record.split(" ")[0].removeprefix("conda:")
            build = record.rsplit("build=", 1)[-1]
            notes[name] = f"conda:{channel} build={build}"
    return notes


def _conda_provenance(names: Iterable[str]) -> dict[str, str]:
    """Return conda build provenance for packs that do not come from PyPI.

    ``gym`` is the concrete case: pyqlib requires it, but no wheel exists for CPython 3.12, so it
    is satisfied by a conda package whose build string must be recorded for reproducibility.
    """
    conda = Path(sys.prefix) / "Scripts" / "conda.exe"
    if not conda.is_file():
        conda = Path(sys.prefix) / "bin" / "conda"
    if not conda.is_file():  # pragma: no cover - conda-less environment
        return {}
    try:
        completed = subprocess.run([str(conda), "list", "--json"], capture_output=True, text=True, check=False)
        payload = json.loads(completed.stdout)
    except Exception:  # pragma: no cover - conda CLI unavailable
        return {}
    wanted = {_canonical(name) for name in names}
    provenance: dict[str, str] = {}
    for entry in payload:
        key = _canonical(str(entry.get("name", "")))
        if key in wanted:
            provenance[key] = (
                f"conda:{entry.get('channel', '?')} version={entry.get('version')} build={entry.get('build_string')}"
            )
    return provenance


def _pypi_digests(name: str, version: str, *, timeout: float = 30.0) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every file of a release, read from the PyPI JSON API."""
    try:
        with urllib.request.urlopen(PYPI_JSON.format(name=name, version=version), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        print(f"warning: could not fetch digests for {name}=={version}: {error}", file=sys.stderr)
        return {}
    digests: dict[str, str] = {}
    for file_info in payload.get("urls", []):
        digest = str(file_info.get("digests", {}).get("sha256", ""))
        filename = str(file_info.get("filename", ""))
        if digest and filename:
            digests[filename] = digest
    return digests


def _has_compatible_wheel(filenames: Iterable[str]) -> bool:
    """Report whether any wheel matches this interpreter (cp312 or pure-python, this platform)."""
    for filename in filenames:
        if not filename.endswith(".whl"):
            continue
        tags = filename[: -len(".whl")].rsplit("-", 3)[-3:]
        if len(tags) != 3:
            continue
        python_tag, _abi, platform_tag = tags
        python_ok = any(token in {"cp312", "py3", "py2", "py2.py3"} for token in python_tag.split("."))
        platform_ok = platform_tag in {"any", "win_amd64"} if platform.system() == "Windows" else True
        if python_ok and platform_ok:
            return True
    return False


def _version_lock_lines(closure: dict[str, str], provenance: dict[str, str]) -> list[str]:
    """Render the version lock body plus the provenance comment block."""
    lines = [f"{name}=={version}" for name, version in sorted(closure.items())]
    if provenance:
        lines += ["", "# --- provenance of distributions that are NOT plain PyPI-wheel installs ---"]
        lines += [f"# {name}: {detail}" for name, detail in sorted(provenance.items())]
    return lines


def _header(title: str, extra: Iterable[str] = ()) -> list[str]:
    """Return the provenance header shared by both lock artefacts."""
    header = [
        "# =====================================================================================",
        f"# {title}",
        "#",
        "# GENERATED FILE - do not edit by hand. Regenerate with:",
        "#     D:\\Anaconda3\\python.exe scripts/lock_requirements.py [--hashes]",
        "#",
        f"# Generated  : {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# Python     : {platform.python_version()} on {platform.system()} {platform.machine()}",
        f"# Interpreter: {sys.executable}",
        "#",
        "# Rationale (PROJECT_SPEC.md 3.1): a silent version or build drift in numpy/scipy/torch",
        "# perturbs the low-order bits of the IC series and can move a Newey-West p-value across a",
        "# significance threshold. The lock makes such drift impossible to introduce unnoticed.",
        "# =====================================================================================",
    ]
    header.extend(extra)
    return header


def _write_version_lock(closure: dict[str, str], provenance: dict[str, str], output: Path) -> None:
    """Write ``requirements.lock.txt``: the exact version closure of the verified environment."""
    header = _header(
        "requirements.lock.txt - EXACT DEPENDENCY CLOSURE OF THE VERIFIED ENVIRONMENT (INF-01)",
        [
            "# Method : transitive Requires-Dist walk over the INSTALLED, contract-verified",
            "#          distributions (offline, deterministic).  Not a fresh PyPI resolution: the",
            "#          closure recorded here is the one that actually produced the verified numbers.",
            "# Install: D:\\Anaconda3\\python.exe -m pip install --no-deps -r requirements.lock.txt",
            "# Digests: see requirements.lock.hashes.txt (pip install --require-hashes)",
        ],
    )
    lines = [*header, "", *_version_lock_lines(closure, provenance)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hash_lock(closure: dict[str, str], output: Path) -> tuple[int, list[str]]:
    """Write the digest lock; return ``(pinned_count, names_without_compatible_wheel)``."""
    body: list[str] = ["--require-hashes", ""]
    pinned = 0
    no_wheel: list[str] = []
    provenance: dict[str, str] = {}
    for name, version in sorted(closure.items()):
        digests = _pypi_digests(name, version)
        if not digests or not _has_compatible_wheel(digests):
            # No wheel for this interpreter: it is NOT emitted into the --require-hashes body,
            # because pip would then try to build it from source during a hermetic install.
            no_wheel.append(f"{name}=={version}")
            provenance[name] = "no wheel for this interpreter; provided by conda (see requirements.lock.txt)"
            continue
        body.append(f"{name}=={version} \\")
        digest_lines = [f"    --hash=sha256:{digest}" for digest in sorted(digests.values())]
        body.extend(digest_lines)
        pinned += 1
    if provenance:
        body += ["", "# --- distributions above WITHOUT a compatible PyPI wheel on this interpreter ---"]
        body += [f"# {name}: {detail}" for name, detail in sorted(provenance.items())]
    header = _header(
        "requirements.lock.hashes.txt - DIGEST-PINNED CLOSURE (INF-01)",
        [
            "# Digests are the sha256 values published by the PyPI JSON API for that release.",
            "# Install: D:\\Anaconda3\\python.exe -m pip install --require-hashes \\",
            "#              -r requirements.lock.hashes.txt",
        ],
    )
    output.write_text("\n".join([*header, *body]) + "\n", encoding="utf-8")
    return pinned, no_wheel


def main(argv: list[str] | None = None) -> int:
    """Generate the lock artefacts and report exactly what was and was not covered."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashes", action="store_true", help="also fetch sha256 digests from PyPI")
    args = parser.parse_args(argv)

    missing = [path.name for path in REQ_FILES if not path.exists()]
    if missing:
        raise SystemExit(f"missing input requirement file(s): {', '.join(missing)}")

    direct = _direct_requirements(REQ_FILES)
    closure = _closure(direct)
    conda_records = _conda_provenance(closure)
    provenance = _provenance_notes(closure, conda_records)

    _write_version_lock(closure, provenance, VERSION_LOCK)
    print(f"wrote {VERSION_LOCK.name}: {len(closure)} distributions in the closure")
    for name, detail in sorted(provenance.items()):
        print(f"  provenance note: {name} -> {detail}")

    if not args.hashes:
        print("digest lock skipped (pass --hashes to fetch sha256 digests from PyPI)")
        return 0

    pinned, no_wheel = _write_hash_lock(closure, HASH_LOCK)
    print(f"wrote {HASH_LOCK.name}: {pinned} distributions digest-pinned")
    if no_wheel:
        print("  without a compatible PyPI wheel (recorded as comments, NOT hash-verified):")
        for entry in no_wheel:
            print(f"    - {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
