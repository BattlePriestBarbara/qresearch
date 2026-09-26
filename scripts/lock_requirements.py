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
``requirements.lock.pip.txt``
    The same closure MINUS the distributions that conda manages (see ADR-003).  This is the file
    referenced by the ``pip:`` block of ``environment.yml``.
``requirements.lock.hashes.txt``
    The pip-managed set with ``--hash=sha256:...`` digests, usable as
    ``pip install --require-hashes -r requirements.lock.hashes.txt``.

Hybrid environment strategy (ADR-003)
-------------------------------------
The closure is classified by wheel availability for the running interpreter:

* **wheel** - a compatible wheel exists; pip manages it, digests are hash-pinned.
* **sdist_pure_python** - the release ships no wheel at all (pure-Python project); pip builds it
  without any compiler, so it stays pip-managed with the sdist digest pinned.
* **conda** - wheels exist but none compatible with this interpreter (native extensions built
  before CPython 3.12); conda manages these and ``environment.yml`` pins them.

Usage
-----
.. code-block:: powershell

    python scripts/lock_requirements.py            # version lock (offline)
    python scripts/lock_requirements.py --hashes   # + pip & digest locks (network)
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType

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
                "run `python scripts/check_env.py` to diagnose the environment"
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


def _interpreter_wheel_tag() -> str:
    """Return the CPython wheel tag of the running interpreter, e.g. ``cp312``."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _platform_tags() -> frozenset[str]:
    """Return the platform tags a wheel may carry to be installable on this machine."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        arch = "win_amd64" if machine in {"amd64", "x86_64"} else "win32"
        return frozenset({"any", arch})
    if system == "Darwin":
        return frozenset({"any", f"macosx_10_12_{'arm64' if machine in {'arm64', 'aarch64'} else 'x86_64'}"})
    return frozenset({"any", "manylinux2014_x86_64", "manylinux_2_17_x86_64", "linux_x86_64"})


def _wheel_is_compatible(filename: str) -> bool:
    """Report whether a wheel file is installable on the running interpreter and platform.

    Two wheel flavours are accepted:

    * **version-specific** wheels whose python tag equals this interpreter (``cp312``);
    * **stable-ABI** wheels tagged ``cp3X-abi3`` with ``X <= <this minor version>``, which are
      forward compatible by design.  Ignoring this class was a real defect: it wrongly excluded
      ``clarabel``, ``cryptography``, ``tornado`` and ``argon2-cffi-bindings``, all of which ship
      usable ``cp3X-abi3`` wheels.
    * **pure-python** wheels (``py3``/``py2.py3``).
    """
    if not filename.endswith(".whl"):
        return False
    parts = filename[: -len(".whl")].rsplit("-", 3)
    if len(parts) != 4:
        return False
    _name_version, python_tag, abi_tag, platform_tag = parts
    if platform_tag not in _platform_tags():
        return False
    expected = _interpreter_wheel_tag()
    for tag in python_tag.split("."):
        if tag in {"py3", "py2"}:
            return True
        if tag == expected:
            return True
        stable_abi = re.fullmatch(r"cp3(\d+)", tag)
        if stable_abi and abi_tag.startswith("abi3") and int(stable_abi.group(1)) <= sys.version_info.minor:
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
    """Return the provenance header shared by both lock artefacts.

    Only the interpreter's *name* is recorded, never its location (``ADR-005``): a lock file is a
    tracked file, so a machine path in it would both leak the developer's layout and reappear on every
    regeneration, which would make the pre-publish path gate impossible to keep green.
    """
    header = [
        "# =====================================================================================",
        f"# {title}",
        "#",
        "# GENERATED FILE - do not edit by hand. Regenerate with:",
        "#     python scripts/lock_requirements.py [--hashes]",
        "#",
        f"# Generated  : {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# Python     : {platform.python_version()} on {platform.system()} {platform.machine()}",
        f"# Interpreter: {Path(sys.executable).name}",
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
            "# Install: python -m pip install --no-deps -r requirements.lock.txt",
            "# Digests: see requirements.lock.hashes.txt (pip install --require-hashes)",
        ],
    )
    lines = [*header, "", *_version_lock_lines(closure, provenance)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Hybrid environment strategy (see docs/adr/ADR-003)
#
# Every distribution in the closure is classified as one of:
#   * "wheel"              - a wheel compatible with this interpreter/platform exists on PyPI;
#                            pip manages it and its digests are hash-pinned.
#   * "sdist_pure_python"  - the release ships NO wheel at all, which is the signature of a
#                            pure-Python project (e.g. gym, pulled in by pyqlib).  pip can build
#                            it without any compiler, so it stays pip-managed and its sdist
#                            digests are hash-pinned.
#   * "conda"              - the release ships wheels, but none compatible with this interpreter
#                            (e.g. psutil 5.9.0, pywin32, pywinpty: CPython extensions built
#                            before 3.12).  Compiling these with pip would be fragile, so conda
#                            manages them and they are listed in environment.yml.
# ---------------------------------------------------------------------------------------
CATEGORY_WHEEL = "wheel"
CATEGORY_SDIST_PURE_PYTHON = "sdist_pure_python"
CATEGORY_CONDA = "conda"
PIP_LOCK = REPO_ROOT / "requirements.lock.pip.txt"


@dataclass(frozen=True)
class ClassifiedDistribution:
    """One closure member together with its management category and PyPI digests."""

    name: str
    version: str
    category: str
    digests: Mapping[str, str] = field(default_factory=dict)

    @property
    def pinned_requirement(self) -> str:
        """Return ``name==version``."""
        return f"{self.name}=={self.version}"

    @property
    def pip_managed(self) -> bool:
        """Return whether pip (not conda) is responsible for this distribution."""
        return self.category != CATEGORY_CONDA

    @property
    def needs_build(self) -> bool:
        """Return whether pip would have to build this distribution from an sdist."""
        return self.category == CATEGORY_SDIST_PURE_PYTHON


def _classify(name: str, version: str) -> ClassifiedDistribution:
    """Classify one closure member against PyPI wheel availability."""
    digests = _pypi_digests(name, version)
    if not digests:
        return ClassifiedDistribution(name, version, CATEGORY_CONDA)
    if any(_wheel_is_compatible(filename) for filename in digests):
        return ClassifiedDistribution(name, version, CATEGORY_WHEEL, MappingProxyType(digests))
    if not any(filename.endswith(".whl") for filename in digests):
        return ClassifiedDistribution(name, version, CATEGORY_SDIST_PURE_PYTHON, MappingProxyType(digests))
    return ClassifiedDistribution(name, version, CATEGORY_CONDA)


def _classify_closure(closure: dict[str, str]) -> dict[str, ClassifiedDistribution]:
    """Classify every distribution in the closure."""
    return {name: _classify(name, version) for name, version in sorted(closure.items())}


def _split_by_category(
    classified: Mapping[str, ClassifiedDistribution],
) -> tuple[list[ClassifiedDistribution], list[ClassifiedDistribution], list[ClassifiedDistribution]]:
    """Return ``(pip_wheel, pip_sdist, conda)`` groupings, each sorted by name."""
    pip_wheel = [item for item in classified.values() if item.category == CATEGORY_WHEEL]
    pip_sdist = [item for item in classified.values() if item.category == CATEGORY_SDIST_PURE_PYTHON]
    conda = [item for item in classified.values() if item.category == CATEGORY_CONDA]
    return (
        sorted(pip_wheel, key=lambda i: i.name),
        sorted(pip_sdist, key=lambda i: i.name),
        sorted(conda, key=lambda i: i.name),
    )


def _write_pip_lock(classified: Mapping[str, ClassifiedDistribution], output: Path) -> list[str]:
    """Write ``requirements.lock.pip.txt``; return the names delegated to conda."""
    pip_managed = sorted(item.pinned_requirement for item in classified.values() if item.pip_managed)
    conda_managed = sorted(item.pinned_requirement for item in classified.values() if not item.pip_managed)
    header = _header(
        "requirements.lock.pip.txt - PIP-MANAGED CLOSURE (hybrid strategy, ADR-003)",
        [
            "# Contents: the dependency closure MINUS the distributions that conda manages.",
            "# Conda-managed (pinned in environment.yml instead):",
            *([f"#   - {requirement}" for requirement in conda_managed] or ["#   (none)"]),
            "#   Reason: those releases ship no wheel compatible with this interpreter, and building",
            "#   native extensions with pip is fragile (ADR-003).",
            f"# Pip-managed distributions: {len(pip_managed)}",
            "# Install: referenced from environment.yml; hermetically via",
            "#          pip install --require-hashes -r requirements.lock.hashes.txt",
        ],
    )
    output.write_text("\n".join([*header, "", *pip_managed]) + "\n", encoding="utf-8")
    return conda_managed


def _write_hash_lock(
    classified: Mapping[str, ClassifiedDistribution], output: Path
) -> tuple[int, list[str], list[str]]:
    """Write ``requirements.lock.hashes.txt``; return ``(pinned, sdist_only, conda_managed)``."""
    pip_wheel, pip_sdist, conda = _split_by_category(classified)
    body: list[str] = ["--require-hashes", ""]
    for item in [*pip_wheel, *pip_sdist]:
        body.append(f"{item.pinned_requirement} \\")
        body.extend(f"    --hash=sha256:{digest}" for digest in sorted(item.digests.values()))
    notes: list[str] = []
    if pip_sdist:
        notes += ["", "# --- sdist-only, pure Python: pip builds these WITHOUT a compiler ---"]
        notes += [f"# {item.pinned_requirement}" for item in pip_sdist]
    if conda:
        notes += ["", "# --- NOT hash-pinned here: managed by conda (see environment.yml) ---"]
        notes += [f"# {item.pinned_requirement}" for item in conda]
    header = _header(
        "requirements.lock.hashes.txt - DIGEST-PINNED PIP CLOSURE (INF-01, ADR-003)",
        [
            "# Digests are the sha256 values published by the PyPI JSON API for that release.",
            "# Install: pip install --require-hashes -r requirements.lock.hashes.txt",
        ],
    )
    output.write_text("\n".join([*header, *body, *notes]) + "\n", encoding="utf-8")
    return (
        len(pip_wheel) + len(pip_sdist),
        [item.pinned_requirement for item in pip_sdist],
        [item.name for item in conda],
    )


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

    if not args.hashes:
        print("classification skipped (pass --hashes to query PyPI and write the pip + digest locks)")
        return 0

    classified = _classify_closure(closure)
    conda_in_pip_lock = _write_pip_lock(classified, PIP_LOCK)
    pinned, sdist_only, conda_names = _write_hash_lock(classified, HASH_LOCK)

    print(f"wrote {PIP_LOCK.name}: {len(classified) - len(conda_names)} pip-managed distributions")
    print(f"wrote {HASH_LOCK.name}: {pinned} distributions digest-pinned")
    if sdist_only:
        print("  sdist-only (pure Python, pip builds them without a compiler):")
        for entry in sdist_only:
            print(f"    - {entry}")
    if conda_names:
        print("  conda-managed (pinned in environment.yml, deliberately NOT hash-pinned by pip):")
        for entry in conda_names:
            print(f"    - {entry}")
    assert len(conda_in_pip_lock) == len(conda_names), "pip lock and digest lock disagree on the conda set"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
