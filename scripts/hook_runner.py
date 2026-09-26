"""Portable pre-commit hook runner (``ADR-005``, task ``INF-02``).

Why this file exists
--------------------
A ``language: system`` hook executes its ``entry`` directly, so two failure modes were possible before
this runner existed:

* spelling the interpreter out (``<drive>:/.../python.exe -m mypy``) makes the gate work on exactly one
  machine, which is the defect ``ADR-005`` removes; and
* a bare ``python -m mypy`` resolves through ``PATH``, which on this machine finds an MSYS2 build
  without the quant stack - the failure ``ADR-001`` exists to prevent.

The hook entries therefore call this script with the *bare* interpreter (it needs only the standard
library) and this script re-executes the real tool with the pinned interpreter, resolving

    ``QRESEARCH_PYTHON``  ->  ``%CONDA_PREFIX%/python[.exe]``  ->  the running interpreter

in that order.  The resolution is a deliberate mirror of
:func:`qresearch.config.paths.candidate_interpreters` instead of an import, because importing
``qresearch`` would run the environment verification itself and abort before any tool could run;
``tests/test_hook_runner.py`` asserts that the two implementations agree on the same input, so the
duplication cannot drift silently.

If no usable interpreter is found the runner exits non-zero and does **not** substitute another one:
``ADR-001`` requires every hook to run inside the pinned environment, and a silent fallback would
weaken the gate to the point where a green run means nothing.

Exit codes
----------
0 - the tool succeeded
1 - the tool failed (its own exit status is propagated)
2 - no usable interpreter was found, or the arguments were malformed

Usage
-----
.. code-block:: text

    python scripts/hook_runner.py -m black --check --diff
    python scripts/hook_runner.py -m mypy --config-file=.mypy.ini
    python scripts/hook_runner.py --show-interpreter
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

INTERPRETER_ENV_VAR: Final[str] = "QRESEARCH_PYTHON"
CONDA_PREFIX_ENV_VAR: Final[str] = "CONDA_PREFIX"
CONDA_PYTHON_RELATIVE: Final[tuple[str, ...]] = ("python.exe",) if os.name == "nt" else ("bin", "python")
"""Path of the interpreter inside an activated conda prefix; mirrors qresearch.config.paths."""

EXIT_TOOL_FAILED: Final[int] = 1
EXIT_NO_INTERPRETER: Final[int] = 2

SHOW_INTERPRETER_FLAG: Final[str] = "--show-interpreter"


def candidate_interpreters(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return the interpreter candidates in priority order (pure; mirrors the library version)."""
    candidates: list[str] = []
    explicit = (environ.get(INTERPRETER_ENV_VAR) or "").strip()
    if explicit:
        candidates.append(explicit)
    prefix = (environ.get(CONDA_PREFIX_ENV_VAR) or "").strip()
    if prefix:
        candidates.append(str(Path(prefix).joinpath(*CONDA_PYTHON_RELATIVE)))
    running = sys.executable
    if running not in candidates:
        candidates.append(running)
    return tuple(candidates)


def resolve_interpreter(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the interpreter the hooks must run with, or ``None`` when none is usable.

    An explicit ``QRESEARCH_PYTHON`` is authoritative: when it is set but does not exist, the runner
    fails instead of silently using another interpreter, because running the gate on the wrong
    binaries is the outcome ``ADR-001`` exists to prevent.
    """
    lookup = os.environ if environ is None else environ
    explicit = (lookup.get(INTERPRETER_ENV_VAR) or "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for candidate in candidate_interpreters(lookup):
        if Path(candidate).is_file():
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested tool with the pinned interpreter and propagate its exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            f"usage: {Path(__file__).name} [-m MODULE | TOOL] [args...]  |  {SHOW_INTERPRETER_FLAG}",
            file=sys.stderr,
        )
        return EXIT_NO_INTERPRETER

    interpreter = resolve_interpreter()
    if interpreter is None:
        print(
            "FATAL: no usable interpreter found for the hook.\n"
            f"  Looked for {INTERPRETER_ENV_VAR}, then {CONDA_PREFIX_ENV_VAR}, then {sys.executable}.\n"
            "  Remediation: activate the pinned environment (see environment.yml / ADR-003) or set "
            f"{INTERPRETER_ENV_VAR} to its interpreter.",
            file=sys.stderr,
        )
        return EXIT_NO_INTERPRETER

    if arguments[0] == SHOW_INTERPRETER_FLAG:
        print(interpreter)
        return 0

    try:
        completed = subprocess.run([interpreter, *arguments], check=False)
    except OSError as error:  # pragma: no cover - depends on the machine
        print(f"FATAL: cannot launch {interpreter}: {error}", file=sys.stderr)
        return EXIT_NO_INTERPRETER
    if completed.returncode != 0:
        return completed.returncode or EXIT_TOOL_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
