"""CLI wrapper for the INF-01 environment check (``python scripts/check_env.py``).

The implementation lives in :mod:`qresearch.env` so that it is importable, testable and
reusable from any entry point.  This script only adapts command-line invocation and adds the
src-layout bootstrap needed when the package has not been installed yet.

Note that importing :mod:`qresearch` performs the environment verification itself, so a
contract violation surfaces here as an :class:`EnvironmentContractError` raised *during the
import*, before any Qlib initialisation can happen.

Exit codes
----------
0 - contract satisfied
1 - environment contract violated (wrong interpreter / python / distribution version)
2 - read-only provider violation (missing provider, cache inside provider, provider in repo)
3 - cache-like pollution detected inside the read-only provider
4 - the qresearch package cannot be imported at all (not installed and src/ not usable)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - bootstrap for uninstalled checkouts
    sys.path.insert(0, str(_SRC))


def main() -> int:
    """Run the environment verification and map its outcome to an exit code.

    Importing :mod:`qresearch` performs the verification, so a contract violation arrives here
    as an exception raised *by the import statement itself*.  A ``RuntimeError`` is a contract
    violation (exit 1); any other exception type is a defect in this codebase and is re-raised
    so that it crashes loudly instead of being disguised as an environment problem.
    """
    try:
        from qresearch.env import main as env_main
    except ImportError as import_error:  # pragma: no cover - packaging failure
        print(f"FATAL: cannot import the qresearch package: {import_error!r}", file=sys.stderr)
        print("Hint: python -m pip install -e . --no-deps", file=sys.stderr)
        return 4
    except Exception as error:  # narrowed to RuntimeError immediately below
        if isinstance(error, RuntimeError):
            print(str(error), file=sys.stderr)
            return 1
        raise
    return env_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
