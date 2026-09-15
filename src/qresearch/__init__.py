"""qresearch - temporal-neural Alpha factor mining on Qlib under strict statistical inference.

This package implements the system specified by ``PROJECT_SPEC.md`` (project root ``D:\\Qlib``).
The governing specification is normative: interfaces, formulas and naming conventions defined
there MUST NOT be changed in code without an Architecture Decision Record.

Import-time environment enforcement (task ``INF-01``)
----------------------------------------------------
Importing this package immediately verifies the pinned runtime contract: CPython ``3.12.7`` on
``D:\\Anaconda3\\python.exe`` with exact patch versions of every float-critical distribution.
The verification runs *before* any ``qlib.init`` call in user code, because a version drift
changes floating-point results and therefore invalidates statistical inference
(``PROJECT_SPEC.md`` 1.2.2, 3.1).

A mismatch raises :class:`~qresearch.env.EnvironmentContractError` (a :class:`RuntimeError`)
and terminates the process.  There is deliberately no bypass flag: silently continuing on the
wrong binaries is the failure mode this check exists to prevent.

Examples
--------
.. code-block:: python

    import qlib
    import qresearch
    from qresearch.env import ProviderWriteGuard, build_qlib_init_kwargs

    provider_uri = qresearch.PROVIDER_URI
    with ProviderWriteGuard(provider_uri):
        qlib.init(**build_qlib_init_kwargs(provider_uri))
"""

from __future__ import annotations

from .env import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_PROVIDER_URI,
    EXPECTED_INTERPRETER,
    PYTHON_VERSION_REQUIRED,
    EnvironmentContractError,
    EnvironmentReport,
    verify_environment,
)
from .version import __version__

# --- Import-time enforcement of the pinned contract (task INF-01). Never bypassed. -------
ENVIRONMENT_REPORT: EnvironmentReport = verify_environment()

# Convenience aliases for the canonical paths of this project.
PROVIDER_URI = DEFAULT_PROVIDER_URI
CACHE_ROOT = DEFAULT_CACHE_ROOT

__all__ = [
    "CACHE_ROOT",
    "ENVIRONMENT_REPORT",
    "EXPECTED_INTERPRETER",
    "PROVIDER_URI",
    "PYTHON_VERSION_REQUIRED",
    "EnvironmentContractError",
    "EnvironmentReport",
    "__version__",
    "verify_environment",
]
