"""Package version resolution (task ``INF-02``).

The version is read from the installed distribution metadata so that a single source of truth
(``pyproject.toml``) drives packaging, provenance blocks and experiment tags.  A static
fallback keeps the module importable from a source checkout that has not been installed.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata

_FALLBACK_VERSION = "0.1.0"
_DISTRIBUTION = "qresearch"


def _resolve_version() -> str:
    """Return the installed distribution version, or the static fallback."""
    try:
        return importlib_metadata.version(_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        return _FALLBACK_VERSION


__version__: str = _resolve_version()

__all__ = ["__version__"]
