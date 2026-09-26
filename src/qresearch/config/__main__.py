"""Module entry point: ``python -m qresearch.config`` (task ``INF-10``).

The console script ``qresearch-config`` (declared in ``pyproject.toml``) calls the same function.
A ``__main__`` module is used instead of putting the guard inside ``loader.py`` so that running the
module does not load ``qresearch.config.loader`` twice (once through the package ``__init__`` and once
as ``__main__``), which ``runpy`` reports as a warning.
"""

from __future__ import annotations

from .loader import main

if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
