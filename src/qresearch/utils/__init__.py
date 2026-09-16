"""Cross-cutting utilities: logging, determinism, artifact IO and type aliases.

Implemented by tasks ``INF-09`` (logging, IO, seeding) and ``INF-02`` (type aliases).  Logging
delegates to the Qlib logger; ``print()`` is forbidden in ``src/`` (``PROJECT_SPEC.md`` 5.4).
"""

from __future__ import annotations
