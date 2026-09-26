"""Configuration and path errors (``ADR-005``, task ``INF-10``).

One exception type governs every configuration defect, so that callers (and tests) can catch a
single, meaningful class instead of guessing which of a dozen ``ValueError``/``KeyError`` sites a
malformed configuration will hit.

:class:`ConfigError` inherits from :class:`RuntimeError` on purpose: a configuration defect is not
a recoverable data condition, and a ``RuntimeError`` is not swallowed by the ``except Exception``
handlers written for data problems elsewhere in the codebase.

Hierarchy
---------
``RuntimeError``
  ``ConfigError``
    ``qresearch.env.EnvironmentContractError``  - pinned runtime contract violated
    ``qresearch.env.ReadOnlyViolationError``    - read-only market data would be written
"""

from __future__ import annotations

__all__ = ["ConfigError"]


class ConfigError(RuntimeError):
    """Raised when configuration, an environment override or a path is invalid.

    Every raise site MUST state the offending value, the rule that rejects it and the remediation,
    because a configuration error is only useful if the operator can fix it without reading the
    source (see ``ADR-005`` for the resolution policy).
    """
