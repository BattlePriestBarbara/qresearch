"""Logging bound to the Qlib logger (task ``INF-09``, ``PROJECT_SPEC.md`` 5.4).

``5.4`` makes two rules normative: ``print()`` is forbidden in ``src/``, and logging delegates to the
Qlib logger so that a run's output is captured by the same machinery Qlib already configures (and
therefore reaches the recorder).  :func:`get_logger` implements the delegation; console entry points
in ``scripts/`` are the only place allowed to ``print``.

The Qlib import is lazy: importing :mod:`qresearch.utils` must not drag the whole framework in, and
must keep working in an interpreter where Qlib is absent.  Qlib's own ``get_module_logger`` nests the
requested name under its ``qlib`` root (``get_module_logger("qresearch.stats")`` becomes
``qlib.qresearch.stats``), which is the intended behaviour: one logger tree, one configuration.
"""

from __future__ import annotations

import logging
from typing import Final

__all__ = ["LOGGER_NAMESPACE", "configure_logging", "get_logger", "log_applied_overrides"]

LOGGER_NAMESPACE: Final[str] = "qresearch"


def get_logger(name: str = LOGGER_NAMESPACE) -> logging.Logger:
    """Return a logger under the ``qresearch`` namespace, delegating to Qlib's logger if available.

    Parameters
    ----------
    name : str
        Module or component name; ``"qresearch"`` itself and already-qualified names are used as
        given, anything else is nested under the namespace.

    Returns
    -------
    logging.Logger
        Qlib's configured logger for the name, or a stdlib logger with a :class:`logging.NullHandler`
        attached when Qlib is not importable (a library must not print on import).
    """
    qualified = _qualify(name)
    try:
        # Lazy on purpose: importing Qlib pulls in the whole framework, and `qresearch.utils` must stay
        # importable (and cheap) in an interpreter where Qlib is absent.
        from qlib.log import get_module_logger  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - qlib is part of the pinned closure
        logger = logging.getLogger(qualified)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger
    # qlib ships no PEP 561 marker, so its return value is `Any`; the local annotation keeps mypy's
    # `warn_return_any` satisfied without weakening the declared return type.
    qlib_logger: logging.Logger = get_module_logger(qualified)
    return qlib_logger


def _qualify(name: str) -> str:
    """Return ``name`` nested under :data:`LOGGER_NAMESPACE` unless it already is."""
    if not name:
        return LOGGER_NAMESPACE
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return name
    return f"{LOGGER_NAMESPACE}.{name}"


def configure_logging(level: int | str = logging.INFO, name: str = LOGGER_NAMESPACE) -> logging.Logger:
    """Set the level of the ``qresearch`` logger; intended for the ``scripts/`` entry points."""
    logger = get_logger(name)
    logger.setLevel(level)
    return logger


def log_applied_overrides(logger: logging.Logger | None = None) -> str:
    """Log every honoured environment override (``PROJECT_SPEC.md`` 3.5: overrides MUST be logged).

    Returns
    -------
    str
        The rendered description, so a caller can also embed it in an artifact.
    """
    # Lazy on purpose: same cycle as `qresearch.utils.io`, and the description is only needed when a
    # caller actually wants to log it.
    from ..config.paths import describe_overrides  # pylint: disable=import-outside-toplevel

    description = describe_overrides()
    (logger or get_logger()).info(description)
    return description
