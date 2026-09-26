"""Global determinism controls (task ``INF-09``, ``PROJECT_SPEC.md`` 5.3, 3.8).

``3.8`` and ``5.3`` make reproducibility a property of the *process*, not of one estimator: a single
seed must drive ``random``, ``numpy`` and ``torch``, and the seeded state must be recorded so that a
run can be reconstructed.  This module does exactly that and returns the state it established.

``PYTHONHASHSEED`` is reported but cannot be set from here: it must be present in the environment
*before* the interpreter starts, so :func:`seed_everything` documents the requirement instead of
pretending to satisfy it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

__all__ = ["PYTHONHASHSEED_ENV_VAR", "SeedState", "numpy_generator", "seed_everything"]

PYTHONHASHSEED_ENV_VAR: Final[str] = "PYTHONHASHSEED"


@dataclass(frozen=True)
class SeedState:
    """The determinism state established by :func:`seed_everything`, ready for the provenance block."""

    seed: int
    torch_seeded: bool
    torch_deterministic: bool
    details: MappingProxyType[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "seed": self.seed,
            "torch_seeded": self.torch_seeded,
            "torch_deterministic": self.torch_deterministic,
            "details": dict(self.details),
        }


def seed_everything(seed: int, *, torch_deterministic: bool = True) -> SeedState:
    """Seed ``random``, the legacy ``numpy`` global state and ``torch``; return the state.

    Parameters
    ----------
    seed : int
        Non-negative seed.  The same value MUST reproduce the same panel, the same split and the
        same network initialization.
    torch_deterministic : bool
        When ``True`` (default), request cuDNN determinism and disable its benchmark autotuner.  The
        autotuner picks a kernel by timing, which is a source of run-to-run variation.

    Returns
    -------
    SeedState
        What was actually seeded, including whether ``torch`` was importable.

    Notes
    -----
    The global ``numpy`` state is seeded (not only a ``Generator``) because Qlib and third-party
    code call ``np.random.*`` directly; :func:`numpy_generator` is provided for new code, which
    SHOULD carry an explicit generator instead.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    hash_seed = os.environ.get(PYTHONHASHSEED_ENV_VAR, "unset (set it before launching for full determinism)")
    details: dict[str, str] = {
        "python_random": "seeded",
        "numpy_legacy_global": "seeded",
        PYTHONHASHSEED_ENV_VAR: hash_seed,
    }
    try:
        # Lazy on purpose: torch is a heavy import and the seeding helper must stay usable (and fast to
        # import) in a process that only needs `random` and `numpy` determinism.
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - torch is part of the pinned closure
        details["torch"] = "not installed"
        return SeedState(seed, torch_seeded=False, torch_deterministic=False, details=MappingProxyType(details))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_deterministic_applied = torch_deterministic
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    details["torch"] = f"manual_seed({seed})"
    details["torch_cuda_available"] = str(torch.cuda.is_available())
    return SeedState(
        seed=seed,
        torch_seeded=True,
        torch_deterministic=torch_deterministic_applied,
        details=MappingProxyType(details),
    )


def numpy_generator(seed: int) -> np.random.Generator:
    """Return an explicit ``numpy`` generator; preferred over the legacy global state."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return np.random.default_rng(seed)
