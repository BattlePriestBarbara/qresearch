"""Type aliases for the array and frame contracts (``PROJECT_SPEC.md`` 5.2).

Centralizing the aliases keeps the annotations in the data layer readable while still satisfying
``disallow_any_generics``: ``np.ndarray`` is generic over shape *and* dtype, so strict mypy requires
the parameter even when only the dtype is contractual.

The dtype of every alias used in the tensor bundle is part of the contract of
``PROJECT_SPEC.md`` 3.4.3 and is asserted by ``tests/test_handler_mask.py``; the permissive
:data:`Array` alias is for intermediate arrays whose dtype is not part of any interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# NOTE: the `TypeAlias` annotation is used instead of PEP 695's `type X = ...` because mypy 1.11
# (the pinned version) reports "PEP 695 type aliases are not yet supported". The equivalent modern
# syntax is therefore suppressed for this module in pyproject.toml (`UP040`), with this comment as
# the justification.

# --- arrays -------------------------------------------------------------------------------
Array: TypeAlias = NDArray[np.generic]
"""A numpy array of any dtype: for intermediate values whose dtype carries no contract."""

FloatArray: TypeAlias = NDArray[np.floating[Any]]
"""A floating-point array (``float32`` or ``float64``)."""

BoolArray: TypeAlias = NDArray[np.bool_]
"""A boolean mask array."""

IntArray: TypeAlias = NDArray[np.integer[Any]]
"""An integer array, e.g. positional fold assignments."""

# --- frames -------------------------------------------------------------------------------
ScoreFrame: TypeAlias = pd.Series
"""A model score or label series on ``MultiIndex(datetime, instrument)``."""

PanelFrame: TypeAlias = pd.DataFrame
"""A wide ``(date x instrument)`` panel with one column per instrument."""

# --- callables ----------------------------------------------------------------------------
ModelCallable: TypeAlias = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]
"""An audit model: ``(train_x, train_y, test_x) -> test_prediction``."""

__all__ = [
    "Array",
    "BoolArray",
    "FloatArray",
    "IntArray",
    "ModelCallable",
    "PanelFrame",
    "ScoreFrame",
]
