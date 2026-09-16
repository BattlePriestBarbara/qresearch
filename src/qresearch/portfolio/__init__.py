"""Layer 5 - cost-aware allocation (``PROJECT_SPEC.md`` 2.4, 3.4.4).

Implemented by phase 3 tasks ``PO-01`` .. ``PO-05``.  Turnover is an interior term of the
optimization program, never a post-hoc diagnostic, and buy/sell cost asymmetry is preserved.
"""

from __future__ import annotations
