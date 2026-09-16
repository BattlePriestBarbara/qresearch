"""Typed configuration schema and loaders (``PROJECT_SPEC.md`` 3.5).

Implemented by task ``INF-10``.  Configuration is declarative YAML consumed through
``qlib.utils.init_instance_by_config``; library code MUST NOT hard-code study parameters.
"""

from __future__ import annotations
