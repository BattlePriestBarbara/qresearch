"""Content-addressed artifact IO and hashing (task ``INF-09``, ``PROJECT_SPEC.md`` 3.7).

``3.7`` requires every run to open a recorder context and to log a configuration hash, a data
snapshot hash and the git SHA next to the artifacts it produces; ``3.2`` requires those artifacts to
be *content-addressable*.  This module provides the primitives for both, and nothing else:

* :func:`sha256_bytes` / :func:`sha256_file` / :func:`hash_mapping` - the digests;
* :func:`canonical_json` - one byte-identical serialization for one logical document, so a hash is
  reproducible across platforms (sorted keys, fixed separators, UTF-8);
* :func:`write_json` / :func:`read_json` / :func:`write_yaml` / :func:`read_yaml` - artifact IO;
* :func:`artifact_dir` / :func:`content_addressed_path` - the layout of ``3.7``.

Root resolution is **lazy**: the functions here never import
:mod:`qresearch.config.paths` at module load, and every entry point accepts an explicit ``root``, so
``qresearch.utils`` stays importable and side-effect free (no directory is created until a caller
asks for one).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "ARTIFACT_KINDS",
    "HASH_CHUNK_BYTES",
    "artifact_dir",
    "canonical_json",
    "content_addressed_path",
    "hash_mapping",
    "read_json",
    "read_yaml",
    "sha256_bytes",
    "sha256_file",
    "write_json",
    "write_yaml",
]

HASH_CHUNK_BYTES: Final[int] = 1024 * 1024
"""Read size for :func:`sha256_file`: one MiB keeps memory flat on multi-gigabyte artifacts."""

HASH_PREFIX_CHARS: Final[int] = 12
"""Characters of the digest kept in a content-addressed filename (48 bits, collision-free here)."""

ARTIFACT_KINDS: Final[tuple[str, ...]] = ("mlruns", "predictions", "models", "reports", "figures", "tables")
"""The artifact directory names fixed by ``3.7``; an unknown kind is a typo, not a new category."""


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the file at ``path``, streamed in :data:`HASH_CHUNK_BYTES` blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: object) -> str:
    """Return a deterministic JSON rendering of ``payload``.

    Sorted keys and compact separators make the rendering a pure function of the *document* rather
    than of the insertion order of a ``dict``, which is what allows ``config_hash`` to be compared
    between runs, machines and languages.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_mapping(payload: Mapping[str, Any] | object) -> str:
    """Return the SHA-256 digest of the canonical JSON rendering of ``payload``."""
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def write_json(path: Path, payload: object) -> Path:
    """Write ``payload`` as UTF-8 JSON with a trailing newline; create parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    target.write_text(serialized + "\n", encoding="utf-8")
    return target


def read_json(path: Path) -> object:
    """Read a JSON document; raise :class:`ValueError` when the file does not contain one."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: object) -> Path:
    """Write ``payload`` as UTF-8 YAML, preserving key order (it is the resolved configuration order)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    target.write_text(text, encoding="utf-8")
    return target


def read_yaml(path: Path) -> object:
    """Read a YAML document with :func:`yaml.safe_load` (no arbitrary object construction)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def artifact_dir(kind: str, run_id: str | None = None, *, root: Path | None = None) -> Path:
    """Return (creating it) the artifact directory of ``3.7`` for ``kind`` and ``run_id``.

    Parameters
    ----------
    kind : str
        One of :data:`ARTIFACT_KINDS`; anything else is rejected, because a free-form name would let
        a run scatter artifacts outside the documented layout.
    run_id : str | None
        Recorder id; ``None`` returns the category root itself (``artifacts/reports/``).
    root : Path | None
        Artifact root; ``None`` resolves :func:`qresearch.config.paths.artifact_root` at call time.

    Returns
    -------
    Path
        The existing directory.
    """
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; allowed: {', '.join(ARTIFACT_KINDS)}")
    if root is None:
        # Lazy on purpose: `qresearch.config.loader` imports this module at module level, so a
        # top-level import here would close the cycle config -> utils -> config.  It also keeps
        # `import qresearch.utils` free of filesystem or environment access.
        from ..config.paths import artifact_root  # pylint: disable=import-outside-toplevel

        base = artifact_root()
    else:
        base = Path(root)
    target = base / kind
    if run_id is not None:
        if not run_id:
            raise ValueError("run_id must be a non-empty string when it is given")
        target = target / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def content_addressed_path(root: Path, stem: str, payload: object, *, suffix: str = ".json") -> Path:
    """Return ``root/<stem>-<digest><suffix>``: the path that *is* the content of ``payload``.

    Two runs that produce the same document therefore produce the same filename, and a changed
    document cannot silently overwrite the previous artifact - which is the property ``3.2`` asks
    for ("artifacts/ MUST be content-addressable").
    """
    if not stem:
        raise ValueError("stem must be a non-empty string")
    digest = hash_mapping(payload)[:HASH_PREFIX_CHARS]
    return Path(root) / f"{stem}-{digest}{suffix}"
