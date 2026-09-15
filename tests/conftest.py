"""Shared fixtures for the qresearch test suite (tasks ``INF-01``, ``INF-02``).

Fixtures here are deliberately filesystem-only and import nothing from Qlib: the INF phase
tests must run in milliseconds and must not depend on the market-data store being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the project root (``D:\\Qlib``)."""
    return REPO_ROOT


@pytest.fixture()
def fake_provider(tmp_path: Path) -> Path:
    """Create a minimal read-only-provider look-alike outside the repository.

    The structure mirrors a Qlib binary store (``calendars``, ``features``, ``instruments``)
    so that fingerprinting and pollution detection are exercised on a realistic tree.
    """
    provider = tmp_path / "cn_data"
    for name in ("calendars", "features", "instruments"):
        directory = provider / name
        directory.mkdir(parents=True)
        (directory / f"{name}.bin").write_bytes(b"qresearch-provider-fixture")
    return provider


@pytest.fixture()
def fake_cache(tmp_path: Path) -> Path:
    """Create a writable cache root that is guaranteed to sit outside the provider."""
    cache = tmp_path / "artifacts" / "qlib_cache"
    cache.mkdir(parents=True)
    return cache
