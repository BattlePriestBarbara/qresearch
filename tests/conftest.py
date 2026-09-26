"""Shared fixtures for the qresearch test suite (tasks ``INF-01``, ``INF-02``, ``INF-11``).

The filesystem fixtures are deliberately light and import nothing from Qlib: the INF-phase tests must
run in milliseconds and must not depend on the market-data store being present.  The one heavy fixture
is :func:`known_signal_panel`, the synthetic ground-truth panel that ``PROJECT_SPEC.md`` 3.8 requires
and that every later statistical test reuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qresearch.data.synthetic import SyntheticPanel, SyntheticPanelSpec, generate_panel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the project root, derived from this file rather than declared."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_panel_spec() -> SyntheticPanelSpec:
    """The DGP knobs the statistical tests share: ``N = 100``, ``T = 1000``, injected IC ``0.05``."""
    return SyntheticPanelSpec()


@pytest.fixture(scope="session")
def known_signal_panel(synthetic_panel_spec: SyntheticPanelSpec) -> SyntheticPanel:
    """The synthetic panel of ``PROJECT_SPEC.md`` 3.8 / task ``INF-11``.

    ``T = 1000`` decisions on ``N = 100`` instruments with a known population correlation of ``0.05``,
    Student-:math:`t` noise (fat tails) and persistent features *and* noise (autocorrelated IC series).
    Ground truth - including the Monte-Carlo standard error of the expected IC - travels with the panel
    in :attr:`SyntheticPanel.truth`, so a test never has to hard-code a tolerance.
    """
    return generate_panel(synthetic_panel_spec)


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
