"""Read-only provider guard tests for task ``INF-01``.

The market-data store is a read-only input.  These tests prove that the project can *detect*
a write into it, so that a silent cache inside ``D:\\qlib_data\\cn_data`` can never be mistaken
for a clean data snapshot.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from qresearch.env import (
    ProviderWriteGuard,
    ReadOnlyViolationError,
    _assert_no_path_under,
    assert_provider_isolation,
    build_qlib_init_kwargs,
    fingerprint_tree,
)


@pytest.mark.unit
def test_fingerprint_requires_existing_directory(tmp_path: Path) -> None:
    """A missing provider means no read-only guarantee can be established."""
    with pytest.raises(ReadOnlyViolationError, match="does not exist"):
        fingerprint_tree(tmp_path / "absent")


@pytest.mark.unit
def test_fingerprint_is_stable_and_complete(fake_provider: Path) -> None:
    """The fingerprint counts files and directories for a realistic provider tree."""
    fingerprint = fingerprint_tree(fake_provider)
    assert fingerprint.n_files == 3
    assert fingerprint.n_dirs == 3
    assert fingerprint.total_bytes > 0
    assert fingerprint == fingerprint_tree(fake_provider)


@pytest.mark.unit
def test_guard_passes_when_provider_is_untouched(fake_provider: Path) -> None:
    """Legitimate read-only work must not trip the guard."""
    with ProviderWriteGuard(fake_provider) as before:
        for path in sorted(fake_provider.rglob("*")):
            if path.is_file():
                path.read_bytes()
    assert before == fingerprint_tree(fake_provider)


@pytest.mark.unit
def test_guard_detects_a_new_file(fake_provider: Path) -> None:
    """Any file created inside the provider is a violation, not a warning."""
    with (
        pytest.raises(ReadOnlyViolationError, match="changed during execution"),
        ProviderWriteGuard(fake_provider),
    ):
        (fake_provider / "expression_cache").mkdir()


@pytest.mark.unit
def test_guard_detects_in_place_modification(fake_provider: Path) -> None:
    """Rewriting an existing artifact must change the fingerprint."""
    target = fake_provider / "features" / "features.bin"
    with pytest.raises(ReadOnlyViolationError), ProviderWriteGuard(fake_provider):
        target.write_bytes(b"qresearch-provider-fixture-mutated-payload")


@pytest.mark.unit
def test_guard_reports_the_delta_fields(fake_provider: Path) -> None:
    """The violation message must localise which aggregate changed."""
    guard = ProviderWriteGuard(fake_provider)
    guard.__enter__()
    (fake_provider / "qlib_cache").mkdir()
    with pytest.raises(ReadOnlyViolationError) as error_info:
        guard.check()
    message = str(error_info.value)
    assert "n_files" in message or "n_dirs" in message
    assert "newest_mtime_ns" in message


@pytest.mark.unit
def test_isolation_accepts_external_provider(fake_provider: Path, fake_cache: Path) -> None:
    """A provider outside the repository with caches elsewhere is valid."""
    assert_provider_isolation(fake_provider, fake_cache)


@pytest.mark.unit
def test_isolation_rejects_cache_inside_provider(fake_provider: Path) -> None:
    """A cache inside the provider would write into read-only data."""
    with pytest.raises(ReadOnlyViolationError, match="lies inside the read-only provider"):
        assert_provider_isolation(fake_provider, fake_provider / "cache")


@pytest.mark.unit
def test_isolation_rejects_provider_inside_repository(repo_root: Path, fake_cache: Path) -> None:
    """Market data must never live inside the working tree."""
    inside = repo_root / "artifacts" / "_pytest_provider_inside_repo"
    inside.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(ReadOnlyViolationError, match="lies inside the repository"):
            assert_provider_isolation(inside, fake_cache)
    finally:
        shutil.rmtree(inside, ignore_errors=True)


@pytest.mark.unit
def test_isolation_rejects_polluted_provider(fake_provider: Path, fake_cache: Path) -> None:
    """Pre-existing cache entries are evidence of an earlier write and must fail loudly."""
    (fake_provider / "qlib_cache").mkdir()
    with pytest.raises(ReadOnlyViolationError, match="cache-like entries"):
        assert_provider_isolation(fake_provider, fake_cache)


@pytest.mark.unit
def test_path_scan_rejects_provider_paths(fake_provider: Path) -> None:
    """The recursive scan is the last line of defence on generated qlib.init kwargs."""
    with pytest.raises(ReadOnlyViolationError, match="points inside the read-only provider"):
        _assert_no_path_under({"nested": [{"expression_cache": str(fake_provider / "cache")}]}, fake_provider)


@pytest.mark.unit
def test_build_qlib_init_kwargs_is_cache_free_by_default(fake_provider: Path, fake_cache: Path) -> None:
    """Default initialization writes nothing and routes the MLflow store under the cache root."""
    kwargs = build_qlib_init_kwargs(fake_provider, cache_root=fake_cache)
    assert kwargs["provider_uri"] == str(fake_provider.resolve())
    assert kwargs["kernels"] == 1
    assert kwargs["expression_cache"] is None
    assert kwargs["dataset_cache"] is None
    exp_manager = kwargs["exp_manager"]
    assert isinstance(exp_manager, dict)
    mlflow_uri = str(exp_manager["kwargs"]["uri"])  # type: ignore[index]
    assert mlflow_uri.startswith("file:")
    assert "mlruns" in mlflow_uri
    assert str(fake_provider.resolve()) not in mlflow_uri


@pytest.mark.unit
def test_build_qlib_init_kwargs_enables_caches_outside_provider(fake_provider: Path, fake_cache: Path) -> None:
    """Opt-in caching must land in the writable cache root, never in the provider."""
    kwargs = build_qlib_init_kwargs(fake_provider, cache_root=fake_cache, enable_cache=True)
    expression_cache = Path(str(kwargs["expression_cache"]))
    dataset_cache = Path(str(kwargs["dataset_cache"]))
    assert expression_cache.is_dir()
    assert dataset_cache.is_dir()
    assert expression_cache.is_relative_to(fake_cache)
    assert not expression_cache.is_relative_to(fake_provider)
    assert not dataset_cache.is_relative_to(fake_provider)


@pytest.mark.unit
def test_build_qlib_init_kwargs_refuses_cache_in_provider(fake_provider: Path) -> None:
    """Configuration that would write into the provider must be rejected outright."""
    with pytest.raises(ReadOnlyViolationError):
        build_qlib_init_kwargs(fake_provider, cache_root=fake_provider / "qlib_cache")
