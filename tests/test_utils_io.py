"""Artifact IO, hashing and determinism helpers (task ``INF-09``).

``3.2`` requires artifacts to be content-addressable, ``3.7`` requires a configuration hash and a data
snapshot hash next to every artifact, and ``3.8``/``5.3`` require determinism.  These tests pin each
primitive against a closed form (``hashlib`` itself, ``random``) rather than against its own output.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from qresearch.utils import io, seeding
from qresearch.utils.logging import LOGGER_NAMESPACE, get_logger, log_applied_overrides


@pytest.mark.unit
def test_hashes_match_hashlib_and_survive_reordering(tmp_path: Path) -> None:
    """A digest must be a function of the *document*, not of a dictionary's insertion order."""
    payload = b"qresearch"
    assert io.sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()

    artifact = tmp_path / "blob.bin"
    artifact.write_bytes(payload)
    assert io.sha256_file(artifact) == io.sha256_bytes(payload)

    first = {"alpha": 1, "beta": [2, 3]}
    second = {"beta": [2, 3], "alpha": 1}
    assert io.hash_mapping(first) == io.hash_mapping(second)
    assert io.hash_mapping(first) != io.hash_mapping({"alpha": 1, "beta": [2, 4]})
    assert json.loads(io.canonical_json(first)) == first


@pytest.mark.unit
def test_json_and_yaml_round_trip(tmp_path: Path) -> None:
    """Artifacts must survive a write/read cycle unchanged and be UTF-8 by default."""
    payload = {"study": "baseline", "pi": 3.14159, "labels": ["a", "b"]}
    json_path = io.write_json(tmp_path / "nested" / "report.json", payload)
    yaml_path = io.write_yaml(tmp_path / "resolved.yaml", payload)

    assert io.read_json(json_path) == payload
    assert io.read_yaml(yaml_path) == payload
    assert json_path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.unit
def test_content_addressed_paths_are_stable_and_content_aware(tmp_path: Path) -> None:
    """The same document maps to the same path; a changed document cannot overwrite the artifact."""
    document = {"ic": 0.05, "t_nw": 13.2}
    first = io.content_addressed_path(tmp_path, "signal_report", document)
    assert first == io.content_addressed_path(tmp_path, "signal_report", {"t_nw": 13.2, "ic": 0.05})
    other = io.content_addressed_path(tmp_path, "signal_report", {"ic": 0.05, "t_nw": 13.3})
    assert first != other
    assert first.suffix == ".json"
    with pytest.raises(ValueError, match="stem"):
        io.content_addressed_path(tmp_path, "", document)


@pytest.mark.unit
def test_artifact_directories_follow_the_layout_of_section_3_7(tmp_path: Path) -> None:
    """Every artifact category is created under the given root, and unknown kinds are refused."""
    category = io.artifact_dir("reports", root=tmp_path)
    assert category == tmp_path / "reports"
    assert category.is_dir()

    run = io.artifact_dir("reports", "run-42", root=tmp_path)
    assert run == tmp_path / "reports" / "run-42"
    assert run.is_dir()
    assert set(io.ARTIFACT_KINDS) == {"mlruns", "predictions", "models", "reports", "figures", "tables"}

    with pytest.raises(ValueError, match="unknown artifact kind"):
        io.artifact_dir("dashboards", root=tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        io.artifact_dir("reports", "", root=tmp_path)


@pytest.mark.unit
def test_seeding_makes_the_global_generators_reproducible() -> None:
    """``3.8``/``5.3``: one seed must control ``random`` and ``numpy``; ``torch`` when it is present."""
    first = seeding.seed_everything(7)
    random_draw = random.random()
    numpy_draw = float(np.random.random())
    second = seeding.seed_everything(7)

    assert random.random() == random_draw
    assert float(np.random.random()) == numpy_draw
    assert first.to_dict() == second.to_dict()
    assert first.seed == 7
    assert "PYTHONHASHSEED" in first.details
    with pytest.raises(ValueError, match="seed must be"):
        seeding.seed_everything(-1)


@pytest.mark.unit
def test_explicit_generator_is_preferred_and_reproducible() -> None:
    """New code carries a ``Generator``; two draws from the same seed must agree bit for bit."""
    left = seeding.numpy_generator(3).standard_normal(5)
    right = seeding.numpy_generator(3).standard_normal(5)
    assert np.array_equal(left, right)
    assert not np.array_equal(left, seeding.numpy_generator(4).standard_normal(5))


@pytest.mark.unit
def test_logging_is_namespaced_and_never_prints(capsys: pytest.CaptureFixture[str]) -> None:
    """``5.4``: library logging goes through the namespace logger, not through ``print``.

    Qlib nests every logger under its own ``qlib`` root, so the namespace shows up at the *end* of the
    name; what matters is that one ``qresearch`` subtree exists and that nothing prints.
    """
    logger = get_logger("stats")
    assert logger.name.endswith("qresearch.stats")
    assert get_logger().name.endswith(LOGGER_NAMESPACE)
    assert get_logger("qresearch.other").name.endswith("qresearch.other")

    described = log_applied_overrides(logger)
    assert "environment overrides" in described
    assert capsys.readouterr().out == ""
