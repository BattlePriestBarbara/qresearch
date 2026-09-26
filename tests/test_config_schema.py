"""Typed configuration schema and loader (task ``INF-10``, ``PROJECT_SPEC.md`` 3.5).

The specification is unusually explicit about this component: *"The config loader MUST reject unknown
keys, missing segments, non-overlapping date ranges, ``fit_end_time`` later than the ``train`` end, and
any ``*_cost`` equal to zero in a workflow that produces reported numbers."*  Each of those clauses has
its own test below, together with the ADR-005 path rules the loader implements.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from qresearch.config import loader, paths, schema
from qresearch.config.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_STUDY = REPO_ROOT / "configs" / "study_baseline.yaml"


@pytest.fixture(autouse=True)
def clean_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from the documented defaults."""
    for name in (paths.DATA_DIR_ENV_VAR, paths.PROVIDER_URI_ENV_VAR, paths.ARTIFACT_ROOT_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    paths.clear_applied_overrides()


@pytest.fixture()
def document() -> dict[str, object]:
    """A fresh, valid study document, equivalent to ``configs/study_baseline.yaml``."""
    return copy.deepcopy(loader.load_yaml_document(REFERENCE_STUDY))


@pytest.fixture()
def config(document: dict[str, object]) -> schema.StudyConfig:
    """The validated configuration of the reference document."""
    return loader.build_study_config(document, source="reference")


@pytest.mark.unit
def test_model_defaults_are_the_verified_configuration_of_section_3_5() -> None:
    """Every default lives in the schema dataclass, and it must be the documented one."""
    params = schema.ModelHyperparameters()
    assert (params.d_feat, params.seq_len, params.hidden_size, params.num_layers) == (158, 20, 128, 2)
    assert (params.encoder, params.loss, params.lambda_rank) == ("gru", "student_t", 0.3)
    assert (params.n_epochs, params.batch_size, params.early_stop, params.seed) == (100, 1024, 20, 42)
    assert params.lr == 1.0e-3
    assert params.nu_min == 2.05
    assert schema.ExchangeCosts().trade_unit == 100


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value"),
    [("encoder", "transformer"), ("loss", "huber"), ("nu_min", 1.5), ("lr", 0.0), ("seq_len", 0), ("seed", -1)],
)
def test_model_hyperparameters_reject_unusable_values(field_name: str, value: object) -> None:
    """A structurally unusable hyper-parameter cannot be constructed at all."""
    with pytest.raises(ConfigError):
        schema.ModelHyperparameters(**{field_name: value})


@pytest.mark.unit
def test_segments_reject_an_overlap() -> None:
    """Overlapping segments silently train on evaluation data - the one thing PIT-2 forbids."""
    with pytest.raises(ConfigError, match="MUST NOT overlap"):
        schema.SegmentSpec(
            train=("2010-01-01", "2016-12-31"),
            valid=("2016-01-01", "2018-12-31"),
            test=("2019-01-01", "2020-12-31"),
        )


@pytest.mark.unit
def test_segments_reject_a_reversed_range_and_a_negative_purge() -> None:
    """A reversed range and a negative purge radius are meaningless, not merely unusual."""
    with pytest.raises(ConfigError, match="precedes start"):
        schema.SegmentSpec(
            train=("2016-12-31", "2010-01-01"),
            valid=("2017-01-01", "2018-12-31"),
            test=("2019-01-01", "2020-12-31"),
        )
    with pytest.raises(ConfigError, match="purge_radius"):
        schema.SegmentSpec(
            train=("2010-01-01", "2016-12-31"),
            valid=("2017-01-01", "2018-12-31"),
            test=("2019-01-01", "2020-12-31"),
            purge_radius=-1,
        )


@pytest.mark.unit
def test_data_handler_rejects_a_fit_window_outside_the_data_window() -> None:
    """A processor cannot be fitted on data the handler never sees (PIT-2)."""
    with pytest.raises(ConfigError, match="outside the data window"):
        schema.DataHandlerConfig(
            start_time="2010-01-01",
            end_time="2016-12-31",
            fit_start_time="2010-01-01",
            fit_end_time="2017-06-30",
            instruments="csi300",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block", "kwargs"),
    [
        ("exchange", {"trade_unit": 10}),
        ("exchange", {"open_cost": -0.001}),
        ("strategy", {"n_drop": 50}),
        ("strategy", {"rebalance": "hourly"}),
        ("strategy", {"risk_degree": 0.0}),
    ],
)
def test_cost_and_strategy_blocks_reject_unusable_values(block: str, kwargs: dict[str, object]) -> None:
    """The cost model mirrors ``qlib.backtest.exchange``; a wrong lot size or cadence is refused."""
    target = schema.ExchangeCosts if block == "exchange" else schema.StrategyConfig
    with pytest.raises(ConfigError):
        target(**kwargs)


@pytest.mark.unit
def test_backtest_rejects_a_non_daily_step() -> None:
    """``3.4.5`` fixes the executor step at ``day``; cadence belongs to the strategy."""
    with pytest.raises(ConfigError, match="time_per_step"):
        schema.BacktestConfig(start_time="2019-01-01", end_time="2020-12-31", time_per_step="week")


@pytest.mark.unit
def test_study_rejects_a_fit_end_beyond_the_train_segment(config: schema.StudyConfig) -> None:
    """``3.5`` names this rule explicitly: fit_end_time MUST NOT pass the train end."""
    with pytest.raises(ConfigError, match="PIT-2"):
        schema.StudyConfig(
            name=config.name,
            qlib_init=config.qlib_init,
            data_handler=schema.DataHandlerConfig(
                start_time="2008-01-01",
                end_time="2024-12-31",
                fit_start_time="2008-01-01",
                fit_end_time="2017-06-30",
                instruments="csi300",
            ),
            segments=config.segments,
            backtest=config.backtest,
        )


@pytest.mark.unit
def test_study_rejects_a_backtest_outside_the_test_segment(config: schema.StudyConfig) -> None:
    """The acceptance criteria of 1.4 are binding on the test segment, so the window sits inside it."""
    with pytest.raises(ConfigError, match="test segment only"):
        schema.StudyConfig(
            name=config.name,
            qlib_init=config.qlib_init,
            data_handler=config.data_handler,
            segments=config.segments,
            backtest=schema.BacktestConfig(start_time="2017-01-01", end_time="2018-12-31"),
        )


@pytest.mark.unit
def test_reference_document_loads_and_resolves_the_documented_paths(config: schema.StudyConfig) -> None:
    """The committed reference study is valid, and its paths come from ADR-005's resolution."""
    assert config.name == "baseline_csi300"
    assert config.seed == 42
    assert Path(config.qlib_init.provider_uri) == paths.data_dir()
    assert config.qlib_init.kernels == 1
    assert config.qlib_init.mlflow_uri == f"file:{paths.artifact_root().as_posix()}/mlruns"
    assert config.model.params.d_feat == 158
    assert config.backtest.exchange.open_cost == 0.0015
    assert [spec.class_name for spec in config.data_handler.infer_processors] == ["RobustZScoreNorm", "Fillna"]


@pytest.mark.unit
def test_unknown_keys_are_rejected_at_every_level(document: dict[str, object]) -> None:
    """A typo must be an error, not a silent fallback to the dataclass default."""
    top_level = copy.deepcopy(document)
    top_level["backtestt"] = top_level["backtest"]
    with pytest.raises(ConfigError, match="unknown key"):
        loader.build_study_config(top_level, source="typo")

    nested = copy.deepcopy(document)
    nested["model"]["kwargs"]["learning_rate"] = 0.01
    with pytest.raises(ConfigError, match="unknown key"):
        loader.build_study_config(nested, source="typo")

    strategy = copy.deepcopy(document)
    strategy["backtest"]["strategy"]["kwargs"]["top_k"] = 10
    with pytest.raises(ConfigError, match="unknown key"):
        loader.build_study_config(strategy, source="typo")


@pytest.mark.unit
def test_missing_blocks_and_segments_are_rejected(document: dict[str, object]) -> None:
    """``3.5`` requires the loader to reject a missing segment (and a missing block is the same defect)."""
    without_model = copy.deepcopy(document)
    del without_model["model"]
    with pytest.raises(ConfigError, match="missing the required block"):
        loader.build_study_config(without_model, source="incomplete")

    without_test = copy.deepcopy(document)
    del without_test["segments"]["test"]
    with pytest.raises(ConfigError, match="missing the segment"):
        loader.build_study_config(without_test, source="incomplete")


@pytest.mark.unit
def test_type_mismatches_are_diagnosed_by_name(document: dict[str, object]) -> None:
    """The most common configuration defect is a quoted number; it must be named, not crashed into."""
    wrong_type = copy.deepcopy(document)
    wrong_type["model"]["kwargs"]["seq_len"] = "20"
    with pytest.raises(ConfigError, match=r"model.kwargs.seq_len must be int"):
        loader.build_study_config(wrong_type, source="typed")

    boolean_as_int = copy.deepcopy(document)
    boolean_as_int["model"]["kwargs"]["n_epochs"] = True
    with pytest.raises(ConfigError, match=r"model.kwargs.n_epochs must be int"):
        loader.build_study_config(boolean_as_int, source="typed")

    numeric_string = copy.deepcopy(document)
    numeric_string["model"]["kwargs"]["lr"] = "1.0e-3"
    assert loader.build_study_config(numeric_string, source="typed").model.params.lr == 1.0e-3


@pytest.mark.unit
def test_zero_transaction_cost_is_rejected_unless_opted_in(document: dict[str, object]) -> None:
    """``3.5`` forbids a zero cost in a workflow that produces reported numbers."""
    costless = copy.deepcopy(document)
    costless["backtest"]["exchange_kwargs"]["open_cost"] = 0.0
    with pytest.raises(ConfigError, match="zero"):
        loader.build_study_config(costless, source="costless")
    opted_in = loader.build_study_config(costless, source="costless", allow_zero_cost=True)
    assert opted_in.backtest.exchange.open_cost == 0.0


@pytest.mark.unit
def test_explicit_provider_argument_wins_over_the_document(document: dict[str, object], tmp_path: Path) -> None:
    """A CLI flag must be able to redirect the data store without editing the document."""
    config = loader.build_study_config(document, source="override", provider_uri=str(tmp_path))
    assert Path(config.qlib_init.provider_uri) == tmp_path


@pytest.mark.unit
def test_provider_uri_must_be_absolute_after_substitution(document: dict[str, object]) -> None:
    """A relative provider depends on the working directory, so it is refused."""
    relative = copy.deepcopy(document)
    relative["qlib_init"]["provider_uri"] = "cn_data"
    with pytest.raises(ConfigError, match="must be absolute"):
        loader.build_study_config(relative, source="relative")


@pytest.mark.unit
def test_only_path_placeholders_may_be_substituted() -> None:
    """Routing behaviour through an environment variable is what ``3.5`` forbids."""
    assert loader.expand_placeholders("uri: ${QLIB_DATA_DIR}", source="x").startswith("uri: ")
    with pytest.raises(ConfigError, match="not an allowed placeholder"):
        loader.expand_placeholders("kernels: ${KERNELS}", source="x")


@pytest.mark.unit
def test_artifact_root_placeholder_supports_the_documented_shorthand() -> None:
    """``${ARTIFACT_ROOT:-artifacts}`` resolves without a machine layout being written down."""
    expanded = loader.expand_placeholders('uri: "file:${ARTIFACT_ROOT:-artifacts}/mlruns"', source="x")
    assert expanded == f'uri: "file:{paths.artifact_root().as_posix()}/mlruns"'
    with pytest.raises(ConfigError, match="not a supported fallback"):
        loader.expand_placeholders("uri: ${ARTIFACT_ROOT:-relative/path}", source="x")


@pytest.mark.unit
def test_expanded_paths_keep_the_yaml_parseable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A substituted path lands inside a quoted YAML scalar, so it must be POSIX-form."""
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, str(tmp_path))
    document = loader.load_yaml_document(REFERENCE_STUDY)
    provider = document["qlib_init"]["provider_uri"]
    assert provider == tmp_path.as_posix()
    assert "\\" not in provider


@pytest.mark.unit
def test_config_hash_is_deterministic_and_content_sensitive(document: dict[str, object]) -> None:
    """``3.5``/``3.7`` require a SHA-256 ``config_hash``: same document, same digest; changed, changed."""
    first = loader.build_study_config(document, source="a")
    second = loader.build_study_config(copy.deepcopy(document), source="a")
    assert loader.config_hash(first) == loader.config_hash(second)
    assert len(loader.config_hash(first)) == 64

    changed = copy.deepcopy(document)
    changed["model"]["kwargs"]["hidden_size"] = 64
    assert loader.config_hash(loader.build_study_config(changed, source="a")) != loader.config_hash(first)


@pytest.mark.unit
def test_resolved_document_records_the_applied_overrides(
    config: schema.StudyConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``3.5``: overrides MUST be logged when applied, so they travel with the resolved config."""
    monkeypatch.setenv(paths.ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    paths.artifact_root()
    resolved = loader.resolved_document(config)
    recorded = resolved[paths.APPLIED_OVERRIDES_SECTION]
    assert isinstance(recorded, list)
    assert [entry["name"] for entry in recorded] == [paths.ARTIFACT_ROOT_ENV_VAR]


@pytest.mark.unit
def test_dump_resolved_config_writes_the_document(config: schema.StudyConfig, tmp_path: Path) -> None:
    """The resolved configuration is dumped next to the artifacts and can be read back."""
    target = loader.dump_resolved_config(config, tmp_path)
    assert target.name == "resolved_config.yaml"
    reloaded = loader.load_yaml_document(target)
    assert reloaded["study"]["name"] == config.name
    assert reloaded["model"]["kwargs"]["d_feat"] == config.model.params.d_feat


@pytest.mark.unit
def test_console_entry_point_checks_and_materializes(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """``qresearch-config`` validates a document and materializes a template (needed by ``qrun``)."""
    assert loader.main(["--check", str(REFERENCE_STUDY)]) == 0
    output = capsys.readouterr().out
    assert "VALID" in output
    assert "config_hash" in output

    template = REPO_ROOT / "configs" / "qlib_init.yaml"
    materialized = tmp_path / "qlib_init.yaml"
    assert loader.main(["--expand", str(template), "--out", str(materialized)]) == 0
    text = materialized.read_text(encoding="utf-8")
    assert "${" not in text
    assert paths.data_dir().as_posix() in text

    broken = tmp_path / "broken.yaml"
    broken.write_text("study: [1, 2\n", encoding="utf-8")
    assert loader.main(["--check", str(broken)]) == 1
    assert "invalid YAML" in capsys.readouterr().err
