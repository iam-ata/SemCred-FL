"""Fast gates for the Phase 12 Step 5 clean utility matrix."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from conftest import Phase4CorpusFixture
from test_phase12_step3 import _upgrade_fixture

from semcred_fl.data.contracts import DataContractError
from semcred_fl.experiments.phase12_clean_matrix import (
    CONFIRMATORY_SEEDS,
    METHODS,
    Phase12CleanConditionConfig,
    Phase12CleanConditionRequest,
    Phase12CleanMatrixRequest,
    _aggregate_standard,
    _artifact_paths,
    _client_index,
    _execution,
    _load_completed_condition,
    _load_completed_method,
    _metric_records,
    inspect_phase12_clean_matrix,
    run_phase12_clean_condition,
    run_phase12_clean_matrix,
)
from semcred_fl.federation.updates import DefenseUpdate
from semcred_fl.identity import run_id


def _config() -> Phase12CleanConditionConfig:
    return Phase12CleanConditionConfig(
        dataset="edge_iiotset",
        profile="dirichlet_0_3",
        seed=17,
        total_clients=5,
        clients_per_round=5,
        total_rounds=2,
        local_epochs=1,
        batch_size=8,
        hidden_dimensions=(8,),
        dropout=0.0,
        burn_in_rounds=1,
        validation_batch_size=16,
        cache_scan_batch_size=17,
        trim_count=1,
        krum_byzantine_count=1,
        krum_selected_count=4,
        prefer_fused_adam=False,
    )


def _request(
    fixture: Phase4CorpusFixture,
    tmp_path: Path,
    *,
    max_rounds: int | None,
) -> Phase12CleanConditionRequest:
    prior_path = fixture.preprocessing_report.parent / "semantic_prior_bundle.json"
    if not prior_path.exists():
        prior_path, _ = _upgrade_fixture(fixture)
    return Phase12CleanConditionRequest(
        preprocessing_report=fixture.preprocessing_report,
        client_partition_manifest=fixture.client_partition_manifest,
        semantic_prior_path=prior_path,
        output_directory=tmp_path / "condition",
        cache_root=tmp_path / "cache",
        device="cpu",
        max_rounds_this_call=max_rounds,
        config=_config(),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "unknown"},
        {"profile": "unknown"},
        {"clients_per_round": 2},
        {"total_rounds": 1},
        {"local_epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"proximal_mu": -1.0},
        {"maximum_class_weight": 0.5},
        {"hidden_dimensions": ()},
        {"burn_in_rounds": 2},
        {"trim_count": 3},
        {"krum_byzantine_count": 2},
        {"krum_selected_count": 5},
        {"methods": ()},
        {"methods": ("unknown",)},
        {"methods": ("fedavg", "fedavg")},
    ],
)
def test_step5_config_rejects_invalid_controls(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_config(), **change)  # type: ignore[arg-type]


def test_step5_full_budget_and_evidence_scope() -> None:
    edge = Phase12CleanConditionConfig("edge_iiotset", "iid")
    cic = Phase12CleanConditionConfig("cic_iot_2023", "iid")
    assert edge.full_budget is True
    assert edge.evidence_scope == "validation_development_confirmatory_candidate"
    assert cic.evidence_scope == "validation_secondary_license_provenance_blocked"
    stable = edge.stable_record()
    assert stable["performance_claims_allowed"] is False
    assert cast(dict[str, object], stable["aggregation"])["clipping_multiplier"] == 2.5
    assert cast(dict[str, object], stable["semantic_coefficients"]) == {
        "loss": 0.5,
        "attribution": 0.5,
    }
    assert cast(dict[str, object], stable["numerical_coefficients"])["history"] == 0.25
    assert CONFIRMATORY_SEEDS == tuple(range(20260816, 20260826))


def test_step5_matrix_dry_run_is_read_only_and_counts_registry(tmp_path: Path) -> None:
    request = Phase12CleanMatrixRequest(
        project_root=tmp_path,
        output_directory=tmp_path / "matrix",
        cache_root=tmp_path / "cache",
        datasets=("edge_iiotset",),
        profiles=("iid", "dirichlet_0_3"),
        seeds=(17, 18),
        config_overrides={"methods": ("fedavg", "semcred_fl")},
    )
    report = inspect_phase12_clean_matrix(request)
    assert report["status"] == "blocked"
    assert report["conditions_total"] == 4
    assert report["method_runs_total"] == 8
    assert len(cast(list[str], report["missing_artifacts"])) == 3
    assert not (tmp_path / "matrix").exists()

    cuda_report = inspect_phase12_clean_matrix(replace(request, device="cuda"))
    assert cuda_report["matrix_input_id"] != report["matrix_input_id"]
    fusion_report = inspect_phase12_clean_matrix(
        replace(request, config_overrides={"semantic_fusion_coefficient": 0.4})
    )
    assert fusion_report["matrix_input_id"] != report["matrix_input_id"]


def test_step5_matrix_dry_run_validates_every_resolved_condition(tmp_path: Path) -> None:
    request = Phase12CleanMatrixRequest(
        project_root=tmp_path,
        output_directory=tmp_path / "matrix",
        cache_root=tmp_path / "cache",
        datasets=("edge_iiotset",),
        profiles=("iid",),
        seeds=(17,),
        config_overrides={"clients_per_round": 4, "trim_count": 2},
    )
    with pytest.raises(ValueError, match="trimmed-mean"):
        inspect_phase12_clean_matrix(request)
    assert not (tmp_path / "matrix").exists()


def _update(client_id: str, values: list[float]) -> DefenseUpdate:
    tensor = torch.tensor(values)
    return DefenseUpdate(
        round_id=0,
        client_id=client_id,
        delta=tensor,
        declared_sample_count=1,
        mean_local_loss=0.1,
        update_norm=float(torch.linalg.vector_norm(tensor).item()),
        finite=True,
        missing_class_ids=(),
    )


@pytest.mark.parametrize("method", METHODS[:-1])
def test_step5_standard_aggregators_are_dispatched(method: str) -> None:
    updates = tuple(
        _update(f"client-{index:02d}", [float(index), float(index + 1)]) for index in range(5)
    )
    delta, metadata = _aggregate_standard(method, updates, _config())
    assert delta.shape == (2,)
    assert torch.isfinite(delta).all().item()
    assert "controls" in metadata and "weights" in metadata


def test_step5_condition_resumes_and_emits_five_methods(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
) -> None:
    interrupted = run_phase12_clean_condition(_request(phase4_corpus, tmp_path, max_rounds=1))
    assert interrupted["status"] == "interrupted"
    methods = cast(list[dict[str, object]], interrupted["methods"])
    assert [item["method"] for item in methods] == list(METHODS)
    assert {item["next_round"] for item in methods} == {1}

    completed = run_phase12_clean_condition(_request(phase4_corpus, tmp_path, max_rounds=None))
    assert completed["status"] == "complete"
    results = cast(list[dict[str, object]], completed["method_results"])
    assert [item["method"] for item in results] == list(METHODS)
    assert all(
        cast(dict[str, object], item["validation_metrics"])["records"] == 40 for item in results
    )
    assert len((tmp_path / "condition/run_metrics.jsonl").read_text().splitlines()) == 35
    runtime = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "condition/methods").glob("*/runtime/round-*.json")
    ]
    assert sum(item["updates_reused"] for item in runtime) >= 20

    repeated = run_phase12_clean_condition(_request(phase4_corpus, tmp_path, max_rounds=None))
    assert repeated == completed


def test_step5_condition_supports_development_method_subset(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
) -> None:
    request = _request(phase4_corpus, tmp_path, max_rounds=None)
    request = replace(
        request,
        config=replace(request.config, methods=("fedavg", "semcred_fl")),
    )
    completed = run_phase12_clean_condition(request)
    assert completed["status"] == "complete"
    results = cast(list[dict[str, object]], completed["method_results"])
    assert [item["method"] for item in results] == ["fedavg", "semcred_fl"]
    gates = cast(dict[str, object], completed["gates"])
    assert gates["requested_methods_complete"] is True
    assert gates["five_standardized_methods_complete"] is False
    assert len((tmp_path / "condition/run_metrics.jsonl").read_text().splitlines()) == 14


def test_step5_matrix_validates_filters_and_artifact_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Phase12CleanMatrixRequest(tmp_path, tmp_path, tmp_path, datasets=())
    with pytest.raises(ValueError):
        Phase12CleanMatrixRequest(tmp_path, tmp_path, tmp_path, profiles=("bad",))
    with pytest.raises(ValueError):
        Phase12CleanMatrixRequest(tmp_path, tmp_path, tmp_path, device="tpu")
    with pytest.raises(ValueError):
        Phase12CleanMatrixRequest(tmp_path, tmp_path, tmp_path, max_rounds_this_call=0)
    with pytest.raises(ValueError):
        Phase12CleanMatrixRequest(tmp_path, tmp_path, tmp_path, max_conditions_this_call=0)
    preprocessing, partitions, prior = _artifact_paths(tmp_path, "edge_iiotset")
    assert preprocessing.name == "edge_iiotset_phase3c_preprocessed.json"
    assert partitions.name == "edge_iiotset_phase3d_clients.json"
    assert prior.name == "semantic_prior_bundle.json"


def test_step5_matrix_assembles_completed_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = 0

    def fake_condition(request: Phase12CleanConditionRequest) -> dict[str, object]:
        nonlocal counter
        counter += 1
        config = request.config
        return {
            "status": "complete",
            "dataset": config.dataset,
            "profile": config.profile,
            "seed": config.seed,
            "condition_input_id": f"input-{counter}",
            "condition_result_id": f"result-{counter}",
        }

    monkeypatch.setattr(
        "semcred_fl.experiments.phase12_clean_matrix.run_phase12_clean_condition",
        fake_condition,
    )
    report = run_phase12_clean_matrix(
        Phase12CleanMatrixRequest(
            project_root=tmp_path,
            output_directory=tmp_path / "matrix",
            cache_root=tmp_path / "cache",
            datasets=("edge_iiotset",),
            profiles=("iid", "dirichlet_0_3"),
            seeds=(17,),
            device="cpu",
            config_overrides={
                "total_clients": 5,
                "clients_per_round": 5,
                "total_rounds": 2,
                "local_epochs": 1,
                "batch_size": 8,
                "hidden_dimensions": (8,),
                "dropout": 0.0,
                "burn_in_rounds": 1,
                "validation_batch_size": 16,
                "cache_scan_batch_size": 17,
                "trim_count": 1,
                "krum_byzantine_count": 1,
                "krum_selected_count": 4,
                "prefer_fused_adam": False,
            },
        )
    )
    assert report["status"] == "complete"
    assert len(cast(list[object], report["condition_results"])) == 2
    assert report["gates"]["test_partition_accessed"] is False  # type: ignore[index]
    repeated = run_phase12_clean_matrix(
        Phase12CleanMatrixRequest(
            project_root=tmp_path,
            output_directory=tmp_path / "matrix",
            cache_root=tmp_path / "cache",
            datasets=("edge_iiotset",),
            profiles=("iid", "dirichlet_0_3"),
            seeds=(17,),
            device="cpu",
            config_overrides={
                "total_clients": 5,
                "clients_per_round": 5,
                "total_rounds": 2,
                "local_epochs": 1,
                "batch_size": 8,
                "hidden_dimensions": (8,),
                "dropout": 0.0,
                "burn_in_rounds": 1,
                "validation_batch_size": 16,
                "cache_scan_batch_size": 17,
                "trim_count": 1,
                "krum_byzantine_count": 1,
                "krum_selected_count": 4,
                "prefer_fused_adam": False,
            },
        )
    )
    assert repeated == report


def test_step5_request_execution_and_client_id_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = (tmp_path,) * 5
    with pytest.raises(ValueError, match="device"):
        Phase12CleanConditionRequest(*paths, device="tpu", config=config)
    with pytest.raises(ValueError, match="maximum rounds"):
        Phase12CleanConditionRequest(*paths, max_rounds_this_call=0, config=config)
    request = Phase12CleanConditionRequest(*paths, device="auto", config=config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _execution(request).device == "cpu"
    cuda_request = replace(request, device="cuda")
    with pytest.raises(RuntimeError, match="unavailable"):
        _execution(cuda_request)
    assert _client_index("client-07") == 7
    with pytest.raises(DataContractError, match="malformed"):
        _client_index("client-seven")


def test_step5_artifact_and_metric_readers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="Cannot read Step 5 method"):
        _load_completed_method(tmp_path / "missing.json", "method-id")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(DataContractError, match="method report is malformed"):
        _load_completed_method(malformed, "method-id")

    method_stable: dict[str, object] = {
        "schema": "fixture",
        "method_run_id": "method-id",
    }
    method = {
        **method_stable,
        "method_result_id": run_id(method_stable, length=64),
        "status": "complete",
        "completed_at_utc": "fixture",
    }
    method_path = tmp_path / "method.json"
    method_path.write_text(json.dumps(method), encoding="utf-8")
    assert _load_completed_method(method_path, "method-id") == method
    method["method_result_id"] = "bad"
    method_path.write_text(json.dumps(method), encoding="utf-8")
    with pytest.raises(DataContractError, match="identity"):
        _load_completed_method(method_path, "method-id")

    condition_path = tmp_path / "condition.json"
    condition_path.write_text("[]", encoding="utf-8")
    with pytest.raises(DataContractError, match="condition report is malformed"):
        _load_completed_condition(condition_path, "condition-id")
    condition_path.write_text("{}", encoding="utf-8")
    with pytest.raises(DataContractError, match="identity"):
        _load_completed_condition(condition_path, "condition-id")

    with pytest.raises(DataContractError, match="method metrics"):
        _metric_records(({},), config=_config(), condition_result_id="result")
    bad_metrics = {
        "method": "fedavg",
        "method_run_id": "run",
        "validation_metrics": {
            "accuracy": True,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "worst_class_recall": 0.0,
            "macro_false_positive_rate": 0.0,
        },
    }
    with pytest.raises(DataContractError, match="validation metric"):
        _metric_records((bad_metrics,), config=_config(), condition_result_id="result")


def test_step5_unknown_standard_aggregate_is_rejected() -> None:
    updates = tuple(
        _update(f"client-{index:02d}", [float(index), float(index + 1)]) for index in range(5)
    )
    with pytest.raises(ValueError, match="Unknown"):
        _aggregate_standard("unknown", updates, _config())


def test_step5_matrix_interruption_and_condition_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "semcred_fl.experiments.phase12_clean_matrix.run_phase12_clean_condition",
        lambda request: {
            "status": "interrupted",
            "dataset": request.config.dataset,
            "profile": request.config.profile,
            "seed": request.config.seed,
        },
    )
    base = Phase12CleanMatrixRequest(
        tmp_path,
        tmp_path / "interrupted",
        tmp_path / "cache",
        datasets=("edge_iiotset",),
        profiles=("iid",),
        seeds=(17,),
        config_overrides={
            "total_clients": 5,
            "clients_per_round": 5,
            "total_rounds": 2,
            "local_epochs": 1,
            "batch_size": 8,
            "hidden_dimensions": (8,),
            "dropout": 0.0,
            "burn_in_rounds": 1,
            "validation_batch_size": 16,
            "cache_scan_batch_size": 17,
            "trim_count": 1,
            "krum_byzantine_count": 1,
            "krum_selected_count": 4,
            "prefer_fused_adam": False,
        },
    )
    interrupted = run_phase12_clean_matrix(base)
    assert interrupted["status"] == "interrupted"
    assert "active_condition" in interrupted

    counter = 0

    def complete(request: Phase12CleanConditionRequest) -> dict[str, object]:
        nonlocal counter
        counter += 1
        return {
            "status": "complete",
            "dataset": request.config.dataset,
            "profile": request.config.profile,
            "seed": request.config.seed,
            "condition_input_id": f"input-{counter}",
            "condition_result_id": f"result-{counter}",
        }

    monkeypatch.setattr(
        "semcred_fl.experiments.phase12_clean_matrix.run_phase12_clean_condition",
        complete,
    )
    limited = run_phase12_clean_matrix(
        replace(
            base,
            output_directory=tmp_path / "limited",
            profiles=("iid", "dirichlet_0_3"),
            max_conditions_this_call=1,
        )
    )
    assert limited["status"] == "interrupted"
    assert limited["next_condition_index"] == 1
