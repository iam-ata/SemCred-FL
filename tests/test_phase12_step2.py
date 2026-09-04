"""Fast, resumable, and leakage-safe gates for Phase 12 Step 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from conftest import Phase4CorpusFixture

from semcred_fl.data.client_cache import ClientCacheBuildRequest, build_selected_client_cache
from semcred_fl.data.contracts import DataContractError
from semcred_fl.experiments import phase12_clean as phase12
from semcred_fl.experiments.phase12_clean import (
    Phase12CleanConfig,
    Phase12CleanRequest,
    run_phase12_clean_comparison,
)
from semcred_fl.identity import run_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _upgrade_fixture_for_semantic_runtime(fixture: Phase4CorpusFixture) -> Path:
    report = json.loads(fixture.preprocessing_report.read_text(encoding="utf-8"))
    preprocessor_path = fixture.preprocessing_report.parent / "preprocessor.json"
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    corpus_id = "a" * 64
    preprocessor_id = "b" * 64
    anchor_id = "c" * 64
    split_id = "d" * 64
    coordinates = [
        {
            "coordinate": f"x{index:04d}",
            "semantic_group": "location" if index < 2 else "magnitude",
        }
        for index in range(4)
    ]
    preprocessor.update(
        {
            "preprocessor_id": preprocessor_id,
            "anchor_id": anchor_id,
            "split_id": split_id,
            "coordinates": coordinates,
        }
    )
    _write_json(preprocessor_path, preprocessor)
    report.update(
        {
            "preprocessed_corpus_id": corpus_id,
            "anchor_id": anchor_id,
            "split_id": split_id,
        }
    )
    report["preprocessor"].update(
        {
            "preprocessor_id": preprocessor_id,
            "sha256": _sha256(preprocessor_path),
        }
    )
    _write_json(fixture.preprocessing_report, report)

    clients = json.loads(fixture.client_partition_manifest.read_text(encoding="utf-8"))
    clients.update(
        {
            "preprocessed_corpus_id": corpus_id,
            "preprocessor_id": preprocessor_id,
            "anchor_id": anchor_id,
            "split_id": split_id,
        }
    )
    _write_json(fixture.client_partition_manifest, clients)

    stable = {
        "schema_version": 1,
        "phase": "7-semantic-prior-bundle",
        "dataset": "edge_iiotset",
        "preprocessed_corpus_id": corpus_id,
        "preprocessor_id": preprocessor_id,
        "anchor_id": anchor_id,
        "alpha_l": 0.5,
        "epsilon": 1e-12,
        "class_names": ["benign", "attack"],
        "feature_group_names": ["location", "magnitude"],
        "fused_priors": [[0.7, 0.3], [0.3, 0.7]],
        "gates": {
            "response_schema_valid": True,
            "class_and_group_coverage_exact": True,
            "anchor_calibration_only": True,
            "prompt_public_only": True,
            "client_records_sent_to_llm": False,
            "client_updates_sent_to_llm": False,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "scoring_anchor_accessed": False,
            "online_llm_call_required": False,
            "performance_claims_allowed": False,
        },
    }
    prior_path = fixture.preprocessing_report.parent / "semantic_prior_bundle.json"
    _write_json(prior_path, {"bundle_id": run_id(stable, length=64), **stable})
    return prior_path


@pytest.fixture
def step2_request(phase4_corpus: Phase4CorpusFixture, tmp_path: Path) -> Phase12CleanRequest:
    prior_path = _upgrade_fixture_for_semantic_runtime(phase4_corpus)
    cache_directory = tmp_path / "cache"
    build_selected_client_cache(
        ClientCacheBuildRequest(
            preprocessing_report=phase4_corpus.preprocessing_report,
            client_partition_manifest=phase4_corpus.client_partition_manifest,
            profile="iid",
            selected_client_indices=(0, 1),
            schedule_id="e" * 64,
            output_directory=cache_directory,
            work_database=cache_directory / "cache.sqlite3",
            batch_size=32,
        )
    )
    config = Phase12CleanConfig(
        total_clients=2,
        clients_per_round=2,
        total_rounds=2,
        local_epochs=1,
        batch_size=8,
        hidden_dimensions=(8,),
        dropout=0.0,
        burn_in_rounds=1,
        validation_batch_size=16,
        prefer_fused_adam=False,
        materialize_client_batches=True,
    )
    return Phase12CleanRequest(
        preprocessing_report=phase4_corpus.preprocessing_report,
        client_partition_manifest=phase4_corpus.client_partition_manifest,
        semantic_prior_path=prior_path,
        client_cache_manifest=cache_directory / "client_cache.json",
        output_directory=tmp_path / "comparison",
        device="cpu",
        max_rounds_this_call=1,
        config=config,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"profile": "dirichlet_0_1"},
        {"clients_per_round": 1},
        {"total_rounds": 0},
        {"local_epochs": 0},
        {"batch_size": 0},
        {"validation_batch_size": 0},
        {"learning_rate": 0.0},
        {"proximal_mu": -1.0},
        {"maximum_class_weight": 0.5},
        {"hidden_dimensions": ()},
        {"dropout": 1.0},
        {"burn_in_rounds": 3},
        {"attribution_interval": 0},
    ],
)
def test_step2_config_rejects_invalid_pilot_controls(change: dict[str, object]) -> None:
    values: dict[str, object] = {}
    values.update(change)
    with pytest.raises(ValueError):
        Phase12CleanConfig(**values)  # type: ignore[arg-type]


def test_step2_request_rejects_invalid_execution_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        Phase12CleanRequest(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, device="tpu")
    with pytest.raises(ValueError, match="maximum rounds"):
        Phase12CleanRequest(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            max_rounds_this_call=0,
        )


def test_step2_execution_resolution_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    automatic = phase12._resolve_execution(
        Phase12CleanRequest(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, device="auto")
    )
    assert automatic.device == "cpu"
    assert automatic.optimizer_backend == "standard"
    with pytest.raises(RuntimeError, match="CUDA"):
        phase12._resolve_execution(
            Phase12CleanRequest(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, device="cuda")
        )


def test_step2_immutable_and_tensor_artifacts_fail_closed(tmp_path: Path) -> None:
    immutable = tmp_path / "immutable.json"
    phase12._write_json_immutable(immutable, {"value": 1})
    phase12._write_json_immutable(immutable, {"value": 1})
    with pytest.raises(DataContractError, match="Immutable"):
        phase12._write_json_immutable(immutable, {"value": 2})

    tensor_path = tmp_path / "tensor.pt"
    digest = phase12._save_torch(tensor_path, {"value": torch.tensor([1.0])})
    assert phase12._load_torch(tensor_path, digest)["value"].item() == 1.0
    with pytest.raises(DataContractError, match="checksum"):
        phase12._load_torch(tensor_path, "0" * 64)
    malformed_path = tmp_path / "malformed.pt"
    torch.save([1, 2], malformed_path)
    with pytest.raises(DataContractError, match="dictionary"):
        phase12._load_torch(malformed_path)
    with pytest.raises(DataContractError, match="client ID"):
        phase12._client_index("malformed")


def _valid_cached_update_payload() -> dict[str, object]:
    delta = torch.tensor([3.0, 4.0])
    return {
        "checkpoint_version": phase12._CHECKPOINT_VERSION,
        "update_input_id": "input-id",
        "delta": delta,
        "delta_sha256": phase12._tensor_sha256(delta),
        "round_id": 0,
        "client_id": "client-00",
        "declared_sample_count": 2,
        "mean_local_loss": 0.25,
        "epoch_durations_seconds": [0.1],
        "update_norm": 5.0,
        "missing_class_ids": [],
    }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("checkpoint_version", 99, "version"),
        ("update_input_id", "other", "different inputs"),
        ("delta", torch.tensor([[3.0, 4.0]]), "tensor"),
        ("delta_sha256", "0" * 64, "identity"),
        ("mean_local_loss", "bad", "metrics"),
        ("missing_class_ids", ["bad"], "missing-class"),
        ("epoch_durations_seconds", ["bad"], "timing"),
        ("update_norm", 9.0, "finite/norm"),
    ],
)
def test_step2_cached_updates_reject_corruption(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    payload = _valid_cached_update_payload()
    payload[field] = replacement
    path = tmp_path / f"{field}.pt"
    torch.save(payload, path)
    with pytest.raises(DataContractError, match=message):
        phase12._load_cached_update(path, expected_input_id="input-id", expected_values=2)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "client states"),
        ({1: {}}, "client-state entry"),
        ({"client-00": {"accepted_rounds": "bad"}}, "accepted rounds"),
        (
            {"client-00": {"accepted_rounds": [], "accepted_history": "bad"}},
            "accepted history",
        ),
    ],
)
def test_step2_checkpoint_states_reject_corruption(value: object, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        phase12._load_states(value)


def test_step2_metric_records_reject_malformed_reports() -> None:
    with pytest.raises(DataContractError, match="method metrics"):
        phase12._comparison_metrics(({},), seed=1, comparison_result_id="result")
    with pytest.raises(DataContractError, match="validation metric"):
        phase12._comparison_metrics(
            (
                {
                    "method": "fedavg",
                    "method_run_id": "run",
                    "validation_metrics": {
                        name: (True if name == "accuracy" else 0.5)
                        for name in (
                            "accuracy",
                            "macro_precision",
                            "macro_recall",
                            "macro_f1",
                            "balanced_accuracy",
                            "worst_class_recall",
                            "macro_false_positive_rate",
                        )
                    },
                },
            ),
            seed=1,
            comparison_result_id="result",
        )


def test_step2_resumes_and_reuses_identical_round_zero_updates(
    step2_request: Phase12CleanRequest,
) -> None:
    interrupted = run_phase12_clean_comparison(step2_request)
    assert interrupted["status"] == "interrupted"
    methods = interrupted["methods"]
    assert isinstance(methods, list)
    assert {item["next_round"] for item in methods} == {1}

    completed = run_phase12_clean_comparison(
        Phase12CleanRequest(
            preprocessing_report=step2_request.preprocessing_report,
            client_partition_manifest=step2_request.client_partition_manifest,
            semantic_prior_path=step2_request.semantic_prior_path,
            client_cache_manifest=step2_request.client_cache_manifest,
            output_directory=step2_request.output_directory,
            device="cpu",
            config=step2_request.config,
        )
    )
    assert completed["status"] == "complete"
    gates = cast(dict[str, object], completed["gates"])
    assert gates["validation_partition_accessed"] is True
    assert gates["test_partition_accessed"] is False
    assert gates["performance_claims_allowed"] is False
    method_results = cast(list[dict[str, object]], completed["method_results"])
    assert {item["method"] for item in method_results} == {"fedavg", "semcred_fl"}
    assert all(
        cast(dict[str, object], item["validation_metrics"])["records"] == 40
        for item in method_results
    )

    fedavg_runtime = json.loads(
        (step2_request.output_directory / "methods/fedavg/runtime/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    semcred_runtime = json.loads(
        (step2_request.output_directory / "methods/semcred_fl/runtime/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert fedavg_runtime["updates_trained"] == 2
    assert semcred_runtime["updates_reused"] == 2
    assert len(tuple((step2_request.output_directory / "shared_updates").glob("*.pt"))) == 4
    assert (step2_request.output_directory / "reporting/macro_f1_plot.svg").is_file()
    first_path = step2_request.output_directory / "comparison_report.json"
    before = first_path.read_bytes()
    repeated = run_phase12_clean_comparison(
        Phase12CleanRequest(
            preprocessing_report=step2_request.preprocessing_report,
            client_partition_manifest=step2_request.client_partition_manifest,
            semantic_prior_path=step2_request.semantic_prior_path,
            client_cache_manifest=step2_request.client_cache_manifest,
            output_directory=step2_request.output_directory,
            device="cpu",
            config=step2_request.config,
        )
    )
    assert repeated == completed
    assert first_path.read_bytes() == before

    first_path.unlink()
    reused_methods = run_phase12_clean_comparison(
        Phase12CleanRequest(
            preprocessing_report=step2_request.preprocessing_report,
            client_partition_manifest=step2_request.client_partition_manifest,
            semantic_prior_path=step2_request.semantic_prior_path,
            client_cache_manifest=step2_request.client_cache_manifest,
            output_directory=step2_request.output_directory,
            device="cpu",
            config=step2_request.config,
        )
    )
    assert reused_methods["comparison_result_id"] == completed["comparison_result_id"]

    corrupted = json.loads(first_path.read_text(encoding="utf-8"))
    corrupted["comparison_result_id"] = "0" * 64
    _write_json(first_path, corrupted)
    with pytest.raises(DataContractError, match="comparison report"):
        run_phase12_clean_comparison(
            Phase12CleanRequest(
                preprocessing_report=step2_request.preprocessing_report,
                client_partition_manifest=step2_request.client_partition_manifest,
                semantic_prior_path=step2_request.semantic_prior_path,
                client_cache_manifest=step2_request.client_cache_manifest,
                output_directory=step2_request.output_directory,
                device="cpu",
                config=step2_request.config,
            )
        )
