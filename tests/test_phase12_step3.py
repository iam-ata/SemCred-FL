"""Fast gates for the resumable Phase 12 Step 3 attacked comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from conftest import Phase4CorpusFixture

from semcred_fl.attacks.artifacts import (
    Phase6BResolution,
    ResolvedCoordinate,
    TriggerCoordinate,
)
from semcred_fl.data.client_cache import SelectedClientCache
from semcred_fl.data.contracts import DataContractError
from semcred_fl.experiments.phase12_attack import (
    Phase12AttackConfig,
    Phase12AttackRequest,
    _AttackedBatchSource,
    _client_assessment,
    _client_index,
    _load_cached_update,
    _resolve_execution,
    run_phase12_attack_comparison,
)
from semcred_fl.experiments.phase12_clean import _tensor_sha256
from semcred_fl.federation.state import ClientState
from semcred_fl.identity import run_id, to_primitive


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _upgrade_fixture(fixture: Phase4CorpusFixture) -> tuple[Path, Path]:
    corpus_id = "a" * 64
    preprocessor_id = "b" * 64
    split_id = "c" * 64
    anchor_id = "d" * 64
    partition_id = "e" * 64
    report = cast(
        dict[str, object], json.loads(fixture.preprocessing_report.read_text(encoding="utf-8"))
    )
    preprocessor_record = cast(dict[str, object], report["preprocessor"])
    preprocessor_path = fixture.preprocessing_report.parent / cast(str, preprocessor_record["path"])
    preprocessor = cast(
        dict[str, object], json.loads(preprocessor_path.read_text(encoding="utf-8"))
    )
    preprocessor.update(
        {
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
            "coordinates": [
                {
                    "coordinate": f"x{index:04d}",
                    "semantic_group": "location" if index < 2 else "magnitude",
                }
                for index in range(4)
            ],
        }
    )
    _write_json(preprocessor_path, preprocessor)
    report.update(
        {
            "preprocessed_corpus_id": corpus_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    preprocessor_record.update(
        {"preprocessor_id": preprocessor_id, "sha256": _sha256(preprocessor_path)}
    )
    _write_json(fixture.preprocessing_report, report)

    manifest = cast(
        dict[str, object],
        json.loads(fixture.client_partition_manifest.read_text(encoding="utf-8")),
    )
    assignment_files = cast(list[dict[str, object]], manifest["assignment_files"])
    assignment_path = fixture.client_partition_manifest.parent / cast(
        str, assignment_files[0]["path"]
    )
    assignment = pq.read_table(assignment_path)
    client_ids = np.asarray([index % 5 for index in range(80)], dtype=np.int16)
    rewritten = assignment.append_column(
        "dirichlet_0_3_client_id",
        pa.array(client_ids, type=pa.int16()),
    )
    pq.write_table(rewritten, assignment_path)
    assignment_files[0].update(
        {"byte_size": assignment_path.stat().st_size, "sha256": _sha256(assignment_path)}
    )
    manifest.update(
        {
            "client_partition_id": partition_id,
            "preprocessed_corpus_id": corpus_id,
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
            "num_clients": 5,
            "assignment_columns": {
                "iid": "iid_client_id",
                "dirichlet_0_3": "dirichlet_0_3_client_id",
            },
            "profiles": {
                "dirichlet_0_3": {
                    "client_totals": [16] * 5,
                    "family_client_counts": {
                        "benign": [8] * 5,
                        "attack": [8] * 5,
                    },
                }
            },
        }
    )
    _write_json(fixture.client_partition_manifest, manifest)

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

    ranked = tuple(
        ResolvedCoordinate(
            coordinate=f"x{index:04d}",
            index=index,
            source_feature=f"feature-{index}",
            lower_bound=-2.0,
            upper_bound=2.0,
            source_median=1.0,
            benign_median=-1.0,
            absolute_median_gap=2.0,
        )
        for index in range(4)
    )
    resolution = Phase6BResolution(
        dataset="edge_iiotset",
        phase6a_contract_id="f" * 64,
        preprocessed_corpus_id=corpus_id,
        preprocessor_id=preprocessor_id,
        split_id=split_id,
        anchor_id=anchor_id,
        client_partition_id=partition_id,
        source_family="attack",
        source_family_id=1,
        target_family="benign",
        target_family_id=0,
        source_training_records=40,
        target_training_records=40,
        federated_training_records=80,
        ranked_coordinates=ranked,
        feature_coordinates=ranked[:3],
        trigger_coordinates=tuple(
            TriggerCoordinate(
                coordinate=item.coordinate,
                index=item.index,
                source_feature=item.source_feature,
                endpoint="lower",
                value=-2.0,
            )
            for item in ranked[:3]
        ),
        trigger_training_collisions=0,
    )
    resolution_path = fixture.preprocessing_report.parent / "resolution.json"
    _write_json(
        resolution_path,
        {
            "resolution_id": resolution.resolution_id,
            **cast(dict[str, object], to_primitive(resolution)),
            "gates": {
                "source_and_benign_medians_exact_over_federated_train": True,
                "feature_coordinates_nonconstant_and_train_ranked": True,
                "trigger_from_top_twelve_coordinates": True,
                "trigger_has_zero_federated_training_collisions": True,
                "scanner_filter_excludes_validation_test_and_anchors": True,
                "test_partition_accessed": False,
                "performance_claims_allowed": False,
            },
        },
    )
    return prior_path, resolution_path


@pytest.fixture
def step3_request(phase4_corpus: Phase4CorpusFixture, tmp_path: Path) -> Phase12AttackRequest:
    prior_path, resolution_path = _upgrade_fixture(phase4_corpus)
    config = Phase12AttackConfig(
        seed=17,
        malicious_ratio=0.2,
        poison_fraction=0.5,
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
        prefer_fused_adam=False,
    )
    cache = tmp_path / "cache"
    return Phase12AttackRequest(
        preprocessing_report=phase4_corpus.preprocessing_report,
        client_partition_manifest=phase4_corpus.client_partition_manifest,
        resolution_path=resolution_path,
        semantic_prior_path=prior_path,
        output_directory=tmp_path / "comparison",
        cache_directory=cache,
        cache_work_database=cache / "cache.sqlite3",
        device="cpu",
        max_rounds_this_call=1,
        config=config,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"profile": "iid"},
        {"attack_name": "backdoor"},
        {"malicious_ratio": 0.5},
        {"poison_fraction": 0.0},
        {"clients_per_round": 2},
        {"total_rounds": 1},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"proximal_mu": -1.0},
        {"maximum_class_weight": 0.5},
        {"hidden_dimensions": ()},
        {"dropout": 1.0},
        {"attribution_interval": 0},
        {"burn_in_rounds": 3},
    ],
)
def test_step3_config_rejects_invalid_controls(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Phase12AttackConfig(**change)  # type: ignore[arg-type]


def test_step3_request_rejects_invalid_runtime_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        Phase12AttackRequest(
            tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, device="tpu"
        )


def test_step3_execution_profiles_and_client_ids(
    step3_request: Phase12AttackRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    automatic = _resolve_execution(replace_request(step3_request, device="auto"))
    assert automatic.device == "cpu"
    assert automatic.optimizer_backend == "standard"

    streamed_config = Phase12AttackConfig(
        seed=17,
        malicious_ratio=0.2,
        poison_fraction=0.5,
        total_clients=5,
        clients_per_round=5,
        total_rounds=2,
        hidden_dimensions=(8,),
        materialize_client_batches=False,
    )
    streamed = _resolve_execution(
        replace_request(step3_request, device="cpu", config=streamed_config)
    )
    assert streamed.batch_delivery == "streamed_attacked_batches_v1"

    with pytest.raises(RuntimeError, match="CUDA"):
        _resolve_execution(replace_request(step3_request, device="cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fused = _resolve_execution(
        replace_request(step3_request, device="cuda", config=streamed_config)
    )
    assert fused.optimizer_backend == "fused"
    assert _client_index("client-09") == 9
    with pytest.raises(DataContractError, match="client ID"):
        _client_index("client-x")


def test_step3_rejects_duplicate_poison_ids(step3_request: Phase12AttackRequest) -> None:
    with pytest.raises(DataContractError, match="duplicate"):
        _AttackedBatchSource(
            cache=cast(SelectedClientCache, object()),
            client_index=0,
            batch_size=8,
            device="cpu",
            malicious=True,
            selected_record_ids=("same", "same"),
            resolution=cast(Phase6BResolution, object()),
            materialize=True,
        )


def _valid_update_payload() -> dict[str, object]:
    delta = torch.tensor([3.0, 4.0])
    return {
        "checkpoint_version": 1,
        "update_input_id": "input-id",
        "delta": delta,
        "delta_sha256": _tensor_sha256(delta),
        "missing_class_ids": [1],
        "epoch_durations_seconds": [0.1],
        "is_malicious": False,
        "round_id": 0,
        "client_id": "client-00",
        "declared_sample_count": 8,
        "mean_local_loss": 0.25,
        "update_norm": 5.0,
        "attack_id": "attack-id",
        "selection_id": None,
        "selected_poison_records": 0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_version", 2, "version"),
        ("update_input_id", "other", "different inputs"),
        ("delta", torch.ones((1, 2)), "tensor"),
        ("delta_sha256", "wrong", "identity"),
        ("missing_class_ids", [True], "missing-class"),
        ("epoch_durations_seconds", [True], "timing"),
        ("is_malicious", 1, "simulation label"),
        ("update_norm", 4.0, "finite/norm"),
    ],
)
def test_step3_cached_update_rejects_corruption(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _valid_update_payload()
    payload[field] = value
    path = tmp_path / "update.pt"
    torch.save(payload, path)
    with pytest.raises(DataContractError, match=message):
        _load_cached_update(path, expected_input_id="input-id", expected_values=2)


def test_step3_cached_update_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "update.pt"
    torch.save(_valid_update_payload(), path)
    update = _load_cached_update(path, expected_input_id="input-id", expected_values=2)
    assert update.client_id == "client-00"
    assert update.update_norm == 5.0


@pytest.mark.parametrize(
    "round_payload",
    [
        {},
        {"semcred": {}, "selected_client_ids": ["client-00"]},
        {
            "semcred": {
                "participating_trust": {"client-00": True},
                "eligible_client_ids": [],
            },
            "selected_client_ids": ["client-00"],
        },
    ],
)
def test_step3_client_assessment_rejects_malformed_evidence(
    tmp_path: Path, round_payload: dict[str, object]
) -> None:
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    _write_json(rounds / "round-0000.json", round_payload)
    states = {"client-00": ClientState(0.5, None, 0, 0)}
    with pytest.raises(DataContractError):
        _client_assessment(
            tmp_path,
            total_rounds=1,
            malicious_client_ids=("client-00",),
            states=states,
        )
    with pytest.raises(ValueError, match="maximum rounds"):
        Phase12AttackRequest(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            max_rounds_this_call=0,
        )


def test_step3_resumes_reuses_updates_and_joins_truth_after_decision(
    step3_request: Phase12AttackRequest,
) -> None:
    interrupted = run_phase12_attack_comparison(step3_request)
    assert interrupted["status"] == "interrupted"
    methods = cast(list[dict[str, object]], interrupted["methods"])
    assert {item["next_round"] for item in methods} == {1}

    completed = run_phase12_attack_comparison(
        replace_request(step3_request, max_rounds_this_call=None)
    )
    assert completed["status"] == "complete"
    gates = cast(dict[str, object], completed["gates"])
    assert gates["one_attack_one_non_iid_profile_one_seed"] is True
    assert gates["simulation_truth_excluded_from_defense_inputs"] is True
    assert gates["test_partition_accessed"] is False
    assert gates["performance_claims_allowed"] is False
    method_results = cast(list[dict[str, object]], completed["method_results"])
    assert {item["method"] for item in method_results} == {"fedavg", "semcred_fl"}
    assert all(
        cast(dict[str, object], item["validation_metrics"])["records"] == 40
        for item in method_results
    )
    semcred = next(item for item in method_results if item["method"] == "semcred_fl")
    assessment = cast(dict[str, object], semcred["client_assessment_metrics"])
    assert assessment["participation_events"] == 10
    assert (
        cast(dict[str, object], assessment["gates"])[
            "simulation_truth_joined_after_defense_decision"
        ]
        is True
    )

    round_record = json.loads(
        (step3_request.output_directory / "methods/semcred_fl/rounds/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert "malicious" not in json.dumps(round_record["semcred"])
    assert round_record["post_decision_attack_audit"]["joined_after_defense_decision"] is True
    selection = json.loads(
        (
            step3_request.output_directory / "attack_contract/poison_selection_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert {item["selected_count"] for item in selection["client_selections"]} == {4}
    fedavg_runtime = json.loads(
        (step3_request.output_directory / "methods/fedavg/runtime/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    semcred_runtime = json.loads(
        (step3_request.output_directory / "methods/semcred_fl/runtime/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert fedavg_runtime["updates_trained"] == 5
    assert semcred_runtime["updates_reused"] == 5
    assert (step3_request.output_directory / "reporting/macro_f1_plot.svg").is_file()
    metric_lines = (step3_request.output_directory / "run_metrics.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(metric_lines.splitlines()) == 20

    repeated = run_phase12_attack_comparison(
        replace_request(step3_request, max_rounds_this_call=None)
    )
    assert repeated == completed


def replace_request(
    request: Phase12AttackRequest,
    *,
    max_rounds_this_call: int | None = None,
    device: str | None = None,
    config: Phase12AttackConfig | None = None,
) -> Phase12AttackRequest:
    return Phase12AttackRequest(
        preprocessing_report=request.preprocessing_report,
        client_partition_manifest=request.client_partition_manifest,
        resolution_path=request.resolution_path,
        semantic_prior_path=request.semantic_prior_path,
        output_directory=request.output_directory,
        cache_directory=request.cache_directory,
        cache_work_database=request.cache_work_database,
        device=request.device if device is None else device,
        max_rounds_this_call=max_rounds_this_call,
        config=request.config if config is None else config,
    )
