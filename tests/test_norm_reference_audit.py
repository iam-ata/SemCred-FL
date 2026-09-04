"""Tests for the checksum-bound Phase 6B.2 closure audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from semcred_fl.attacks.artifacts import benign_norm_reference
from semcred_fl.attacks.norm_calibration import phase6b_norm_calibration_config
from semcred_fl.attacks.norm_reference_audit import (
    NormReferenceAuditRequest,
    audit_phase6b_norm_references,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import run_id, to_primitive


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reference_fixture(root: Path) -> Path:
    config = phase6b_norm_calibration_config()
    dataset = "edge_iiotset"
    profile = "iid"
    corpus_id = "a" * 64
    partition_id = "b" * 64
    detector = {
        "input_dimension": 4,
        "num_classes": 2,
        "hidden_dimensions": list(config.hidden_dimensions),
        "dropout": config.dropout,
    }
    local_training = {
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate,
        "proximal_mu": config.proximal_mu,
        "maximum_class_weight": config.maximum_class_weight,
        "seed": config.base_seed,
        "device": "cpu",
        "optimizer": "adam",
    }
    detector_id = run_id(detector, length=64)
    local_id = run_id(local_training, length=64)
    engine_id = "c" * 64
    rows: list[dict[str, object]] = []
    development_ids: list[str] = []
    norms: list[float] = []
    for round_id in range(config.total_rounds):
        for client_index in range(config.clients_per_round):
            norm = 1.0 + round_id + client_index / 100.0
            development_id = run_id(
                {
                    "phase": "fixture",
                    "round_id": round_id,
                    "client_id": client_index,
                    "norm": norm,
                },
                length=64,
            )
            rows.append(
                {
                    "development_run_id": development_id,
                    "round_id": round_id,
                    "client_id": f"client-{client_index:02d}",
                    "update_norm": norm,
                }
            )
            development_ids.append(development_id)
            norms.append(norm)
    reference = benign_norm_reference(
        dataset=dataset,
        profile=profile,
        preprocessed_corpus_id=corpus_id,
        client_partition_id=partition_id,
        detector_config_id=detector_id,
        local_training_config_id=local_id,
        development_run_ids=development_ids,
        update_norms=norms,
    )
    calibration_input_id = "d" * 64
    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "6B.2",
        "dataset": dataset,
        "profile": profile,
        "preprocessed_corpus_id": corpus_id,
        "client_partition_id": partition_id,
        "schedule_id": "e" * 64,
        "contract_id": config.contract_id,
        "device": "cpu",
        "calibration_input_id": calibration_input_id,
        "cache_id": "f" * 64,
        "federated_run_id": engine_id,
        "detector_config_id": detector_id,
        "detector": detector,
        "local_training_config_id": local_id,
        "local_training": local_training,
        "reference": {
            "reference_id": reference.reference_id,
            **cast(dict[str, object], to_primitive(reference)),
        },
        "update_norm_records": rows,
        "gates": {
            "all_scheduled_updates_accepted": True,
            "all_development_clients_honest": True,
            "cache_bound_to_schedule": True,
            "same_dataset_model_partition_and_training_contract": True,
            "reference_uses_prior_benign_development_updates_only": True,
            "current_round_benign_updates_accessed": False,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "anchor_records_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    artifact = {
        "calibration_id": run_id(stable, length=64),
        "created_at_utc": "2026-08-31T00:00:00+00:00",
        **stable,
    }
    path = root / dataset / profile / "benign_norm_reference.json"
    _write_json(path, artifact)
    return path


def test_closure_audit_is_reproducible_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "references"
    artifact_path = _reference_fixture(root)
    output = root / "phase6b_norm_reference_index.json"
    request = NormReferenceAuditRequest(
        root_directory=root,
        output_path=output,
        expected_references=(("edge_iiotset", "iid"),),
    )
    first = audit_phase6b_norm_references(request)
    second = audit_phase6b_norm_references(request)
    assert first == second
    assert first["expected_reference_count"] == 1
    assert first["total_development_updates"] == 60
    gates = cast(dict[str, object], first["gates"])
    assert gates["all_reference_quartiles_recomputed"] is True
    assert gates["test_partition_accessed"] is False

    artifact = cast(dict[str, object], json.loads(artifact_path.read_text(encoding="utf-8")))
    reference = cast(dict[str, object], artifact["reference"])
    reference["upper_norm"] = 999.0
    _write_json(artifact_path, artifact)
    with pytest.raises(DataContractError):
        audit_phase6b_norm_references(request)
