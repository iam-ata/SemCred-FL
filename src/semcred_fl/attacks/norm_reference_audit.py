"""Closure audit for the complete Phase 6B.2 benign norm-reference matrix."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from semcred_fl.attacks.artifacts import benign_norm_reference
from semcred_fl.attacks.norm_calibration import (
    Phase6BNormCalibrationConfig,
    phase6b_norm_calibration_config,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import run_id, to_primitive

_EXPECTED_REFERENCES = (
    ("edge_iiotset", "iid"),
    ("edge_iiotset", "dirichlet_0_1"),
    ("edge_iiotset", "dirichlet_0_3"),
    ("edge_iiotset", "dirichlet_1_0"),
    ("cic_iot_2023", "iid"),
    ("cic_iot_2023", "dirichlet_0_1"),
    ("cic_iot_2023", "dirichlet_0_3"),
    ("cic_iot_2023", "dirichlet_1_0"),
)


@dataclass(frozen=True, slots=True)
class NormReferenceAuditRequest:
    """Paths and expected matrix for one immutable Phase 6B.2 closure audit."""

    root_directory: Path
    output_path: Path
    expected_references: tuple[tuple[str, str], ...] = _EXPECTED_REFERENCES
    config: Phase6BNormCalibrationConfig = field(default_factory=phase6b_norm_calibration_config)

    def __post_init__(self) -> None:
        if (
            not self.expected_references
            or len(set(self.expected_references)) != len(self.expected_references)
            or any(
                not dataset or profile not in self.config.profiles
                for dataset, profile in self.expected_references
            )
        ):
            raise ValueError("Phase 6B.2 closure matrix is invalid")


@dataclass(frozen=True, slots=True)
class _AuditedReference:
    inventory: dict[str, object]
    contract_id: str
    schedule_id: str
    local_training_config_id: str
    preprocessed_corpus_id: str
    client_partition_id: str
    detector_config_id: str
    federated_run_id: str
    calibration_input_id: str
    calibration_id: str
    cache_id: str
    reference_id: str
    development_run_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise DataContractError(f"Phase 6B.2 {key} must be a full SHA-256 identity")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _audit_one(
    path: Path,
    *,
    dataset: str,
    profile: str,
    config: Phase6BNormCalibrationConfig,
) -> _AuditedReference:
    artifact = _load_json(path, f"{dataset}/{profile} norm reference")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("phase") != "6B.2"
        or artifact.get("dataset") != dataset
        or artifact.get("profile") != profile
        or artifact.get("contract_id") != config.contract_id
    ):
        raise DataContractError("Phase 6B.2 artifact scope or contract differs")
    calibration_id = _string(artifact, "calibration_id")
    stable_artifact = {
        key: value
        for key, value in artifact.items()
        if key not in {"calibration_id", "created_at_utc"}
    }
    if run_id(stable_artifact, length=64) != calibration_id:
        raise DataContractError("Phase 6B.2 calibration identity is invalid")

    gates = artifact.get("gates")
    required_true = (
        "all_scheduled_updates_accepted",
        "all_development_clients_honest",
        "cache_bound_to_schedule",
        "same_dataset_model_partition_and_training_contract",
        "reference_uses_prior_benign_development_updates_only",
    )
    required_false = (
        "current_round_benign_updates_accessed",
        "validation_partition_accessed",
        "test_partition_accessed",
        "anchor_records_accessed",
        "performance_claims_allowed",
    )
    if (
        not isinstance(gates, dict)
        or not all(gates.get(name) is True for name in required_true)
        or not all(gates.get(name) is False for name in required_false)
    ):
        raise DataContractError("Phase 6B.2 artifact evidence-boundary gates failed")

    detector = artifact.get("detector")
    local_training = artifact.get("local_training")
    if not isinstance(detector, dict) or not isinstance(local_training, dict):
        raise DataContractError("Phase 6B.2 model or local-training record is missing")
    detector_record = cast(dict[str, object], detector)
    local_record = cast(dict[str, object], local_training)
    detector_config_id = _string(artifact, "detector_config_id")
    local_training_config_id = _string(artifact, "local_training_config_id")
    if (
        run_id(detector_record, length=64) != detector_config_id
        or run_id(local_record, length=64) != local_training_config_id
        or detector_record.get("hidden_dimensions") != list(config.hidden_dimensions)
        or detector_record.get("dropout") != config.dropout
        or local_record.get("local_epochs") != config.local_epochs
        or local_record.get("learning_rate") != config.learning_rate
        or local_record.get("proximal_mu") != config.proximal_mu
        or local_record.get("maximum_class_weight") != config.maximum_class_weight
        or local_record.get("seed") != config.base_seed
        or local_record.get("optimizer") != "adam"
        or local_record.get("device") != artifact.get("device")
    ):
        raise DataContractError("Phase 6B.2 model or local-training identity differs")

    reference_value = artifact.get("reference")
    updates_value = artifact.get("update_norm_records")
    if not isinstance(reference_value, dict) or not isinstance(updates_value, list):
        raise DataContractError("Phase 6B.2 reference or update population is missing")
    reference_record = cast(dict[str, object], reference_value)
    update_records: list[dict[str, object]] = []
    for value in updates_value:
        if not isinstance(value, dict):
            raise DataContractError("Phase 6B.2 update-norm record is malformed")
        update_records.append(cast(dict[str, object], value))
    expected_updates = config.total_rounds * config.clients_per_round
    if len(update_records) != expected_updates:
        raise DataContractError("Phase 6B.2 update population is incomplete")
    round_counts: Counter[int] = Counter()
    round_clients: dict[int, set[str]] = defaultdict(set)
    development_ids: list[str] = []
    norms: list[float] = []
    for record in update_records:
        round_id_value = record.get("round_id")
        client_id = record.get("client_id")
        development_id = record.get("development_run_id")
        norm = record.get("update_norm")
        if (
            not isinstance(round_id_value, int)
            or not 0 <= round_id_value < config.total_rounds
            or not isinstance(client_id, str)
            or not isinstance(development_id, str)
            or len(development_id) != 64
            or not isinstance(norm, int | float)
        ):
            raise DataContractError("Phase 6B.2 update-norm record is invalid")
        round_counts[round_id_value] += 1
        round_clients[round_id_value].add(client_id)
        development_ids.append(development_id)
        norms.append(float(norm))
    if any(
        round_counts[round_id] != config.clients_per_round
        or len(round_clients[round_id]) != config.clients_per_round
        for round_id in range(config.total_rounds)
    ):
        raise DataContractError("Phase 6B.2 round schedule cardinality is invalid")

    reference_id = _string(reference_record, "reference_id")
    stored_reference = {
        key: value for key, value in reference_record.items() if key != "reference_id"
    }
    rebuilt = benign_norm_reference(
        dataset=dataset,
        profile=profile,
        preprocessed_corpus_id=_string(artifact, "preprocessed_corpus_id"),
        client_partition_id=_string(artifact, "client_partition_id"),
        detector_config_id=detector_config_id,
        local_training_config_id=local_training_config_id,
        development_run_ids=development_ids,
        update_norms=norms,
    )
    if (
        rebuilt.reference_id != reference_id
        or to_primitive(rebuilt) != stored_reference
        or reference_record.get("dataset") != dataset
        or reference_record.get("profile") != profile
    ):
        raise DataContractError("Phase 6B.2 norm reference cannot be reproduced")

    relative_path = path.relative_to(path.parents[2]).as_posix()
    inventory = {
        "dataset": dataset,
        "profile": profile,
        "path": relative_path,
        "byte_size": path.stat().st_size,
        "sha256": _sha256(path),
        "calibration_input_id": _string(artifact, "calibration_input_id"),
        "calibration_id": calibration_id,
        "reference_id": reference_id,
        "update_count": rebuilt.update_count,
        "lower_norm": rebuilt.lower_norm,
        "upper_norm": rebuilt.upper_norm,
    }
    return _AuditedReference(
        inventory=inventory,
        contract_id=_string(artifact, "contract_id"),
        schedule_id=_string(artifact, "schedule_id"),
        local_training_config_id=local_training_config_id,
        preprocessed_corpus_id=_string(artifact, "preprocessed_corpus_id"),
        client_partition_id=_string(artifact, "client_partition_id"),
        detector_config_id=detector_config_id,
        federated_run_id=_string(artifact, "federated_run_id"),
        calibration_input_id=_string(artifact, "calibration_input_id"),
        calibration_id=calibration_id,
        cache_id=_string(artifact, "cache_id"),
        reference_id=reference_id,
        development_run_ids=tuple(development_ids),
    )


def _unique(values: Sequence[str], expected: int, description: str) -> None:
    if len(set(values)) != expected:
        raise DataContractError(f"Phase 6B.2 {description} are not globally unique")


def audit_phase6b_norm_references(
    request: NormReferenceAuditRequest,
) -> dict[str, object]:
    """Audit all expected norm references and write one checksum-bound closure index."""

    audited: list[_AuditedReference] = []
    by_dataset: dict[str, list[_AuditedReference]] = defaultdict(list)
    for dataset, profile in request.expected_references:
        path = request.root_directory / dataset / profile / "benign_norm_reference.json"
        if not path.is_file():
            raise DataContractError(f"Missing Phase 6B.2 reference: {dataset}/{profile}")
        audited_reference = _audit_one(
            path, dataset=dataset, profile=profile, config=request.config
        )
        audited.append(audited_reference)
        by_dataset[dataset].append(audited_reference)

    expected_count = len(request.expected_references)
    _unique([item.calibration_input_id for item in audited], expected_count, "input identities")
    _unique([item.calibration_id for item in audited], expected_count, "calibration identities")
    _unique([item.cache_id for item in audited], expected_count, "cache identities")
    _unique([item.reference_id for item in audited], expected_count, "reference identities")
    all_development_ids = [identity for item in audited for identity in item.development_run_ids]
    _unique(
        all_development_ids,
        expected_count * request.config.total_rounds * request.config.clients_per_round,
        "development-update identities",
    )
    if (
        {item.contract_id for item in audited} != {request.config.contract_id}
        or len({item.schedule_id for item in audited}) != 1
        or len({item.local_training_config_id for item in audited}) != 1
    ):
        raise DataContractError("Phase 6B.2 shared contract or training controls differ")
    engine_ids_by_dataset: dict[str, str] = {}
    for dataset, items in by_dataset.items():
        if (
            len({item.preprocessed_corpus_id for item in items}) != 1
            or len({item.client_partition_id for item in items}) != 1
            or len({item.detector_config_id for item in items}) != 1
            or len({item.federated_run_id for item in items}) != 1
        ):
            raise DataContractError("Phase 6B.2 dataset-scoped inputs differ across profiles")
        engine_ids_by_dataset[dataset] = items[0].federated_run_id

    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "6B.2-closure",
        "contract_id": request.config.contract_id,
        "expected_reference_count": expected_count,
        "total_development_updates": len(all_development_ids),
        "federated_engine_run_ids_by_dataset": dict(sorted(engine_ids_by_dataset.items())),
        "references": [item.inventory for item in audited],
        "gates": {
            "all_expected_dataset_profiles_present": True,
            "all_artifact_and_reference_identities_recomputed": True,
            "all_reference_file_hashes_recorded": True,
            "all_reference_quartiles_recomputed": True,
            "all_scientific_identities_unique": True,
            "all_update_populations_complete": True,
            "dataset_model_and_partition_consistent_across_profiles": True,
            "shared_schedule_and_local_training_contract_consistent": True,
            "federated_engine_ids_scoped_by_calibration_directories": True,
            "current_round_benign_updates_accessed": False,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "anchor_records_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    closure_id = run_id(stable, length=64)
    if request.output_path.exists():
        existing = _load_json(request.output_path, "Phase 6B.2 closure index")
        existing_stable = {
            key: value
            for key, value in existing.items()
            if key not in {"closure_id", "created_at_utc"}
        }
        if existing.get("closure_id") != closure_id or existing_stable != stable:
            raise DataContractError("Existing Phase 6B.2 closure index differs")
        return existing
    closure: dict[str, object] = {
        "closure_id": closure_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **stable,
    }
    _write_json(request.output_path, closure)
    return closure
