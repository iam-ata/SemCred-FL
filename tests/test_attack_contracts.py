"""Tests for the frozen Phase 6A attack definitions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from semcred_fl.attacks.contracts import (
    AdaptiveNormPolicy,
    AttackContractError,
    CoordinationPolicy,
    DataPoisoningPolicy,
    DatasetAttackProfile,
    FeaturePoisoningPolicy,
    Phase6AContract,
    TriggerPolicy,
    phase6a_contract,
)
from semcred_fl.config import DatasetName
from semcred_fl.identity import run_id, to_primitive

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase6a_yaml_exactly_matches_typed_contract() -> None:
    loaded = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "attacks" / "phase6a.yaml").read_text(encoding="utf-8")
    )
    contract = phase6a_contract()
    assert loaded == to_primitive(contract)
    assert run_id(contract, length=64) == run_id(loaded, length=64)


def test_phase6a_validation_artifact_matches_contract() -> None:
    report = json.loads(
        (
            PROJECT_ROOT / "artifacts" / "attacks" / "phase_6a" / "phase6a_contract_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["contract_id"] == run_id(phase6a_contract(), length=64)
    assert report["status"] == "complete"
    assert not report["attack_execution_ready"]
    assert not report["test_partition_accessed"]
    assert not report["performance_claims_allowed"]


def test_dataset_sources_are_train_only_minority_families() -> None:
    profiles = {profile.dataset: profile for profile in phase6a_contract().profiles}
    edge = profiles[DatasetName.EDGE_IIOTSET]
    cic = profiles[DatasetName.CIC_IOT_2023]
    assert (edge.source_family, edge.target_family) == ("injection", "benign")
    assert edge.observed_federated_training_records == 46_519
    assert (cic.source_family, cic.target_family) == ("web_based", "benign")
    assert cic.observed_federated_training_records == 17_060


def test_data_policy_has_exact_clean_no_op_and_sealed_partitions() -> None:
    policy = DataPoisoningPolicy()
    assert policy.no_op_fraction == 0.0
    assert policy.allowed_partition == "federated_train_after_anchor_removal"
    with pytest.raises(AttackContractError, match="anchors, validation, or test"):
        replace(policy, allowed_partition="test")


def test_clean_label_and_trigger_resolution_are_training_only() -> None:
    feature_policy = FeaturePoisoningPolicy()
    assert feature_policy.preserve_labels
    assert feature_policy.resolution_partition == "federated_train_after_anchor_removal"
    with pytest.raises(AttackContractError, match="preserve labels"):
        replace(feature_policy, preserve_labels=False)

    trigger = TriggerPolicy()
    assert trigger.training_collision_requirement == 0
    assert trigger.triggered_test_source == "clean_test_source_family_rows"
    with pytest.raises(AttackContractError, match="zero clean-training collisions"):
        replace(trigger, training_collision_requirement=1)


def test_coalition_and_adaptive_attacker_cannot_read_defense_private_data() -> None:
    coordination = CoordinationPolicy()
    assert not coordination.may_observe_current_benign_updates
    assert not coordination.may_access_anchor_records
    with pytest.raises(AttackContractError, match="exceeds"):
        replace(coordination, may_access_anchor_records=True)

    adaptive = AdaptiveNormPolicy()
    assert adaptive.reference_artifact_required
    assert not adaptive.current_round_benign_updates_visible
    with pytest.raises(AttackContractError, match="cannot see"):
        replace(adaptive, current_round_benign_updates_visible=True)


def test_contract_rejects_test_access_or_performance_claims() -> None:
    contract = phase6a_contract()
    with pytest.raises(AttackContractError, match="cannot access test data"):
        replace(contract, test_partition_accessed=True)
    with pytest.raises(AttackContractError, match="cannot access test data"):
        replace(contract, performance_claims_allowed=True)


def test_profile_rejects_nonbenign_target() -> None:
    profile = phase6a_contract().profiles[0]
    with pytest.raises(AttackContractError, match="benign class"):
        replace(profile, target_family="malware", target_family_id=4)


def test_contract_requires_both_supported_datasets() -> None:
    profile = phase6a_contract().profiles[0]
    with pytest.raises(AttackContractError, match="one profile"):
        Phase6AContract(profiles=(profile,))


def test_source_profile_requires_training_floor() -> None:
    with pytest.raises(AttackContractError, match="training floor"):
        DatasetAttackProfile(
            dataset=DatasetName.EDGE_IIOTSET,
            split_id="split",
            preprocessor_id="prep",
            source_family="injection",
            source_family_id=3,
            target_family="benign",
            target_family_id=0,
            source_selection_rule="rule",
            minimum_training_records=10_000,
            observed_federated_training_records=9_999,
        )
