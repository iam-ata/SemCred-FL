"""Tests for the frozen dataset intake contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from semcred_fl.data.ciciot2023 import CIC_SOURCE_DIRECTORY_TO_FINE
from semcred_fl.data.preprocessing import CIC_PREPARATION_POLICY, EDGE_PREPARATION_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"


def _load(name: str) -> dict[str, Any]:
    loaded = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_contract_validator_accepts_frozen_configs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_dataset_contracts.py"),
            "--config-dir",
            str(CONFIG_DIR),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dataset_contracts=2" in result.stdout


def test_every_fine_label_has_exactly_one_family() -> None:
    for name in ("ciciot2023.yaml", "edge_iiotset.yaml"):
        contract = _load(name)
        labels = contract["labels"]
        assert set(labels["fine_labels"]) == set(labels["fine_to_family"])
        assert set(labels["fine_to_family"].values()) <= set(labels["family_labels"])


def test_archive_eligibility_matches_the_frozen_evidence_boundary() -> None:
    cic = _load("ciciot2023.yaml")
    edge = _load("edge_iiotset.yaml")

    assert cic["local_archive"]["confirmatory_eligible"] is False
    assert edge["local_archive"]["confirmatory_eligible"] is True
    assert (
        edge["local_archive"]["eligibility_scope"]
        == "primary_family_with_frozen_capture_and_mitm_flow_claim_boundary"
    )


def test_cic_derivative_and_edge_ml_subset_are_restricted() -> None:
    cic = _load("ciciot2023.yaml")
    edge = _load("edge_iiotset.yaml")
    assert cic["quality_gates"]["allow_smoke_input_for_confirmatory_runs"] is False
    assert edge["quality_gates"]["allow_ml_subset_as_independent_experiment_corpus"] is False


def test_cic_official_source_is_present_but_not_yet_confirmatory() -> None:
    cic = _load("ciciot2023.yaml")
    source = cic["official_source_archive"]
    labels = cic["labels"]
    assert source["archive_file_entries"] == 310
    assert source["extracted_csv_files"] == 309
    assert source["sha256"] == "b18c8b58ce8c2ee5f058eb72ffad1f2f9ee40afe89b6d3b53a8f3189551cfe5d"
    assert source["confirmatory_eligible"] is False
    assert len(labels["source_directory_to_fine"]) == 34
    assert set(labels["source_directory_to_fine"].values()) == set(labels["fine_labels"])
    assert labels["source_directory_to_fine"] == CIC_SOURCE_DIRECTORY_TO_FINE


def test_yaml_phase2c_evidence_and_feature_partitions_match_typed_policies() -> None:
    for name, policy in (
        ("ciciot2023.yaml", CIC_PREPARATION_POLICY),
        ("edge_iiotset.yaml", EDGE_PREPARATION_POLICY),
    ):
        contract = _load(name)
        evidence = contract["phase_2c_evidence"]
        preparation = contract["preparation_policy"]

        assert preparation["policy_id"] == policy.policy_id
        assert evidence["phase_2b_audit_id"] == policy.evidence.phase_2b_audit_id
        assert evidence["phase_2c_analysis_id"] == policy.evidence.phase_2c_analysis_id
        assert evidence["valid_rows"] == policy.evidence.valid_rows
        assert evidence["raw_unique_fingerprints"] == policy.evidence.raw_unique_fingerprints
        assert (
            evidence["raw_primary_consistent_unique_fingerprints"]
            == policy.evidence.raw_primary_consistent_unique_fingerprints
        )
        assert tuple(preparation["categorical_columns"]) == policy.features.categorical_columns
        assert tuple(preparation["continuous_columns"]) == policy.features.continuous_columns
        assert tuple(preparation["dropped_columns"]) == policy.features.dropped_columns


def test_conflict_and_deduplication_gates_are_mandatory() -> None:
    for name in ("ciciot2023.yaml", "edge_iiotset.yaml"):
        gates = _load(name)["quality_gates"]
        assert gates["require_primary_conflict_component_exclusion"] is True
        assert gates["require_fine_conflict_exclusion_from_secondary_analysis"] is True
        assert gates["require_canonical_deduplication_before_splitting"] is True
