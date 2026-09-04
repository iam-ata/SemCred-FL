"""Validate static SemCred-FL dataset intake contracts without reading large data files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a list of strings")
    return value


def validate_contract(path: Path) -> str:
    """Validate one YAML contract and return its dataset identifier."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(document, str(path))
    if root.get("schema_version") != 2:
        raise ValueError(f"{path}: schema_version must be 2")

    dataset = _mapping(root.get("dataset"), f"{path}: dataset")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"{path}: dataset.id must be a non-empty string")

    archive = _mapping(root.get("local_archive"), f"{path}: local_archive")
    sha256 = archive.get("sha256")
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{path}: local_archive.sha256 must be lowercase SHA-256")
    if not isinstance(archive.get("byte_size"), int) or archive["byte_size"] <= 0:
        raise ValueError(f"{path}: local_archive.byte_size must be positive")
    if not isinstance(archive.get("confirmatory_eligible"), bool):
        raise ValueError(f"{path}: local_archive.confirmatory_eligible must be boolean")

    labels = _mapping(root.get("labels"), f"{path}: labels")
    fine_labels = _string_list(labels.get("fine_labels"), f"{path}: labels.fine_labels")
    family_labels = _string_list(labels.get("family_labels"), f"{path}: labels.family_labels")
    label_map = _mapping(labels.get("fine_to_family"), f"{path}: labels.fine_to_family")

    if len(fine_labels) != len(set(fine_labels)):
        raise ValueError(f"{path}: fine labels must be unique")
    if len(family_labels) != len(set(family_labels)):
        raise ValueError(f"{path}: family labels must be unique")
    if set(label_map) != set(fine_labels):
        missing = sorted(set(fine_labels) - set(label_map))
        extra = sorted(set(label_map) - set(fine_labels))
        raise ValueError(f"{path}: label-map mismatch; missing={missing}, extra={extra}")

    unknown_families = sorted(set(label_map.values()) - set(family_labels))
    if unknown_families:
        raise ValueError(f"{path}: unknown mapped families: {unknown_families}")

    source_map_value = labels.get("source_directory_to_fine")
    if source_map_value is not None:
        source_map = _mapping(source_map_value, f"{path}: labels.source_directory_to_fine")
        unknown_source_labels = sorted(set(source_map.values()) - set(fine_labels))
        if unknown_source_labels:
            raise ValueError(
                f"{path}: source directories map to unknown labels: {unknown_source_labels}"
            )
        if len(set(source_map.values())) != len(fine_labels):
            raise ValueError(f"{path}: every fine label must have one source directory")

    evidence = _mapping(root.get("phase_2c_evidence"), f"{path}: phase_2c_evidence")
    for key in ("phase_2b_audit_id", "phase_2c_analysis_id"):
        identifier = evidence.get(key)
        if not isinstance(identifier, str) or SHA256_PATTERN.fullmatch(identifier) is None:
            raise ValueError(f"{path}: phase_2c_evidence.{key} must be a full identifier")
    if evidence.get("complete") is not True:
        raise ValueError(f"{path}: Phase 2C evidence must be complete")
    count_keys = (
        "valid_rows",
        "raw_unique_fingerprints",
        "primary_conflict_fingerprints",
        "primary_conflict_occurrences",
        "fine_only_conflict_fingerprints",
        "fine_only_conflict_occurrences",
        "raw_primary_consistent_unique_fingerprints",
    )
    if any(not isinstance(evidence.get(key), int) or evidence[key] < 0 for key in count_keys):
        raise ValueError(f"{path}: Phase 2C evidence counts must be non-negative integers")
    expected_consistent = (
        evidence["raw_unique_fingerprints"] - evidence["primary_conflict_fingerprints"]
    )
    if evidence["raw_primary_consistent_unique_fingerprints"] != expected_consistent:
        raise ValueError(f"{path}: inconsistent Phase 2C primary-consistent count")

    preparation = _mapping(root.get("preparation_policy"), f"{path}: preparation_policy")
    policy_id = preparation.get("policy_id")
    if not isinstance(policy_id, str) or SHA256_PATTERN.fullmatch(policy_id) is None:
        raise ValueError(f"{path}: preparation_policy.policy_id must be a full identifier")
    if preparation.get("fingerprint_stage") != "after_validation_and_canonicalization":
        raise ValueError(f"{path}: final fingerprints must follow canonicalization")
    required_actions = {
        "primary_conflict_action": "exclude_entire_component",
        "fine_only_conflict_action": "keep_for_primary_exclude_from_fine_analysis",
        "duplicate_action": "keep_one_canonical_representative",
    }
    for key, expected in required_actions.items():
        if preparation.get(key) != expected:
            raise ValueError(f"{path}: invalid {key}")
    if preparation.get("representative_order") != ["source_file", "source_row"]:
        raise ValueError(f"{path}: duplicate representative order is not canonical")
    categorical = _string_list(
        preparation.get("categorical_columns"), f"{path}: categorical_columns"
    )
    continuous = _string_list(preparation.get("continuous_columns"), f"{path}: continuous_columns")
    dropped = _string_list(preparation.get("dropped_columns"), f"{path}: dropped_columns")
    prepared_columns = categorical + continuous + dropped
    if len(prepared_columns) != len(set(prepared_columns)):
        raise ValueError(f"{path}: prepared feature partitions must be disjoint")
    domains_value = preparation.get("closed_category_domains")
    if domains_value is None:
        if categorical:
            raise ValueError(f"{path}: categorical columns require closed raw domains")
    elif set(_mapping(domains_value, f"{path}: closed_category_domains")) != set(categorical):
        raise ValueError(f"{path}: closed category domains must match categorical columns")

    quality_gates = _mapping(root.get("quality_gates"), f"{path}: quality_gates")
    required_true = (
        "require_source_hash",
        "require_group_disjoint_splits",
        "require_exact_duplicate_components_within_one_split",
        "require_primary_conflict_component_exclusion",
        "require_fine_conflict_exclusion_from_secondary_analysis",
        "require_canonical_deduplication_before_splitting",
        "fit_preprocessor_on_federated_train_only",
    )
    disabled = [name for name in required_true if quality_gates.get(name) is not True]
    if disabled:
        raise ValueError(f"{path}: mandatory quality gates disabled: {disabled}")

    return dataset_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/datasets"),
        help="directory containing dataset YAML contracts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(args.config_dir.glob("*.yaml"))
    if not paths:
        raise SystemExit(f"no dataset contracts found in {args.config_dir}")

    dataset_ids = [validate_contract(path) for path in paths]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise SystemExit("dataset IDs must be unique")

    print(f"valid dataset_contracts={len(paths)} ids={','.join(sorted(dataset_ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
