"""Validate the frozen Phase 6A contract against completed Phase 3 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.contracts import (  # noqa: E402
    DatasetAttackProfile,
    phase6a_contract,
)
from semcred_fl.identity import run_id, to_primitive  # noqa: E402


def _object(value: object, *, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} must be a JSON/YAML object")
    return cast(dict[str, object], value)


def _integer(value: object, *, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    return value


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _read_json(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), description=str(path))


def _artifact_paths(project_root: Path, profile: DatasetAttackProfile) -> tuple[Path, Path, Path]:
    dataset = profile.dataset.value
    split_root = project_root / "artifacts" / "splits" / "phase_3b" / dataset
    return (
        split_root / f"{dataset}_phase3b_split.json",
        split_root / "anchor_manifest.json",
        project_root / "artifacts" / "preprocessing" / "phase_3c" / dataset / "preprocessor.json",
    )


def _validate_profile(project_root: Path, profile: DatasetAttackProfile) -> dict[str, object]:
    split_path, anchor_path, preprocessor_path = _artifact_paths(project_root, profile)
    split = _read_json(split_path)
    anchors = _read_json(anchor_path)
    preprocessor = _read_json(preprocessor_path)

    if _text(split.get("split_id"), description="split_id") != profile.split_id:
        raise ValueError(f"{profile.dataset.value}: split ID mismatch")
    if _text(preprocessor.get("preprocessor_id"), description="preprocessor_id") != (
        profile.preprocessor_id
    ):
        raise ValueError(f"{profile.dataset.value}: preprocessor ID mismatch")

    label_ids = _object(preprocessor.get("family_label_to_id"), description="family_label_to_id")
    if _integer(label_ids.get(profile.source_family), description="source family ID") != (
        profile.source_family_id
    ):
        raise ValueError(f"{profile.dataset.value}: source family ID mismatch")
    if _integer(label_ids.get(profile.target_family), description="target family ID") != (
        profile.target_family_id
    ):
        raise ValueError(f"{profile.dataset.value}: target family ID mismatch")

    family_splits = _object(split.get("family_split_counts"), description="family_split_counts")
    anchor_counts = _object(anchors.get("family_counts"), description="anchor family_counts")
    eligible_counts: dict[str, int] = {}
    for family, raw_counts in family_splits.items():
        if family == "benign":
            continue
        counts = _object(raw_counts, description=f"split counts for {family}")
        train_count = _integer(counts.get("train"), description=f"train count for {family}")
        anchor_count = _integer(anchor_counts.get(family), description=f"anchor count for {family}")
        available = train_count - anchor_count
        if available >= profile.minimum_training_records:
            eligible_counts[family] = available
    if not eligible_counts:
        raise ValueError(f"{profile.dataset.value}: no source family meets the training floor")
    selected_family, selected_count = min(
        eligible_counts.items(), key=lambda item: (item[1], item[0])
    )
    if selected_family != profile.source_family:
        raise ValueError(
            f"{profile.dataset.value}: expected train-only source {selected_family}, "
            f"found {profile.source_family}"
        )
    if selected_count != profile.observed_federated_training_records:
        raise ValueError(f"{profile.dataset.value}: frozen source record count mismatch")

    return {
        "dataset": profile.dataset.value,
        "split_id": profile.split_id,
        "preprocessor_id": profile.preprocessor_id,
        "source_family": selected_family,
        "target_family": profile.target_family,
        "federated_training_source_records": selected_count,
        "test_partition_accessed": False,
        "status": "passed",
    }


def validate(project_root: Path, contract_path: Path) -> dict[str, object]:
    """Return a deterministic validation report or raise on any mismatch."""

    contract = phase6a_contract()
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    expected = to_primitive(contract)
    if loaded != expected:
        raise ValueError("The YAML contract differs from the typed Phase 6A contract")
    profile_reports = [_validate_profile(project_root, profile) for profile in contract.profiles]
    return {
        "schema_version": 1,
        "phase": "6A",
        "contract_id": run_id(contract, length=64),
        "status": "complete",
        "profile_checks": profile_reports,
        "resolution_artifacts_pending": [
            "clean_label_feature_resolution",
            "backdoor_trigger_resolution",
            "benign_update_norm_reference",
        ],
        "attack_execution_ready": False,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/attacks/phase6a.yaml"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    contract_path = args.contract
    if not contract_path.is_absolute():
        contract_path = project_root / contract_path
    report = validate(project_root, contract_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
