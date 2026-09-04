"""Integration tests for the resumable train-only Phase 6B resolver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from semcred_fl.attacks import resolution as resolution_module
from semcred_fl.attacks.resolution import (
    Phase6BResolutionRequest,
    build_phase6b_resolution,
)
from semcred_fl.config import DatasetName

IDENTITY = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_artifacts(root: Path) -> tuple[Path, Path]:
    coordinates = [
        {
            "coordinate": f"x{index:04d}",
            "kind": "continuous_standardized",
            "source_feature": f"feature-{index}",
            "semantic_group": f"feature-{index}",
            "category": None,
        }
        for index in range(12)
    ]
    continuous = {
        f"feature-{index}": {
            "clip_lower": -100.0,
            "clip_upper": 100.0,
            "mean_after_clipping": 0.0,
            "scale": 1.0,
        }
        for index in range(12)
    }
    preprocessor = {
        "preprocessor_id": IDENTITY,
        "dataset": DatasetName.EDGE_IIOTSET.value,
        "split_id": IDENTITY,
        "anchor_id": IDENTITY,
        "model_input_dimension": 12,
        "family_label_to_id": {"benign": 0, "other-1": 1, "other-2": 2, "injection": 3},
        "coordinates": coordinates,
        "continuous_features": continuous,
    }
    preprocessor_path = root / "preprocessor.json"
    _write_json(preprocessor_path, preprocessor)

    partitions = ["train"] * 8 + ["validation", "test"]
    families = [0, 0, 0, 0, 3, 3, 1, 2, 0, 3]
    data: dict[str, object] = {
        "record_id": [f"row-{index}" for index in range(10)],
        "partition": partitions,
        "family_id": families,
    }
    for index in range(12):
        multiplier = float(index + 1)
        values = [0.0, 0.2, 0.4, 0.6, multiplier, multiplier + 0.5, 0.1, 0.3]
        values.extend([9999.0, -9999.0])
        data[f"x{index:04d}"] = values
    table = pa.table(data)
    shard_path = root / "shard.parquet"
    pq.write_table(table, shard_path)
    report = {
        "dataset": DatasetName.EDGE_IIOTSET.value,
        "preprocessed_corpus_id": IDENTITY,
        "split_id": IDENTITY,
        "anchor_id": IDENTITY,
        "records": 10,
        "partition_counts": {
            "train": 8,
            "validation": 1,
            "test": 1,
            "anchor_calibration": 0,
            "anchor_scoring": 0,
        },
        "preprocessor": {
            "path": "preprocessor.json",
            "sha256": _sha256(preprocessor_path),
            "preprocessor_id": IDENTITY,
        },
        "transformed_files": [
            {
                "path": "shard.parquet",
                "row_count": 10,
                "byte_size": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
            }
        ],
        "gates": {
            "fit_cardinality_matches_federated_training_pool": True,
            "fit_excludes_validation_test_and_anchors": True,
            "no_nonfinite_model_input": True,
            "partition_cardinality_matches_phase3b": True,
            "preprocessing_fitted": True,
            "stable_feature_order_and_dimension": True,
            "transformed_cardinality_matches_canonical": True,
        },
    }
    report_path = root / "preprocessed.json"
    _write_json(report_path, report)
    manifest = {
        "dataset": DatasetName.EDGE_IIOTSET.value,
        "preprocessed_corpus_id": IDENTITY,
        "preprocessor_id": IDENTITY,
        "split_id": IDENTITY,
        "anchor_id": IDENTITY,
        "client_partition_id": IDENTITY,
        "training_records": 8,
        "family_training_counts": {
            "benign": 4,
            "other-1": 1,
            "other-2": 1,
            "injection": 2,
        },
        "gates": {
            "all_client_ids_in_range": True,
            "client_partition_created": True,
            "every_training_record_assigned_once_per_profile": True,
            "family_cardinality_conserved": True,
            "minimum_client_size_satisfied": True,
            "no_record_duplicated_within_a_profile": True,
            "only_federated_training_records_assigned": True,
            "quota_matrices_match_plan": True,
            "validation_test_and_anchors_excluded": True,
            "model_training_performed": False,
            "transformed_features_duplicated": False,
        },
    }
    manifest_path = root / "clients.json"
    _write_json(manifest_path, manifest)
    return report_path, manifest_path


def test_full_resolver_is_train_only_deterministic_and_resumable(
    tmp_path: Path, monkeypatch: object
) -> None:
    report_path, manifest_path = _fixture_artifacts(tmp_path)
    original = resolution_module._profile(DatasetName.EDGE_IIOTSET)
    fixture_profile = replace(
        original,
        split_id=IDENTITY,
        preprocessor_id=IDENTITY,
        minimum_training_records=1,
        observed_federated_training_records=2,
    )
    monkeypatch.setattr(resolution_module, "_profile", lambda dataset: fixture_profile)
    output = tmp_path / "output"
    request = Phase6BResolutionRequest(
        dataset=DatasetName.EDGE_IIOTSET,
        preprocessing_report=report_path,
        client_partition_manifest=manifest_path,
        output_directory=output,
        work_database=output / "resume.sqlite3",
        batch_size=3,
    )
    first = build_phase6b_resolution(request)
    second = build_phase6b_resolution(request)
    assert first == second
    assert first["resolution_id"]
    assert first["source_training_records"] == 2
    assert first["target_training_records"] == 4
    assert first["trigger_training_collisions"] == 0
    assert first["gates"]["scanner_filter_excludes_validation_test_and_anchors"]
    ranked = first["ranked_coordinates"]
    assert ranked[0]["coordinate"] == "x0011"
    assert first["feature_coordinates"] == ranked[:3]
    assert not first["test_partition_accessed"]
    assert not first["validation_partition_accessed"]
