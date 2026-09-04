"""Shared deterministic fixtures for repository tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class Phase4CorpusFixture:
    """Paths for a tiny valid Phase 3C/3D artifact pair."""

    preprocessing_report: Path
    client_partition_manifest: Path


@pytest.fixture
def phase4_corpus(tmp_path: Path) -> Phase4CorpusFixture:
    """Create a small linearly separable transformed corpus."""

    artifact = tmp_path / "preprocessing"
    shards = artifact / "shards"
    shards.mkdir(parents=True)
    rng = np.random.default_rng(20260816)
    partitions = (
        ["train"] * 80
        + ["validation"] * 40
        + ["test"] * 20
        + ["anchor_calibration"] * 4
        + ["anchor_scoring"] * 4
    )
    family_ids = np.asarray(
        [index % 2 for index in range(80)]
        + [index % 2 for index in range(40)]
        + [index % 2 for index in range(20)]
        + [0, 1, 0, 1]
        + [0, 1, 0, 1],
        dtype=np.int16,
    )
    centers = np.where(family_ids[:, None] == 0, -1.0, 1.0)
    features = centers + rng.normal(0.0, 0.12, size=(len(partitions), 4))
    table = pa.table(
        {
            "record_id": pa.array(
                [f"record-{index:04d}" for index in range(len(partitions))], type=pa.string()
            ),
            "partition": pa.array(partitions, type=pa.string()),
            "family_id": pa.array(family_ids, type=pa.int16()),
            **{
                f"x{index:04d}": pa.array(features[:, index], type=pa.float32())
                for index in range(4)
            },
        }
    )
    shard = shards / "0000-preprocessed.parquet"
    pq.write_table(table, shard)
    preprocessor = {
        "schema_version": 1,
        "dataset": "edge_iiotset",
        "split_id": "split-test",
        "anchor_id": "anchor-test",
        "preprocessor_id": "preprocessor-test",
        "model_input_dimension": 4,
        "family_label_to_id": {"benign": 0, "attack": 1},
    }
    preprocessor_path = artifact / "preprocessor.json"
    _write_json(preprocessor_path, preprocessor)
    report = {
        "schema_version": 1,
        "dataset": "edge_iiotset",
        "preprocessed_corpus_id": "corpus-test",
        "split_id": "split-test",
        "anchor_id": "anchor-test",
        "records": len(partitions),
        "partition_counts": {
            "train": 80,
            "validation": 40,
            "test": 20,
            "anchor_calibration": 4,
            "anchor_scoring": 4,
        },
        "preprocessor": {
            "path": "preprocessor.json",
            "preprocessor_id": "preprocessor-test",
            "sha256": _sha256(preprocessor_path),
        },
        "transformed_files": [
            {
                "path": "shards/0000-preprocessed.parquet",
                "row_count": len(partitions),
                "byte_size": shard.stat().st_size,
                "sha256": _sha256(shard),
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
    report_path = artifact / "edge_iiotset_phase3c_preprocessed.json"
    _write_json(report_path, report)
    assignment_directory = tmp_path / "assignments"
    assignment_directory.mkdir()
    assignment_path = assignment_directory / "0000-clients.parquet"
    training_client_ids = np.asarray([(index // 2) % 2 for index in range(80)], dtype=np.int16)
    pq.write_table(
        pa.table(
            {
                "record_id": pa.array(
                    [f"record-{index:04d}" for index in range(80)], type=pa.string()
                ),
                "family_id": pa.array(family_ids[:80], type=pa.int16()),
                "iid_client_id": pa.array(training_client_ids, type=pa.int16()),
            }
        ),
        assignment_path,
    )
    client_manifest = {
        "client_partition_id": "clients-test",
        "dataset": "edge_iiotset",
        "preprocessed_corpus_id": "corpus-test",
        "preprocessor_id": "preprocessor-test",
        "split_id": "split-test",
        "anchor_id": "anchor-test",
        "training_records": 80,
        "num_clients": 2,
        "family_training_counts": {"benign": 40, "attack": 40},
        "assignment_columns": {"iid": "iid_client_id"},
        "assignment_files": [
            {
                "path": "assignments/0000-clients.parquet",
                "row_count": 80,
                "byte_size": assignment_path.stat().st_size,
                "sha256": _sha256(assignment_path),
            }
        ],
        "profiles": {
            "iid": {
                "client_totals": [40, 40],
                "family_client_counts": {"benign": [20, 20], "attack": [20, 20]},
            }
        },
        "gates": {
            "all_client_ids_in_range": True,
            "client_partition_created": True,
            "every_training_record_assigned_once_per_profile": True,
            "family_cardinality_conserved": True,
            "minimum_client_size_satisfied": True,
            "model_training_performed": False,
            "no_record_duplicated_within_a_profile": True,
            "only_federated_training_records_assigned": True,
            "quota_matrices_match_plan": True,
            "transformed_features_duplicated": False,
            "validation_test_and_anchors_excluded": True,
        },
    }
    client_path = tmp_path / "edge_iiotset_phase3d_clients.json"
    _write_json(client_path, client_manifest)
    return Phase4CorpusFixture(report_path, client_path)
