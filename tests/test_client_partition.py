"""Tests for exact, resumable Phase 3D client-partition manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data import client_partition as phase3d
from semcred_fl.data.client_partition import (
    ClientPartitionBuildRequest,
    build_phase3d_clients,
    plan_phase3d_clients,
)
from semcred_fl.data.contracts import DataContractError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[ClientPartitionBuildRequest, set[str]]:
    phase3c = tmp_path / "phase3c"
    split_root = tmp_path / "phase3b"
    shard_root = phase3c / "shards"
    shard_root.mkdir(parents=True)
    split_root.mkdir(parents=True)
    split_id = hashlib.sha256(b"split").hexdigest()
    anchor_id = hashlib.sha256(b"anchor").hexdigest()
    preprocessor_id = hashlib.sha256(b"preprocessor").hexdigest()
    corpus_id = hashlib.sha256(b"phase3c-corpus").hexdigest()
    preprocessor = {
        "preprocessor_id": preprocessor_id,
        "family_label_to_id": {"attack": 0, "benign": 1},
        "model_input_dimension": 2,
    }
    preprocessor_path = phase3c / "preprocessor.json"
    _write_json(preprocessor_path, preprocessor)

    partitions = (
        ["train"] * 32
        + ["validation"] * 2
        + ["test"] * 2
        + ["anchor_calibration"] * 2
        + ["anchor_scoring"] * 2
    )
    family_ids = [index % 2 for index in range(40)]
    record_ids = [hashlib.sha256(f"record-{index}".encode()).hexdigest() for index in range(40)]
    rows = [
        {
            "record_id": record_id,
            "partition": partition,
            "family_id": family_id,
            "fine_id": family_id,
            "fine_label_ambiguous": False,
            "x0000": float(index),
            "x0001": float(-index),
        }
        for index, (record_id, partition, family_id) in enumerate(
            zip(record_ids, partitions, family_ids, strict=True)
        )
    ]
    transformed_files: list[dict[str, object]] = []
    for index, part in enumerate((rows[:20], rows[20:])):
        path = shard_root / f"{index:04d}-preprocessed.parquet"
        pq.write_table(pa.Table.from_pylist(part), path, compression="zstd")
        transformed_files.append(
            {
                "path": f"shards/{path.name}",
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
                "row_count": len(part),
            }
        )
    preprocessing_report = phase3c / "edge_iiotset_phase3c_preprocessed.json"
    _write_json(
        preprocessing_report,
        {
            "dataset": DatasetName.EDGE_IIOTSET.value,
            "preprocessed_corpus_id": corpus_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
            "partition_counts": {
                "train": 32,
                "validation": 2,
                "test": 2,
                "anchor_calibration": 2,
                "anchor_scoring": 2,
            },
            "preprocessor": {
                "path": preprocessor_path.name,
                "byte_size": preprocessor_path.stat().st_size,
                "sha256": _sha256(preprocessor_path),
                "preprocessor_id": preprocessor_id,
            },
            "transformed_files": transformed_files,
            "gates": {
                "fit_excludes_validation_test_and_anchors": True,
                "fit_cardinality_matches_federated_training_pool": True,
                "transformed_cardinality_matches_canonical": True,
                "partition_cardinality_matches_phase3b": True,
                "stable_feature_order_and_dimension": True,
                "no_nonfinite_model_input": True,
                "preprocessing_fitted": True,
                "client_partition_created": False,
            },
        },
    )

    anchors = [
        {
            "record_id": record_ids[36],
            "family_label": "attack",
            "subset": "calibration",
        },
        {
            "record_id": record_ids[37],
            "family_label": "benign",
            "subset": "calibration",
        },
        {
            "record_id": record_ids[38],
            "family_label": "attack",
            "subset": "scoring",
        },
        {
            "record_id": record_ids[39],
            "family_label": "benign",
            "subset": "scoring",
        },
    ]
    anchor_path = split_root / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(anchors), anchor_path)
    anchor_manifest = split_root / "anchor_manifest.json"
    _write_json(
        anchor_manifest,
        {
            "dataset": DatasetName.EDGE_IIOTSET.value,
            "anchor_id": anchor_id,
            "record_ids_sha256": hashlib.sha256(
                "\n".join(sorted(row["record_id"] for row in anchors)).encode()
            ).hexdigest(),
            "family_counts": {"attack": 2, "benign": 2},
            "subset_counts": {"calibration": 2, "scoring": 2},
            "file": {
                "path": anchor_path.name,
                "byte_size": anchor_path.stat().st_size,
                "sha256": _sha256(anchor_path),
            },
        },
    )
    split_manifest = split_root / "edge_iiotset_phase3b_split.json"
    _write_json(
        split_manifest,
        {
            "dataset": DatasetName.EDGE_IIOTSET.value,
            "base_seed": 20260816,
            "split_id": split_id,
            "family_split_counts": {
                "attack": {"train": 18, "validation": 1, "test": 1},
                "benign": {"train": 18, "validation": 1, "test": 1},
            },
        },
    )
    output = tmp_path / "output"
    request = ClientPartitionBuildRequest(
        dataset=DatasetName.EDGE_IIOTSET,
        preprocessing_report=preprocessing_report,
        split_manifest=split_manifest,
        anchor_manifest=anchor_manifest,
        output_directory=output,
        work_database=output / "resume.sqlite3",
        num_clients=4,
        minimum_samples_per_client=2,
        maximum_partition_attempts=5_000,
        parquet_batch_size=7,
    )
    expected_train = {
        record_id
        for record_id, partition in zip(record_ids, partitions, strict=True)
        if partition == "train"
    }
    return request, expected_train


def test_plan_and_build_conserve_every_profile_and_resume(tmp_path: Path) -> None:
    request, expected_train = _fixture(tmp_path)
    plan = plan_phase3d_clients(request)
    first = build_phase3d_clients(request)
    second = build_phase3d_clients(request)

    assert set(cast(dict[str, object], plan["profiles"])) == {
        "iid",
        "dirichlet_0_1",
        "dirichlet_0_3",
        "dirichlet_1_0",
    }
    assert first["client_partition_id"] == second["client_partition_id"]
    assert first["training_records"] == 32
    assert len(cast(list[object], first["assignment_files"])) == 2
    assert cast(dict[str, bool], first["gates"])["model_training_performed"] is False
    tables = [
        pq.read_table(request.output_directory / item["path"])
        for item in cast(list[dict[str, object]], first["assignment_files"])
    ]
    rows = pa.concat_tables(tables).to_pylist()
    assert {row["record_id"] for row in rows} == expected_train
    assert len(rows) == len({row["record_id"] for row in rows}) == 32
    for column in cast(dict[str, str], first["assignment_columns"]).values():
        assert all(0 <= row[column] < 4 for row in rows)


def test_iid_is_family_balanced_and_dirichlet_profiles_meet_minimum(tmp_path: Path) -> None:
    request, _ = _fixture(tmp_path)
    plan = plan_phase3d_clients(request)
    profiles = cast(dict[str, dict[str, object]], plan["profiles"])
    iid = cast(dict[str, list[int]], profiles["iid"]["family_client_counts"])
    assert all(max(counts) - min(counts) <= 1 for counts in iid.values())
    for profile in profiles.values():
        totals = cast(list[int], profile["client_totals"])
        assert sum(totals) == 32
        assert min(totals) >= 2
        matrices = cast(dict[str, list[int]], profile["family_client_counts"])
        assert sum(matrices["attack"]) == 16
        assert sum(matrices["benign"]) == 16


def test_resume_rejects_changed_output_and_identity(tmp_path: Path) -> None:
    request, _ = _fixture(tmp_path)
    build_phase3d_clients(request)
    output = request.output_directory / "assignments" / "0000-clients.parquet"
    output.write_bytes(output.read_bytes() + b"changed")
    with pytest.raises(DataContractError, match="missing or changed"):
        build_phase3d_clients(request)

    changed_output = tmp_path / "changed"
    changed = ClientPartitionBuildRequest(
        dataset=request.dataset,
        preprocessing_report=request.preprocessing_report,
        split_manifest=request.split_manifest,
        anchor_manifest=request.anchor_manifest,
        output_directory=changed_output,
        work_database=request.work_database,
        num_clients=request.num_clients,
        minimum_samples_per_client=1,
        maximum_partition_attempts=request.maximum_partition_attempts,
        parquet_batch_size=request.parquet_batch_size,
    )
    with pytest.raises(DataContractError, match="resume database identity"):
        build_phase3d_clients(changed)


def test_build_rejects_changed_phase3c_shard(tmp_path: Path) -> None:
    request, _ = _fixture(tmp_path)
    source = request.preprocessing_report.parent / "shards" / "0000-preprocessed.parquet"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(DataContractError, match="missing or changed"):
        build_phase3d_clients(request)


def test_request_scalar_controls_are_validated(tmp_path: Path) -> None:
    common: dict[str, object] = {
        "dataset": DatasetName.EDGE_IIOTSET,
        "preprocessing_report": tmp_path,
        "split_manifest": tmp_path,
        "anchor_manifest": tmp_path,
        "output_directory": tmp_path,
        "work_database": tmp_path,
    }
    cases = (
        ("base_seed", -1, "base_seed"),
        ("num_clients", 1, "at least two"),
        ("minimum_samples_per_client", 0, "must be positive"),
        ("dirichlet_alphas", (), "At least one"),
        ("dirichlet_alphas", (0.0,), "finite and positive"),
        ("dirichlet_alphas", (0.1, 0.1), "unique"),
        ("dirichlet_alphas", (1.0, 0.1), "ascending"),
        ("maximum_partition_attempts", 0, "must be positive"),
        ("parquet_batch_size", 0, "must be positive"),
    )
    for field, value, message in cases:
        with pytest.raises(ValueError, match=message):
            ClientPartitionBuildRequest(**common, **{field: value})  # type: ignore[arg-type]


def test_plan_rejects_identity_gate_and_count_drift(tmp_path: Path) -> None:
    request, _ = _fixture(tmp_path)
    preprocessing = json.loads(request.preprocessing_report.read_text())
    split = json.loads(request.split_manifest.read_text())
    anchor = json.loads(request.anchor_manifest.read_text())
    paths = {
        "preprocessing": request.preprocessing_report,
        "split": request.split_manifest,
        "anchor": request.anchor_manifest,
    }
    originals = {"preprocessing": preprocessing, "split": split, "anchor": anchor}
    cases: tuple[tuple[str, str, object, str], ...] = (
        ("preprocessing", "dataset", "wrong", "dataset identity"),
        ("split", "base_seed", 7, "root seed"),
        ("preprocessing", "split_id", "wrong", "split or anchor"),
        (
            "preprocessing",
            "gates",
            {**preprocessing["gates"], "preprocessing_fitted": False},
            "Phase 3C gates",
        ),
        (
            "preprocessing",
            "gates",
            {**preprocessing["gates"], "client_partition_created": True},
            "already claims",
        ),
        ("preprocessing", "partition_counts", {"train": 31}, "do not cover"),
        ("preprocessing", "transformed_files", [], "inventory is missing"),
    )
    for target, key, value, message in cases:
        for name, document in originals.items():
            _write_json(paths[name], document)
        changed = dict(originals[target])
        changed[key] = value
        _write_json(paths[target], changed)
        with pytest.raises(DataContractError, match=message):
            plan_phase3d_clients(request)


def test_internal_allocation_guards(tmp_path: Path) -> None:
    assert phase3d._largest_remainder(5, [1.0, 1.0]) == [3, 2]
    with pytest.raises(DataContractError, match="positive mass"):
        phase3d._largest_remainder(5, [0.0, 0.0])
    assert phase3d._quantile([1.0, 3.0], 0.5) == 2.0
    with pytest.raises(DataContractError, match="empty client"):
        phase3d._quantile([], 0.5)
    with pytest.raises(DataContractError, match="ML dependencies"):
        phase3d._dependency("semcred_fl_dependency_that_does_not_exist")
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(DataContractError, match="Cannot read malformed"):
        phase3d._load_json(malformed, "malformed")
    with pytest.raises(DataContractError, match="escapes"):
        phase3d._resolved_file(tmp_path, {"path": "../outside"}, "test")
    request, _ = _fixture(tmp_path / "progress")
    with pytest.raises(ValueError, match="progress_every"):
        build_phase3d_clients(request, progress_every=0)
