"""Tests for the leakage-safe, resumable Phase 3C preprocessing artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data import preprocessing_artifact as phase3c
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.preprocessing import (
    ABSENT_CATEGORY,
    CIC_PREPARATION_POLICY,
    EDGE_PREPARATION_POLICY,
    UNKNOWN_CATEGORY,
)
from semcred_fl.data.preprocessing_artifact import (
    PreprocessingBuildRequest,
    build_phase3c_preprocessing,
    plan_phase3c_preprocessing,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _feature_values(dataset: DatasetName, value: float, method: str) -> dict[str, object]:
    if dataset is DatasetName.CIC_IOT_2023:
        return {name: value + index / 100 for index, name in enumerate(CIC_SOURCE_COLUMNS)}
    policy = EDGE_PREPARATION_POLICY.features
    result: dict[str, object] = dict.fromkeys(policy.continuous_columns, value)
    result.update(
        {
            "http.request.method": method,
            "http.request.version": "HTTP/1.1",
            "mqtt.conack.flags": "0",
            "mqtt.protoname": ABSENT_CATEGORY,
        }
    )
    return result


def _fixture(tmp_path: Path, dataset: DatasetName) -> PreprocessingBuildRequest:
    input_root = tmp_path / dataset.value
    canonical_root = input_root / "canonical"
    split_root = input_root / "split"
    canonical_shards = canonical_root / "shards"
    assignment_root = split_root / "assignments"
    canonical_shards.mkdir(parents=True)
    assignment_root.mkdir(parents=True)

    partitions = (
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    )
    families = ("benign", "attack") * 5
    values = (0.0, 10.0, 20.0, 30.0, 1_000.0, -1_000.0, 50_000.0, -50_000.0, 90_000.0, -90_000.0)
    methods = ("POST", "POST", "POST", "POST", "POST", "POST", "GET", "GET", "OPTIONS", "OPTIONS")
    record_ids = [hashlib.sha256(f"record-{index}".encode()).hexdigest() for index in range(10)]
    canonical_rows = [
        {"record_id": record_id, **_feature_values(dataset, value, method)}
        for record_id, value, method in zip(record_ids, values, methods, strict=True)
    ]
    assignment_rows = [
        {
            "record_id": record_id,
            "split": split,
            "family_label": family,
            "fine_label": family,
            "fine_label_ambiguous": False,
        }
        for record_id, split, family in zip(record_ids, partitions, families, strict=True)
    ]
    canonical_inventory: list[dict[str, object]] = []
    assignment_inventory: list[dict[str, object]] = []
    for index, (canonical_part, assignment_part) in enumerate(
        ((canonical_rows[:5], assignment_rows[:5]), (canonical_rows[5:], assignment_rows[5:]))
    ):
        canonical_path = canonical_shards / f"{index:04d}.parquet"
        assignment_path = assignment_root / f"{index:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(canonical_part), canonical_path)
        pq.write_table(pa.Table.from_pylist(assignment_part), assignment_path)
        canonical_inventory.append(
            {
                "file": f"shards/{canonical_path.name}",
                "byte_size": canonical_path.stat().st_size,
                "sha256": _sha256(canonical_path),
                "row_count": len(canonical_part),
            }
        )
        assignment_inventory.append(
            {
                "path": f"assignments/{assignment_path.name}",
                "byte_size": assignment_path.stat().st_size,
                "sha256": _sha256(assignment_path),
                "row_count": len(assignment_part),
            }
        )

    policy = (
        CIC_PREPARATION_POLICY if dataset is DatasetName.CIC_IOT_2023 else EDGE_PREPARATION_POLICY
    )
    corpus_id = hashlib.sha256(f"{dataset.value}-corpus".encode()).hexdigest()
    canonical_manifest = canonical_root / f"{dataset.value}_canonical.json"
    _write_json(
        canonical_manifest,
        {
            "corpus_id": corpus_id,
            "provenance": {
                "dataset": dataset.value,
                "preparation_policy_id": policy.policy_id,
            },
            "shards": canonical_inventory,
        },
    )

    anchors = [
        {"record_id": record_ids[4], "subset": "calibration"},
        {"record_id": record_ids[5], "subset": "scoring"},
    ]
    anchor_path = split_root / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(anchors), anchor_path)
    anchor_record_hash = hashlib.sha256(
        "\n".join(sorted(row["record_id"] for row in anchors)).encode()
    ).hexdigest()
    anchor_id = hashlib.sha256(f"{dataset.value}-anchors".encode()).hexdigest()
    anchor_manifest = split_root / "anchor_manifest.json"
    _write_json(
        anchor_manifest,
        {
            "dataset": dataset.value,
            "canonical_corpus_id": corpus_id,
            "anchor_id": anchor_id,
            "record_ids_sha256": anchor_record_hash,
            "subset_counts": {"calibration": 1, "scoring": 1},
            "file": {
                "path": anchor_path.name,
                "byte_size": anchor_path.stat().st_size,
                "sha256": _sha256(anchor_path),
            },
            "gates": {
                "all_anchors_selected_from_train": True,
                "calibration_scoring_disjoint": True,
                "exact_class_balance": True,
            },
        },
    )

    split_id = hashlib.sha256(f"{dataset.value}-split".encode()).hexdigest()
    split_manifest = split_root / f"{dataset.value}_phase3b_split.json"
    family_counts = {
        "attack": {"train": 3, "validation": 1, "test": 1},
        "benign": {"train": 3, "validation": 1, "test": 1},
    }
    _write_json(
        split_manifest,
        {
            "dataset": dataset.value,
            "base_seed": 20260816,
            "records": 10,
            "canonical_corpus_id": corpus_id,
            "split_id": split_id,
            "federated_train_records_after_anchor_removal": 4,
            "split_counts": {"train": 6, "validation": 2, "test": 2},
            "family_split_counts": family_counts,
            "fine_split_counts": family_counts,
            "assignment_files": assignment_inventory,
            "anchor_manifest": {"anchor_id": anchor_id, "records": 2},
            "gates": {
                "assignment_cardinality_matches_canonical": True,
                "primary_family_cardinality_matches_canonical": True,
                "every_primary_family_in_every_split": True,
                "group_disjoint_by_construction": True,
                "fingerprint_disjoint_by_canonical_deduplication": True,
                "anchors_disjoint_and_train_only": True,
                "preprocessing_fitted": False,
                "client_partition_created": False,
            },
        },
    )

    output = tmp_path / "output" / dataset.value
    return PreprocessingBuildRequest(
        dataset=dataset,
        canonical_manifest=canonical_manifest,
        split_manifest=split_manifest,
        anchor_manifest=anchor_manifest,
        output_directory=output,
        work_database=output / "resume.sqlite3",
        lower_quantile=0.0,
        upper_quantile=1.0,
        quantile_sample_target=100,
        parquet_batch_size=3,
    )


@pytest.mark.parametrize("dataset", tuple(DatasetName))
def test_build_is_training_only_complete_and_resumable(
    tmp_path: Path, dataset: DatasetName
) -> None:
    request = _fixture(tmp_path, dataset)
    plan = plan_phase3c_preprocessing(request)
    first = build_phase3c_preprocessing(request)
    second = build_phase3c_preprocessing(request)

    assert plan["federated_training_records"] == 4
    assert first["preprocessed_corpus_id"] == second["preprocessed_corpus_id"]
    assert first["records"] == 10
    assert first["partition_counts"] == {
        "train": 4,
        "validation": 2,
        "test": 2,
        "anchor_calibration": 1,
        "anchor_scoring": 1,
    }
    preprocessor = json.loads((request.output_directory / "preprocessor.json").read_text())
    assert preprocessor["fit_records"] == 4
    assert preprocessor["quantile_sample_records"] == 4
    first_feature = request.dataset.value == DatasetName.CIC_IOT_2023.value
    feature_name = (
        CIC_SOURCE_COLUMNS[0]
        if first_feature
        else EDGE_PREPARATION_POLICY.features.continuous_columns[0]
    )
    assert preprocessor["continuous_features"][feature_name]["clip_lower"] == 0.0
    assert preprocessor["continuous_features"][feature_name]["clip_upper"] == 30.0
    gates = cast(dict[str, bool], first["gates"])
    assert gates["preprocessing_fitted"] is True
    assert gates["client_partition_created"] is False
    assert all(value for name, value in gates.items() if name != "client_partition_created")


def test_edge_vocabulary_is_learned_only_from_non_anchor_training_rows(tmp_path: Path) -> None:
    request = _fixture(tmp_path, DatasetName.EDGE_IIOTSET)
    build_phase3c_preprocessing(request)
    artifact = json.loads((request.output_directory / "preprocessor.json").read_text())
    method_vocabulary = artifact["categorical_vocabulary"]["http.request.method"]
    assert method_vocabulary == ["POST", UNKNOWN_CATEGORY]
    assert "GET" not in method_vocabulary
    unknown_coordinate = next(
        item["coordinate"]
        for item in artifact["coordinates"]
        if item["source_feature"] == "http.request.method" and item["category"] == UNKNOWN_CATEGORY
    )
    transformed = pa.concat_tables(
        [
            pq.read_table(path)
            for path in sorted((request.output_directory / "shards").glob("*.parquet"))
        ]
    ).to_pylist()
    validation = next(row for row in transformed if row["partition"] == "validation")
    assert validation[unknown_coordinate] == 1.0


def test_phase3c_rejects_seed_anchor_and_population_drift(tmp_path: Path) -> None:
    request = _fixture(tmp_path, DatasetName.CIC_IOT_2023)
    with pytest.raises(DataContractError, match="root seed"):
        plan_phase3c_preprocessing(
            PreprocessingBuildRequest(
                dataset=request.dataset,
                canonical_manifest=request.canonical_manifest,
                split_manifest=request.split_manifest,
                anchor_manifest=request.anchor_manifest,
                output_directory=request.output_directory,
                work_database=request.work_database,
                base_seed=7,
            )
        )

    anchor = json.loads(request.anchor_manifest.read_text())
    anchor["record_ids_sha256"] = "0" * 64
    _write_json(request.anchor_manifest, anchor)
    with pytest.raises(DataContractError, match="identity digest"):
        build_phase3c_preprocessing(request)

    anchor_path = request.anchor_manifest.parent / "anchors.parquet"
    ids = sorted(pq.read_table(anchor_path, columns=["record_id"]).column(0).to_pylist())
    anchor["record_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    _write_json(request.anchor_manifest, anchor)
    split = json.loads(request.split_manifest.read_text())
    split["federated_train_records_after_anchor_removal"] = 5
    _write_json(request.split_manifest, split)
    population_output = tmp_path / "population-drift"
    population_request = PreprocessingBuildRequest(
        dataset=request.dataset,
        canonical_manifest=request.canonical_manifest,
        split_manifest=request.split_manifest,
        anchor_manifest=request.anchor_manifest,
        output_directory=population_output,
        work_database=population_output / "resume.sqlite3",
        lower_quantile=request.lower_quantile,
        upper_quantile=request.upper_quantile,
        quantile_sample_target=request.quantile_sample_target,
        parquet_batch_size=request.parquet_batch_size,
    )
    with pytest.raises(DataContractError, match="training population"):
        build_phase3c_preprocessing(population_request)


def test_request_controls_and_resume_identity_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Quantiles"):
        PreprocessingBuildRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            lower_quantile=0.9,
            upper_quantile=0.1,
        )
    request = _fixture(tmp_path, DatasetName.CIC_IOT_2023)
    build_phase3c_preprocessing(request)
    changed = PreprocessingBuildRequest(
        dataset=request.dataset,
        canonical_manifest=request.canonical_manifest,
        split_manifest=request.split_manifest,
        anchor_manifest=request.anchor_manifest,
        output_directory=request.output_directory,
        work_database=request.work_database,
        lower_quantile=0.01,
        upper_quantile=request.upper_quantile,
        quantile_sample_target=request.quantile_sample_target,
        parquet_batch_size=request.parquet_batch_size,
    )
    with pytest.raises(DataContractError, match="resume database identity"):
        build_phase3c_preprocessing(changed)


def test_request_rejects_each_invalid_scalar_control(tmp_path: Path) -> None:
    common: dict[str, object] = {
        "dataset": DatasetName.CIC_IOT_2023,
        "canonical_manifest": tmp_path,
        "split_manifest": tmp_path,
        "anchor_manifest": tmp_path,
        "output_directory": tmp_path,
        "work_database": tmp_path,
    }
    cases = (
        ("base_seed", -1, "base_seed"),
        ("quantile_sample_target", 1, "at least two"),
        ("standard_deviation_epsilon", 0.0, "must be positive"),
        ("parquet_batch_size", 0, "must be positive"),
    )
    for field, value, message in cases:
        with pytest.raises(ValueError, match=message):
            PreprocessingBuildRequest(**common, **{field: value})  # type: ignore[arg-type]


def test_plan_rejects_manifest_identity_and_gate_drift(tmp_path: Path) -> None:
    request = _fixture(tmp_path, DatasetName.CIC_IOT_2023)
    canonical = json.loads(request.canonical_manifest.read_text())
    split = json.loads(request.split_manifest.read_text())
    anchor = json.loads(request.anchor_manifest.read_text())

    cases: tuple[tuple[str, str, str, object], ...] = (
        ("canonical", "provenance", "Canonical manifest dataset", {"dataset": "wrong"}),
        ("split", "dataset", "Phase 3B dataset", "wrong"),
        ("canonical", "corpus_id", "canonical corpus identity", "wrong"),
        ("split", "anchor_manifest", "manifest identities", {"anchor_id": "wrong"}),
        (
            "split",
            "gates",
            "split gates",
            {**split["gates"], "assignment_cardinality_matches_canonical": False},
        ),
        (
            "anchor",
            "gates",
            "anchor gates",
            {**anchor["gates"], "exact_class_balance": False},
        ),
        ("split", "federated_train_records_after_anchor_removal", "count is invalid", 1),
        (
            "canonical",
            "provenance",
            "preparation policy mismatch",
            {"dataset": request.dataset.value, "preparation_policy_id": "wrong"},
        ),
        ("canonical", "shards", "inventories", []),
        ("split", "assignment_files", "inventories", []),
    )
    originals = {"canonical": canonical, "split": split, "anchor": anchor}
    paths = {
        "canonical": request.canonical_manifest,
        "split": request.split_manifest,
        "anchor": request.anchor_manifest,
    }
    for target, key, message, value in cases:
        for name, document in originals.items():
            _write_json(paths[name], document)
        changed = dict(originals[target])
        changed[key] = value
        _write_json(paths[target], changed)
        with pytest.raises(DataContractError, match=message):
            plan_phase3c_preprocessing(request)


def test_internal_guards_cover_malformed_files_and_sampling(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(DataContractError, match="Cannot read malformed"):
        phase3c._load_json(malformed, "malformed")
    with pytest.raises(DataContractError, match="ML dependencies"):
        phase3c._dependency("semcred_fl_dependency_that_does_not_exist")
    with pytest.raises(DataContractError, match="escapes"):
        phase3c._resolved_file(tmp_path, {"path": "../outside"}, "test file")
    assert phase3c._sample_selected("record", 1, 0) is False
    assert phase3c._merge_moments(1, [2.0], [0.0], 0, [0.0], [0.0]) == (
        1,
        [2.0],
        [0.0],
    )
    with pytest.raises(DataContractError, match="escaped the frozen domain"):
        phase3c._ordered_vocabulary(
            EDGE_PREPARATION_POLICY,
            {
                name: ({"INVALID"} if name == "http.request.method" else set())
                for name in EDGE_PREPARATION_POLICY.features.categorical_columns
            },
        )
    request = _fixture(tmp_path / "progress", DatasetName.CIC_IOT_2023)
    with pytest.raises(ValueError, match="progress_every"):
        build_phase3c_preprocessing(request, progress_every=0)
