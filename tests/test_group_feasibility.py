"""Tests for Phase 3A grouping and split-feasibility evidence."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.canonical import CanonicalBuildRequest, build_canonical_corpus
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.group_feasibility import (
    GroupFeasibilityRequest,
    _edge_time,
    analyze_group_feasibility,
)

_HASH = "b" * 64


def _write_rows(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CIC_SOURCE_COLUMNS)
        writer.writerows([[str(value)] * len(CIC_SOURCE_COLUMNS) for value in values])


def _canonical_cic(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "raw"
    _write_rows(root / "DDoS-HTTP_Flood" / "a.csv", [100, 101])
    _write_rows(root / "DDoS-HTTP_Flood" / "b.csv", [100, 102])
    _write_rows(root / "DDoS-HTTP_Flood" / "c.csv", [103])
    _write_rows(root / "DoS-HTTP_Flood" / "d.csv", [201])
    _write_rows(root / "DoS-HTTP_Flood" / "e.csv", [202])
    _write_rows(root / "DoS-HTTP_Flood" / "f.csv", [203])
    output = tmp_path / "canonical"
    build_canonical_corpus(
        CanonicalBuildRequest(
            dataset=DatasetName.CIC_IOT_2023,
            root=root,
            archive_sha256=_HASH,
            output_directory=output,
            work_database=output / "cic_iot_2023_canonical.sqlite3",
            row_batch_size=2,
            parquet_batch_size=2,
        ),
        progress_every=100,
    )
    return output / "cic_iot_2023_canonical.json", output / "cic_iot_2023_canonical.sqlite3"


def _request(tmp_path: Path, manifest: Path, database: Path) -> GroupFeasibilityRequest:
    output = tmp_path / "phase3a"
    return GroupFeasibilityRequest(
        dataset=DatasetName.CIC_IOT_2023,
        canonical_manifest=manifest,
        canonical_database=database,
        output_directory=output,
        work_database=output / "cic_iot_2023_phase3a.sqlite3",
        calibration_anchors_per_class=1,
        scoring_anchors_per_class=1,
        parquet_batch_size=2,
        connectivity_batch_size=1,
    )


def test_group_feasibility_resumes_and_connects_duplicate_sources(tmp_path: Path) -> None:
    manifest, database = _canonical_cic(tmp_path)
    request = _request(tmp_path, manifest, database)

    first = analyze_group_feasibility(request, progress_every=100)
    resumed = analyze_group_feasibility(request, progress_every=100)

    assert resumed["analysis_id"] == first["analysis_id"]
    assert first["records_scanned"] == 7
    assert first["split_assignment_created"] is False
    connectivity = cast(dict[str, object], first["source_connectivity"])
    assert connectivity["cross_source_fingerprints"] == 1
    assert connectivity["source_files"] == 6
    assert connectivity["connected_components"] == 5
    policies = cast(dict[str, dict[str, object]], first["candidate_policies"])
    assert policies["source_file"]["status"] == "candidate_only"
    assert policies["source_directory"]["status"] == "necessary_conditions_fail"
    connected = policies["duplicate_connected_source_file"]
    assert connected["status"] == "necessary_conditions_fail"
    families = cast(dict[str, dict[str, object]], connected["family"])
    assert families["ddos"]["groups"] == 2
    assert families["dos"]["groups"] == 3
    assert (request.output_directory / "cic_iot_2023_phase3a.json").is_file()
    markdown = (request.output_directory / "cic_iot_2023_phase3a.md").read_text(encoding="utf-8")
    assert "Train/validation/test assignment created: no" in markdown
    with sqlite3.connect(request.work_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM completed_shards").fetchone()[0] == 6


def test_resume_rejects_changed_anchor_contract(tmp_path: Path) -> None:
    manifest, database = _canonical_cic(tmp_path)
    request = _request(tmp_path, manifest, database)
    analyze_group_feasibility(request, progress_every=100)

    changed = GroupFeasibilityRequest(
        dataset=request.dataset,
        canonical_manifest=manifest,
        canonical_database=database,
        output_directory=request.output_directory,
        work_database=request.work_database,
        calibration_anchors_per_class=2,
        scoring_anchors_per_class=1,
    )
    with pytest.raises(DataContractError, match="identity mismatch"):
        analyze_group_feasibility(changed)


@pytest.mark.parametrize(
    ("value", "category", "has_stamp"),
    [
        ("2021 17:58:23.150209000", "parseable", True),
        ("2021 25:58:23.1", "other", False),
        ("249.66.134.149", "ip_like", False),
        ("0.0", "zero", False),
        ("2022-01-01 12:00:00", "other", False),
    ],
)
def test_edge_time_quality_classification(value: str, category: str, has_stamp: bool) -> None:
    actual_category, stamp = _edge_time(value)
    assert actual_category == category
    assert (stamp is not None) is has_stamp


def test_request_rejects_invalid_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Anchor counts"):
        GroupFeasibilityRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path / "manifest.json",
            tmp_path / "canonical.db",
            tmp_path,
            tmp_path / "work.db",
            calibration_anchors_per_class=0,
        )
    with pytest.raises(ValueError, match="Batch sizes"):
        GroupFeasibilityRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path / "manifest.json",
            tmp_path / "canonical.db",
            tmp_path,
            tmp_path / "work.db",
            parquet_batch_size=0,
        )
