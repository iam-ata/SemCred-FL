"""Tests for official per-capture CICIoT2023 ingestion."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from semcred_fl.data.ciciot2023 import (
    CIC_SOURCE_COLUMNS,
    CIC_SOURCE_DIRECTORY_TO_FINE,
    CICIoT2023Adapter,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.validation import inspect_adapter


def _write_cic(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CIC_SOURCE_COLUMNS)
        writer.writerows(rows)


def test_cic_adapter_maps_directory_label_and_rejects_non_finite(tmp_path: Path) -> None:
    source = tmp_path / "Benign_Final" / "capture.pcap.csv"
    valid = ["1"] * len(CIC_SOURCE_COLUMNS)
    non_finite = valid.copy()
    non_finite[0] = "nan"
    malformed = valid[:-1]
    _write_cic(source, [valid, non_finite, malformed])

    adapter = CICIoT2023Adapter(tmp_path)
    result = adapter.load_file(source, max_rows=3)
    assert len(result.records) == 1
    record = result.records[0]
    assert record.fine_label == "BenignTraffic"
    assert record.family_label == "benign"
    assert record.grouping_map["capture_file"] == "capture.pcap.csv"
    assert record.feature_map["Header_Length"] == "1"
    assert result.report.rows_seen == 3
    assert result.report.invalid_width == 1
    assert result.report.non_finite_numeric == 1
    assert result.report.invalid_rows == 2


def test_cic_discovery_and_inspection_are_deterministic(tmp_path: Path) -> None:
    row = ["2"] * len(CIC_SOURCE_COLUMNS)
    _write_cic(tmp_path / "XSS" / "z.csv", [row])
    _write_cic(tmp_path / "Backdoor_Malware" / "a.csv", [row])
    adapter = CICIoT2023Adapter(tmp_path)
    files = adapter.discover_files()
    assert [path.name for path in files] == ["a.csv", "z.csv"]

    summary = inspect_adapter(adapter, max_files=2, max_rows_per_file=1)
    assert summary.discovered_files == 2
    assert summary.rows_emitted == 2
    assert summary.family_labels_observed == ("web_based",)


def test_cic_unknown_label_directory_fails(tmp_path: Path) -> None:
    source = tmp_path / "Unknown" / "capture.csv"
    _write_cic(source, [["1"] * len(CIC_SOURCE_COLUMNS)])
    with pytest.raises(DataContractError, match="Unknown CICIoT2023"):
        CICIoT2023Adapter(tmp_path).discover_files()


def test_cic_source_directory_mapping_covers_34_official_directories() -> None:
    assert len(CIC_SOURCE_DIRECTORY_TO_FINE) == 34
    assert len(set(CIC_SOURCE_DIRECTORY_TO_FINE.values())) == 34
