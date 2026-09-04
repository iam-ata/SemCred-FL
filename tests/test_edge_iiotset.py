"""Tests for raw per-scenario Edge-IIoTset ingestion."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.edge_iiotset import (
    EDGE_PROHIBITED_MODEL_COLUMNS,
    EDGE_SOURCE_COLUMNS,
    EdgeIIoTsetAdapter,
)


def _edge_row(*, fine_label: str, binary_label: str) -> list[str]:
    values = ["0"] * len(EDGE_SOURCE_COLUMNS)
    indices = {column: index for index, column in enumerate(EDGE_SOURCE_COLUMNS)}
    values[indices["frame.time"]] = "2021 11:44:10.000000"
    values[indices["ip.src_host"]] = "192.168.0.1"
    values[indices["ip.dst_host"]] = "192.168.0.2"
    values[indices["Attack_label"]] = binary_label
    values[indices["Attack_type"]] = fine_label
    return values


def _write_edge(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EDGE_SOURCE_COLUMNS)
        writer.writerows(rows)


def test_edge_adapter_separates_grouping_and_prohibited_fields(tmp_path: Path) -> None:
    source = tmp_path / "Attack traffic" / "MITM_attack.csv"
    _write_edge(source, [_edge_row(fine_label="MITM", binary_label="1")])
    result = EdgeIIoTsetAdapter(tmp_path).load_file(source, max_rows=1)
    record = result.records[0]

    assert record.fine_label == "MITM"
    assert record.family_label == "man_in_the_middle"
    assert record.grouping_map["endpoint_pair"] == "192.168.0.1|192.168.0.2"
    assert set(EDGE_PROHIBITED_MODEL_COLUMNS).isdisjoint(record.feature_map)
    assert "Attack_label" not in record.feature_map
    assert "Attack_type" not in record.feature_map


def test_edge_adapter_counts_malformed_width(tmp_path: Path) -> None:
    source = tmp_path / "Normal traffic" / "Sensor" / "Sensor.csv"
    valid = _edge_row(fine_label="Normal", binary_label="0")
    _write_edge(source, [valid[:-1], valid])
    result = EdgeIIoTsetAdapter(tmp_path).load_file(source, max_rows=2)
    assert result.report.invalid_width == 1
    assert result.report.rows_emitted == 1


def test_edge_adapter_normalizes_provider_os_fingerprinting_label(tmp_path: Path) -> None:
    source = tmp_path / "Attack traffic" / "OS_Fingerprinting_attack.csv"
    _write_edge(source, [_edge_row(fine_label="OS_Fingerprinting", binary_label="1")])

    record = EdgeIIoTsetAdapter(tmp_path).load_file(source, max_rows=1).records[0]

    assert record.fine_label == "Fingerprinting"
    assert record.family_label == "information_gathering"


@pytest.mark.parametrize(
    ("fine_label", "binary_label", "message"),
    [
        ("Unknown", "1", "Unknown Edge-IIoTset label"),
        ("MITM", "0", "Binary/fine label mismatch"),
        ("Normal", "0", "Source/row label mismatch"),
    ],
)
def test_edge_adapter_fails_unknown_or_inconsistent_labels(
    tmp_path: Path, fine_label: str, binary_label: str, message: str
) -> None:
    source = tmp_path / "Attack traffic" / "MITM_attack.csv"
    _write_edge(source, [_edge_row(fine_label=fine_label, binary_label=binary_label)])
    with pytest.raises(DataContractError, match=message):
        EdgeIIoTsetAdapter(tmp_path).load_file(source, max_rows=1)


def test_edge_discovery_excludes_selected_ml_files(tmp_path: Path) -> None:
    attack = tmp_path / "Attack traffic" / "MITM_attack.csv"
    normal = tmp_path / "Normal traffic" / "Sensor" / "Sensor.csv"
    selected = tmp_path / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
    row_attack = _edge_row(fine_label="MITM", binary_label="1")
    row_normal = _edge_row(fine_label="Normal", binary_label="0")
    _write_edge(attack, [row_attack])
    _write_edge(normal, [row_normal])
    _write_edge(selected, [row_normal])

    files = EdgeIIoTsetAdapter(tmp_path).discover_files()
    assert set(files) == {attack, normal}
