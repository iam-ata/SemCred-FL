"""Tests for resumable, exact source-data auditing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.audit import AuditRequest, CardinalitySketch, ColumnAccumulator, run_audit
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.edge_iiotset import EDGE_SOURCE_COLUMNS

_HASH = "a" * 64


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _request(dataset: DatasetName, root: Path, output: Path) -> AuditRequest:
    return AuditRequest(
        dataset=dataset,
        root=root,
        archive_sha256=_HASH,
        output_directory=output,
        work_database=output / f"{dataset.value}.sqlite3",
    )


def _edge_row(*, fine: str, binary: str, first_feature: str = "0") -> list[str]:
    row = ["0"] * len(EDGE_SOURCE_COLUMNS)
    indices = {name: index for index, name in enumerate(EDGE_SOURCE_COLUMNS)}
    row[indices["frame.time"]] = "2022 12:00:00"
    row[indices["ip.src_host"]] = "10.0.0.1"
    row[indices["ip.dst_host"]] = "10.0.0.2"
    row[indices["arp.opcode"]] = first_feature
    row[indices["Attack_type"]] = fine
    row[indices["Attack_label"]] = binary
    return row


def test_cic_audit_is_resumable_and_counts_exact_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "cic"
    output = tmp_path / "artifacts"
    base = ["1"] * len(CIC_SOURCE_COLUMNS)
    distinct = ["2"] * len(CIC_SOURCE_COLUMNS)
    non_finite = base.copy()
    non_finite[0] = "nan"
    _write_csv(root / "Benign_Final" / "a.csv", CIC_SOURCE_COLUMNS, [base, base, non_finite])
    _write_csv(root / "XSS" / "b.csv", CIC_SOURCE_COLUMNS, [base, distinct, base[:-1]])

    request = _request(DatasetName.CIC_IOT_2023, root, output)
    first = run_audit(request, progress_every=2)
    resumed = run_audit(request, progress_every=2)

    assert first["audit_id"] == resumed["audit_id"]
    assert first["coverage"] == {
        "completed_files": 2,
        "rows_seen": 6,
        "valid_rows": 4,
        "invalid_width": 1,
        "invalid_label": 0,
        "non_finite_numeric": 1,
        "header_failures": [],
    }
    duplicates = first["duplicates"]
    assert isinstance(duplicates, dict)
    assert duplicates["unique_model_feature_fingerprints"] == 2
    assert duplicates["duplicate_occurrences_after_first"] == 2
    assert duplicates["within_file_duplicate_occurrences"] == 1
    assert duplicates["fingerprints_present_in_multiple_files"] == 1
    assert duplicates["occurrences_outside_first_source_file"] == 1
    assert duplicates["fingerprints_with_conflicting_fine_labels"] == 1
    assert (output / "cic_iot_2023_audit.json").is_file()
    assert (output / "cic_iot_2023_audit.md").is_file()
    persisted = json.loads((output / "cic_iot_2023_audit.json").read_text(encoding="utf-8"))
    assert persisted["audit_id"] == first["audit_id"]


def test_edge_audit_profiles_mixed_types_and_invalid_labels(tmp_path: Path) -> None:
    root = tmp_path / "edge"
    source = root / "Attack traffic" / "MITM_attack.csv"
    rows = [
        _edge_row(fine="MITM", binary="1", first_feature="0x10"),
        _edge_row(fine="MITM", binary="1", first_feature="1.5"),
        _edge_row(fine="MITM", binary="0", first_feature="text"),
    ]
    _write_csv(source, EDGE_SOURCE_COLUMNS, rows)

    summary = run_audit(_request(DatasetName.EDGE_IIOTSET, root, tmp_path / "out"))

    coverage = summary["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["rows_seen"] == 3
    assert coverage["valid_rows"] == 2
    assert coverage["invalid_label"] == 1
    columns = summary["columns"]
    assert isinstance(columns, dict)
    arp = columns["arp.opcode"]
    assert isinstance(arp, dict)
    assert arp["kind_counts"] == {"finite_decimal": 1, "hex_integer": 1, "text": 1}
    assert arp["approximate_cardinality"] == 3
    assert summary["model_readiness"]["ready"] is False  # type: ignore[index]


def test_edge_audit_accepts_provider_os_fingerprinting_alias(tmp_path: Path) -> None:
    root = tmp_path / "edge"
    source = root / "Attack traffic" / "OS_Fingerprinting_attack.csv"
    rows = [_edge_row(fine="OS_Fingerprinting", binary="1")]
    _write_csv(source, EDGE_SOURCE_COLUMNS, rows)

    summary = run_audit(_request(DatasetName.EDGE_IIOTSET, root, tmp_path / "out"))

    coverage = summary["coverage"]
    labels = summary["labels"]
    assert isinstance(coverage, dict)
    assert isinstance(labels, dict)
    assert coverage["invalid_label"] == 0
    assert labels["fine_row_counts"] == {"Fingerprinting": 1}


def test_changed_completed_source_and_mismatched_identity_fail(tmp_path: Path) -> None:
    root = tmp_path / "cic"
    source = root / "Benign_Final" / "a.csv"
    _write_csv(source, CIC_SOURCE_COLUMNS, [["1"] * len(CIC_SOURCE_COLUMNS)])
    request = _request(DatasetName.CIC_IOT_2023, root, tmp_path / "out")
    run_audit(request)

    with source.open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    with pytest.raises(DataContractError, match="changed"):
        run_audit(request)

    different_hash = AuditRequest(
        dataset=request.dataset,
        root=request.root,
        archive_sha256="b" * 64,
        output_directory=request.output_directory,
        work_database=request.work_database,
    )
    with pytest.raises(DataContractError, match="different dataset root"):
        run_audit(different_hash)


def test_audit_validation_and_cardinality_serialization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full SHA-256"):
        _request(DatasetName.CIC_IOT_2023, tmp_path, tmp_path / "out").__class__(
            DatasetName.CIC_IOT_2023, tmp_path, "bad", tmp_path, tmp_path / "a.db"
        )
    request = _request(DatasetName.CIC_IOT_2023, tmp_path, tmp_path / "out")
    with pytest.raises(ValueError, match="positive"):
        run_audit(request, progress_every=0)

    accumulator = ColumnAccumulator()
    for value in ("1", "2", "2", "null"):
        accumulator.add(value)
    restored = ColumnAccumulator.from_dict(accumulator.to_dict(include_sketch=True))
    assert restored.total == 4
    assert restored.kinds["missing"] == 1
    assert 2 <= restored.cardinality.estimate() <= 5
    with pytest.raises(ValueError, match="different precisions"):
        CardinalitySketch().merge(CardinalitySketch(registers=bytearray(2)))
    with pytest.raises(ValueError, match="Invalid cardinality"):
        CardinalitySketch.decode("AA==")
