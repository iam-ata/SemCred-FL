"""Tests for resumable Phase 2C conflict and raw-domain diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.audit import AuditRequest, run_audit
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.edge_iiotset import EDGE_SOURCE_COLUMNS
from semcred_fl.data.phase2c import Phase2CRequest, run_phase2c

_HASH = "c" * 64


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _audit(dataset: DatasetName, root: Path, output: Path) -> tuple[Path, Path]:
    database = output / f"{dataset.value}_audit.sqlite3"
    run_audit(
        AuditRequest(dataset, root, _HASH, output, database),
        progress_every=100,
    )
    return output / f"{dataset.value}_audit.json", database


def _request(
    dataset: DatasetName,
    root: Path,
    audit_json: Path,
    audit_database: Path,
    output: Path,
) -> Phase2CRequest:
    return Phase2CRequest(
        dataset=dataset,
        root=root,
        audit_json=audit_json,
        audit_database=audit_database,
        output_directory=output,
        work_database=output / f"{dataset.value}_phase2c.sqlite3",
        reported_top_values=10,
    )


def test_cic_phase2c_separates_primary_and_fine_only_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "cic"
    first = ["1"] * len(CIC_SOURCE_COLUMNS)
    second = ["2"] * len(CIC_SOURCE_COLUMNS)
    _write_csv(root / "DDoS-HTTP_Flood" / "a.csv", CIC_SOURCE_COLUMNS, [first, second])
    _write_csv(root / "DDoS-SYN_Flood" / "b.csv", CIC_SOURCE_COLUMNS, [first])
    _write_csv(root / "DoS-HTTP_Flood" / "c.csv", CIC_SOURCE_COLUMNS, [second])
    audit_json, audit_database = _audit(DatasetName.CIC_IOT_2023, root, tmp_path / "audit")
    request = _request(
        DatasetName.CIC_IOT_2023,
        root,
        audit_json,
        audit_database,
        tmp_path / "phase2c",
    )

    first_report = run_phase2c(request, progress_every=100)
    resumed = run_phase2c(request, progress_every=100)

    assert first_report["analysis_id"] == resumed["analysis_id"]
    assert first_report["coverage"] == {
        "discovered_files": 3,
        "completed_files": 3,
        "rows_seen": 4,
        "valid_rows": 4,
        "conflict_occurrences_observed": 4,
        "complete": True,
    }
    conflicts = first_report["label_conflicts"]
    assert isinstance(conflicts, dict)
    assert conflicts["primary_family_conflict_fingerprints"] == 1
    assert conflicts["primary_family_conflict_occurrences"] == 2
    assert conflicts["fine_only_conflict_fingerprints"] == 1
    assert conflicts["fine_only_conflict_occurrences"] == 2
    assert conflicts["incomplete_fingerprints"] == 0
    assert first_report["raw_value_domains"] == {}
    assert (request.output_directory / "cic_iot_2023_phase2c.json").is_file()
    assert (request.output_directory / "cic_iot_2023_phase2c.md").is_file()


def _edge_row(
    *, fine: str, binary: str, method: str, arp_opcode: str, feature_seed: str = "0"
) -> list[str]:
    row = [feature_seed] * len(EDGE_SOURCE_COLUMNS)
    indices = {name: index for index, name in enumerate(EDGE_SOURCE_COLUMNS)}
    row[indices["frame.time"]] = "2022 12:00:00"
    row[indices["ip.src_host"]] = "10.0.0.1"
    row[indices["ip.dst_host"]] = "10.0.0.2"
    row[indices["http.request.method"]] = method
    row[indices["arp.opcode"]] = arp_opcode
    row[indices["Attack_type"]] = fine
    row[indices["Attack_label"]] = binary
    return row


def test_edge_phase2c_collects_exact_mixed_domains(tmp_path: Path) -> None:
    root = tmp_path / "edge"
    shared_mitm = _edge_row(fine="MITM", binary="1", method="GET", arp_opcode="request")
    shared_xss = _edge_row(fine="XSS", binary="1", method="GET", arp_opcode="request")
    distinct = _edge_row(
        fine="MITM", binary="1", method="POST", arp_opcode="0x01", feature_seed="2"
    )
    _write_csv(
        root / "Attack traffic" / "MITM_attack.csv",
        EDGE_SOURCE_COLUMNS,
        [shared_mitm, distinct],
    )
    _write_csv(root / "Attack traffic" / "XSS_attack.csv", EDGE_SOURCE_COLUMNS, [shared_xss])
    audit_json, audit_database = _audit(DatasetName.EDGE_IIOTSET, root, tmp_path / "audit")
    request = _request(
        DatasetName.EDGE_IIOTSET,
        root,
        audit_json,
        audit_database,
        tmp_path / "phase2c",
    )

    report = run_phase2c(request, progress_every=100)

    conflicts = report["label_conflicts"]
    assert isinstance(conflicts, dict)
    assert conflicts["primary_family_conflict_fingerprints"] == 1
    assert conflicts["primary_family_conflict_occurrences"] == 2
    domains = report["raw_value_domains"]
    assert isinstance(domains, dict)
    methods = domains["http.request.method"]
    arp = domains["arp.opcode"]
    assert isinstance(methods, dict)
    assert isinstance(arp, dict)
    assert methods["exact_distinct_collected_values"] == 2
    assert methods["collected_kind_counts"] == {"text": 3}
    all_values = cast(list[dict[str, object]], methods["all_values"])
    assert {item["raw_value"] for item in all_values} == {"GET", "POST"}
    assert arp["exceptional_values"] == [
        {"raw_value": "request", "value_kind": "text", "occurrence_count": 2}
    ]
    persisted = json.loads(
        (request.output_directory / "edge_iiotset_phase2c.json").read_text(encoding="utf-8")
    )
    assert persisted["analysis_id"] == report["analysis_id"]


def test_phase2c_rejects_bad_limits_and_mismatched_audit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        Phase2CRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path,
            tmp_path / "audit.json",
            tmp_path / "audit.sqlite3",
            tmp_path,
            tmp_path / "phase2c.sqlite3",
            max_exact_domain_values=0,
        )
    with pytest.raises(ValueError, match="positive"):
        Phase2CRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path,
            tmp_path / "audit.json",
            tmp_path / "audit.sqlite3",
            tmp_path,
            tmp_path / "phase2c.sqlite3",
            reported_top_values=0,
        )

    root = tmp_path / "cic"
    _write_csv(
        root / "Benign_Final" / "a.csv",
        CIC_SOURCE_COLUMNS,
        [["1"] * len(CIC_SOURCE_COLUMNS)],
    )
    audit_json, audit_database = _audit(DatasetName.CIC_IOT_2023, root, tmp_path / "audit")
    request = _request(
        DatasetName.CIC_IOT_2023,
        root,
        audit_json,
        audit_database,
        tmp_path / "phase2c",
    )
    with pytest.raises(ValueError, match="positive"):
        run_phase2c(request, progress_every=0)

    changed = json.loads(audit_json.read_text(encoding="utf-8"))
    changed["provenance"]["dataset"] = DatasetName.EDGE_IIOTSET.value
    audit_json.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(DataContractError, match="dataset does not match"):
        run_phase2c(request)
