"""Tests for reproducible Phase 3B split and anchor manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.split_manifest import (
    SplitBuildRequest,
    _assign_grouped_family,
    _Group,
    _record_split,
    build_phase3b_split,
    plan_phase3b_split,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_fixture(tmp_path: Path, *, rows_per_family: int = 300) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical"
    shard_directory = canonical / "shards"
    shard_directory.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for family in ("benign", "dos_ddos"):
        for index in range(rows_per_family):
            fingerprint = hashlib.sha256(f"{family}-fingerprint-{index}".encode()).hexdigest()
            rows.append(
                {
                    "record_id": hashlib.sha256(f"record-{fingerprint}".encode()).hexdigest(),
                    "canonical_fingerprint": fingerprint,
                    "family_label": family,
                    "fine_label": family,
                    "fine_label_ambiguous": False,
                    "source_file": f"source/{family}.csv",
                    "source_row": index + 1,
                }
            )
    shards: list[dict[str, object]] = []
    midpoint = len(rows) // 2
    for shard_index, shard_rows in enumerate((rows[:midpoint], rows[midpoint:])):
        path = shard_directory / f"{shard_index:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), path, compression="zstd")
        shards.append(
            {
                "file": f"shards/{path.name}",
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
                "row_count": len(shard_rows),
            }
        )
    manifest = {
        "corpus_id": hashlib.sha256(b"synthetic-corpus").hexdigest(),
        "provenance": {"dataset": DatasetName.CIC_IOT_2023.value},
        "shards": shards,
        "family_label_counts": {"benign": rows_per_family, "dos_ddos": rows_per_family},
    }
    manifest_path = canonical / "cic_iot_2023_canonical.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    database = canonical / "cic_iot_2023_canonical.sqlite3"
    database.touch()
    return manifest_path, database


def _ipv4_udp_frame(source: str, destination: str, port: int) -> bytes:
    source_ip = bytes(int(part) for part in source.split("."))
    destination_ip = bytes(int(part) for part in destination.split("."))
    ethernet = bytes.fromhex("00112233445566778899aabb0800")
    ipv4 = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        28,
        1,
        0,
        64,
        17,
        0,
        source_ip,
        destination_ip,
    )
    return ethernet + ipv4 + struct.pack("!HHHH", port, port, 8, 0)


def _edge_request(tmp_path: Path) -> SplitBuildRequest:
    mitm_source = "Attack traffic/MITM_attack.csv"
    flows = [
        ("192.168.0.10", "192.168.0.20", 1000),
        ("192.168.0.10", "192.168.0.30", 2000),
        ("192.168.0.10", "192.168.0.40", 3000),
    ]
    packet_rows: list[tuple[str, str, int]] = []
    for index in range(120):
        source, destination, port = flows[index % len(flows)]
        if index % 2:
            source, destination = destination, source
        packet_rows.append((source, destination, port))
    raw_csv = tmp_path / "MITM_attack.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ip.src_host", "ip.dst_host", "Attack_type"])
        writer.writerows([[source, destination, "MITM"] for source, destination, _ in packet_rows])
    pcap = tmp_path / "MITM.pcap"
    with pcap.open("wb") as stream:
        stream.write(b"\xd4\xc3\xb2\xa1")
        stream.write(struct.pack("<HHIIII", 2, 4, 0, 0, 65_535, 1))
        for index, (source, destination, port) in enumerate(packet_rows):
            frame = _ipv4_udp_frame(source, destination, port)
            stream.write(struct.pack("<IIII", 100 + index, 0, len(frame), len(frame)))
            stream.write(frame)

    canonical = tmp_path / "edge-canonical"
    shard_directory = canonical / "shards"
    shard_directory.mkdir(parents=True)
    inventories: list[dict[str, object]] = []
    sources = [
        (mitm_source, "man_in_the_middle", 120),
        ("Normal traffic/a.csv", "benign", 10),
        ("Normal traffic/b.csv", "benign", 10),
        ("Normal traffic/c.csv", "benign", 10),
    ]
    for file_index, (source_file, family, count) in enumerate(sources):
        records: list[dict[str, object]] = []
        for row in range(1, count + 1):
            fingerprint = hashlib.sha256(f"{source_file}:{row}".encode()).hexdigest()
            records.append(
                {
                    "record_id": hashlib.sha256(f"record:{fingerprint}".encode()).hexdigest(),
                    "canonical_fingerprint": fingerprint,
                    "family_label": family,
                    "fine_label": "MITM" if family == "man_in_the_middle" else "Normal",
                    "fine_label_ambiguous": False,
                    "source_file": source_file,
                    "source_row": row,
                }
            )
        path = shard_directory / f"{file_index:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        inventories.append(
            {
                "file": f"shards/{path.name}",
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
                "row_count": count,
            }
        )
    manifest_path = canonical / "edge_iiotset_canonical.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_id": "edge-synthetic-corpus",
                "provenance": {"dataset": DatasetName.EDGE_IIOTSET.value},
                "shards": inventories,
                "family_label_counts": {"benign": 30, "man_in_the_middle": 120},
            }
        ),
        encoding="utf-8",
    )
    canonical_database = canonical / "edge_iiotset_canonical.sqlite3"
    with sqlite3.connect(canonical_database) as connection:
        connection.execute("CREATE TABLE completed_files(file_index INTEGER, source_file TEXT)")
        connection.executemany(
            "INSERT INTO completed_files VALUES (?, ?)",
            [(index, source[0]) for index, source in enumerate(sources)],
        )
    phase3a_database = tmp_path / "edge_iiotset_phase3a.sqlite3"
    with sqlite3.connect(phase3a_database) as connection:
        connection.execute(
            "CREATE TABLE source_label_counts("
            "source_file TEXT, task TEXT, label TEXT, record_count INTEGER)"
        )
        connection.executemany(
            "INSERT INTO source_label_counts VALUES (?, 'family', ?, ?)",
            [(source, family, count) for source, family, count in sources],
        )
    phase3a1 = tmp_path / "edge_iiotset_phase3a1.json"
    phase3a1.write_text(
        json.dumps(
            {
                "analysis_id": "edge-phase3a1-synthetic",
                "dataset": DatasetName.EDGE_IIOTSET.value,
                "canonical_corpus_id": "edge-synthetic-corpus",
                "candidate_policies": {
                    "source_capture_plus_mitm_pcap_bidirectional_flow": {
                        "primary_necessary_conditions_pass": True
                    }
                },
                "csv_pcap_alignment": {"packet_ordinal_alignment_accepted": True},
                "input_sha256": {"raw_csv": _sha256(raw_csv), "pcap": _sha256(pcap)},
            }
        ),
        encoding="utf-8",
    )
    return SplitBuildRequest(
        dataset=DatasetName.EDGE_IIOTSET,
        canonical_manifest=manifest_path,
        canonical_database=canonical_database,
        output_directory=tmp_path / "edge-output",
        work_database=tmp_path / "edge-output" / "resume.sqlite3",
        calibration_anchors_per_class=1,
        scoring_anchors_per_class=1,
        parquet_batch_size=17,
        phase3a_database=phase3a_database,
        phase3a1_report=phase3a1,
        edge_raw_mitm_csv=raw_csv,
        edge_mitm_pcap=pcap,
    )


def _request(tmp_path: Path, **overrides: object) -> SplitBuildRequest:
    manifest, database = _canonical_fixture(tmp_path)
    values: dict[str, object] = {
        "dataset": DatasetName.CIC_IOT_2023,
        "canonical_manifest": manifest,
        "canonical_database": database,
        "output_directory": tmp_path / "output",
        "work_database": tmp_path / "output" / "resume.sqlite3",
        "calibration_anchors_per_class": 1,
        "scoring_anchors_per_class": 1,
        "parquet_batch_size": 37,
    }
    values.update(overrides)
    return SplitBuildRequest(**values)  # type: ignore[arg-type]


def test_record_split_is_deterministic_and_respects_boundaries() -> None:
    basis_points = {"train": 7000, "validation": 1500, "test": 1500}
    first = [_record_split(f"record-{index}", 91, basis_points) for index in range(1000)]
    second = [_record_split(f"record-{index}", 91, basis_points) for index in range(1000)]

    assert first == second
    assert set(first) == {"train", "validation", "test"}


def test_group_assignment_is_deterministic_and_group_disjoint() -> None:
    groups = [_Group(f"group-{index}", "benign", index + 1) for index in range(8)]
    basis_points = {"train": 7000, "validation": 1500, "test": 1500}

    first = _assign_grouped_family(groups, basis_points, 77)
    second = _assign_grouped_family(list(reversed(groups)), basis_points, 77)

    assert first == second
    assert set(first) == {group.group_id for group in groups}
    assert set(first.values()) == {"train", "validation", "test"}


def test_group_assignment_rejects_fewer_than_three_groups() -> None:
    with pytest.raises(DataContractError, match="fewer than three groups"):
        _assign_grouped_family(
            [_Group("one", "benign", 10), _Group("two", "benign", 5)],
            {"train": 7000, "validation": 1500, "test": 1500},
            1,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_basis_points": 6999}, "sum to 10000"),
        ({"base_seed": -1}, "cannot be negative"),
        ({"calibration_anchors_per_class": 0}, "must be positive"),
        ({"parquet_batch_size": 0}, "must be positive"),
    ],
)
def test_request_rejects_invalid_controls(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(tmp_path, **overrides)


def test_edge_request_requires_grouping_evidence(tmp_path: Path) -> None:
    manifest, database = _canonical_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires Phase 3A/3A.1"):
        SplitBuildRequest(
            dataset=DatasetName.EDGE_IIOTSET,
            canonical_manifest=manifest,
            canonical_database=database,
            output_directory=tmp_path / "output",
            work_database=tmp_path / "resume.sqlite3",
        )


def test_plan_only_writes_a_nonmaterialized_contract(tmp_path: Path) -> None:
    request = _request(tmp_path)

    report = plan_phase3b_split(request)

    assert report["group_policy"] == "fingerprint_deduplicated_record_hash_v1"
    assert report["split_assignment_created"] is False
    assert len(cast(str, report["plan_id"])) == 64
    assert (request.output_directory / "cic_iot_2023_phase3b_plan.json").is_file()
    assert (request.output_directory / "cic_iot_2023_phase3b_plan.md").is_file()


def test_cic_builder_is_resumable_and_creates_disjoint_train_anchors(tmp_path: Path) -> None:
    request = _request(tmp_path)

    first = build_phase3b_split(request, progress_every=2)
    second = build_phase3b_split(request, progress_every=2)

    assert first["split_id"] == second["split_id"]
    assert first["records"] == 600
    assert sum(cast(dict[str, int], first["split_counts"]).values()) == 600
    family_counts = cast(dict[str, dict[str, int]], first["family_split_counts"])
    assert all(count > 0 for counts in family_counts.values() for count in counts.values())
    gates = cast(dict[str, bool], first["gates"])
    assert gates["assignment_cardinality_matches_canonical"] is True
    assert gates["anchors_disjoint_and_train_only"] is True

    anchors = pq.read_table(request.output_directory / "anchors.parquet").to_pylist()
    assert len(anchors) == 4
    assert len({row["record_id"] for row in anchors}) == 4
    assert {row["subset"] for row in anchors} == {"calibration", "scoring"}
    assignment_rows: list[dict[str, object]] = []
    for path in sorted((request.output_directory / "assignments").glob("*.parquet")):
        assignment_rows.extend(pq.read_table(path).to_pylist())
    split_by_record = {row["record_id"]: row["split"] for row in assignment_rows}
    assert all(split_by_record[row["record_id"]] == "train" for row in anchors)


def test_edge_builder_materializes_whole_source_and_mitm_flow_groups(tmp_path: Path) -> None:
    request = _edge_request(tmp_path)

    report = build_phase3b_split(request, progress_every=2)

    assert report["records"] == 150
    assert report["group_policy"] == "source_capture_plus_mitm_pcap_bidirectional_flow_v1"
    family_counts = cast(dict[str, dict[str, int]], report["family_split_counts"])
    assert all(count > 0 for counts in family_counts.values() for count in counts.values())
    plan = cast(dict[str, object], report)
    assert len(cast(str, plan["plan_id"])) == 64
    assignments: list[dict[str, object]] = []
    for path in sorted((request.output_directory / "assignments").glob("*.parquet")):
        assignments.extend(pq.read_table(path).to_pylist())
    group_splits: dict[str, set[str]] = {}
    for row in assignments:
        group_splits.setdefault(cast(str, row["group_id"]), set()).add(cast(str, row["split"]))
    assert len(group_splits) == 6
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_resume_database_rejects_a_changed_plan(tmp_path: Path) -> None:
    request = _request(tmp_path)
    build_phase3b_split(request)
    changed = SplitBuildRequest(
        dataset=request.dataset,
        canonical_manifest=request.canonical_manifest,
        canonical_database=request.canonical_database,
        output_directory=request.output_directory,
        work_database=request.work_database,
        base_seed=request.base_seed + 1,
        calibration_anchors_per_class=1,
        scoring_anchors_per_class=1,
    )

    with pytest.raises(DataContractError, match="identity mismatch"):
        build_phase3b_split(changed)


def test_builder_rejects_a_changed_canonical_shard(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifest = cast(
        dict[str, object], json.loads(request.canonical_manifest.read_text(encoding="utf-8"))
    )
    shard = cast(dict[str, object], cast(list[object], manifest["shards"])[0])
    shard["sha256"] = "0" * 64
    request.canonical_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataContractError, match="checksum mismatch"):
        build_phase3b_split(request)


def test_builder_rejects_family_counts_that_disagree_with_shards(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifest = cast(
        dict[str, object], json.loads(request.canonical_manifest.read_text(encoding="utf-8"))
    )
    family_counts = cast(dict[str, int], manifest["family_label_counts"])
    family_counts["benign"] += 1
    request.canonical_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataContractError, match="family counts do not match"):
        build_phase3b_split(request)


def test_builder_rejects_nonpositive_progress_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_phase3b_split(_request(tmp_path), progress_every=0)
