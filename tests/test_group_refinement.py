"""Tests for the bounded Phase 3A.1 Edge MITM refinement audit."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import sqlite3
import struct
from collections import Counter
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.group_refinement import (
    EdgeMitmRefinementRequest,
    PcapPacket,
    _csv_alignment,
    _csv_ip,
    _ethernet_metadata,
    _hybrid_summary,
    _read_only_sqlite,
    _source_family_counts,
    analyze_edge_mitm_refinement,
    read_classic_pcap,
)


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
    udp = struct.pack("!HHHH", port, port, 8, 0)
    return ethernet + ipv4 + udp


def _write_pcap(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    with path.open("wb") as stream:
        stream.write(b"\xd4\xc3\xb2\xa1")
        stream.write(struct.pack("<HHIIII", 2, 4, 0, 0, 65_535, 1))
        for timestamp, source, destination, port in rows:
            frame = _ipv4_udp_frame(source, destination, port)
            stream.write(struct.pack("<IIII", timestamp, 0, len(frame), len(frame)))
            stream.write(frame)


def _arp_frame(source: str, destination: str) -> bytes:
    source_ip = bytes(int(part) for part in source.split("."))
    destination_ip = bytes(int(part) for part in destination.split("."))
    destination_mac = bytes.fromhex("001122334455")
    source_mac = bytes.fromhex("66778899aabb")
    ethernet = destination_mac + source_mac + bytes.fromhex("0806")
    arp = struct.pack(
        "!HHBBH6s4s6s4s",
        1,
        0x0800,
        6,
        4,
        1,
        source_mac,
        source_ip,
        destination_mac,
        destination_ip,
    )
    return ethernet + arp


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> EdgeMitmRefinementRequest:
    source_file = "Attack traffic/MITM_attack.csv"
    packet_rows = [
        (100, "192.168.0.10", "192.168.0.20", 1000),
        (110, "192.168.0.20", "192.168.0.10", 1000),
        (170, "192.168.0.10", "192.168.0.30", 2000),
        (180, "192.168.0.30", "192.168.0.10", 2000),
        (240, "192.168.0.10", "192.168.0.40", 3000),
        (250, "192.168.0.40", "192.168.0.10", 3000),
    ]
    raw_csv = tmp_path / "MITM_attack.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ip.src_host", "ip.dst_host", "Attack_type"])
        writer.writerows(
            [[source, destination, "MITM"] for _, source, destination, _ in packet_rows]
        )
    pcap = tmp_path / "MITM.pcap"
    _write_pcap(pcap, packet_rows)

    canonical = tmp_path / "canonical"
    shards = canonical / "shards"
    shards.mkdir(parents=True)
    shard = shards / "0000-test.parquet"
    pq.write_table(
        pa.table(
            {
                "source_file": [source_file] * 6,
                "source_row": list(range(1, 7)),
                "family_label": ["man_in_the_middle"] * 6,
            }
        ),
        shard,
    )
    manifest = canonical / "edge_iiotset_canonical.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_id": "corpus-test",
                "provenance": {"dataset": "edge_iiotset"},
                "shards": [
                    {
                        "file": "shards/0000-test.parquet",
                        "byte_size": shard.stat().st_size,
                        "sha256": _sha256(shard),
                        "row_count": 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    canonical_database = canonical / "edge_iiotset_canonical.sqlite3"
    with sqlite3.connect(canonical_database) as connection:
        connection.execute("CREATE TABLE completed_files(file_index INTEGER, source_file TEXT)")
        connection.execute("INSERT INTO completed_files VALUES (?, ?)", (0, source_file))

    phase3a_report = tmp_path / "edge_iiotset_phase3a.json"
    phase3a_report.write_text(
        json.dumps({"dataset": "edge_iiotset", "analysis_id": "phase3a-test"}),
        encoding="utf-8",
    )
    phase3a_database = tmp_path / "edge_iiotset_phase3a.sqlite3"
    with sqlite3.connect(phase3a_database) as connection:
        connection.execute(
            "CREATE TABLE source_label_counts("
            "source_file TEXT, task TEXT, label TEXT, record_count INTEGER)"
        )
        connection.executemany(
            "INSERT INTO source_label_counts VALUES (?, 'family', ?, ?)",
            [
                (source_file, "man_in_the_middle", 6),
                ("Normal traffic/a.csv", "benign", 10),
                ("Normal traffic/b.csv", "benign", 10),
                ("Normal traffic/c.csv", "benign", 10),
            ],
        )
    return EdgeMitmRefinementRequest(
        canonical_manifest=manifest,
        canonical_database=canonical_database,
        phase3a_report=phase3a_report,
        phase3a_database=phase3a_database,
        raw_csv=raw_csv,
        pcap=pcap,
        output_directory=tmp_path / "phase3a1",
        calibration_anchors_per_class=1,
        scoring_anchors_per_class=1,
        time_block_seconds=(60,),
        minimum_ip_alignment_rows=3,
    )


def test_phase3a1_accepts_aligned_pcap_surrogates(tmp_path: Path) -> None:
    request = _fixture(tmp_path)

    report = analyze_edge_mitm_refinement(request)

    alignment = cast(dict[str, object], report["csv_pcap_alignment"])
    assert alignment["packet_ordinal_alignment_accepted"] is True
    assert alignment["ip_rows_compared"] == 6
    candidates = cast(dict[str, dict[str, object]], report["candidate_policies"])
    flow = candidates["source_capture_plus_mitm_pcap_bidirectional_flow"]
    block = candidates["source_capture_plus_mitm_pcap_60s_block"]
    assert flow["mitm_groups"] == 3
    assert block["mitm_groups"] == 3
    assert flow["status"] == "candidate_only"
    assert block["primary_necessary_conditions_pass"] is True
    assert report["split_assignment_created"] is False
    assert (request.output_directory / "edge_iiotset_phase3a1.json").is_file()
    markdown = (request.output_directory / "edge_iiotset_phase3a1.md").read_text(encoding="utf-8")
    assert "within-capture dependence surrogates" in markdown


def test_phase3a1_rejects_packet_ordinal_alignment_mismatch(tmp_path: Path) -> None:
    request = _fixture(tmp_path)
    rows = list(csv.reader(request.raw_csv.read_text(encoding="utf-8").splitlines()))
    rows[1][0] = "192.168.99.99"
    with request.raw_csv.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)

    report = analyze_edge_mitm_refinement(request)

    alignment = cast(dict[str, object], report["csv_pcap_alignment"])
    assert alignment["packet_ordinal_alignment_accepted"] is False
    assert alignment["ip_rows_mismatched"] == 1
    candidates = cast(dict[str, dict[str, object]], report["candidate_policies"])
    assert all(item["status"] == "alignment_gate_failed" for item in candidates.values())


def test_classic_pcap_parser_normalizes_bidirectional_flows(tmp_path: Path) -> None:
    path = tmp_path / "two.pcap"
    _write_pcap(
        path,
        [
            (1, "10.0.0.1", "10.0.0.2", 53),
            (2, "10.0.0.2", "10.0.0.1", 53),
        ],
    )

    packets = read_classic_pcap(path)

    assert packets[0].flow_key == packets[1].flow_key
    assert packets[0].ip_endpoints == ("10.0.0.1", "10.0.0.2")
    assert packets[1].timestamp_ns - packets[0].timestamp_ns == 1_000_000_000


def test_classic_pcap_parser_rejects_unknown_container(tmp_path: Path) -> None:
    path = tmp_path / "capture.pcapng"
    path.write_bytes(b"\x0a\x0d\x0d\x0a")

    with pytest.raises(DataContractError, match="classic PCAP"):
        read_classic_pcap(path)


def test_classic_pcap_parser_extracts_arp_protocol_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "arp.pcap"
    frame = _arp_frame("192.168.0.10", "192.168.0.20")
    with path.open("wb") as stream:
        stream.write(b"\xd4\xc3\xb2\xa1")
        stream.write(struct.pack("<HHIIII", 2, 4, 0, 0, 65_535, 1))
        stream.write(struct.pack("<IIII", 1, 0, len(frame), len(frame)))
        stream.write(frame)

    packet = read_classic_pcap(path)[0]

    assert packet.flow_key == "arp|192.168.0.10<->192.168.0.20"
    assert packet.ip_endpoints == ("192.168.0.10", "192.168.0.20")


def test_ethernet_parser_covers_vlan_ipv6_and_fallbacks() -> None:
    ipv4 = _ipv4_udp_frame("10.0.0.1", "10.0.0.2", 53)
    vlan = ipv4[:12] + bytes.fromhex("810000010800") + ipv4[14:]
    vlan_key, vlan_endpoints = _ethernet_metadata(vlan)
    assert vlan_key.startswith("ipv4|17|")
    assert vlan_endpoints == ("10.0.0.1", "10.0.0.2")

    source = ipaddress.ip_address("2001:db8::1").packed
    destination = ipaddress.ip_address("2001:db8::2").packed
    ethernet = bytes.fromhex("00112233445566778899aabb86dd")
    ipv6 = struct.pack("!IHBB16s16s", 6 << 28, 8, 17, 64, source, destination)
    udp = struct.pack("!HHHH", 5353, 5353, 8, 0)
    ipv6_key, ipv6_endpoints = _ethernet_metadata(ethernet + ipv6 + udp)
    assert ipv6_key.startswith("ipv6|17|")
    assert ipv6_endpoints == ("2001:db8::1", "2001:db8::2")

    assert _ethernet_metadata(b"short") == ("unparsed|short_ethernet", ())
    fallback, endpoints = _ethernet_metadata(bytes.fromhex("00112233445566778899aabb88b5"))
    assert fallback.startswith("ethernet|34997|")
    assert endpoints == ()


@pytest.mark.parametrize(
    ("tail", "link_type", "message"),
    [
        (b"", 1, "no packets"),
        (b"", 101, "link type"),
        (b"short", 1, "packet header"),
        (struct.pack("<IIII", 1, 0, 10, 10) + b"xx", 1, "packet payload"),
    ],
)
def test_classic_pcap_parser_rejects_structural_errors(
    tmp_path: Path, tail: bytes, link_type: int, message: str
) -> None:
    path = tmp_path / f"bad-{link_type}-{len(tail)}.pcap"
    path.write_bytes(
        b"\xd4\xc3\xb2\xa1" + struct.pack("<HHIIII", 2, 4, 0, 0, 65_535, link_type) + tail
    )

    with pytest.raises(DataContractError, match=message):
        read_classic_pcap(path)


def test_refinement_helpers_reject_missing_database_and_classify_values(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="missing"):
        _read_only_sqlite(tmp_path / "missing.sqlite3")
    assert _csv_ip("0.0") is None
    assert _csv_ip("not-an-ip") is None
    assert _csv_ip(" 10.0.0.1 ") == "10.0.0.1"


def test_hybrid_summary_reports_failed_necessary_condition() -> None:
    summary = _hybrid_summary(
        {
            "MITM.csv": ("man_in_the_middle", 2),
            "normal.csv": ("benign", 2),
        },
        "MITM.csv",
        Counter({"one-flow": 2}),
        anchors_per_class=2,
        alignment_accepted=True,
    )

    assert summary["primary_necessary_conditions_pass"] is False
    assert summary["status"] == "necessary_conditions_fail"


def test_csv_alignment_rejects_empty_and_incomplete_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataContractError, match="empty"):
        _csv_alignment(empty, [PcapPacket(0, "flow", ())], 1)

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("ip.src_host,Attack_type\n10.0.0.1,MITM\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="lacks columns"):
        _csv_alignment(incomplete, [PcapPacket(0, "flow", ("10.0.0.1",))], 1)


def test_csv_alignment_counts_wrong_labels_extra_rows_and_bad_width(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(
        "ip.src_host,ip.dst_host,Attack_type\n"
        "10.0.0.1,10.0.0.2,Normal\n"
        "10.0.0.1,10.0.0.2,MITM\n"
        "bad,width\n",
        encoding="utf-8",
    )
    packet = PcapPacket(0, "flow", ("10.0.0.1", "10.0.0.2"))

    result = _csv_alignment(path, [packet], 1)

    assert result["raw_csv_rows"] == 3
    assert result["exact_width_csv_rows"] == 2
    assert result["wrong_exact_width_labels"] == 1
    assert result["packet_ordinal_alignment_accepted"] is False


def test_source_family_counts_rejects_empty_and_mixed_sources(tmp_path: Path) -> None:
    path = tmp_path / "counts.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE source_label_counts("
            "source_file TEXT, task TEXT, label TEXT, record_count INTEGER)"
        )
    with pytest.raises(DataContractError, match="no primary-family"):
        _source_family_counts(path)

    with sqlite3.connect(path) as connection:
        connection.executemany(
            "INSERT INTO source_label_counts VALUES (?, 'family', ?, ?)",
            [("mixed.csv", "benign", 1), ("mixed.csv", "malware", 1)],
        )
    with pytest.raises(DataContractError, match="mixed primary families"):
        _source_family_counts(path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"calibration_anchors_per_class": 0}, "Anchor counts"),
        ({"time_block_seconds": ()}, "Time-block durations"),
        ({"time_block_seconds": (60, 60)}, "unique"),
        ({"minimum_ip_alignment_rows": 0}, "alignment evidence"),
    ],
)
def test_refinement_request_rejects_invalid_controls(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "canonical_manifest": tmp_path / "manifest.json",
        "canonical_database": tmp_path / "canonical.sqlite3",
        "phase3a_report": tmp_path / "phase3a.json",
        "phase3a_database": tmp_path / "phase3a.sqlite3",
        "raw_csv": tmp_path / "raw.csv",
        "pcap": tmp_path / "raw.pcap",
        "output_directory": tmp_path,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        EdgeMitmRefinementRequest(**values)  # type: ignore[arg-type]
