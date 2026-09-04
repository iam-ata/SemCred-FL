"""Phase 3A.1 evidence for Edge-IIoTset within-capture grouping surrogates."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import sqlite3
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Final, cast

from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import run_id

_SCHEMA_VERSION: Final = 2
_PLACEHOLDERS: Final = frozenset({"", "0", "0.0"})
_PCAP_FORMATS: Final = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}


@dataclass(frozen=True, slots=True)
class EdgeMitmRefinementRequest:
    """Inputs and quality gates for the bounded Phase 3A.1 audit."""

    canonical_manifest: Path
    canonical_database: Path
    phase3a_report: Path
    phase3a_database: Path
    raw_csv: Path
    pcap: Path
    output_directory: Path
    calibration_anchors_per_class: int = 32
    scoring_anchors_per_class: int = 32
    time_block_seconds: tuple[int, ...] = (30, 60, 120)
    minimum_ip_alignment_rows: int = 32

    def __post_init__(self) -> None:
        if self.calibration_anchors_per_class < 1 or self.scoring_anchors_per_class < 1:
            raise ValueError("Anchor counts must be positive")
        if not self.time_block_seconds or any(value < 1 for value in self.time_block_seconds):
            raise ValueError("Time-block durations must be positive")
        if len(set(self.time_block_seconds)) != len(self.time_block_seconds):
            raise ValueError("Time-block durations must be unique")
        if self.minimum_ip_alignment_rows < 1:
            raise ValueError("Minimum IP-alignment evidence must be positive")

    @property
    def total_anchors_per_class(self) -> int:
        """Return the two disjoint anchor subsets required per primary class."""

        return self.calibration_anchors_per_class + self.scoring_anchors_per_class


@dataclass(frozen=True, slots=True)
class PcapPacket:
    """Packet metadata needed for grouping and CSV-order validation."""

    timestamp_ns: int
    flow_key: str
    ip_endpoints: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(stream: BinaryIO, size: int, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise DataContractError(f"Truncated PCAP {context}")
    return value


def _pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _mac(value: bytes) -> str:
    return ":".join(f"{item:02x}" for item in value)


def _transport_flow(
    version: str,
    protocol: int,
    source: str,
    destination: str,
    payload: bytes,
) -> str:
    source_port = destination_port = 0
    if protocol in {6, 17} and len(payload) >= 4:
        source_port, destination_port = struct.unpack("!HH", payload[:4])
    left, right = _pair(f"{source}:{source_port}", f"{destination}:{destination_port}")
    return f"{version}|{protocol}|{left}<->{right}"


def _ethernet_metadata(frame: bytes) -> tuple[str, tuple[str, ...]]:
    if len(frame) < 14:
        return "unparsed|short_ethernet", ()
    destination_mac, source_mac = _mac(frame[:6]), _mac(frame[6:12])
    ethertype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    while ethertype in {0x8100, 0x88A8, 0x9100} and len(frame) >= offset + 4:
        ethertype = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
        offset += 4
    payload = frame[offset:]
    if ethertype == 0x0800 and len(payload) >= 20:
        header_length = (payload[0] & 0x0F) * 4
        if header_length >= 20 and len(payload) >= header_length:
            source = str(ipaddress.ip_address(payload[12:16]))
            destination = str(ipaddress.ip_address(payload[16:20]))
            protocol = payload[9]
            return (
                _transport_flow("ipv4", protocol, source, destination, payload[header_length:]),
                _pair(source, destination),
            )
    if ethertype == 0x86DD and len(payload) >= 40:
        source = str(ipaddress.ip_address(payload[8:24]))
        destination = str(ipaddress.ip_address(payload[24:40]))
        protocol = payload[6]
        return (
            _transport_flow("ipv6", protocol, source, destination, payload[40:]),
            _pair(source, destination),
        )
    if ethertype == 0x0806 and len(payload) >= 8:
        hardware_length = payload[4]
        protocol_length = payload[5]
        address_length = 8 + 2 * hardware_length + 2 * protocol_length
        if protocol_length == 4 and len(payload) >= address_length:
            sender_offset = 8 + hardware_length
            target_offset = sender_offset + protocol_length + hardware_length
            source = str(
                ipaddress.ip_address(payload[sender_offset : sender_offset + protocol_length])
            )
            destination = str(
                ipaddress.ip_address(payload[target_offset : target_offset + protocol_length])
            )
            left, right = _pair(source, destination)
            return f"arp|{left}<->{right}", (left, right)
    left, right = _pair(source_mac, destination_mac)
    return f"ethernet|{ethertype}|{left}<->{right}", ()


def read_classic_pcap(path: Path) -> list[PcapPacket]:
    """Read a classic Ethernet PCAP without accepting PCAP-NG or other link types."""

    packets: list[PcapPacket] = []
    with path.open("rb") as stream:
        magic = _read_exact(stream, 4, "global header")
        format_details = _PCAP_FORMATS.get(magic)
        if format_details is None:
            raise DataContractError("Unsupported PCAP magic; classic PCAP is required")
        byte_order, fractional_ns = format_details
        global_header = _read_exact(stream, 20, "global header")
        _, _, _, _, _, link_type = struct.unpack(f"{byte_order}HHIIII", global_header)
        if link_type != 1:
            raise DataContractError(f"Unsupported PCAP link type: {link_type}")
        while True:
            packet_header = stream.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                raise DataContractError("Truncated PCAP packet header")
            seconds, fraction, included_length, _ = struct.unpack(
                f"{byte_order}IIII", packet_header
            )
            frame = _read_exact(stream, included_length, "packet payload")
            flow_key, endpoints = _ethernet_metadata(frame)
            packets.append(
                PcapPacket(
                    timestamp_ns=seconds * 1_000_000_000 + fraction * fractional_ns,
                    flow_key=flow_key,
                    ip_endpoints=endpoints,
                )
            )
    if not packets:
        raise DataContractError("PCAP contains no packets")
    return packets


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error


def _read_only_sqlite(path: Path) -> sqlite3.Connection:
    """Open an existing evidence database without creating journals or lock files."""

    if not path.is_file():
        raise DataContractError(f"Evidence database is missing: {path}")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)


def _mitm_source_and_shard(
    request: EdgeMitmRefinementRequest, manifest: Mapping[str, object]
) -> tuple[str, Path]:
    with _read_only_sqlite(request.canonical_database) as connection:
        rows = connection.execute(
            "SELECT file_index, source_file FROM completed_files "
            "WHERE instr(lower(source_file), 'mitm') > 0"
        ).fetchall()
    if len(rows) != 1:
        raise DataContractError(f"Expected one canonical MITM source, found {len(rows)}")
    file_index, source_file = cast(tuple[int, str], rows[0])
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not 0 <= file_index < len(shards):
        raise DataContractError("Canonical manifest cannot resolve the MITM shard")
    shard = cast(dict[str, object], shards[file_index])
    relative = shard.get("file")
    if not isinstance(relative, str):
        raise DataContractError("Canonical MITM shard has no file path")
    base = request.canonical_manifest.parent.resolve()
    shard_path = (base / relative).resolve()
    try:
        shard_path.relative_to(base)
    except ValueError as error:
        raise DataContractError("Canonical MITM shard escapes its artifact directory") from error
    if not shard_path.is_file() or shard_path.stat().st_size != shard.get("byte_size"):
        raise DataContractError("Canonical MITM shard is missing or has changed")
    if _file_sha256(shard_path) != shard.get("sha256"):
        raise DataContractError("Canonical MITM shard checksum mismatch")
    return source_file, shard_path


def _canonical_source_rows(shard_path: Path, source_file: str) -> list[int]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:
        raise DataContractError("Phase 3A.1 requires the project ML dependencies") from error
    table = pq.read_table(shard_path, columns=["source_file", "source_row", "family_label"])
    values = table.to_pydict()
    sources = cast(list[str], values["source_file"])
    rows = cast(list[int], values["source_row"])
    families = cast(list[str], values["family_label"])
    if not rows or any(value != source_file for value in sources):
        raise DataContractError("Canonical MITM shard source identity mismatch")
    if any(value != "man_in_the_middle" for value in families):
        raise DataContractError("Canonical MITM shard primary-family mismatch")
    if len(rows) != len(set(rows)):
        raise DataContractError("Canonical MITM source rows are not unique")
    return sorted(rows)


def _csv_ip(value: str) -> str | None:
    if value.strip() in _PLACEHOLDERS:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def _csv_alignment(
    path: Path, packets: Sequence[PcapPacket], minimum_ip_rows: int
) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise DataContractError("MITM CSV is empty") from error
        required = {"ip.src_host", "ip.dst_host", "Attack_type"}
        if not required.issubset(header):
            raise DataContractError(f"MITM CSV lacks columns: {sorted(required - set(header))}")
        source_index = header.index("ip.src_host")
        destination_index = header.index("ip.dst_host")
        label_index = header.index("Attack_type")
        raw_rows = exact_width = ip_compared = ip_matched = ip_mismatched = 0
        wrong_labels = 0
        for raw_rows, row in enumerate(reader, start=1):
            if len(row) != len(header):
                continue
            exact_width += 1
            if row[label_index] != "MITM":
                wrong_labels += 1
            if raw_rows > len(packets):
                continue
            csv_endpoints = {
                value
                for value in (_csv_ip(row[source_index]), _csv_ip(row[destination_index]))
                if value is not None
            }
            packet_endpoints = set(packets[raw_rows - 1].ip_endpoints)
            if not csv_endpoints or not packet_endpoints:
                continue
            ip_compared += 1
            if csv_endpoints.issubset(packet_endpoints):
                ip_matched += 1
            else:
                ip_mismatched += 1
    accepted = (
        raw_rows == len(packets)
        and wrong_labels == 0
        and ip_compared >= minimum_ip_rows
        and ip_mismatched == 0
    )
    return {
        "raw_csv_rows": raw_rows,
        "pcap_packets": len(packets),
        "exact_width_csv_rows": exact_width,
        "wrong_exact_width_labels": wrong_labels,
        "ip_rows_compared": ip_compared,
        "ip_rows_matched": ip_matched,
        "ip_rows_mismatched": ip_mismatched,
        "minimum_ip_alignment_rows": minimum_ip_rows,
        "packet_ordinal_alignment_accepted": accepted,
    }


def _source_family_counts(path: Path) -> dict[str, tuple[str, int]]:
    with _read_only_sqlite(path) as connection:
        rows = connection.execute(
            "SELECT source_file, label, record_count FROM source_label_counts "
            "WHERE task='family' ORDER BY source_file, label"
        ).fetchall()
    result: dict[str, tuple[str, int]] = {}
    for source_file, label, count in cast(list[tuple[str, str, int]], rows):
        if source_file in result:
            raise DataContractError(f"Source file has mixed primary families: {source_file}")
        result[source_file] = (label, count)
    if not result:
        raise DataContractError("Phase 3A database has no primary-family source counts")
    return result


def _hybrid_summary(
    source_counts: Mapping[str, tuple[str, int]],
    mitm_source: str,
    mitm_groups: Counter[str],
    anchors_per_class: int,
    alignment_accepted: bool,
) -> dict[str, object]:
    by_family: defaultdict[str, list[int]] = defaultdict(list)
    for source, (family, count) in source_counts.items():
        if source != mitm_source:
            by_family[family].append(count)
    by_family["man_in_the_middle"].extend(mitm_groups.values())
    families: dict[str, dict[str, object]] = {}
    for family, counts in sorted(by_family.items()):
        records = sum(counts)
        families[family] = {
            "records": records,
            "groups": len(counts),
            "largest_group_records": max(counts, default=0),
            "three_way_group_necessary_condition": len(counts) >= 3,
            "anchor_capacity_necessary_condition": records >= anchors_per_class,
        }
    necessary = all(
        cast(bool, item["three_way_group_necessary_condition"])
        and cast(bool, item["anchor_capacity_necessary_condition"])
        for item in families.values()
    )
    return {
        "groups": sum(len(counts) for counts in by_family.values()),
        "largest_group_records": max(
            (max(counts, default=0) for counts in by_family.values()), default=0
        ),
        "mitm_groups": len(mitm_groups),
        "mitm_largest_group_records": max(mitm_groups.values(), default=0),
        "family": families,
        "primary_necessary_conditions_pass": necessary,
        "packet_ordinal_alignment_required": True,
        "status": (
            "candidate_only"
            if necessary and alignment_accepted
            else "alignment_gate_failed"
            if not alignment_accepted
            else "necessary_conditions_fail"
        ),
    }


def _write_markdown(report: Mapping[str, object], path: Path) -> None:
    alignment = cast(dict[str, object], report["csv_pcap_alignment"])
    candidates = cast(dict[str, dict[str, object]], report["candidate_policies"])
    lines = [
        "# Phase 3A.1: Edge-IIoTset MITM grouping refinement",
        "",
        f"- Analysis ID: `{report['analysis_id']}`",
        f"- Canonical corpus ID: `{report['canonical_corpus_id']}`",
        f"- Phase 3A analysis ID: `{report['phase3a_analysis_id']}`",
        f"- Canonical MITM records: {report['canonical_mitm_records']:,}",
        (
            "- CSV-to-PCAP packet-ordinal alignment accepted: "
            f"{alignment['packet_ordinal_alignment_accepted']}"
        ),
        "- Train/validation/test assignment created: no",
        "",
        "## Alignment evidence",
        "",
        f"- Raw CSV rows / PCAP packets: {alignment['raw_csv_rows']:,} / "
        f"{alignment['pcap_packets']:,}",
        f"- Exact-width CSV rows: {alignment['exact_width_csv_rows']:,}",
        f"- Comparable IP rows: {alignment['ip_rows_compared']:,}",
        f"- IP matches / mismatches: {alignment['ip_rows_matched']:,} / "
        f"{alignment['ip_rows_mismatched']:,}",
        "",
        "## Candidate hybrid policies",
        "",
        "| Policy | Total groups | MITM groups | Largest MITM group | Primary gate | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, candidate in candidates.items():
        lines.append(
            f"| {name} | {candidate['groups']:,} | {candidate['mitm_groups']:,} | "
            f"{candidate['mitm_largest_group_records']:,} | "
            f"{candidate['primary_necessary_conditions_pass']} | {candidate['status']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            (
                "The PCAP-derived flow and time-block keys subdivide one continuous MITM "
                "capture. They are within-capture dependence surrogates, not independent "
                "capture sessions. A passing row is only a necessary-condition result and "
                "does not create or freeze a split."
            ),
            "",
            "Fine-grained evaluation remains secondary and is not approved by this audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_edge_mitm_refinement(request: EdgeMitmRefinementRequest) -> dict[str, object]:
    """Evaluate bounded PCAP-derived MITM groups without assigning data splits."""

    manifest = _load_json(request.canonical_manifest, "canonical manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("dataset") != "edge_iiotset":
        raise DataContractError("Phase 3A.1 requires an Edge-IIoTset canonical manifest")
    phase3a = _load_json(request.phase3a_report, "Phase 3A report")
    if phase3a.get("dataset") != "edge_iiotset":
        raise DataContractError("Phase 3A report dataset mismatch")
    source_file, shard_path = _mitm_source_and_shard(request, manifest)
    if request.raw_csv.name.casefold() != Path(source_file).name.casefold():
        raise DataContractError("MITM raw CSV does not match canonical source identity")
    packets = read_classic_pcap(request.pcap)
    canonical_rows = _canonical_source_rows(shard_path, source_file)
    if canonical_rows[-1] > len(packets):
        raise DataContractError("Canonical MITM source row exceeds the PCAP packet count")
    alignment = _csv_alignment(request.raw_csv, packets, request.minimum_ip_alignment_rows)
    selected = [packets[source_row - 1] for source_row in canonical_rows]
    origin = min(packet.timestamp_ns for packet in packets)
    groups: dict[str, Counter[str]] = {
        "source_capture_plus_mitm_pcap_bidirectional_flow": Counter(
            packet.flow_key for packet in selected
        )
    }
    for seconds in sorted(request.time_block_seconds):
        width_ns = seconds * 1_000_000_000
        groups[f"source_capture_plus_mitm_pcap_{seconds}s_block"] = Counter(
            str((packet.timestamp_ns - origin) // width_ns) for packet in selected
        )
    source_counts = _source_family_counts(request.phase3a_database)
    if source_file not in source_counts:
        raise DataContractError("Phase 3A database has no MITM source count")
    expected_mitm = source_counts[source_file]
    if expected_mitm != ("man_in_the_middle", len(canonical_rows)):
        raise DataContractError("Phase 3A and canonical MITM counts disagree")
    accepted = cast(bool, alignment["packet_ordinal_alignment_accepted"])
    candidates = {
        name: _hybrid_summary(
            source_counts,
            source_file,
            values,
            request.total_anchors_per_class,
            accepted,
        )
        for name, values in groups.items()
    }
    stable = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": "edge_iiotset",
        "canonical_corpus_id": manifest["corpus_id"],
        "phase3a_analysis_id": phase3a["analysis_id"],
        "mitm_source_file": source_file,
        "canonical_mitm_records": len(canonical_rows),
        "required_disjoint_anchors_per_primary_class": request.total_anchors_per_class,
        "input_sha256": {
            "raw_csv": _file_sha256(request.raw_csv),
            "pcap": _file_sha256(request.pcap),
            "canonical_shard": _file_sha256(shard_path),
        },
        "csv_pcap_alignment": alignment,
        "candidate_policies": candidates,
        "pcap_flow_parser": {
            "parser_schema_version": _SCHEMA_VERSION,
            "classic_pcap_only": True,
            "ethernet_link_type_required": True,
            "arp_protocol_addresses_included": True,
            "unparsed_selected_packets": sum(
                1 for packet in selected if packet.flow_key.startswith("unparsed|")
            ),
        },
        "split_assignment_created": False,
        "group_policy_frozen": False,
        "fine_grained_evaluation_approved": False,
        "surrogate_boundary": (
            "PCAP-derived flow and time blocks subdivide one continuous MITM capture; "
            "they are not independent capture sessions."
        ),
    }
    report = {
        "analysis_id": run_id(stable),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        **stable,
    }
    request.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = request.output_directory / "edge_iiotset_phase3a1.json"
    markdown_path = request.output_directory / "edge_iiotset_phase3a1.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, markdown_path)
    return report
