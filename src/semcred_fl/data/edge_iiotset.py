"""Schema-first adapter for raw per-scenario Edge-IIoTset CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import (
    DataContractError,
    DatasetRecord,
    DatasetSchema,
    FileLoadResult,
    IngestionReport,
)
from semcred_fl.data.record_ids import normalized_source_file, record_id
from semcred_fl.data.validation import require_positive_limit, validate_header

EDGE_SOURCE_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.opcode",
    "arp.hw.size",
    "arp.src.proto_ipv4",
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "icmp.unused",
    "http.file_data",
    "http.content_length",
    "http.request.uri.query",
    "http.request.method",
    "http.referer",
    "http.request.full_uri",
    "http.request.version",
    "http.response",
    "http.tls_port",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.options",
    "tcp.payload",
    "tcp.seq",
    "tcp.srcport",
    "udp.port",
    "udp.stream",
    "udp.time_delta",
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
    "dns.retransmit_request_in",
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.msg",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver",
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
    "Attack_label",
    "Attack_type",
)

EDGE_PROHIBITED_MODEL_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "http.referer",
    "icmp.transmit_timestamp",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "dns.qry.name",
    "mqtt.msg",
    "mqtt.topic",
)

EDGE_LABEL_COLUMNS = ("Attack_label", "Attack_type")
EDGE_MODEL_FEATURE_COLUMNS = tuple(
    column
    for column in EDGE_SOURCE_COLUMNS
    if column not in set(EDGE_PROHIBITED_MODEL_COLUMNS) | set(EDGE_LABEL_COLUMNS)
)
EDGE_CATEGORICAL_COLUMNS = (
    "http.request.method",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
)
EDGE_UNRESOLVED_COLUMNS = tuple(
    column for column in EDGE_MODEL_FEATURE_COLUMNS if column not in EDGE_CATEGORICAL_COLUMNS
)

EDGE_FINE_TO_FAMILY = {
    "Backdoor": "malware",
    "DDoS_HTTP": "dos_ddos",
    "DDoS_ICMP": "dos_ddos",
    "DDoS_TCP": "dos_ddos",
    "DDoS_UDP": "dos_ddos",
    "Fingerprinting": "information_gathering",
    "MITM": "man_in_the_middle",
    "Normal": "benign",
    "Password": "malware",
    "Port_Scanning": "information_gathering",
    "Ransomware": "malware",
    "SQL_injection": "injection",
    "Uploading": "injection",
    "Vulnerability_scanner": "information_gathering",
    "XSS": "injection",
}

EDGE_RAW_FINE_TO_CANONICAL = {
    **{fine_label: fine_label for fine_label in EDGE_FINE_TO_FAMILY},
    "OS_Fingerprinting": "Fingerprinting",
}


def canonical_edge_fine_label(raw_label: str) -> str | None:
    """Return the canonical fine label for an exact raw Edge-IIoTset label."""

    return EDGE_RAW_FINE_TO_CANONICAL.get(raw_label)


EDGE_ATTACK_FILE_TO_FINE = {
    "Backdoor_attack.csv": "Backdoor",
    "DDoS_HTTP_Flood_attack.csv": "DDoS_HTTP",
    "DDoS_ICMP_Flood_attack.csv": "DDoS_ICMP",
    "DDoS_TCP_SYN_Flood_attack.csv": "DDoS_TCP",
    "DDoS_UDP_Flood_attack.csv": "DDoS_UDP",
    "MITM_attack.csv": "MITM",
    "OS_Fingerprinting_attack.csv": "Fingerprinting",
    "Password_attack.csv": "Password",
    "Port_Scanning_attack.csv": "Port_Scanning",
    "Ransomware_attack.csv": "Ransomware",
    "SQL_injection_attack.csv": "SQL_injection",
    "Uploading_attack.csv": "Uploading",
    "Vulnerability_scanner_attack.csv": "Vulnerability_scanner",
    "XSS_attack.csv": "XSS",
}

EDGE_SCHEMA = DatasetSchema(
    dataset=DatasetName.EDGE_IIOTSET,
    source_columns=EDGE_SOURCE_COLUMNS,
    label_columns=EDGE_LABEL_COLUMNS,
    model_feature_columns=EDGE_MODEL_FEATURE_COLUMNS,
    grouping_columns=("source_file", "frame_time", "endpoint_pair"),
    categorical_columns=EDGE_CATEGORICAL_COLUMNS,
    continuous_columns=(),
    unresolved_columns=EDGE_UNRESOLVED_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class EdgeIIoTsetAdapter:
    """Read raw Edge-IIoTset sources while isolating leakage-prone fields."""

    root: Path
    schema: DatasetSchema = field(default=EDGE_SCHEMA, init=False)

    def discover_files(self) -> tuple[Path, ...]:
        """Return raw attack and normal CSVs, excluding selected ML/DNN files."""

        if not self.root.is_dir():
            raise FileNotFoundError(f"Edge-IIoTset source directory not found: {self.root}")
        files = tuple(
            sorted(
                (
                    *self.root.glob("Attack traffic/*.csv"),
                    *self.root.glob("Normal traffic/*/*.csv"),
                ),
                key=lambda path: normalized_source_file(self.root, path).casefold(),
            )
        )
        if not files:
            raise DataContractError(f"No raw Edge-IIoTset CSV files found under {self.root}")
        for source_file in files:
            self.expected_fine_label(source_file)
        return files

    def expected_fine_label(self, source_file: Path) -> str:
        """Return the canonical fine label implied by a raw source file's identity."""

        relative = normalized_source_file(self.root, source_file)
        parts = Path(relative).parts
        if parts and parts[0] == "Normal traffic":
            return "Normal"
        try:
            return EDGE_ATTACK_FILE_TO_FINE[source_file.name]
        except KeyError as exc:
            raise DataContractError(f"Unknown Edge-IIoTset source file: {relative}") from exc

    def load_file(self, source_file: Path, *, max_rows: int) -> FileLoadResult:
        """Load a bounded prefix and reject malformed or mislabeled records."""

        require_positive_limit(max_rows, "max_rows")
        relative = normalized_source_file(self.root, source_file)
        expected_label = self.expected_fine_label(source_file)
        indices = {column: index for index, column in enumerate(self.schema.source_columns)}
        records: list[DatasetRecord] = []
        rows_seen = 0
        invalid_width = 0
        try:
            with source_file.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header is None:
                    raise DataContractError(f"Empty Edge-IIoTset source file: {relative}")
                validate_header(header, self.schema.source_columns, relative)
                for source_row, row in enumerate(reader, start=1):
                    if rows_seen >= max_rows:
                        break
                    rows_seen += 1
                    current_id = record_id(self.schema.dataset, relative, source_row)
                    if len(row) != len(self.schema.source_columns):
                        invalid_width += 1
                        continue
                    raw_fine_label = row[indices["Attack_type"]].strip()
                    fine_label = canonical_edge_fine_label(raw_fine_label)
                    if fine_label is None:
                        raise DataContractError(
                            f"Unknown Edge-IIoTset label {raw_fine_label!r} "
                            f"in {relative}:{source_row}"
                        )
                    if fine_label != expected_label:
                        raise DataContractError(
                            f"Source/row label mismatch in {relative}:{source_row}; "
                            f"expected={expected_label}, actual={fine_label}"
                        )
                    binary_label = row[indices["Attack_label"]].strip()
                    expected_binary = "0" if fine_label == "Normal" else "1"
                    if binary_label != expected_binary:
                        raise DataContractError(
                            f"Binary/fine label mismatch in {relative}:{source_row}"
                        )

                    feature_values = tuple(
                        (column, row[indices[column]].strip())
                        for column in self.schema.model_feature_columns
                    )
                    endpoints = sorted(
                        (
                            row[indices["ip.src_host"]].strip(),
                            row[indices["ip.dst_host"]].strip(),
                        )
                    )
                    grouping = (
                        ("source_file", relative),
                        ("frame_time", row[indices["frame.time"]].strip()),
                        ("endpoint_pair", "|".join(endpoints)),
                    )
                    records.append(
                        DatasetRecord(
                            dataset=self.schema.dataset,
                            record_id=current_id,
                            source_file=relative,
                            source_row=source_row,
                            fine_label=fine_label,
                            family_label=EDGE_FINE_TO_FAMILY[fine_label],
                            features=feature_values,
                            grouping=grouping,
                        )
                    )
        except csv.Error as exc:
            raise DataContractError(f"CSV parsing failed for {relative}: {exc}") from exc

        return FileLoadResult(
            records=tuple(records),
            report=IngestionReport(
                source_file=relative,
                rows_seen=rows_seen,
                rows_emitted=len(records),
                invalid_width=invalid_width,
            ),
        )
