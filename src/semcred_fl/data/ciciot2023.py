"""Schema-first adapter for the official per-capture CICIoT2023 CSV files."""

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
from semcred_fl.data.validation import is_finite_number, require_positive_limit, validate_header

CIC_SOURCE_COLUMNS = (
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
)

CIC_SOURCE_DIRECTORY_TO_FINE = {
    "Backdoor_Malware": "Backdoor_Malware",
    "Benign_Final": "BenignTraffic",
    "BrowserHijacking": "BrowserHijacking",
    "CommandInjection": "CommandInjection",
    "DDoS-ACK_Fragmentation": "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood": "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood": "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation": "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_FLOOD": "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFLOOD": "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood": "DDoS-SYN_Flood",
    "DDoS-SlowLoris": "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood": "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood": "DDoS-TCP_Flood",
    "DDoS-UDP_Flood": "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation": "DDoS-UDP_Fragmentation",
    "DNS_Spoofing": "DNS_Spoofing",
    "DictionaryBruteForce": "DictionaryBruteForce",
    "DoS-HTTP_Flood": "DoS-HTTP_Flood",
    "DoS-SYN_Flood": "DoS-SYN_Flood",
    "DoS-TCP_Flood": "DoS-TCP_Flood",
    "DoS-UDP_Flood": "DoS-UDP_Flood",
    "MITM-ArpSpoofing": "MITM-ArpSpoofing",
    "Mirai-greeth_flood": "Mirai-greeth_flood",
    "Mirai-greip_flood": "Mirai-greip_flood",
    "Mirai-udpplain": "Mirai-udpplain",
    "Recon-HostDiscovery": "Recon-HostDiscovery",
    "Recon-OSScan": "Recon-OSScan",
    "Recon-PingSweep": "Recon-PingSweep",
    "Recon-PortScan": "Recon-PortScan",
    "SqlInjection": "SqlInjection",
    "Uploading_Attack": "Uploading_Attack",
    "VulnerabilityScan": "VulnerabilityScan",
    "XSS": "XSS",
}

CIC_FINE_TO_FAMILY = {
    "Backdoor_Malware": "web_based",
    "BenignTraffic": "benign",
    "BrowserHijacking": "web_based",
    "CommandInjection": "web_based",
    "DDoS-ACK_Fragmentation": "ddos",
    "DDoS-HTTP_Flood": "ddos",
    "DDoS-ICMP_Flood": "ddos",
    "DDoS-ICMP_Fragmentation": "ddos",
    "DDoS-PSHACK_Flood": "ddos",
    "DDoS-RSTFINFlood": "ddos",
    "DDoS-SYN_Flood": "ddos",
    "DDoS-SlowLoris": "ddos",
    "DDoS-SynonymousIP_Flood": "ddos",
    "DDoS-TCP_Flood": "ddos",
    "DDoS-UDP_Flood": "ddos",
    "DDoS-UDP_Fragmentation": "ddos",
    "DNS_Spoofing": "spoofing",
    "DictionaryBruteForce": "brute_force",
    "DoS-HTTP_Flood": "dos",
    "DoS-SYN_Flood": "dos",
    "DoS-TCP_Flood": "dos",
    "DoS-UDP_Flood": "dos",
    "MITM-ArpSpoofing": "spoofing",
    "Mirai-greeth_flood": "mirai",
    "Mirai-greip_flood": "mirai",
    "Mirai-udpplain": "mirai",
    "Recon-HostDiscovery": "recon",
    "Recon-OSScan": "recon",
    "Recon-PingSweep": "recon",
    "Recon-PortScan": "recon",
    "SqlInjection": "web_based",
    "Uploading_Attack": "web_based",
    "VulnerabilityScan": "recon",
    "XSS": "web_based",
}

CIC_SCHEMA = DatasetSchema(
    dataset=DatasetName.CIC_IOT_2023,
    source_columns=CIC_SOURCE_COLUMNS,
    label_columns=(),
    model_feature_columns=CIC_SOURCE_COLUMNS,
    grouping_columns=("source_file", "capture_file", "source_directory"),
    categorical_columns=(),
    continuous_columns=CIC_SOURCE_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class CICIoT2023Adapter:
    """Read official CICIoT2023 CSVs without losing capture identity."""

    root: Path
    schema: DatasetSchema = field(default=CIC_SCHEMA, init=False)

    def discover_files(self) -> tuple[Path, ...]:
        """Return all original CSV files in deterministic relative-path order."""

        if not self.root.is_dir():
            raise FileNotFoundError(f"CICIoT2023 source directory not found: {self.root}")
        files = tuple(
            sorted(
                self.root.rglob("*.csv"),
                key=lambda path: normalized_source_file(self.root, path).casefold(),
            )
        )
        if not files:
            raise DataContractError(f"No CICIoT2023 CSV files found under {self.root}")
        unknown = sorted({path.parent.name for path in files} - set(CIC_SOURCE_DIRECTORY_TO_FINE))
        if unknown:
            raise DataContractError(f"Unknown CICIoT2023 source directories: {unknown}")
        return files

    def load_file(self, source_file: Path, *, max_rows: int) -> FileLoadResult:
        """Load a bounded prefix of one official per-capture CSV file."""

        require_positive_limit(max_rows, "max_rows")
        relative = normalized_source_file(self.root, source_file)
        source_directory = source_file.parent.name
        try:
            fine_label = CIC_SOURCE_DIRECTORY_TO_FINE[source_directory]
            family_label = CIC_FINE_TO_FAMILY[fine_label]
        except KeyError as exc:
            raise DataContractError(
                f"Unknown CICIoT2023 label directory: {source_directory}"
            ) from exc

        records: list[DatasetRecord] = []
        rows_seen = 0
        invalid_width = 0
        non_finite = 0
        try:
            with source_file.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header is None:
                    raise DataContractError(f"Empty CICIoT2023 source file: {relative}")
                validate_header(header, self.schema.source_columns, relative)
                for source_row, row in enumerate(reader, start=1):
                    if rows_seen >= max_rows:
                        break
                    rows_seen += 1
                    current_id = record_id(self.schema.dataset, relative, source_row)
                    if len(row) != len(self.schema.source_columns):
                        invalid_width += 1
                        continue
                    if not all(is_finite_number(value.strip()) for value in row):
                        non_finite += 1
                        continue
                    features = tuple(
                        (column, value.strip())
                        for column, value in zip(
                            self.schema.model_feature_columns, row, strict=True
                        )
                    )
                    grouping = (
                        ("source_file", relative),
                        ("capture_file", source_file.name),
                        ("source_directory", source_directory),
                    )
                    records.append(
                        DatasetRecord(
                            dataset=self.schema.dataset,
                            record_id=current_id,
                            source_file=relative,
                            source_row=source_row,
                            fine_label=fine_label,
                            family_label=family_label,
                            features=features,
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
                non_finite_numeric=non_finite,
            ),
        )
