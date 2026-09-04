"""Resumable, streaming source-data quality audit for SemCred-FL datasets."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.ciciot2023 import (
    CIC_FINE_TO_FAMILY,
    CIC_SOURCE_DIRECTORY_TO_FINE,
    CICIoT2023Adapter,
)
from semcred_fl.data.contracts import DataContractError, DatasetSchema
from semcred_fl.data.edge_iiotset import (
    EDGE_FINE_TO_FAMILY,
    EdgeIIoTsetAdapter,
    canonical_edge_fine_label,
)
from semcred_fl.data.record_ids import normalized_source_file
from semcred_fl.data.validation import DatasetAdapter
from semcred_fl.identity import canonical_json, run_id

_INTEGER_RE: Final = re.compile(r"^[+-]?\d+$")
_HEX_RE: Final = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
_MISSING_TOKENS: Final = frozenset({"", "?", "na", "n/a", "nan", "none", "null"})
_HLL_PRECISION: Final = 10
_HLL_REGISTERS: Final = 1 << _HLL_PRECISION
_UPSERT_BATCH: Final = 10_000


def _value_kind(value: str) -> str:
    """Classify a stripped source value without silently coercing it."""

    lowered = value.casefold()
    if lowered in _MISSING_TOKENS:
        return "missing"
    if _HEX_RE.fullmatch(value):
        return "hex_integer"
    if _INTEGER_RE.fullmatch(value):
        return "decimal_integer"
    try:
        parsed = float(value)
    except ValueError:
        return "text"
    return "finite_decimal" if math.isfinite(parsed) else "non_finite"


@dataclass(slots=True)
class CardinalitySketch:
    """Fixed-memory HyperLogLog sketch used only for approximate cardinality."""

    registers: bytearray = field(default_factory=lambda: bytearray(_HLL_REGISTERS))

    def add(self, value: str) -> None:
        """Add a value to the sketch."""

        digest = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")
        index = digest >> (256 - _HLL_PRECISION)
        remainder_bits = 256 - _HLL_PRECISION
        remainder = digest & ((1 << remainder_bits) - 1)
        rank = remainder_bits + 1 if remainder == 0 else remainder_bits - remainder.bit_length() + 1
        self.registers[index] = max(self.registers[index], rank)

    def merge(self, other: CardinalitySketch) -> None:
        """Merge another compatible sketch."""

        if len(other.registers) != len(self.registers):
            raise ValueError("Cardinality sketches use different precisions")
        for index, value in enumerate(other.registers):
            self.registers[index] = max(self.registers[index], value)

    def estimate(self) -> int:
        """Return the rounded HyperLogLog cardinality estimate."""

        register_count = len(self.registers)
        alpha = 0.7213 / (1.0 + 1.079 / register_count)
        raw = alpha * register_count**2 / sum(2.0 ** (-value) for value in self.registers)
        zeros = self.registers.count(0)
        if raw <= 2.5 * register_count and zeros:
            raw = register_count * math.log(register_count / zeros)
        return round(raw)

    def encode(self) -> str:
        """Encode registers for JSON persistence."""

        return base64.b64encode(self.registers).decode("ascii")

    @classmethod
    def decode(cls, encoded: str) -> CardinalitySketch:
        """Restore a persisted sketch."""

        registers = bytearray(base64.b64decode(encoded.encode("ascii"), validate=True))
        if len(registers) != _HLL_REGISTERS:
            raise ValueError("Invalid cardinality sketch length")
        return cls(registers=registers)


@dataclass(slots=True)
class ColumnAccumulator:
    """Streaming type, completeness, and cardinality statistics for one column."""

    total: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    max_length: int = 0
    first_digest: str | None = None
    is_constant: bool = True
    cardinality: CardinalitySketch = field(default_factory=CardinalitySketch)

    def add(self, value: str) -> None:
        """Observe one stripped source value."""

        self.total += 1
        self.kinds[_value_kind(value)] += 1
        self.max_length = max(self.max_length, len(value))
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if self.first_digest is None:
            self.first_digest = digest
        elif digest != self.first_digest:
            self.is_constant = False
        self.cardinality.add(value)

    def merge(self, other: ColumnAccumulator) -> None:
        """Merge file-level statistics into a dataset-level profile."""

        if self.first_digest is None:
            self.first_digest = other.first_digest
            self.is_constant = other.is_constant
        elif other.first_digest is not None:
            self.is_constant = (
                self.is_constant and other.is_constant and self.first_digest == other.first_digest
            )
        self.total += other.total
        self.kinds.update(other.kinds)
        self.max_length = max(self.max_length, other.max_length)
        self.cardinality.merge(other.cardinality)

    def to_dict(self, *, include_sketch: bool) -> dict[str, object]:
        """Serialize the profile, optionally retaining its mergeable sketch."""

        result: dict[str, object] = {
            "total": self.total,
            "kind_counts": dict(sorted(self.kinds.items())),
            "missing_count": self.kinds["missing"],
            "max_length": self.max_length,
            "constant": self.is_constant and self.total > 0,
            "approximate_cardinality": self.cardinality.estimate(),
            "cardinality_method": "HyperLogLog(p=10)",
        }
        if include_sketch:
            result["first_digest"] = self.first_digest
            result["cardinality_sketch"] = self.cardinality.encode()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ColumnAccumulator:
        """Restore a file-level profile from the audit database."""

        kinds = Counter(cast(dict[str, int], value["kind_counts"]))
        return cls(
            total=cast(int, value["total"]),
            kinds=kinds,
            max_length=cast(int, value["max_length"]),
            first_digest=cast(str | None, value["first_digest"]),
            is_constant=cast(bool, value["constant"]),
            cardinality=CardinalitySketch.decode(cast(str, value["cardinality_sketch"])),
        )


@dataclass(frozen=True, slots=True)
class AuditRequest:
    """Immutable identity and paths for one source audit."""

    dataset: DatasetName
    root: Path
    archive_sha256: str
    output_directory: Path
    work_database: Path

    def __post_init__(self) -> None:
        digest = self.archive_sha256.casefold()
        if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
            raise ValueError("archive_sha256 must be a full SHA-256 hexadecimal digest")


def model_feature_fingerprint(values: tuple[str, ...]) -> bytes:
    """Hash ordered, stripped model-feature values without labels or grouping fields."""

    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _open_database(request: AuditRequest, schema: DatasetSchema) -> sqlite3.Connection:
    request.work_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(request.work_database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS completed_files (
            source_file TEXT PRIMARY KEY,
            byte_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fingerprints (
            fingerprint BLOB PRIMARY KEY,
            total_count INTEGER NOT NULL,
            first_file_count INTEGER NOT NULL,
            source_file_count INTEGER NOT NULL,
            last_file TEXT NOT NULL,
            first_fine_label TEXT NOT NULL,
            label_conflict INTEGER NOT NULL,
            within_file_duplicates INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )
    identity = canonical_json(
        {
            "audit_schema_version": 1,
            "dataset": request.dataset,
            "dataset_schema_id": schema.schema_id,
            "root": request.root.resolve(),
            "archive_sha256": request.archive_sha256.casefold(),
        }
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = 'audit_identity'"
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('audit_identity', ?)", (identity,)
        )
        connection.commit()
    elif existing[0] != identity:
        connection.close()
        raise DataContractError(
            "The work database belongs to a different dataset root, schema, or archive hash"
        )
    return connection


def _adapter(request: AuditRequest) -> DatasetAdapter:
    if request.dataset is DatasetName.CIC_IOT_2023:
        return CICIoT2023Adapter(request.root)
    return EdgeIIoTsetAdapter(request.root)


def _labels(
    adapter: DatasetAdapter, source_file: Path, row: list[str], indices: dict[str, int]
) -> tuple[str, str] | None:
    if isinstance(adapter, CICIoT2023Adapter):
        fine = CIC_SOURCE_DIRECTORY_TO_FINE[source_file.parent.name]
        return fine, CIC_FINE_TO_FAMILY[fine]
    edge = cast(EdgeIIoTsetAdapter, adapter)
    raw_fine = row[indices["Attack_type"]].strip()
    edge_fine = canonical_edge_fine_label(raw_fine)
    binary = row[indices["Attack_label"]].strip()
    if edge_fine is None:
        return None
    expected_binary = "0" if edge_fine == "Normal" else "1"
    if edge_fine != edge.expected_fine_label(source_file) or binary != expected_binary:
        return None
    return edge_fine, EDGE_FINE_TO_FAMILY[edge_fine]


_FINGERPRINT_UPSERT: Final = """
INSERT INTO fingerprints(
    fingerprint, total_count, first_file_count, source_file_count, last_file,
    first_fine_label, label_conflict, within_file_duplicates
) VALUES (?, ?, ?, 1, ?, ?, 0, ?)
ON CONFLICT(fingerprint) DO UPDATE SET
    total_count = total_count + excluded.total_count,
    first_file_count = first_file_count + CASE
        WHEN source_file_count = 1 AND last_file = excluded.last_file
        THEN excluded.total_count ELSE 0 END,
    source_file_count = source_file_count + CASE
        WHEN last_file = excluded.last_file THEN 0 ELSE 1 END,
    within_file_duplicates = within_file_duplicates + CASE
        WHEN last_file = excluded.last_file THEN excluded.total_count
        ELSE excluded.total_count - 1 END,
    label_conflict = CASE
        WHEN label_conflict = 1 OR first_fine_label != excluded.first_fine_label THEN 1
        ELSE 0 END,
    last_file = excluded.last_file
"""


def _flush_fingerprints(
    connection: sqlite3.Connection,
    counts: Counter[bytes],
    source_file: str,
    fine_label: str,
) -> None:
    connection.executemany(
        _FINGERPRINT_UPSERT,
        (
            (fingerprint, count, count, source_file, fine_label, count - 1)
            for fingerprint, count in counts.items()
        ),
    )
    counts.clear()


def _scan_file(
    connection: sqlite3.Connection,
    adapter: DatasetAdapter,
    source_file: Path,
) -> dict[str, object]:
    schema = adapter.schema
    relative = normalized_source_file(adapter.root, source_file)
    stat = source_file.stat()
    columns = {name: ColumnAccumulator() for name in schema.model_feature_columns}
    report: dict[str, object] = {
        "source_file": relative,
        "byte_size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "header_valid": False,
        "rows_seen": 0,
        "valid_rows": 0,
        "invalid_width": 0,
        "invalid_label": 0,
        "non_finite_numeric": 0,
        "fine_label_counts": {},
        "family_label_counts": {},
    }
    fine_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    pending: Counter[bytes] = Counter()
    pending_rows = 0
    with source_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            report["columns"] = {
                name: value.to_dict(include_sketch=True) for name, value in columns.items()
            }
            return report
        report["header_valid"] = tuple(header) == schema.source_columns
        if len(header) != len(schema.source_columns) or set(header) != set(schema.source_columns):
            for _ in reader:
                report["rows_seen"] = cast(int, report["rows_seen"]) + 1
                report["invalid_width"] = cast(int, report["invalid_width"]) + 1
            report["columns"] = {
                name: value.to_dict(include_sketch=True) for name, value in columns.items()
            }
            return report
        indices = {name: header.index(name) for name in schema.source_columns}
        for row in reader:
            report["rows_seen"] = cast(int, report["rows_seen"]) + 1
            if len(row) != len(header):
                report["invalid_width"] = cast(int, report["invalid_width"]) + 1
                continue
            stripped = tuple(row[indices[name]].strip() for name in schema.model_feature_columns)
            for name, value in zip(schema.model_feature_columns, stripped, strict=True):
                columns[name].add(value)
            labels = _labels(adapter, source_file, row, indices)
            if labels is None:
                report["invalid_label"] = cast(int, report["invalid_label"]) + 1
                continue
            fine_label, family_label = labels
            if isinstance(adapter, CICIoT2023Adapter) and any(
                _value_kind(value) not in {"decimal_integer", "finite_decimal"}
                for value in stripped
            ):
                report["non_finite_numeric"] = cast(int, report["non_finite_numeric"]) + 1
                continue
            report["valid_rows"] = cast(int, report["valid_rows"]) + 1
            fine_counts[fine_label] += 1
            family_counts[family_label] += 1
            pending[model_feature_fingerprint(stripped)] += 1
            pending_rows += 1
            if pending_rows >= _UPSERT_BATCH:
                _flush_fingerprints(connection, pending, relative, fine_label)
                pending_rows = 0
        if pending:
            label = next(iter(fine_counts), "unknown")
            _flush_fingerprints(connection, pending, relative, label)
    report["fine_label_counts"] = dict(sorted(fine_counts.items()))
    report["family_label_counts"] = dict(sorted(family_counts.items()))
    report["columns"] = {
        name: value.to_dict(include_sketch=True) for name, value in columns.items()
    }
    return report


def _completed_report(
    connection: sqlite3.Connection, relative: str, source_file: Path
) -> dict[str, object] | None:
    row = connection.execute(
        "SELECT byte_size, modified_ns, report_json FROM completed_files WHERE source_file = ?",
        (relative,),
    ).fetchone()
    if row is None:
        return None
    stat = source_file.stat()
    if row[0] != stat.st_size or row[1] != stat.st_mtime_ns:
        raise DataContractError(f"Previously audited source file changed: {relative}")
    return cast(dict[str, object], json.loads(cast(str, row[2])))


def _aggregate(
    connection: sqlite3.Connection,
    request: AuditRequest,
    schema: DatasetSchema,
) -> dict[str, object]:
    rows = connection.execute(
        "SELECT source_file, report_json FROM completed_files ORDER BY source_file"
    ).fetchall()
    file_reports = [cast(dict[str, object], json.loads(item[1])) for item in rows]
    columns = {name: ColumnAccumulator() for name in schema.model_feature_columns}
    fine_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_files: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    header_failures: list[str] = []
    for report in file_reports:
        for key in (
            "rows_seen",
            "valid_rows",
            "invalid_width",
            "invalid_label",
            "non_finite_numeric",
        ):
            totals[key] += cast(int, report[key])
        if not cast(bool, report["header_valid"]):
            header_failures.append(cast(str, report["source_file"]))
        current_family = cast(dict[str, int], report["family_label_counts"])
        fine_counts.update(cast(dict[str, int], report["fine_label_counts"]))
        family_counts.update(current_family)
        family_files.update(current_family.keys())
        for name, value in cast(dict[str, dict[str, object]], report["columns"]).items():
            columns[name].merge(ColumnAccumulator.from_dict(value))

    duplicate_row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(total_count), 0),
               COALESCE(SUM(total_count - 1), 0),
               COALESCE(SUM(within_file_duplicates), 0),
               COALESCE(SUM(CASE WHEN source_file_count > 1 THEN 1 ELSE 0 END), 0),
               COALESCE(SUM(total_count - first_file_count), 0),
               COALESCE(SUM(label_conflict), 0)
        FROM fingerprints
        """
    ).fetchone()
    assert duplicate_row is not None
    valid_rows = totals["valid_rows"]
    observed_cells = sum(column.total for column in columns.values())
    missing_cells = sum(column.kinds["missing"] for column in columns.values())
    completeness = 100.0 if observed_cells == 0 else 100.0 * (1 - missing_cells / observed_cells)
    structural_errors = totals["invalid_width"] + totals["invalid_label"] + len(header_failures)
    consistency = (
        100.0
        if totals["rows_seen"] == 0
        else 100.0 * max(0.0, 1 - structural_errors / totals["rows_seen"])
    )
    validity = 100.0 if totals["rows_seen"] == 0 else 100.0 * valid_rows / totals["rows_seen"]
    uniqueness = 100.0 if valid_rows == 0 else 100.0 * cast(int, duplicate_row[0]) / valid_rows
    dqs = (30 * completeness + 25 * consistency + 20 * validity + 15 * uniqueness) / 90
    group_feasible = bool(family_files) and all(count >= 3 for count in family_files.values())
    summary: dict[str, object] = {
        "audit_schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "dataset": request.dataset.value,
            "archive_sha256": request.archive_sha256.casefold(),
            "hash_interpretation": "locally_computed_identity_not_independent_authenticity_proof",
            "source_root": str(request.root.resolve()),
            "dataset_schema_id": schema.schema_id,
        },
        "coverage": {
            "completed_files": len(file_reports),
            "rows_seen": totals["rows_seen"],
            "valid_rows": valid_rows,
            "invalid_width": totals["invalid_width"],
            "invalid_label": totals["invalid_label"],
            "non_finite_numeric": totals["non_finite_numeric"],
            "header_failures": header_failures,
        },
        "labels": {
            "fine_row_counts": dict(sorted(fine_counts.items())),
            "family_row_counts": dict(sorted(family_counts.items())),
            "family_source_file_counts": dict(sorted(family_files.items())),
        },
        "columns": {name: value.to_dict(include_sketch=False) for name, value in columns.items()},
        "duplicates": {
            "unique_model_feature_fingerprints": duplicate_row[0],
            "fingerprinted_valid_rows": duplicate_row[1],
            "duplicate_occurrences_after_first": duplicate_row[2],
            "within_file_duplicate_occurrences": duplicate_row[3],
            "fingerprints_present_in_multiple_files": duplicate_row[4],
            "occurrences_outside_first_source_file": duplicate_row[5],
            "fingerprints_with_conflicting_fine_labels": duplicate_row[6],
            "fingerprint_scope": (
                "ordered stripped model-feature values; labels and grouping fields excluded"
            ),
        },
        "split_feasibility": {
            "at_least_three_source_files_per_observed_family": group_feasible,
            "decision": "candidate"
            if group_feasible
            else "not_feasible_from_source_file_groups_alone",
            "note": (
                "This is necessary but not sufficient; duplicate components must remain "
                "within one split."
            ),
        },
        "data_quality_score": {
            "score": round(dqs, 2),
            "completeness": round(completeness, 2),
            "consistency": round(consistency, 2),
            "validity": round(validity, 2),
            "uniqueness": round(uniqueness, 2),
            "timeliness": None,
            "method": (
                "Applicable weights renormalized from 90%; timeliness is not inferable "
                "from these source files."
            ),
        },
        "model_readiness": {
            "ready": False,
            "reason": (
                "Audit output must first be reviewed and frozen into cleaning, typing, "
                "and group-disjoint split rules."
            ),
        },
    }
    summary["audit_id"] = run_id(
        {
            key: value
            for key, value in summary.items()
            if key not in {"generated_at_utc", "audit_id"}
        },
        length=64,
    )
    return summary


def _markdown(summary: dict[str, object]) -> str:
    coverage = cast(dict[str, object], summary["coverage"])
    duplicates = cast(dict[str, object], summary["duplicates"])
    score = cast(dict[str, object], summary["data_quality_score"])
    verdict = (
        "production-ready by score only"
        if cast(float, score["score"]) >= 85
        else "usable only with documented remediation"
        if cast(float, score["score"]) >= 65
        else "remediation required"
    )
    return "\n".join(
        (
            "# SemCred-FL source data audit",
            "",
            f"Audit ID: `{summary['audit_id']}`",
            "",
            "## Bottom line",
            "",
            (
                f"DQS: **{score['score']}/100** — {verdict}. Model readiness remains "
                "**false** until the findings are reviewed and converted into frozen "
                "preprocessing and split rules."
            ),
            "",
            "## Coverage",
            "",
            f"- Files completed: {coverage['completed_files']}",
            f"- Rows inspected: {coverage['rows_seen']}",
            f"- Valid rows: {coverage['valid_rows']}",
            f"- Invalid-width rows: {coverage['invalid_width']}",
            f"- Invalid-label rows: {coverage['invalid_label']}",
            f"- Non-finite mandatory numeric rows: {coverage['non_finite_numeric']}",
            "",
            "## Exact duplicate audit",
            "",
            (
                "- Unique model-feature fingerprints: "
                f"{duplicates['unique_model_feature_fingerprints']}"
            ),
            (
                "- Duplicate occurrences after first: "
                f"{duplicates['duplicate_occurrences_after_first']}"
            ),
            (
                "- Within-file duplicate occurrences: "
                f"{duplicates['within_file_duplicate_occurrences']}"
            ),
            (
                "- Fingerprints present in multiple files: "
                f"{duplicates['fingerprints_present_in_multiple_files']}"
            ),
            (
                "- Fingerprints with conflicting fine labels: "
                f"{duplicates['fingerprints_with_conflicting_fine_labels']}"
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "Cardinalities are fixed-memory HyperLogLog estimates. Duplicate statistics "
                "are exact for valid rows. The DQS does not infer freshness because source "
                "timestamps and collection dates are not reliable timeliness evidence. A "
                "high DQS does not override the mandatory leakage and split gates."
            ),
            "",
        )
    )


def run_audit(request: AuditRequest, *, progress_every: int = 1) -> dict[str, object]:
    """Run or resume a complete audit and write JSON and Markdown reports."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    adapter = _adapter(request)
    files = adapter.discover_files()
    connection = _open_database(request, adapter.schema)
    try:
        for index, source_file in enumerate(files, start=1):
            relative = normalized_source_file(adapter.root, source_file)
            if _completed_report(connection, relative, source_file) is not None:
                if index % progress_every == 0 or index == len(files):
                    print(f"[{index}/{len(files)}] resume: {relative}", flush=True)
                continue
            try:
                connection.execute("BEGIN IMMEDIATE")
                report = _scan_file(connection, adapter, source_file)
                connection.execute(
                    "INSERT INTO completed_files VALUES (?, ?, ?, ?)",
                    (
                        relative,
                        cast(int, report["byte_size"]),
                        cast(int, report["modified_ns"]),
                        json.dumps(
                            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            if index % progress_every == 0 or index == len(files):
                print(f"[{index}/{len(files)}] audited: {relative}", flush=True)
        summary = _aggregate(connection, request, adapter.schema)
    finally:
        connection.close()

    request.output_directory.mkdir(parents=True, exist_ok=True)
    stem = request.dataset.value
    json_path = request.output_directory / f"{stem}_audit.json"
    markdown_path = request.output_directory / f"{stem}_audit.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return summary
