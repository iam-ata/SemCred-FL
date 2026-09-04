"""Resumable construction of canonical, globally deduplicated dataset corpora."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.audit import model_feature_fingerprint
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
from semcred_fl.data.preprocessing import (
    CIC_PREPARATION_POLICY,
    EDGE_PREPARATION_POLICY,
    DatasetPreparationPolicy,
    FeatureValueError,
    normalize_feature_values,
)
from semcred_fl.data.record_ids import normalized_source_file, record_id
from semcred_fl.data.validation import DatasetAdapter, validate_header
from semcred_fl.identity import canonical_json, run_id

_SCHEMA_VERSION: Final = 1
_DECISION_UNCLASSIFIED: Final = 0
_DECISION_EXCLUDE_PRIMARY_CONFLICT: Final = 1
_DECISION_ELIGIBLE_FINE_AMBIGUOUS: Final = 2
_DECISION_ELIGIBLE: Final = 3
_UPSERT_SQL: Final = """
INSERT INTO fingerprints(
    fingerprint, occurrence_count, fine_mask, family_mask, source_mask,
    representative_file_index, representative_row, decision
) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
ON CONFLICT(fingerprint) DO UPDATE SET
    occurrence_count = occurrence_count + excluded.occurrence_count,
    fine_mask = fine_mask | excluded.fine_mask,
    family_mask = family_mask | excluded.family_mask,
    source_mask = blob_or(source_mask, excluded.source_mask),
    representative_file_index = CASE
        WHEN excluded.representative_file_index < representative_file_index
          OR (excluded.representative_file_index = representative_file_index
              AND excluded.representative_row < representative_row)
        THEN excluded.representative_file_index ELSE representative_file_index END,
    representative_row = CASE
        WHEN excluded.representative_file_index < representative_file_index
          OR (excluded.representative_file_index = representative_file_index
              AND excluded.representative_row < representative_row)
        THEN excluded.representative_row ELSE representative_row END
"""


@dataclass(frozen=True, slots=True)
class CanonicalBuildRequest:
    """Inputs and bounded-memory controls for one canonical-corpus build."""

    dataset: DatasetName
    root: Path
    archive_sha256: str
    output_directory: Path
    work_database: Path
    row_batch_size: int = 10_000
    parquet_batch_size: int = 10_000

    def __post_init__(self) -> None:
        digest = self.archive_sha256.casefold()
        if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
            raise ValueError("archive_sha256 must be a full SHA-256 hexadecimal digest")
        if self.row_batch_size < 1 or self.parquet_batch_size < 1:
            raise ValueError("Batch sizes must be positive")


@dataclass(frozen=True, slots=True)
class PreparedRow:
    """One validated row after policy-frozen scalar canonicalization."""

    source_row: int
    fine_label: str
    family_label: str
    features: tuple[tuple[str, str], ...]
    grouping: tuple[tuple[str, str], ...]
    fingerprint: bytes


@dataclass(slots=True)
class ScanStats:
    """Complete row accounting for one source-file scan."""

    rows_seen: int = 0
    valid_rows: int = 0
    invalid_width: int = 0
    invalid_features: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-ready accounting."""

        return {
            "rows_seen": self.rows_seen,
            "valid_rows": self.valid_rows,
            "invalid_width": self.invalid_width,
            "invalid_feature_rows": sum(self.invalid_features.values()),
            "invalid_features": dict(sorted(self.invalid_features.items())),
        }


@dataclass(slots=True)
class _Aggregate:
    occurrence_count: int
    fine_mask: int
    family_mask: int
    representative_row: int


def _adapter(dataset: DatasetName, root: Path) -> DatasetAdapter:
    if dataset is DatasetName.CIC_IOT_2023:
        return CICIoT2023Adapter(root)
    return EdgeIIoTsetAdapter(root)


def _policy(dataset: DatasetName) -> DatasetPreparationPolicy:
    if dataset is DatasetName.CIC_IOT_2023:
        return CIC_PREPARATION_POLICY
    return EDGE_PREPARATION_POLICY


def _labels(dataset: DatasetName) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = CIC_FINE_TO_FAMILY if dataset is DatasetName.CIC_IOT_2023 else EDGE_FINE_TO_FAMILY
    return tuple(sorted(mapping)), tuple(sorted(set(mapping.values())))


def _feature_error_key(error: FeatureValueError) -> str:
    message = str(error)
    return message.split(":", maxsplit=1)[0]


def _edge_labels_and_grouping(
    row: Sequence[str], indices: Mapping[str, int], relative: str, expected_fine: str
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    raw_fine = row[indices["Attack_type"]].strip()
    fine = canonical_edge_fine_label(raw_fine)
    if fine is None:
        raise DataContractError(f"Unknown Edge-IIoTset label {raw_fine!r} in {relative}")
    if fine != expected_fine:
        raise DataContractError(
            f"Source/row label mismatch in {relative}; expected={expected_fine}, actual={fine}"
        )
    binary = row[indices["Attack_label"]].strip()
    expected_binary = "0" if fine == "Normal" else "1"
    if binary != expected_binary:
        raise DataContractError(f"Binary/fine label mismatch in {relative}")
    endpoints = sorted((row[indices["ip.src_host"]].strip(), row[indices["ip.dst_host"]].strip()))
    grouping = (
        ("source_file", relative),
        ("frame_time", row[indices["frame.time"]].strip()),
        ("endpoint_pair", "|".join(endpoints)),
    )
    return fine, EDGE_FINE_TO_FAMILY[fine], grouping


def _iter_prepared_rows(
    dataset: DatasetName,
    root: Path,
    source_file: Path,
    schema: DatasetSchema,
    policy: DatasetPreparationPolicy,
    stats: ScanStats,
) -> Iterator[PreparedRow]:
    relative = normalized_source_file(root, source_file)
    indices = {column: index for index, column in enumerate(schema.source_columns)}
    if dataset is DatasetName.CIC_IOT_2023:
        try:
            fine = CIC_SOURCE_DIRECTORY_TO_FINE[source_file.parent.name]
        except KeyError as exc:
            raise DataContractError(f"Unknown CICIoT2023 label directory: {relative}") from exc
        family = CIC_FINE_TO_FAMILY[fine]
        expected_edge_fine = ""
    else:
        fine = family = ""
        expected_edge_fine = EdgeIIoTsetAdapter(root).expected_fine_label(source_file)

    try:
        with source_file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None:
                raise DataContractError(f"Empty source file: {relative}")
            validate_header(header, schema.source_columns, relative)
            for source_row, row in enumerate(reader, start=1):
                stats.rows_seen += 1
                if len(row) != len(schema.source_columns):
                    stats.invalid_width += 1
                    continue
                if dataset is DatasetName.EDGE_IIOTSET:
                    fine, family, grouping = _edge_labels_and_grouping(
                        row, indices, relative, expected_edge_fine
                    )
                else:
                    grouping = (
                        ("source_file", relative),
                        ("capture_file", source_file.name),
                        ("source_directory", source_file.parent.name),
                    )
                raw_features = {
                    column: row[indices[column]].strip() for column in schema.model_feature_columns
                }
                try:
                    features = normalize_feature_values(raw_features, policy.features)
                except FeatureValueError as exc:
                    stats.invalid_features[_feature_error_key(exc)] += 1
                    continue
                stats.valid_rows += 1
                yield PreparedRow(
                    source_row=source_row,
                    fine_label=fine,
                    family_label=family,
                    features=features,
                    grouping=grouping,
                    fingerprint=model_feature_fingerprint(tuple(value for _, value in features)),
                )
    except csv.Error as exc:
        raise DataContractError(f"CSV parsing failed for {relative}: {exc}") from exc


def _blob_or(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise DataContractError("Source masks have inconsistent lengths")
    return bytes(a | b for a, b in zip(left, right, strict=True))


def _source_mask(file_index: int, file_count: int) -> bytes:
    mask = bytearray((file_count + 7) // 8)
    mask[file_index // 8] = 1 << (file_index % 8)
    return bytes(mask)


def _source_count(mask: bytes) -> int:
    return sum(value.bit_count() for value in mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def _put_metadata(connection: sqlite3.Connection, key: str, value: object) -> None:
    encoded = value if isinstance(value, str) else canonical_json(value)
    connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, encoded))


def _open_database(
    request: CanonicalBuildRequest,
    schema: DatasetSchema,
    policy: DatasetPreparationPolicy,
    source_files: tuple[Path, ...],
) -> sqlite3.Connection:
    request.work_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(request.work_database)
    connection.create_function("blob_or", 2, _blob_or, deterministic=True)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fingerprints(
            fingerprint BLOB PRIMARY KEY,
            occurrence_count INTEGER NOT NULL,
            fine_mask INTEGER NOT NULL,
            family_mask INTEGER NOT NULL,
            source_mask BLOB NOT NULL,
            representative_file_index INTEGER NOT NULL,
            representative_row INTEGER NOT NULL,
            decision INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS completed_files(
            file_index INTEGER PRIMARY KEY,
            source_file TEXT UNIQUE NOT NULL,
            byte_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS emitted_files(
            file_index INTEGER PRIMARY KEY,
            output_file TEXT UNIQUE NOT NULL,
            byte_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        """
    )
    files_identity = [
        {
            "source_file": normalized_source_file(request.root, path),
            "byte_size": path.stat().st_size,
        }
        for path in source_files
    ]
    identity = canonical_json(
        {
            "canonical_schema_version": _SCHEMA_VERSION,
            "dataset": request.dataset,
            "dataset_schema_id": schema.schema_id,
            "preparation_policy_id": policy.policy_id,
            "archive_sha256": request.archive_sha256.casefold(),
            "files": files_identity,
        }
    )
    existing = _metadata(connection).get("identity")
    if existing is not None and existing != identity:
        connection.close()
        raise DataContractError("Existing Phase 2D database belongs to a different build identity")
    if existing is None:
        _put_metadata(connection, "identity", identity)
        connection.commit()
    return connection


def _flush_aggregates(
    connection: sqlite3.Connection,
    pending: dict[bytes, _Aggregate],
    source_mask: bytes,
    file_index: int,
) -> None:
    connection.executemany(
        _UPSERT_SQL,
        (
            (
                fingerprint,
                item.occurrence_count,
                item.fine_mask,
                item.family_mask,
                source_mask,
                file_index,
                item.representative_row,
            )
            for fingerprint, item in pending.items()
        ),
    )
    pending.clear()


def _index_file(
    connection: sqlite3.Connection,
    request: CanonicalBuildRequest,
    schema: DatasetSchema,
    policy: DatasetPreparationPolicy,
    source_file: Path,
    file_index: int,
    file_count: int,
    fine_bits: Mapping[str, int],
    family_bits: Mapping[str, int],
) -> ScanStats:
    stats = ScanStats()
    pending: dict[bytes, _Aggregate] = {}
    mask = _source_mask(file_index, file_count)
    connection.execute("BEGIN IMMEDIATE")
    try:
        for row in _iter_prepared_rows(
            request.dataset, request.root, source_file, schema, policy, stats
        ):
            fine_mask = fine_bits[row.fine_label]
            family_mask = family_bits[row.family_label]
            current = pending.get(row.fingerprint)
            if current is None:
                pending[row.fingerprint] = _Aggregate(1, fine_mask, family_mask, row.source_row)
            else:
                current.occurrence_count += 1
                current.fine_mask |= fine_mask
                current.family_mask |= family_mask
            if len(pending) >= request.row_batch_size:
                _flush_aggregates(connection, pending, mask, file_index)
        if pending:
            _flush_aggregates(connection, pending, mask, file_index)
        relative = normalized_source_file(request.root, source_file)
        source_stat = source_file.stat()
        connection.execute(
            "INSERT INTO completed_files VALUES (?, ?, ?, ?, ?)",
            (
                file_index,
                relative,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                canonical_json(stats.as_dict()),
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return stats


def _is_multi_label(mask: int) -> bool:
    return mask != 0 and mask & (mask - 1) != 0


def _classify(connection: sqlite3.Connection) -> dict[str, int]:
    saved = _metadata(connection).get("classification")
    if saved is not None:
        return cast(dict[str, int], json.loads(saved))
    summary: Counter[str] = Counter()
    updates: list[tuple[int, bytes]] = []
    connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = connection.execute(
            "SELECT fingerprint, occurrence_count, fine_mask, family_mask FROM fingerprints"
        )
        while rows := cursor.fetchmany(10_000):
            for fingerprint, count, fine_mask, family_mask in rows:
                summary["canonical_fingerprints"] += 1
                summary["canonical_valid_occurrences"] += int(count)
                if _is_multi_label(int(family_mask)):
                    decision = _DECISION_EXCLUDE_PRIMARY_CONFLICT
                    summary["primary_conflict_fingerprints"] += 1
                    summary["primary_conflict_occurrences"] += int(count)
                elif _is_multi_label(int(fine_mask)):
                    decision = _DECISION_ELIGIBLE_FINE_AMBIGUOUS
                    summary["eligible_fingerprints"] += 1
                    summary["eligible_occurrences"] += int(count)
                    summary["fine_ambiguous_fingerprints"] += 1
                    summary["fine_ambiguous_occurrences"] += int(count)
                else:
                    decision = _DECISION_ELIGIBLE
                    summary["eligible_fingerprints"] += 1
                    summary["eligible_occurrences"] += int(count)
                updates.append((decision, fingerprint))
            connection.executemany(
                "UPDATE fingerprints SET decision=? WHERE fingerprint=?", updates
            )
            updates.clear()
        summary["eligible_duplicate_occurrences_removed"] = (
            summary["eligible_occurrences"] - summary["eligible_fingerprints"]
        )
        result = dict(sorted(summary.items()))
        _put_metadata(connection, "classification", result)
        connection.commit()
        return result
    except BaseException:
        connection.rollback()
        raise


def _parquet_schema(schema: DatasetSchema, policy: DatasetPreparationPolicy) -> object:
    import pyarrow as pa  # type: ignore[import-untyped]

    fields = [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("canonical_fingerprint", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("fine_label", pa.string(), nullable=False),
        pa.field("family_label", pa.string(), nullable=False),
        pa.field("fine_label_ambiguous", pa.bool_(), nullable=False),
        pa.field("duplicate_count", pa.int64(), nullable=False),
        pa.field("source_file_count", pa.int32(), nullable=False),
    ]
    fields.extend(
        pa.field(column, pa.string(), nullable=False) for column in schema.grouping_columns[1:]
    )
    categorical = set(policy.features.categorical_columns)
    fields.extend(
        pa.field(column, pa.string() if column in categorical else pa.float64(), nullable=False)
        for column in policy.features.prepared_feature_columns
    )
    return pa.schema(fields)


def _output_row(
    request: CanonicalBuildRequest,
    relative: str,
    row: PreparedRow,
    occurrence_count: int,
    source_mask: bytes,
    decision: int,
    policy: DatasetPreparationPolicy,
) -> dict[str, object]:
    output: dict[str, object] = {
        "record_id": record_id(request.dataset, relative, row.source_row),
        "canonical_fingerprint": row.fingerprint.hex(),
        "dataset": request.dataset.value,
        "source_file": relative,
        "source_row": row.source_row,
        "fine_label": row.fine_label,
        "family_label": row.family_label,
        "fine_label_ambiguous": decision == _DECISION_ELIGIBLE_FINE_AMBIGUOUS,
        "duplicate_count": occurrence_count,
        "source_file_count": _source_count(source_mask),
    }
    output.update(dict(row.grouping[1:]))
    categorical = set(policy.features.categorical_columns)
    output.update(
        {column: value if column in categorical else float(value) for column, value in row.features}
    )
    return output


def _lookup(
    connection: sqlite3.Connection, fingerprints: Sequence[bytes]
) -> dict[bytes, tuple[int, int, int, int, bytes]]:
    connection.execute("DELETE FROM batch_lookup")
    connection.executemany(
        "INSERT OR IGNORE INTO batch_lookup(fingerprint) VALUES (?)",
        ((fingerprint,) for fingerprint in fingerprints),
    )
    return {
        bytes(fingerprint): (
            int(decision),
            int(file_index),
            int(representative_row),
            int(count),
            bytes(source_mask),
        )
        for (
            fingerprint,
            decision,
            file_index,
            representative_row,
            count,
            source_mask,
        ) in connection.execute(
            """
            SELECT f.fingerprint, f.decision, f.representative_file_index,
                   f.representative_row, f.occurrence_count, f.source_mask
            FROM fingerprints AS f JOIN batch_lookup AS b USING(fingerprint)
            """
        )
    }


def _emit_file(
    connection: sqlite3.Connection,
    request: CanonicalBuildRequest,
    schema: DatasetSchema,
    policy: DatasetPreparationPolicy,
    source_file: Path,
    file_index: int,
    output_file: Path,
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    relative = normalized_source_file(request.root, source_file)
    arrow_schema = _parquet_schema(schema, policy)
    partial = output_file.with_suffix(output_file.suffix + ".partial")
    partial.unlink(missing_ok=True)
    stats = ScanStats()
    rows: list[PreparedRow] = []
    output_rows: list[dict[str, object]] = []
    fine_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    emitted = 0

    def flush(writer: object) -> None:
        nonlocal emitted
        if not rows:
            return
        decisions = _lookup(connection, [row.fingerprint for row in rows])
        for row in rows:
            decision, representative_file, representative_row, count, source_mask = decisions[
                row.fingerprint
            ]
            if decision not in {
                _DECISION_ELIGIBLE,
                _DECISION_ELIGIBLE_FINE_AMBIGUOUS,
            }:
                continue
            if representative_file != file_index or representative_row != row.source_row:
                continue
            output_rows.append(
                _output_row(request, relative, row, count, source_mask, decision, policy)
            )
            fine_counts[row.fine_label] += 1
            family_counts[row.family_label] += 1
        if output_rows:
            cast("pq.ParquetWriter", writer).write_table(
                pa.Table.from_pylist(output_rows, schema=arrow_schema)
            )
            emitted += len(output_rows)
        rows.clear()
        output_rows.clear()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(partial, arrow_schema, compression="zstd")
    try:
        for prepared in _iter_prepared_rows(
            request.dataset, request.root, source_file, schema, policy, stats
        ):
            rows.append(prepared)
            if len(rows) >= request.parquet_batch_size:
                flush(writer)
        flush(writer)
        if emitted == 0:
            writer.write_table(pa.Table.from_pylist([], schema=arrow_schema))
    finally:
        writer.close()
    indexed_report_row = connection.execute(
        "SELECT report_json FROM completed_files WHERE file_index=?", (file_index,)
    ).fetchone()
    if indexed_report_row is None or json.loads(indexed_report_row[0]) != stats.as_dict():
        partial.unlink(missing_ok=True)
        raise DataContractError(f"Source rescan no longer matches indexed accounting: {relative}")
    partial.replace(output_file)
    report: dict[str, object] = {
        "row_count": emitted,
        "fine_label_counts": dict(sorted(fine_counts.items())),
        "family_label_counts": dict(sorted(family_counts.items())),
        "rescan": stats.as_dict(),
    }
    output_hash = _sha256(output_file)
    connection.execute(
        "INSERT INTO emitted_files VALUES (?, ?, ?, ?, ?, ?)",
        (
            file_index,
            output_file.name,
            output_file.stat().st_size,
            output_hash,
            emitted,
            canonical_json(report),
        ),
    )
    connection.commit()
    return report


def _schema_document(
    request: CanonicalBuildRequest, schema: DatasetSchema, policy: DatasetPreparationPolicy
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "dataset_schema_id": schema.schema_id,
        "preparation_policy_id": policy.policy_id,
        "metadata_columns": [
            "record_id",
            "canonical_fingerprint",
            "dataset",
            "source_file",
            "source_row",
            "fine_label",
            "family_label",
            "fine_label_ambiguous",
            "duplicate_count",
            "source_file_count",
        ],
        "grouping_columns": list(schema.grouping_columns[1:]),
        "categorical_columns": list(policy.features.categorical_columns),
        "continuous_columns": list(policy.features.continuous_columns),
        "dropped_columns": list(policy.features.dropped_columns),
        "prepared_feature_columns": list(policy.features.prepared_feature_columns),
    }


def _write_manifest(
    connection: sqlite3.Connection,
    request: CanonicalBuildRequest,
    schema: DatasetSchema,
    policy: DatasetPreparationPolicy,
    source_files: tuple[Path, ...],
    classification: dict[str, int],
) -> dict[str, object]:
    file_reports = [
        json.loads(report)
        for (report,) in connection.execute(
            "SELECT report_json FROM completed_files ORDER BY file_index"
        )
    ]
    emitted_rows = list(
        connection.execute(
            "SELECT output_file, byte_size, sha256, row_count, report_json "
            "FROM emitted_files ORDER BY file_index"
        )
    )
    fine_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    invalid_features: Counter[str] = Counter()
    shards: list[dict[str, object]] = []
    for output_file, byte_size, digest, row_count, report_json in emitted_rows:
        shard_report = cast(dict[str, object], json.loads(report_json))
        fine_counts.update(cast(dict[str, int], shard_report["fine_label_counts"]))
        family_counts.update(cast(dict[str, int], shard_report["family_label_counts"]))
        shards.append(
            {
                "file": f"shards/{output_file}",
                "byte_size": byte_size,
                "sha256": digest,
                "row_count": row_count,
            }
        )
    rows_seen = sum(cast(int, item["rows_seen"]) for item in file_reports)
    valid_rows = sum(cast(int, item["valid_rows"]) for item in file_reports)
    invalid_width = sum(cast(int, item["invalid_width"]) for item in file_reports)
    invalid_feature_rows = sum(cast(int, item["invalid_feature_rows"]) for item in file_reports)
    emitted_count = sum(cast(int, item["row_count"]) for item in shards)
    coverage: dict[str, object] = {
        "discovered_files": len(source_files),
        "indexed_files": len(file_reports),
        "emitted_shards": len(shards),
        "rows_seen": rows_seen,
        "valid_rows": valid_rows,
        "invalid_width": invalid_width,
        "invalid_feature_rows": invalid_feature_rows,
        "emitted_rows": emitted_count,
        "complete": len(file_reports) == len(source_files) == len(shards),
    }
    for file_report in file_reports:
        invalid_features.update(cast(dict[str, int], file_report["invalid_features"]))
    coverage["invalid_features"] = dict(sorted(invalid_features.items()))
    if emitted_count != classification["eligible_fingerprints"]:
        raise DataContractError("Emitted record count does not equal eligible fingerprints")
    schema_document = _schema_document(request, schema, policy)
    schema_path = request.output_directory / f"{request.dataset.value}_canonical_schema.json"
    schema_path.write_text(json.dumps(schema_document, indent=2) + "\n", encoding="utf-8")
    identity = {
        "schema": schema_document,
        "archive_sha256": request.archive_sha256.casefold(),
        "classification": classification,
        "shards": shards,
        "fine_label_counts": dict(sorted(fine_counts.items())),
        "family_label_counts": dict(sorted(family_counts.items())),
    }
    corpus_id = run_id(identity, length=64)
    manifest: dict[str, object] = {
        "corpus_id": corpus_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "dataset": request.dataset.value,
            "source_root": str(request.root.resolve()),
            "archive_sha256": request.archive_sha256.casefold(),
            "dataset_schema_id": schema.schema_id,
            "preparation_policy_id": policy.policy_id,
            "phase_2b_audit_id": policy.evidence.phase_2b_audit_id,
            "phase_2c_analysis_id": policy.evidence.phase_2c_analysis_id,
        },
        "coverage": coverage,
        "classification": classification,
        "fine_label_counts": dict(sorted(fine_counts.items())),
        "family_label_counts": dict(sorted(family_counts.items())),
        "schema_file": schema_path.name,
        "shards": shards,
        "resume_database": str(request.work_database.resolve()),
    }
    json_path = request.output_directory / f"{request.dataset.value}_canonical.json"
    markdown_path = request.output_directory / f"{request.dataset.value}_canonical.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    indexed_files = len(file_reports)
    discovered_files = len(source_files)
    primary_conflicts = classification.get("primary_conflict_fingerprints", 0)
    fine_ambiguities = classification.get("fine_ambiguous_fingerprints", 0)
    removed_duplicates = classification["eligible_duplicate_occurrences_removed"]
    markdown_path.write_text(
        "\n".join(
            (
                f"# Canonical corpus: {request.dataset.value}",
                "",
                f"- Corpus ID: `{corpus_id}`",
                f"- Indexed source files: {indexed_files}/{discovered_files}",
                f"- Rows seen: {rows_seen:,}",
                f"- Canonical valid occurrences: {classification['canonical_valid_occurrences']:,}",
                f"- Primary-conflict fingerprints excluded: {primary_conflicts:,}",
                "- Fine-ambiguous fingerprints retained for primary analysis: "
                f"{fine_ambiguities:,}",
                f"- Eligible duplicate occurrences removed: {removed_duplicates:,}",
                f"- Canonical records emitted: {emitted_count:,}",
                f"- Parquet shards: {len(shards)}",
                "",
                "No scaler, imputer, or learned encoder is fitted in Phase 2D.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return manifest


def build_canonical_corpus(
    request: CanonicalBuildRequest, *, progress_every: int = 1
) -> dict[str, object]:
    """Build or resume a canonical Parquet corpus with full row accounting."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    adapter = _adapter(request.dataset, request.root)
    schema = adapter.schema
    policy = _policy(request.dataset)
    source_files = tuple(
        sorted(
            adapter.discover_files(),
            key=lambda path: normalized_source_file(request.root, path),
        )
    )
    fine_labels, family_labels = _labels(request.dataset)
    fine_bits = {label: 1 << index for index, label in enumerate(fine_labels)}
    family_bits = {label: 1 << index for index, label in enumerate(family_labels)}
    request.output_directory.mkdir(parents=True, exist_ok=True)
    shards_directory = request.output_directory / "shards"
    shards_directory.mkdir(parents=True, exist_ok=True)
    connection = _open_database(request, schema, policy, source_files)
    try:
        completed = {
            int(row[0]): row for row in connection.execute("SELECT * FROM completed_files")
        }
        for file_index, source_file in enumerate(source_files):
            relative = normalized_source_file(request.root, source_file)
            existing = completed.get(file_index)
            if existing is not None:
                source_stat = source_file.stat()
                if (
                    existing[1] != relative
                    or int(existing[2]) != source_stat.st_size
                    or int(existing[3]) != source_stat.st_mtime_ns
                ):
                    raise DataContractError(f"Indexed source file changed: {relative}")
                continue
            stats = _index_file(
                connection,
                request,
                schema,
                policy,
                source_file,
                file_index,
                len(source_files),
                fine_bits,
                family_bits,
            )
            if (file_index + 1) % progress_every == 0:
                print(
                    f"indexed={file_index + 1}/{len(source_files)} "
                    f"source={relative} valid_rows={stats.valid_rows}",
                    flush=True,
                )
        classification = _classify(connection)
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS batch_lookup("
            "fingerprint BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        emitted = {int(row[0]): row for row in connection.execute("SELECT * FROM emitted_files")}
        for file_index, source_file in enumerate(source_files):
            relative = normalized_source_file(request.root, source_file)
            source_digest = hashlib.sha256(relative.encode()).hexdigest()[:12]
            output_name = f"{file_index:04d}-{source_digest}.parquet"
            output_file = shards_directory / output_name
            existing = emitted.get(file_index)
            if existing is not None:
                if existing[1] != output_name or not output_file.is_file():
                    raise DataContractError(
                        f"Emitted shard is missing or mismatched: {output_name}"
                    )
                if int(existing[2]) != output_file.stat().st_size:
                    raise DataContractError(f"Emitted shard size changed: {output_name}")
                continue
            report = _emit_file(
                connection,
                request,
                schema,
                policy,
                source_file,
                file_index,
                output_file,
            )
            if (file_index + 1) % progress_every == 0:
                print(
                    f"emitted={file_index + 1}/{len(source_files)} "
                    f"source={normalized_source_file(request.root, source_file)} "
                    f"rows={report['row_count']}",
                    flush=True,
                )
        return _write_manifest(connection, request, schema, policy, source_files, classification)
    finally:
        connection.close()
