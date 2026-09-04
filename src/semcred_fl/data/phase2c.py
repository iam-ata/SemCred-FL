"""Resumable diagnostics that turn Phase 2B findings into preprocessing evidence."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
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
from semcred_fl.data.record_ids import normalized_source_file
from semcred_fl.data.validation import DatasetAdapter, is_finite_number
from semcred_fl.identity import canonical_json, run_id

_PHASE2C_SCHEMA_VERSION: Final = 1
_CONFLICT_BATCH_SIZE: Final = 10_000
_DOMAIN_BATCH_SIZE: Final = 5_000
_MISSING_TOKENS: Final = frozenset({"", "?", "na", "n/a", "nan", "none", "null"})


@dataclass(frozen=True, slots=True)
class DomainRule:
    """Exact raw-value collection rule for one model-facing column."""

    column: str
    mode: str
    reason: str

    def __post_init__(self) -> None:
        if self.mode not in {"all", "exceptional"}:
            raise ValueError("Domain mode must be 'all' or 'exceptional'")


@dataclass(frozen=True, slots=True)
class Phase2CRequest:
    """Paths and limits that identify one Phase 2C diagnostic run."""

    dataset: DatasetName
    root: Path
    audit_json: Path
    audit_database: Path
    output_directory: Path
    work_database: Path
    max_exact_domain_values: int = 100_000
    reported_top_values: int = 50

    def __post_init__(self) -> None:
        if self.max_exact_domain_values < 1:
            raise ValueError("max_exact_domain_values must be positive")
        if self.reported_top_values < 1:
            raise ValueError("reported_top_values must be positive")


def _adapter(request: Phase2CRequest) -> DatasetAdapter:
    if request.dataset is DatasetName.CIC_IOT_2023:
        return CICIoT2023Adapter(request.root)
    return EdgeIIoTsetAdapter(request.root)


def _read_audit(request: Phase2CRequest, schema: DatasetSchema) -> dict[str, object]:
    try:
        summary = cast(
            dict[str, object], json.loads(request.audit_json.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Cannot read Phase 2B audit JSON: {request.audit_json}") from exc

    provenance = cast(dict[str, object], summary.get("provenance"))
    if provenance.get("dataset") != request.dataset.value:
        raise DataContractError("Phase 2B audit dataset does not match the Phase 2C request")
    if provenance.get("dataset_schema_id") != schema.schema_id:
        raise DataContractError("Phase 2B audit schema does not match the current adapter schema")
    source_root = Path(cast(str, provenance.get("source_root"))).resolve()
    if source_root != request.root.resolve():
        raise DataContractError("Phase 2B audit source root does not match the Phase 2C root")
    audit_id = summary.get("audit_id")
    if not isinstance(audit_id, str) or len(audit_id) != 64:
        raise DataContractError("Phase 2B audit JSON has no valid full audit_id")
    return summary


def _kind_count(profile: dict[str, object], kind: str) -> int:
    return int(cast(dict[str, int], profile.get("kind_counts", {})).get(kind, 0))


def _domain_rules(
    summary: dict[str, object], schema: DatasetSchema, max_exact_values: int
) -> tuple[DomainRule, ...]:
    if schema.dataset is DatasetName.CIC_IOT_2023:
        return ()

    profiles = cast(dict[str, dict[str, object]], summary["columns"])
    rules: list[DomainRule] = []
    categorical = set(schema.categorical_columns)
    for column in schema.model_feature_columns:
        profile = profiles[column]
        cardinality = cast(int, profile["approximate_cardinality"])
        has_text = _kind_count(profile, "text") > 0
        constant = bool(profile["constant"])
        if column in categorical:
            rules.append(DomainRule(column, "all", "declared_categorical"))
        elif constant:
            rules.append(DomainRule(column, "all", "phase_2b_constant"))
        elif has_text and cardinality <= max_exact_values:
            rules.append(DomainRule(column, "all", "text_bearing_low_cardinality"))
        elif has_text:
            rules.append(DomainRule(column, "exceptional", "text_in_high_cardinality_column"))
    return tuple(rules)


def _open_audit_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2B audit database not found: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA query_only=ON")
    required = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fingerprints'"
    ).fetchone()
    if required is None:
        connection.close()
        raise DataContractError("Phase 2B database has no fingerprints table")
    return connection


def _open_work_database(
    request: Phase2CRequest,
    schema: DatasetSchema,
    summary: dict[str, object],
    rules: tuple[DomainRule, ...],
) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS conflict_labels (
            fingerprint BLOB PRIMARY KEY,
            total_count INTEGER NOT NULL,
            observed_count INTEGER NOT NULL DEFAULT 0,
            fine_mask INTEGER NOT NULL DEFAULT 0,
            family_mask INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS domain_values (
            column_name TEXT NOT NULL,
            raw_value TEXT NOT NULL,
            value_kind TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            PRIMARY KEY(column_name, raw_value)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS completed_files (
            source_file TEXT PRIMARY KEY,
            byte_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            rows_seen INTEGER NOT NULL,
            valid_rows INTEGER NOT NULL,
            conflict_occurrences INTEGER NOT NULL
        );
        """
    )
    identity = canonical_json(
        {
            "phase2c_schema_version": _PHASE2C_SCHEMA_VERSION,
            "dataset": request.dataset,
            "dataset_schema_id": schema.schema_id,
            "root": request.root.resolve(),
            "audit_id": summary["audit_id"],
            "domain_rules": rules,
            "max_exact_domain_values": request.max_exact_domain_values,
        }
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='phase2c_identity'"
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('phase2c_identity', ?)", (identity,)
        )
        connection.commit()
    elif existing[0] != identity:
        connection.close()
        raise DataContractError(
            "The Phase 2C work database belongs to a different audit, root, schema, or policy"
        )
    return connection


def _initialize_conflicts(
    work: sqlite3.Connection,
    audit: sqlite3.Connection,
    expected_conflicts: int,
) -> None:
    initialized = work.execute(
        "SELECT value FROM metadata WHERE key='conflicts_initialized'"
    ).fetchone()
    if initialized is None:
        cursor = audit.execute(
            "SELECT fingerprint, total_count FROM fingerprints WHERE label_conflict=1"
        )
        copied = 0
        work.execute("BEGIN IMMEDIATE")
        try:
            while True:
                rows = cursor.fetchmany(_CONFLICT_BATCH_SIZE)
                if not rows:
                    break
                work.executemany(
                    "INSERT INTO conflict_labels(fingerprint, total_count) VALUES (?, ?)", rows
                )
                copied += len(rows)
            if copied != expected_conflicts:
                raise DataContractError(
                    "Phase 2B JSON and database disagree on fine-label conflict count"
                )
            work.execute("INSERT INTO metadata(key, value) VALUES ('conflicts_initialized', '1')")
            work.commit()
        except Exception:
            work.rollback()
            raise
    actual = cast(int, work.execute("SELECT COUNT(*) FROM conflict_labels").fetchone()[0])
    if actual != expected_conflicts:
        raise DataContractError("Phase 2C conflict index is incomplete")


def _value_kind(value: str) -> str:
    lowered = value.casefold()
    if lowered in _MISSING_TOKENS:
        return "missing"
    try:
        if value.casefold().startswith(("0x", "+0x", "-0x")):
            int(value, 0)
            return "hex_integer"
        parsed = float(value)
    except ValueError:
        return "text"
    return "finite_numeric" if math.isfinite(parsed) else "non_finite"


def _retain_domain_value(rule: DomainRule, kind: str) -> bool:
    return rule.mode == "all" or kind in {"missing", "non_finite", "text"}


def _labels(
    adapter: DatasetAdapter,
    source_file: Path,
    row: list[str],
    indices: dict[str, int],
) -> tuple[str, str] | None:
    if isinstance(adapter, CICIoT2023Adapter):
        fine = CIC_SOURCE_DIRECTORY_TO_FINE[source_file.parent.name]
        return fine, CIC_FINE_TO_FAMILY[fine]
    edge = cast(EdgeIIoTsetAdapter, adapter)
    edge_fine = canonical_edge_fine_label(row[indices["Attack_type"]].strip())
    if edge_fine is None or edge_fine != edge.expected_fine_label(source_file):
        return None
    binary = row[indices["Attack_label"]].strip()
    expected_binary = "0" if edge_fine == "Normal" else "1"
    if binary != expected_binary:
        return None
    return edge_fine, EDGE_FINE_TO_FAMILY[edge_fine]


def _completed_file(connection: sqlite3.Connection, relative: str, source_file: Path) -> bool:
    row = connection.execute(
        "SELECT byte_size, modified_ns FROM completed_files WHERE source_file=?", (relative,)
    ).fetchone()
    if row is None:
        return False
    stat = source_file.stat()
    if row[0] != stat.st_size or row[1] != stat.st_mtime_ns:
        raise DataContractError(f"Previously analyzed source file changed: {relative}")
    return True


def _scan_file(
    connection: sqlite3.Connection,
    adapter: DatasetAdapter,
    source_file: Path,
    conflict_fingerprints: set[bytes],
    rules: tuple[DomainRule, ...],
    fine_bits: dict[str, int],
    family_bits: dict[str, int],
) -> tuple[int, int, int]:
    schema = adapter.schema
    relative = normalized_source_file(adapter.root, source_file)
    rows_seen = 0
    valid_rows = 0
    conflict_hits: Counter[tuple[bytes, int, int]] = Counter()
    domain_counts: Counter[tuple[str, str]] = Counter()
    rule_map = {rule.column: rule for rule in rules}
    with source_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(header) != schema.source_columns:
            raise DataContractError(f"Header changed after Phase 2B audit: {relative}")
        indices = {name: index for index, name in enumerate(header)}
        for row in reader:
            rows_seen += 1
            if len(row) != len(header):
                continue
            stripped = tuple(row[indices[name]].strip() for name in schema.model_feature_columns)
            for column, rule in rule_map.items():
                value = row[indices[column]].strip()
                if rule.mode == "all":
                    domain_counts[(column, value)] += 1
                else:
                    kind = _value_kind(value)
                    if _retain_domain_value(rule, kind):
                        domain_counts[(column, value)] += 1
            labels = _labels(adapter, source_file, row, indices)
            if labels is None:
                continue
            if isinstance(adapter, CICIoT2023Adapter) and not all(
                is_finite_number(value) for value in stripped
            ):
                continue
            valid_rows += 1
            fingerprint = model_feature_fingerprint(stripped)
            if fingerprint in conflict_fingerprints:
                fine, family = labels
                conflict_hits[(fingerprint, fine_bits[fine], family_bits[family])] += 1

    stat = source_file.stat()
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.executemany(
            """
            UPDATE conflict_labels
            SET observed_count=observed_count+?,
                fine_mask=fine_mask | ?,
                family_mask=family_mask | ?
            WHERE fingerprint=?
            """,
            (
                (count, fine_bit, family_bit, fingerprint)
                for (fingerprint, fine_bit, family_bit), count in conflict_hits.items()
            ),
        )
        domain_items = list(domain_counts.items())
        for start in range(0, len(domain_items), _DOMAIN_BATCH_SIZE):
            connection.executemany(
                """
                INSERT INTO domain_values(column_name, raw_value, value_kind, occurrence_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(column_name, raw_value) DO UPDATE SET
                    occurrence_count=occurrence_count+excluded.occurrence_count
                """,
                (
                    (column, value, _value_kind(value), count)
                    for (column, value), count in domain_items[start : start + _DOMAIN_BATCH_SIZE]
                ),
            )
        connection.execute(
            "INSERT INTO completed_files VALUES (?, ?, ?, ?, ?, ?)",
            (
                relative,
                stat.st_size,
                stat.st_mtime_ns,
                rows_seen,
                valid_rows,
                sum(conflict_hits.values()),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return rows_seen, valid_rows, sum(conflict_hits.values())


def _mask_labels(mask: int, labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(label for index, label in enumerate(labels) if mask & (1 << index))


def _conflict_summary(
    connection: sqlite3.Connection,
    fine_labels: tuple[str, ...],
    family_labels: tuple[str, ...],
) -> dict[str, object]:
    totals: Counter[str] = Counter()
    combinations: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    combination_occurrences: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for total_count, observed_count, fine_mask, family_mask in connection.execute(
        "SELECT total_count, observed_count, fine_mask, family_mask FROM conflict_labels"
    ):
        totals["fingerprints"] += 1
        totals["occurrences"] += cast(int, total_count)
        if observed_count != total_count:
            totals["incomplete_fingerprints"] += 1
            totals["unresolved_occurrences"] += abs(
                cast(int, total_count) - cast(int, observed_count)
            )
        fine = _mask_labels(cast(int, fine_mask), fine_labels)
        families = _mask_labels(cast(int, family_mask), family_labels)
        if len(families) > 1:
            bucket = "primary_family_conflict"
        elif len(fine) > 1:
            bucket = "fine_only_conflict"
        else:
            bucket = "unresolved_conflict"
        totals[f"{bucket}_fingerprints"] += 1
        totals[f"{bucket}_occurrences"] += cast(int, total_count)
        key = (fine, families)
        combinations[key] += 1
        combination_occurrences[key] += cast(int, total_count)

    ranked = sorted(
        combinations,
        key=lambda key: (-combinations[key], -combination_occurrences[key], key),
    )
    return {
        "audited_conflict_fingerprints": totals["fingerprints"],
        "audited_conflict_occurrences": totals["occurrences"],
        "primary_family_conflict_fingerprints": totals["primary_family_conflict_fingerprints"],
        "primary_family_conflict_occurrences": totals["primary_family_conflict_occurrences"],
        "fine_only_conflict_fingerprints": totals["fine_only_conflict_fingerprints"],
        "fine_only_conflict_occurrences": totals["fine_only_conflict_occurrences"],
        "unresolved_conflict_fingerprints": totals["unresolved_conflict_fingerprints"],
        "incomplete_fingerprints": totals["incomplete_fingerprints"],
        "unresolved_occurrences": totals["unresolved_occurrences"],
        "label_combinations": [
            {
                "fine_labels": list(key[0]),
                "family_labels": list(key[1]),
                "fingerprint_count": combinations[key],
                "occurrence_count": combination_occurrences[key],
            }
            for key in ranked
        ],
    }


def _domain_summary(
    connection: sqlite3.Connection,
    summary: dict[str, object],
    rules: tuple[DomainRule, ...],
    reported_top_values: int,
) -> dict[str, object]:
    profiles = cast(dict[str, dict[str, object]], summary["columns"])
    result: dict[str, object] = {}
    for rule in rules:
        rows = connection.execute(
            """
            SELECT raw_value, value_kind, occurrence_count
            FROM domain_values WHERE column_name=?
            ORDER BY occurrence_count DESC, raw_value
            """,
            (rule.column,),
        ).fetchall()
        kind_counts: Counter[str] = Counter()
        for _, kind, count in rows:
            kind_counts[cast(str, kind)] += cast(int, count)
        top_values = [
            {"raw_value": value, "value_kind": kind, "occurrence_count": count}
            for value, kind, count in rows[:reported_top_values]
        ]
        exceptional = [
            {"raw_value": value, "value_kind": kind, "occurrence_count": count}
            for value, kind, count in rows
            if kind in {"missing", "non_finite", "text"}
        ]
        item: dict[str, object] = {
            "collection_mode": rule.mode,
            "selection_reason": rule.reason,
            "phase_2b_profile": profiles[rule.column],
            "collected_occurrences": sum(cast(int, row[2]) for row in rows),
            "exact_distinct_collected_values": len(rows),
            "collected_kind_counts": dict(sorted(kind_counts.items())),
            "top_values": top_values,
            "exceptional_values": exceptional,
        }
        if len(rows) <= reported_top_values:
            item["all_values"] = top_values
        result[rule.column] = item
    return result


def _aggregate(
    connection: sqlite3.Connection,
    request: Phase2CRequest,
    schema: DatasetSchema,
    audit_summary: dict[str, object],
    rules: tuple[DomainRule, ...],
    discovered_files: int,
) -> dict[str, object]:
    coverage = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(rows_seen),0), COALESCE(SUM(valid_rows),0), "
        "COALESCE(SUM(conflict_occurrences),0) FROM completed_files"
    ).fetchone()
    assert coverage is not None
    fine_to_family = (
        CIC_FINE_TO_FAMILY if request.dataset is DatasetName.CIC_IOT_2023 else EDGE_FINE_TO_FAMILY
    )
    fine_labels = tuple(sorted(fine_to_family))
    family_labels = tuple(sorted(set(fine_to_family.values())))
    conflicts = _conflict_summary(connection, fine_labels, family_labels)
    complete = (
        cast(int, coverage[0]) == discovered_files
        and cast(int, conflicts["incomplete_fingerprints"]) == 0
    )
    report: dict[str, object] = {
        "phase2c_schema_version": _PHASE2C_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "dataset": request.dataset.value,
            "source_root": str(request.root.resolve()),
            "dataset_schema_id": schema.schema_id,
            "phase_2b_audit_id": audit_summary["audit_id"],
            "phase_2b_audit_json": str(request.audit_json.resolve()),
            "phase_2b_audit_database": str(request.audit_database.resolve()),
        },
        "coverage": {
            "discovered_files": discovered_files,
            "completed_files": coverage[0],
            "rows_seen": coverage[1],
            "valid_rows": coverage[2],
            "conflict_occurrences_observed": coverage[3],
            "complete": complete,
        },
        "label_conflicts": conflicts,
        "raw_value_domains": _domain_summary(
            connection, audit_summary, rules, request.reported_top_values
        ),
        "policy_status": {
            "frozen": False,
            "reason": (
                "The diagnostic must be complete and reviewed before cleaning, typing, "
                "duplicate, and secondary-label policies are frozen."
            ),
        },
    }
    report["analysis_id"] = run_id(
        {
            key: value
            for key, value in report.items()
            if key not in {"generated_at_utc", "analysis_id"}
        },
        length=64,
    )
    return report


def _markdown(report: dict[str, object]) -> str:
    coverage = cast(dict[str, object], report["coverage"])
    conflicts = cast(dict[str, object], report["label_conflicts"])
    domains = cast(dict[str, object], report["raw_value_domains"])
    return "\n".join(
        (
            "# SemCred-FL Phase 2C diagnostic",
            "",
            f"Analysis ID: `{report['analysis_id']}`",
            "",
            "## Coverage",
            "",
            f"- Files completed: {coverage['completed_files']}/{coverage['discovered_files']}",
            f"- Rows inspected: {coverage['rows_seen']}",
            f"- Audit-valid rows reproduced: {coverage['valid_rows']}",
            f"- Complete: {str(coverage['complete']).lower()}",
            "",
            "## Exact label-conflict partition",
            "",
            (
                "- Primary-family-conflict fingerprints: "
                f"{conflicts['primary_family_conflict_fingerprints']} "
                f"({conflicts['primary_family_conflict_occurrences']} occurrences)"
            ),
            (
                "- Fine-only-conflict fingerprints: "
                f"{conflicts['fine_only_conflict_fingerprints']} "
                f"({conflicts['fine_only_conflict_occurrences']} occurrences)"
            ),
            f"- Incomplete conflict fingerprints: {conflicts['incomplete_fingerprints']}",
            "",
            "## Raw-value domain coverage",
            "",
            f"- Targeted columns: {len(domains)}",
            "- Exact values are retained in the resume database; JSON lists bounded summaries.",
            "",
            "## Decision gate",
            "",
            (
                "No preprocessing policy is frozen by this report alone. The completed output "
                "must be reviewed before exclusion, parsing, encoding, and duplicate rules are "
                "written into the dataset contracts."
            ),
            "",
        )
    )


def run_phase2c(request: Phase2CRequest, *, progress_every: int = 1) -> dict[str, object]:
    """Run or resume Phase 2C diagnostics and write JSON and Markdown reports."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    adapter = _adapter(request)
    files = adapter.discover_files()
    audit_summary = _read_audit(request, adapter.schema)
    audit_coverage = cast(dict[str, object], audit_summary["coverage"])
    if cast(int, audit_coverage["completed_files"]) != len(files):
        raise DataContractError("Phase 2B audit does not cover every currently discovered file")
    rules = _domain_rules(audit_summary, adapter.schema, request.max_exact_domain_values)
    work = _open_work_database(request, adapter.schema, audit_summary, rules)
    audit = _open_audit_database(request.audit_database)
    expected_conflicts = cast(
        int,
        cast(dict[str, object], audit_summary["duplicates"])[
            "fingerprints_with_conflicting_fine_labels"
        ],
    )
    try:
        _initialize_conflicts(work, audit, expected_conflicts)
    finally:
        audit.close()

    conflict_fingerprints = {
        cast(bytes, row[0]) for row in work.execute("SELECT fingerprint FROM conflict_labels")
    }
    fine_to_family = (
        CIC_FINE_TO_FAMILY if request.dataset is DatasetName.CIC_IOT_2023 else EDGE_FINE_TO_FAMILY
    )
    fine_bits = {label: 1 << index for index, label in enumerate(sorted(fine_to_family))}
    family_bits = {
        label: 1 << index for index, label in enumerate(sorted(set(fine_to_family.values())))
    }
    try:
        for index, source_file in enumerate(files, start=1):
            relative = normalized_source_file(adapter.root, source_file)
            if _completed_file(work, relative, source_file):
                action = "resume"
            else:
                _scan_file(
                    work,
                    adapter,
                    source_file,
                    conflict_fingerprints,
                    rules,
                    fine_bits,
                    family_bits,
                )
                action = "analyzed"
            if index % progress_every == 0 or index == len(files):
                print(f"[{index}/{len(files)}] {action}: {relative}", flush=True)
        report = _aggregate(
            work, request, adapter.schema, audit_summary, rules, discovered_files=len(files)
        )
    finally:
        work.close()

    request.output_directory.mkdir(parents=True, exist_ok=True)
    stem = request.dataset.value
    json_path = request.output_directory / f"{stem}_phase2c.json"
    markdown_path = request.output_directory / f"{stem}_phase2c.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
