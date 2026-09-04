"""Resumable Phase 3A grouping and split-feasibility evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import canonical_json, run_id

_SCHEMA_VERSION: Final = 1
_EDGE_TIME_RE: Final = re.compile(
    r"^(?P<year>\d{4}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\.(?P<fraction>\d+)$"
)
_IP_LIKE_RE: Final = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_PLACEHOLDERS: Final = frozenset({"", "0", "0.0"})


@dataclass(frozen=True, slots=True)
class GroupFeasibilityRequest:
    """Inputs and bounded-memory controls for a Phase 3A audit."""

    dataset: DatasetName
    canonical_manifest: Path
    canonical_database: Path
    output_directory: Path
    work_database: Path
    calibration_anchors_per_class: int = 32
    scoring_anchors_per_class: int = 32
    parquet_batch_size: int = 50_000
    connectivity_batch_size: int = 10_000

    def __post_init__(self) -> None:
        if self.calibration_anchors_per_class < 1 or self.scoring_anchors_per_class < 1:
            raise ValueError("Anchor counts must be positive")
        if self.parquet_batch_size < 1 or self.connectivity_batch_size < 1:
            raise ValueError("Batch sizes must be positive")

    @property
    def total_anchors_per_class(self) -> int:
        """Return the frozen number of disjoint anchors required per primary class."""

        return self.calibration_anchors_per_class + self.scoring_anchors_per_class


class _UnionFind:
    def __init__(self, size: int, parents: Sequence[int] | None = None) -> None:
        self.parents = list(range(size)) if parents is None else list(parents)
        if len(self.parents) != size:
            raise DataContractError("Invalid persisted source-connectivity state")

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _load_manifest(request: GroupFeasibilityRequest) -> dict[str, object]:
    try:
        manifest = cast(
            dict[str, object], json.loads(request.canonical_manifest.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read canonical manifest: {error}") from error
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("dataset") != request.dataset.value:
        raise DataContractError("Canonical manifest dataset does not match the request")
    if not isinstance(manifest.get("corpus_id"), str):
        raise DataContractError("Canonical manifest has no corpus_id")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DataContractError("Canonical manifest has no shard inventory")
    return manifest


def _identity(request: GroupFeasibilityRequest, manifest: Mapping[str, object]) -> str:
    shards = cast(list[dict[str, object]], manifest["shards"])
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "corpus_id": manifest["corpus_id"],
        "shards": [
            {
                "file": shard["file"],
                "sha256": shard["sha256"],
                "row_count": shard["row_count"],
                "byte_size": shard["byte_size"],
            }
            for shard in shards
        ],
        "calibration_anchors_per_class": request.calibration_anchors_per_class,
        "scoring_anchors_per_class": request.scoring_anchors_per_class,
        "policies": (
            ["source_file", "source_directory"]
            if request.dataset is DatasetName.CIC_IOT_2023
            else ["source_file", "endpoint_pair_global", "source_file_endpoint_pair"]
        ),
    }
    return canonical_json(payload)


def _open_work_database(path: Path, identity: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS completed_shards(
            shard_index INTEGER PRIMARY KEY, shard_file TEXT NOT NULL,
            byte_size INTEGER NOT NULL, row_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS groups(
            policy TEXT NOT NULL, group_hash BLOB NOT NULL, record_count INTEGER NOT NULL,
            PRIMARY KEY(policy, group_hash)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS group_labels(
            policy TEXT NOT NULL, group_hash BLOB NOT NULL, task TEXT NOT NULL,
            label TEXT NOT NULL, record_count INTEGER NOT NULL,
            PRIMARY KEY(policy, group_hash, task, label)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS source_label_counts(
            source_file TEXT NOT NULL, task TEXT NOT NULL, label TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            PRIMARY KEY(source_file, task, label)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS source_stats(
            source_file TEXT PRIMARY KEY, record_count INTEGER NOT NULL,
            parseable_time INTEGER NOT NULL, zero_time INTEGER NOT NULL,
            ip_like_time INTEGER NOT NULL, other_time INTEGER NOT NULL,
            nonmonotonic_time INTEGER NOT NULL, placeholder_endpoint INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS cross_source_fingerprints(
            fingerprint BLOB PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError(
            "Resume database identity mismatch; use a new work database or restore the inputs"
        )
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _group_hash(policy: str, parts: Sequence[str]) -> bytes:
    digest = hashlib.sha256()
    for value in (policy, *parts):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.digest()


def _edge_time(value: str) -> tuple[str, int | None]:
    stripped = value.strip()
    if stripped in _PLACEHOLDERS:
        return "zero", None
    match = _EDGE_TIME_RE.fullmatch(stripped)
    if match is not None:
        hour, minute, second = (int(match[name]) for name in ("hour", "minute", "second"))
        if hour < 24 and minute < 60 and second < 60:
            fraction = match["fraction"][:9].ljust(9, "0")
            stamp = (
                int(match["year"]) * 86_400_000_000_000
                + (hour * 3600 + minute * 60 + second) * 1_000_000_000
                + int(fraction)
            )
            return "parseable", stamp
    if _IP_LIKE_RE.fullmatch(stripped) is not None:
        return "ip_like", None
    return "other", None


def _increment(
    values: dict[tuple[object, ...], int], key: tuple[object, ...], amount: int = 1
) -> None:
    values[key] = values.get(key, 0) + amount


def _policies(dataset: DatasetName, row: Mapping[str, object]) -> list[tuple[str, tuple[str, ...]]]:
    source = cast(str, row["source_file"])
    if dataset is DatasetName.CIC_IOT_2023:
        return [
            ("source_file", (source,)),
            ("source_directory", (cast(str, row["source_directory"]),)),
        ]
    endpoint = cast(str, row["endpoint_pair"])
    return [
        ("source_file", (source,)),
        ("endpoint_pair_global", (endpoint,)),
        ("source_file_endpoint_pair", (source, endpoint)),
    ]


def _flush_counts(
    connection: sqlite3.Connection,
    groups: Mapping[tuple[object, ...], int],
    labels: Mapping[tuple[object, ...], int],
    source_labels: Mapping[tuple[object, ...], int],
) -> None:
    connection.executemany(
        """INSERT INTO groups(policy, group_hash, record_count) VALUES (?, ?, ?)
        ON CONFLICT(policy, group_hash) DO UPDATE SET
        record_count=record_count+excluded.record_count""",
        ((cast(str, key[0]), cast(bytes, key[1]), count) for key, count in groups.items()),
    )
    connection.executemany(
        """INSERT INTO group_labels(policy, group_hash, task, label, record_count)
        VALUES (?, ?, ?, ?, ?) ON CONFLICT(policy, group_hash, task, label) DO UPDATE SET
        record_count=record_count+excluded.record_count""",
        (
            (
                cast(str, key[0]),
                cast(bytes, key[1]),
                cast(str, key[2]),
                cast(str, key[3]),
                count,
            )
            for key, count in labels.items()
        ),
    )
    connection.executemany(
        """INSERT INTO source_label_counts(source_file, task, label, record_count)
        VALUES (?, ?, ?, ?) ON CONFLICT(source_file, task, label) DO UPDATE SET
        record_count=record_count+excluded.record_count""",
        (
            (cast(str, key[0]), cast(str, key[1]), cast(str, key[2]), count)
            for key, count in source_labels.items()
        ),
    )


def _scan_shard(
    request: GroupFeasibilityRequest,
    connection: sqlite3.Connection,
    shard_index: int,
    shard: Mapping[str, object],
) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError(
            "Phase 3A requires pyarrow; install the project ML dependencies"
        ) from error
    relative = cast(str, shard["file"])
    base = request.canonical_manifest.parent.resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise DataContractError(
            f"Shard path escapes the canonical directory: {relative}"
        ) from error
    if not path.is_file() or path.stat().st_size != cast(int, shard["byte_size"]):
        raise DataContractError(f"Canonical shard is missing or has changed: {relative}")

    columns = [
        "canonical_fingerprint",
        "source_file",
        "family_label",
        "fine_label",
        "fine_label_ambiguous",
        "source_file_count",
    ]
    columns.extend(
        ["source_directory"]
        if request.dataset is DatasetName.CIC_IOT_2023
        else ["endpoint_pair", "frame_time"]
    )
    parquet = pq.ParquetFile(path)
    missing = sorted(set(columns) - set(parquet.schema_arrow.names))
    if missing:
        raise DataContractError(f"Canonical shard lacks Phase 3A columns: {missing}")

    rows_seen = 0
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    last_time: dict[str, int] = {}
    for batch in parquet.iter_batches(batch_size=request.parquet_batch_size, columns=columns):
        table = batch.to_pydict()
        groups: dict[tuple[object, ...], int] = {}
        labels: dict[tuple[object, ...], int] = {}
        source_labels: dict[tuple[object, ...], int] = {}
        cross_source: set[bytes] = set()
        digest_cache: dict[tuple[str, ...], bytes] = {}
        for position in range(batch.num_rows):
            row = {column: table[column][position] for column in columns}
            source = cast(str, row["source_file"])
            family = cast(str, row["family_label"])
            fine = cast(str, row["fine_label"])
            fine_eligible = not cast(bool, row["fine_label_ambiguous"])
            _increment(source_labels, (source, "family", family))
            if fine_eligible:
                _increment(source_labels, (source, "fine", fine))
            for policy, parts in _policies(request.dataset, row):
                cache_key = (policy, *parts)
                digest = digest_cache.get(cache_key)
                if digest is None:
                    digest = _group_hash(policy, parts)
                    digest_cache[cache_key] = digest
                _increment(groups, (policy, digest))
                _increment(labels, (policy, digest, "family", family))
                if fine_eligible:
                    _increment(labels, (policy, digest, "fine", fine))
            if cast(int, row["source_file_count"]) > 1:
                cross_source.add(bytes.fromhex(cast(str, row["canonical_fingerprint"])))
            stats = source_stats[source]
            stats["record_count"] += 1
            if request.dataset is DatasetName.EDGE_IIOTSET:
                category, stamp = _edge_time(cast(str, row["frame_time"]))
                stats[f"{category}_time"] += 1
                if stamp is not None:
                    previous = last_time.get(source)
                    if previous is not None and stamp < previous:
                        stats["nonmonotonic_time"] += 1
                    last_time[source] = stamp
                endpoints = cast(str, row["endpoint_pair"]).split("|")
                if len(endpoints) != 2 or any(item.strip() in _PLACEHOLDERS for item in endpoints):
                    stats["placeholder_endpoint"] += 1
            rows_seen += 1
        _flush_counts(connection, groups, labels, source_labels)
        connection.executemany(
            "INSERT OR IGNORE INTO cross_source_fingerprints(fingerprint) VALUES (?)",
            ((fingerprint,) for fingerprint in cross_source),
        )

    if rows_seen != cast(int, shard["row_count"]):
        raise DataContractError(
            f"Shard row count mismatch for {relative}: "
            f"expected={shard['row_count']}, actual={rows_seen}"
        )
    connection.executemany(
        """INSERT INTO source_stats(
        source_file, record_count, parseable_time, zero_time, ip_like_time, other_time,
        nonmonotonic_time, placeholder_endpoint) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
        record_count=record_count+excluded.record_count,
        parseable_time=parseable_time+excluded.parseable_time,
        zero_time=zero_time+excluded.zero_time,
        ip_like_time=ip_like_time+excluded.ip_like_time,
        other_time=other_time+excluded.other_time,
        nonmonotonic_time=nonmonotonic_time+excluded.nonmonotonic_time,
        placeholder_endpoint=placeholder_endpoint+excluded.placeholder_endpoint""",
        (
            (
                source,
                stats["record_count"],
                stats["parseable_time"],
                stats["zero_time"],
                stats["ip_like_time"],
                stats["other_time"],
                stats["nonmonotonic_time"],
                stats["placeholder_endpoint"],
            )
            for source, stats in source_stats.items()
        ),
    )
    connection.execute(
        "INSERT INTO completed_shards VALUES (?, ?, ?, ?)",
        (shard_index, relative, cast(int, shard["byte_size"]), rows_seen),
    )
    connection.commit()


def _source_connectivity(
    request: GroupFeasibilityRequest, work: sqlite3.Connection
) -> tuple[dict[str, int], dict[str, object]]:
    canonical = sqlite3.connect(request.canonical_database)
    try:
        files = canonical.execute(
            "SELECT file_index, source_file FROM completed_files ORDER BY file_index"
        ).fetchall()
        if [row[0] for row in files] != list(range(len(files))):
            raise DataContractError("Canonical source-file indices are not contiguous")
        persisted = work.execute(
            "SELECT value FROM metadata WHERE key='connectivity_state'"
        ).fetchone()
        cursor = b""
        processed = 0
        parents: Sequence[int] | None = None
        if persisted is not None:
            persisted_state = cast(dict[str, object], json.loads(cast(str, persisted[0])))
            cursor = bytes.fromhex(cast(str, persisted_state["cursor"]))
            processed = cast(int, persisted_state["processed"])
            parents = cast(list[int], persisted_state["parents"])
        union = _UnionFind(len(files), parents)
        canonical.execute("CREATE TEMP TABLE IF NOT EXISTS phase3a_lookup(fp BLOB PRIMARY KEY)")
        cross_total = cast(
            int, work.execute("SELECT COUNT(*) FROM cross_source_fingerprints").fetchone()[0]
        )
        while True:
            fingerprints = [
                cast(bytes, row[0])
                for row in work.execute(
                    "SELECT fingerprint FROM cross_source_fingerprints "
                    "WHERE fingerprint > ? ORDER BY fingerprint LIMIT ?",
                    (cursor, request.connectivity_batch_size),
                ).fetchall()
            ]
            if not fingerprints:
                break
            canonical.execute("DELETE FROM phase3a_lookup")
            canonical.executemany(
                "INSERT INTO phase3a_lookup(fp) VALUES (?)", ((item,) for item in fingerprints)
            )
            masks = canonical.execute(
                "SELECT fingerprint, source_mask FROM fingerprints "
                "JOIN phase3a_lookup ON fingerprint=fp ORDER BY fingerprint"
            ).fetchall()
            if len(masks) != len(fingerprints):
                raise DataContractError("Canonical database is missing a cross-source fingerprint")
            for _, raw_mask in masks:
                mask = bytes(raw_mask)
                members = [
                    index for index in range(len(files)) if mask[index // 8] & (1 << (index % 8))
                ]
                for member in members[1:]:
                    union.union(members[0], member)
            cursor = fingerprints[-1]
            processed += len(fingerprints)
            state_json = canonical_json(
                {"cursor": cursor.hex(), "processed": processed, "parents": union.parents}
            )
            work.execute(
                "INSERT INTO metadata(key, value) VALUES ('connectivity_state', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (state_json,),
            )
            work.commit()
            print(f"phase3a_connectivity={processed}/{cross_total}", flush=True)
        component_by_source = {
            cast(str, source): union.find(cast(int, index)) for index, source in files
        }
        components = Counter(component_by_source.values())
        return component_by_source, {
            "source_files": len(files),
            "cross_source_fingerprints": cross_total,
            "processed_cross_source_fingerprints": processed,
            "connected_components": len(components),
            "largest_component_source_files": max(components.values(), default=0),
            "component_size_histogram": {
                str(size): count for size, count in sorted(Counter(components.values()).items())
            },
        }
    except sqlite3.Error as error:
        raise DataContractError(f"Canonical database contract error: {error}") from error
    finally:
        canonical.close()


def _policy_summary(
    connection: sqlite3.Connection,
    policy: str,
    expected_family: Iterable[str],
    expected_fine: Iterable[str],
    anchors: int,
) -> dict[str, object]:
    group_row = connection.execute(
        "SELECT COUNT(*), COALESCE(MAX(record_count), 0), COALESCE(SUM(record_count), 0) "
        "FROM groups WHERE policy=?",
        (policy,),
    ).fetchone()
    task_results: dict[str, dict[str, object]] = {}
    for task, expected in (("family", expected_family), ("fine", expected_fine)):
        rows = {
            cast(str, label): (cast(int, records), cast(int, groups), cast(int, maximum))
            for label, records, groups, maximum in connection.execute(
                "SELECT label, SUM(record_count), COUNT(*), MAX(record_count) "
                "FROM group_labels WHERE policy=? AND task=? GROUP BY label ORDER BY label",
                (policy, task),
            )
        }
        labels: dict[str, object] = {}
        for label in sorted(set(expected)):
            records, groups, maximum = rows.get(label, (0, 0, 0))
            entry: dict[str, object] = {
                "records": records,
                "groups": groups,
                "largest_group_records": maximum,
                "largest_group_share": 0.0 if records == 0 else maximum / records,
                "three_way_group_necessary_condition": groups >= 3,
            }
            if task == "family":
                entry["global_anchor_count_necessary_condition"] = records >= anchors
            labels[label] = entry
        task_results[task] = labels
    mixed: dict[str, int] = {}
    for task in ("family", "fine"):
        mixed[task] = cast(
            int,
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT group_hash FROM group_labels "
                "WHERE policy=? AND task=? GROUP BY group_hash HAVING COUNT(*) > 1)",
                (policy, task),
            ).fetchone()[0],
        )
    family_entries = cast(dict[str, dict[str, object]], task_results["family"])
    fine_entries = cast(dict[str, dict[str, object]], task_results["fine"])
    primary_ok = all(
        cast(bool, item["three_way_group_necessary_condition"])
        and cast(bool, item["global_anchor_count_necessary_condition"])
        for item in family_entries.values()
    )
    secondary_ok = all(
        cast(bool, item["three_way_group_necessary_condition"]) for item in fine_entries.values()
    )
    return {
        "groups": cast(int, group_row[0]),
        "records": cast(int, group_row[2]),
        "largest_group_records": cast(int, group_row[1]),
        "mixed_label_groups": mixed,
        "family": task_results["family"],
        "fine_unambiguous_only": task_results["fine"],
        "primary_necessary_conditions_pass": primary_ok,
        "secondary_necessary_conditions_pass": secondary_ok,
        "status": "candidate_only" if primary_ok and secondary_ok else "necessary_conditions_fail",
    }


def _connected_source_summary(
    connection: sqlite3.Connection,
    component_by_source: Mapping[str, int],
    expected_family: Iterable[str],
    expected_fine: Iterable[str],
    anchors: int,
) -> dict[str, object]:
    grouped: dict[str, dict[int, Counter[str]]] = {
        "family": defaultdict(Counter),
        "fine": defaultdict(Counter),
    }
    for source, task, label, count in connection.execute(
        "SELECT source_file, task, label, record_count FROM source_label_counts"
    ):
        component = component_by_source.get(cast(str, source))
        if component is None:
            raise DataContractError(f"Canonical database lacks source mapping for {source}")
        grouped[cast(str, task)][component][cast(str, label)] += cast(int, count)
    labels_by_task: dict[str, dict[str, object]] = {}
    mixed: dict[str, int] = {}
    all_components: set[int] = set()
    for task, expected in (("family", expected_family), ("fine", expected_fine)):
        by_component = grouped[task]
        all_components.update(by_component)
        mixed[task] = sum(len(counts) > 1 for counts in by_component.values())
        entries: dict[str, object] = {}
        for label in sorted(set(expected)):
            counts = [values[label] for values in by_component.values() if values[label] > 0]
            records = sum(counts)
            entry: dict[str, object] = {
                "records": records,
                "groups": len(counts),
                "largest_group_records": max(counts, default=0),
                "largest_group_share": 0.0 if records == 0 else max(counts) / records,
                "three_way_group_necessary_condition": len(counts) >= 3,
            }
            if task == "family":
                entry["global_anchor_count_necessary_condition"] = records >= anchors
            entries[label] = entry
        labels_by_task[task] = entries
    family_entries = cast(dict[str, dict[str, object]], labels_by_task["family"])
    fine_entries = cast(dict[str, dict[str, object]], labels_by_task["fine"])
    primary_ok = all(
        cast(bool, item["three_way_group_necessary_condition"])
        and cast(bool, item["global_anchor_count_necessary_condition"])
        for item in family_entries.values()
    )
    secondary_ok = all(
        cast(bool, item["three_way_group_necessary_condition"]) for item in fine_entries.values()
    )
    return {
        "groups": len(grouped["family"]),
        "records": sum(sum(values.values()) for values in grouped["family"].values()),
        "largest_group_records": max(
            (sum(values.values()) for values in grouped["family"].values()), default=0
        ),
        "mixed_label_groups": mixed,
        "family": labels_by_task["family"],
        "fine_unambiguous_only": labels_by_task["fine"],
        "primary_necessary_conditions_pass": primary_ok,
        "secondary_necessary_conditions_pass": secondary_ok,
        "status": "candidate_only" if primary_ok and secondary_ok else "necessary_conditions_fail",
    }


def _time_evidence(connection: sqlite3.Connection, dataset: DatasetName) -> dict[str, object]:
    if dataset is DatasetName.CIC_IOT_2023:
        return {
            "available": False,
            "reason": "CIC canonical records have capture provenance, not timestamps",
        }
    row = connection.execute(
        "SELECT COALESCE(SUM(record_count),0), COALESCE(SUM(parseable_time),0), "
        "COALESCE(SUM(zero_time),0), COALESCE(SUM(ip_like_time),0), "
        "COALESCE(SUM(other_time),0), COALESCE(SUM(nonmonotonic_time),0), "
        "COALESCE(SUM(placeholder_endpoint),0) FROM source_stats"
    ).fetchone()
    total = cast(int, row[0])
    parseable = cast(int, row[1])
    return {
        "available": True,
        "records": total,
        "parseable_frame_time": parseable,
        "parseable_share": 0.0 if total == 0 else parseable / total,
        "zero_frame_time": cast(int, row[2]),
        "ip_like_frame_time": cast(int, row[3]),
        "other_unparseable_frame_time": cast(int, row[4]),
        "nonmonotonic_parseable_transitions": cast(int, row[5]),
        "placeholder_endpoint_records": cast(int, row[6]),
        "timestamp_alone_accepted_as_group": False,
    }


def _write_markdown(report: Mapping[str, object], path: Path) -> None:
    policies = cast(dict[str, dict[str, object]], report["candidate_policies"])
    lines = [
        f"# Phase 3A grouping feasibility: {report['dataset']}",
        "",
        f"- Analysis ID: `{report['analysis_id']}`",
        f"- Canonical corpus ID: `{report['canonical_corpus_id']}`",
        f"- Canonical records scanned: {report['records_scanned']:,}",
        f"- Required disjoint anchors per primary class: {report['anchors_per_primary_class']}",
        "- Train/validation/test assignment created: no",
        "",
        "## Candidate policies",
        "",
        "| Policy | Groups | Largest group | Primary gate | Fine gate | Status |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, policy in policies.items():
        lines.append(
            f"| {name} | {policy['groups']:,} | {policy['largest_group_records']:,} | "
            f"{policy['primary_necessary_conditions_pass']} | "
            f"{policy['secondary_necessary_conditions_pass']} | {policy['status']} |"
        )
    lines.extend(
        [
            "",
            (
                "A passing row is only a necessary-condition result. It does not freeze a "
                "grouping policy or prove that a feasible stratified split exists."
            ),
            "",
            "## Scientific boundary",
            "",
            (
                "Exact duplicate provenance is represented by duplicate-connected source-file "
                "components. Fine-label feasibility excludes canonical records marked as "
                "fine-label ambiguous. Edge timestamps are quality-audited but are not accepted "
                "as a grouping key in Phase 3A."
            ),
            "",
            "## Next decision",
            "",
            cast(str, report["next_decision"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_group_feasibility(
    request: GroupFeasibilityRequest, *, progress_every: int = 1
) -> dict[str, object]:
    """Build or resume exact candidate-group evidence without assigning splits."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    manifest = _load_manifest(request)
    identity = _identity(request, manifest)
    connection = _open_work_database(request.work_database, identity)
    try:
        shards = cast(list[dict[str, object]], manifest["shards"])
        completed = {
            cast(int, row[0])
            for row in connection.execute("SELECT shard_index FROM completed_shards")
        }
        for index, shard in enumerate(shards):
            if index in completed:
                continue
            _scan_shard(request, connection, index, shard)
            if (index + 1) % progress_every == 0 or index + 1 == len(shards):
                print(f"phase3a_scan={index + 1}/{len(shards)} file={shard['file']}", flush=True)
        component_by_source, connectivity = _source_connectivity(request, connection)
        family_counts = cast(dict[str, int], manifest.get("family_label_counts", {}))
        fine_counts = cast(dict[str, int], manifest.get("fine_label_counts", {}))
        policies = (
            ["source_file", "source_directory"]
            if request.dataset is DatasetName.CIC_IOT_2023
            else ["source_file", "endpoint_pair_global", "source_file_endpoint_pair"]
        )
        summaries = {
            policy: _policy_summary(
                connection,
                policy,
                family_counts,
                fine_counts,
                request.total_anchors_per_class,
            )
            for policy in policies
        }
        summaries["duplicate_connected_source_file"] = _connected_source_summary(
            connection,
            component_by_source,
            family_counts,
            fine_counts,
            request.total_anchors_per_class,
        )
        records = cast(
            int,
            connection.execute(
                "SELECT COALESCE(SUM(row_count),0) FROM completed_shards"
            ).fetchone()[0],
        )
        stable = {
            "schema_version": _SCHEMA_VERSION,
            "dataset": request.dataset.value,
            "canonical_corpus_id": manifest["corpus_id"],
            "records_scanned": records,
            "anchors_per_primary_class": request.total_anchors_per_class,
            "candidate_policies": summaries,
            "source_connectivity": connectivity,
            "edge_grouping_field_quality": _time_evidence(connection, request.dataset),
            "split_assignment_created": False,
            "group_policy_frozen": False,
            "fine_task_excludes_ambiguous_labels": True,
            "limitations": [
                (
                    "No split fraction is frozen, so this audit cannot assign train, validation, "
                    "or test records."
                ),
                (
                    "Passing necessary conditions does not prove a feasible class-stratified "
                    "grouped split."
                ),
                (
                    "Endpoint groups use the retained canonical representative; duplicate "
                    "provenance across non-representative raw rows requires the "
                    "source-connectivity result."
                ),
            ],
            "next_decision": (
                "Freeze a deduplicated record-level CIC policy and split fractions before "
                "Phase 3B; the provider-shuffled CSVs do not expose capture identities."
                if request.dataset is DatasetName.CIC_IOT_2023
                else (
                    "Use the measured timestamp and endpoint quality to freeze an Edge run/block "
                    "rule before Phase 3B."
                )
            ),
        }
        report = {"analysis_id": run_id(stable), "created_at_utc": _utc_now(), **stable}
        request.output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"{request.dataset.value}_phase3a"
        (request.output_directory / f"{stem}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_markdown(report, request.output_directory / f"{stem}.md")
        return report
    finally:
        connection.close()
