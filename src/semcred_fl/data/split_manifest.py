"""Resumable Phase 3B split assignment and trusted-anchor manifests."""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.group_refinement import (
    EdgeMitmRefinementRequest,
    _canonical_source_rows,
    _csv_alignment,
    _file_sha256,
    _mitm_source_and_shard,
    _source_family_counts,
    read_classic_pcap,
)
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.seeding import derive_seed

_SCHEMA_VERSION: Final = 1
_SPLITS: Final = ("train", "validation", "test")
_EDGE_POLICY: Final = "source_capture_plus_mitm_pcap_bidirectional_flow_v1"
_CIC_POLICY: Final = "fingerprint_deduplicated_record_hash_v1"


@dataclass(frozen=True, slots=True)
class SplitBuildRequest:
    """Frozen inputs and controls for one Phase 3B dataset split."""

    dataset: DatasetName
    canonical_manifest: Path
    canonical_database: Path
    output_directory: Path
    work_database: Path
    base_seed: int = 20260816
    train_basis_points: int = 7000
    validation_basis_points: int = 1500
    test_basis_points: int = 1500
    calibration_anchors_per_class: int = 32
    scoring_anchors_per_class: int = 32
    parquet_batch_size: int = 50_000
    phase3a_database: Path | None = None
    phase3a1_report: Path | None = None
    edge_raw_mitm_csv: Path | None = None
    edge_mitm_pcap: Path | None = None

    def __post_init__(self) -> None:
        fractions = (
            self.train_basis_points,
            self.validation_basis_points,
            self.test_basis_points,
        )
        if any(value < 1 for value in fractions) or sum(fractions) != 10_000:
            raise ValueError("Split basis points must be positive and sum to 10000")
        if self.base_seed < 0:
            raise ValueError("base_seed cannot be negative")
        if self.calibration_anchors_per_class < 1 or self.scoring_anchors_per_class < 1:
            raise ValueError("Anchor counts must be positive")
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive")
        if self.dataset is DatasetName.EDGE_IIOTSET and any(
            value is None
            for value in (
                self.phase3a_database,
                self.phase3a1_report,
                self.edge_raw_mitm_csv,
                self.edge_mitm_pcap,
            )
        ):
            raise ValueError("Edge Phase 3B requires Phase 3A/3A.1 and MITM source evidence")

    @property
    def anchors_per_class(self) -> int:
        """Return the total disjoint anchor requirement per primary class."""

        return self.calibration_anchors_per_class + self.scoring_anchors_per_class

    @property
    def split_basis_points(self) -> dict[str, int]:
        """Return the requested partition proportions in basis points."""

        return dict(
            zip(
                _SPLITS,
                (
                    self.train_basis_points,
                    self.validation_basis_points,
                    self.test_basis_points,
                ),
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _Group:
    group_id: str
    family: str
    records: int


@dataclass(frozen=True, slots=True)
class _Plan:
    report: dict[str, object]
    edge_source_groups: Mapping[str, str]
    edge_mitm_groups: Mapping[int, str]
    edge_group_splits: Mapping[str, str]
    edge_mitm_source: str | None


_AnchorCandidate = tuple[str, str, str, str, int]


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error


def _hash_text(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def _seeded_digest(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()


def _record_split(record_id: str, split_seed: int, basis_points: Mapping[str, int]) -> str:
    bucket = int.from_bytes(_seeded_digest(split_seed, "record-split", record_id)[:8], "big")
    bucket %= 10_000
    boundary = 0
    for split in _SPLITS:
        boundary += basis_points[split]
        if bucket < boundary:
            return split
    raise AssertionError("Split basis-point boundaries are incomplete")


def _objective(counts: Mapping[str, int], total: int, basis_points: Mapping[str, int]) -> float:
    return sum(
        ((counts[split] - total * basis_points[split] / 10_000) / max(total, 1)) ** 2
        for split in _SPLITS
    )


def _candidate_group_assignment(
    ordered: Sequence[_Group],
    initial: Sequence[str],
    basis_points: Mapping[str, int],
) -> dict[str, str]:
    assignment = {group.group_id: split for group, split in zip(ordered[:3], initial, strict=True)}
    counts = Counter[str]()
    group_counts = Counter[str]()
    for group in ordered[:3]:
        split = assignment[group.group_id]
        counts[split] += group.records
        group_counts[split] += 1
    total = sum(group.records for group in ordered)
    for group in ordered[3:]:
        scored: list[tuple[float, str]] = []
        for split in _SPLITS:
            trial = Counter(counts)
            trial[split] += group.records
            scored.append((_objective(trial, total, basis_points), split))
        split = min(scored)[1]
        assignment[group.group_id] = split
        counts[split] += group.records
        group_counts[split] += 1
    improved = True
    while improved:
        improved = False
        current = _objective(counts, total, basis_points)
        for group in ordered:
            source = assignment[group.group_id]
            if group_counts[source] <= 1:
                continue
            for target in _SPLITS:
                if target == source:
                    continue
                trial = Counter(counts)
                trial[source] -= group.records
                trial[target] += group.records
                score = _objective(trial, total, basis_points)
                if score + 1e-18 < current:
                    assignment[group.group_id] = target
                    counts = trial
                    group_counts[source] -= 1
                    group_counts[target] += 1
                    improved = True
                    current = score
    return assignment


def _assign_grouped_family(
    groups: Sequence[_Group], basis_points: Mapping[str, int], split_seed: int
) -> dict[str, str]:
    if len(groups) < 3:
        raise DataContractError(f"Primary family {groups[0].family!r} has fewer than three groups")
    ordered = sorted(
        groups,
        key=lambda group: (
            -group.records,
            _seeded_digest(split_seed, "group-order", group.group_id),
        ),
    )
    candidates = [
        _candidate_group_assignment(ordered, permutation, basis_points)
        for permutation in itertools.permutations(_SPLITS)
    ]
    total = sum(group.records for group in ordered)

    def score(assignment: Mapping[str, str]) -> tuple[float, str]:
        counts = Counter[str]()
        for group in ordered:
            counts[assignment[group.group_id]] += group.records
        return _objective(counts, total, basis_points), canonical_json(
            dict(sorted(assignment.items()))
        )

    return min(candidates, key=score)


def _edge_plan(
    request: SplitBuildRequest, manifest: Mapping[str, object], split_seed: int
) -> tuple[list[_Group], dict[str, str], dict[int, str], dict[str, str], dict[str, object]]:
    phase3a_database = cast(Path, request.phase3a_database)
    phase3a1_path = cast(Path, request.phase3a1_report)
    raw_csv = cast(Path, request.edge_raw_mitm_csv)
    pcap_path = cast(Path, request.edge_mitm_pcap)
    phase3a1 = _load_json(phase3a1_path, "Phase 3A.1 report")
    if phase3a1.get("dataset") != request.dataset.value:
        raise DataContractError("Phase 3A.1 dataset mismatch")
    if phase3a1.get("canonical_corpus_id") != manifest.get("corpus_id"):
        raise DataContractError("Phase 3A.1 canonical corpus mismatch")
    candidate_policies = phase3a1.get("candidate_policies")
    flow_candidate = (
        candidate_policies.get("source_capture_plus_mitm_pcap_bidirectional_flow")
        if isinstance(candidate_policies, dict)
        else None
    )
    if not isinstance(flow_candidate, dict) or not flow_candidate.get(
        "primary_necessary_conditions_pass"
    ):
        raise DataContractError("Phase 3A.1 bidirectional-flow candidate did not pass")
    alignment_report = phase3a1.get("csv_pcap_alignment")
    if not isinstance(alignment_report, dict) or not alignment_report.get(
        "packet_ordinal_alignment_accepted"
    ):
        raise DataContractError("Phase 3A.1 packet-ordinal alignment did not pass")
    hashes = phase3a1.get("input_sha256")
    if not isinstance(hashes, dict):
        raise DataContractError("Phase 3A.1 input hashes are missing")
    if _file_sha256(raw_csv) != hashes.get("raw_csv") or _file_sha256(pcap_path) != hashes.get(
        "pcap"
    ):
        raise DataContractError("Edge MITM sources changed after Phase 3A.1")
    refinement_request = EdgeMitmRefinementRequest(
        canonical_manifest=request.canonical_manifest,
        canonical_database=request.canonical_database,
        phase3a_report=phase3a1_path,
        phase3a_database=phase3a_database,
        raw_csv=raw_csv,
        pcap=pcap_path,
        output_directory=request.output_directory,
    )
    mitm_source, mitm_shard = _mitm_source_and_shard(refinement_request, manifest)
    packets = read_classic_pcap(pcap_path)
    alignment = _csv_alignment(raw_csv, packets, refinement_request.minimum_ip_alignment_rows)
    if not alignment["packet_ordinal_alignment_accepted"]:
        raise DataContractError("Recomputed Edge MITM packet alignment failed")
    source_rows = _canonical_source_rows(mitm_shard, mitm_source)
    source_counts = _source_family_counts(phase3a_database)
    source_groups: dict[str, str] = {}
    groups: list[_Group] = []
    for source, (family, records) in source_counts.items():
        if source == mitm_source:
            continue
        group_id = _hash_text(_EDGE_POLICY, f"source:{source}")
        source_groups[source] = group_id
        groups.append(_Group(group_id, family, records))
    mitm_groups: dict[int, str] = {}
    flow_counts: Counter[str] = Counter()
    for source_row in source_rows:
        flow_key = packets[source_row - 1].flow_key
        group_id = _hash_text(_EDGE_POLICY, f"mitm-flow:{flow_key}")
        mitm_groups[source_row] = group_id
        flow_counts[group_id] += 1
    groups.extend(
        _Group(group_id, "man_in_the_middle", records) for group_id, records in flow_counts.items()
    )
    by_family: defaultdict[str, list[_Group]] = defaultdict(list)
    for group in groups:
        by_family[group.family].append(group)
    assignments: dict[str, str] = {}
    for family_groups in by_family.values():
        assignments.update(
            _assign_grouped_family(family_groups, request.split_basis_points, split_seed)
        )
    realized: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        realized[group.family][assignments[group.group_id]] += group.records
    evidence = {
        "phase3a1_analysis_id": phase3a1["analysis_id"],
        "mitm_source_file": mitm_source,
        "mitm_canonical_records": len(source_rows),
        "mitm_flow_groups": len(flow_counts),
        "group_assignment_sha256": hashlib.sha256(
            canonical_json(dict(sorted(assignments.items()))).encode()
        ).hexdigest(),
        "recomputed_alignment": alignment,
        "family_split_counts": {
            family: {split: counts[split] for split in _SPLITS}
            for family, counts in sorted(realized.items())
        },
    }
    return groups, source_groups, mitm_groups, assignments, evidence


def _plan(request: SplitBuildRequest) -> _Plan:
    manifest = _load_json(request.canonical_manifest, "canonical manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("dataset") != request.dataset.value:
        raise DataContractError("Canonical manifest dataset mismatch")
    corpus_id = manifest.get("corpus_id")
    shards = manifest.get("shards")
    family_counts = manifest.get("family_label_counts")
    if not isinstance(corpus_id, str) or not isinstance(shards, list) or not shards:
        raise DataContractError("Canonical manifest inventory is incomplete")
    if not isinstance(family_counts, dict) or not family_counts:
        raise DataContractError("Canonical manifest family counts are missing")
    split_seed = derive_seed(request.base_seed, f"phase3b-split:{request.dataset.value}")
    anchor_seed = derive_seed(request.base_seed, f"phase3b-anchor:{request.dataset.value}")
    edge_source: dict[str, str] = {}
    edge_mitm: dict[int, str] = {}
    edge_assignments: dict[str, str] = {}
    edge_mitm_source: str | None = None
    policy = _CIC_POLICY
    realized: dict[str, object] = {
        "mode": "deterministic_hash_threshold_expected_counts",
        "expected_family_split_counts": {
            str(family): {
                split: round(cast(int, records) * request.split_basis_points[split] / 10_000)
                for split in _SPLITS
            }
            for family, records in sorted(family_counts.items())
        },
    }
    group_count: int | None = None
    if request.dataset is DatasetName.EDGE_IIOTSET:
        policy = _EDGE_POLICY
        groups, edge_source, edge_mitm, edge_assignments, realized = _edge_plan(
            request, manifest, split_seed
        )
        edge_mitm_source = cast(str, realized["mitm_source_file"])
        group_count = len(groups)
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "canonical_corpus_id": corpus_id,
        "group_policy": policy,
        "group_count": group_count,
        "base_seed": request.base_seed,
        "derived_seeds": {"split": split_seed, "anchor": anchor_seed},
        "target_basis_points": request.split_basis_points,
        "anchors_per_primary_class": {
            "calibration": request.calibration_anchors_per_class,
            "scoring": request.scoring_anchors_per_class,
        },
        "primary_family_counts": dict(sorted(family_counts.items())),
        "realized_or_expected": realized,
        "secondary_fine_label_status": "retained_but_not_capture_disjoint_confirmatory",
        "claim_boundary": (
            "CIC is a deduplicated record-level split over provider-shuffled feature windows; "
            "it does not estimate capture-session generalization."
            if request.dataset is DatasetName.CIC_IOT_2023
            else (
                "Edge source captures are disjoint except MITM, whose groups are "
                "within-capture bidirectional-flow surrogates rather than independent sessions."
            )
        ),
        "split_assignment_created": False,
    }
    report: dict[str, object] = {"plan_id": run_id(stable, length=64), **stable}
    return _Plan(report, edge_source, edge_mitm, edge_assignments, edge_mitm_source)


def _write_plan(report: Mapping[str, object], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{report['dataset']}_phase3b_plan.json"
    md_path = output_directory / f"{report['dataset']}_phase3b_plan.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Phase 3B split plan: {report['dataset']}",
        "",
        f"- Plan ID: `{report['plan_id']}`",
        f"- Canonical corpus ID: `{report['canonical_corpus_id']}`",
        f"- Group policy: `{report['group_policy']}`",
        f"- Root seed: {report['base_seed']}",
        "- Target split: 70% train, 15% validation, 15% test",
        "- Split assignment created: no",
        "",
        "## Scientific boundary",
        "",
        cast(str, report["claim_boundary"]),
        "",
        "A full run must materialize record-ID assignments and pass all overlap and anchor gates.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def plan_phase3b_split(request: SplitBuildRequest) -> dict[str, object]:
    """Create the immutable Phase 3B plan without scanning canonical records."""

    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    return plan.report


def _open_work_database(request: SplitBuildRequest, identity: str) -> sqlite3.Connection:
    request.work_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(request.work_database)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS completed_shards(
            shard_index INTEGER PRIMARY KEY, input_file TEXT NOT NULL,
            input_sha256 TEXT NOT NULL, input_rows INTEGER NOT NULL,
            output_file TEXT NOT NULL, output_size INTEGER NOT NULL,
            output_sha256 TEXT NOT NULL, report_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS anchor_candidates(
            family TEXT NOT NULL, score TEXT NOT NULL, record_id TEXT NOT NULL,
            canonical_fingerprint TEXT NOT NULL, source_file TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            PRIMARY KEY(family, score, record_id)
        ) WITHOUT ROWID;
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError("Phase 3B resume database identity mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _assignment_schema() -> object:
    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("canonical_fingerprint", pa.string(), nullable=False),
            pa.field("family_label", pa.string(), nullable=False),
            pa.field("fine_label", pa.string(), nullable=False),
            pa.field("fine_label_ambiguous", pa.bool_(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ]
    )


def _group_and_split(
    request: SplitBuildRequest,
    plan: _Plan,
    record_id: str,
    fingerprint: str,
    source_file: str,
    source_row: int,
) -> tuple[str, str]:
    if request.dataset is DatasetName.CIC_IOT_2023:
        group_id = _hash_text(_CIC_POLICY, f"fingerprint:{fingerprint}")
        split_seed = cast(dict[str, int], plan.report["derived_seeds"])["split"]
        return group_id, _record_split(record_id, split_seed, request.split_basis_points)
    if source_file in plan.edge_source_groups:
        group_id = plan.edge_source_groups[source_file]
    else:
        if source_file != plan.edge_mitm_source:
            raise DataContractError(f"Edge record has unknown source group: {source_file}")
        try:
            group_id = plan.edge_mitm_groups[source_row]
        except KeyError as error:
            raise DataContractError(
                f"Edge record has no frozen group: {source_file}:{source_row}"
            ) from error
    return group_id, plan.edge_group_splits[group_id]


def _retain_candidate(
    values: list[_AnchorCandidate], candidate: _AnchorCandidate, limit: int
) -> None:
    bisect.insort(values, candidate)
    if len(values) > limit:
        values.pop()


def _merge_anchor_candidates(
    connection: sqlite3.Connection,
    local: Mapping[str, Sequence[_AnchorCandidate]],
    limit: int,
) -> None:
    for family, candidates in local.items():
        existing = [
            cast(_AnchorCandidate, tuple(row))
            for row in connection.execute(
                "SELECT score, record_id, canonical_fingerprint, source_file, source_row "
                "FROM anchor_candidates WHERE family=? ORDER BY score, record_id",
                (family,),
            )
        ]
        merged = sorted([*existing, *candidates])[:limit]
        connection.execute("DELETE FROM anchor_candidates WHERE family=?", (family,))
        connection.executemany(
            "INSERT INTO anchor_candidates VALUES (?, ?, ?, ?, ?, ?)",
            ((family, *candidate) for candidate in merged),
        )


def _scan_shard(
    request: SplitBuildRequest,
    plan: _Plan,
    connection: sqlite3.Connection,
    shard_index: int,
    shard: Mapping[str, object],
) -> dict[str, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:
        raise DataContractError("Phase 3B requires the project ML dependencies") from error
    relative = shard.get("file")
    if not isinstance(relative, str):
        raise DataContractError("Canonical shard has no file path")
    base = request.canonical_manifest.parent.resolve()
    source_path = (base / relative).resolve()
    try:
        source_path.relative_to(base)
    except ValueError as error:
        raise DataContractError("Canonical shard escapes its artifact directory") from error
    if not source_path.is_file() or source_path.stat().st_size != shard.get("byte_size"):
        raise DataContractError(f"Canonical shard is missing or changed: {relative}")
    if _file_sha256(source_path) != shard.get("sha256"):
        raise DataContractError(f"Canonical shard checksum mismatch: {relative}")
    output_relative = f"assignments/{shard_index:04d}-{source_path.stem}-split.parquet"
    output_path = request.output_directory / output_relative
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    parquet = pq.ParquetFile(source_path)
    columns = [
        "record_id",
        "canonical_fingerprint",
        "family_label",
        "fine_label",
        "fine_label_ambiguous",
        "source_file",
        "source_row",
    ]
    if missing := sorted(set(columns) - set(parquet.schema_arrow.names)):
        raise DataContractError(f"Canonical shard lacks Phase 3B columns: {missing}")
    schema = _assignment_schema()
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    split_counts: Counter[str] = Counter()
    family_split: Counter[str] = Counter()
    fine_split: Counter[str] = Counter()
    group_split: dict[str, str] = {}
    candidates: defaultdict[str, list[_AnchorCandidate]] = defaultdict(list)
    rows_seen = 0
    anchor_seed = cast(dict[str, int], plan.report["derived_seeds"])["anchor"]
    try:
        for batch in parquet.iter_batches(batch_size=request.parquet_batch_size, columns=columns):
            source = batch.to_pydict()
            output_rows: list[dict[str, object]] = []
            for position in range(batch.num_rows):
                record_id = cast(str, source["record_id"][position])
                fingerprint = cast(str, source["canonical_fingerprint"][position])
                family = cast(str, source["family_label"][position])
                fine = cast(str, source["fine_label"][position])
                source_file = cast(str, source["source_file"][position])
                source_row = cast(int, source["source_row"][position])
                group_id, split = _group_and_split(
                    request, plan, record_id, fingerprint, source_file, source_row
                )
                previous = group_split.setdefault(group_id, split)
                if previous != split:
                    raise DataContractError("One group was assigned to multiple splits")
                output_rows.append(
                    {
                        "record_id": record_id,
                        "canonical_fingerprint": fingerprint,
                        "family_label": family,
                        "fine_label": fine,
                        "fine_label_ambiguous": bool(source["fine_label_ambiguous"][position]),
                        "source_file": source_file,
                        "source_row": source_row,
                        "group_id": group_id,
                        "split": split,
                    }
                )
                split_counts[split] += 1
                family_split[f"{family}\0{split}"] += 1
                fine_split[f"{fine}\0{split}"] += 1
                if split == "train":
                    score = _seeded_digest(anchor_seed, "anchor-rank", record_id).hex()
                    _retain_candidate(
                        candidates[family],
                        (score, record_id, fingerprint, source_file, source_row),
                        request.anchors_per_class,
                    )
                rows_seen += 1
            if output_rows:
                writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
    finally:
        writer.close()
    if rows_seen != shard.get("row_count"):
        partial.unlink(missing_ok=True)
        raise DataContractError(
            f"Phase 3B shard row mismatch: expected={shard.get('row_count')}, actual={rows_seen}"
        )
    partial.replace(output_path)
    report: dict[str, object] = {
        "row_count": rows_seen,
        "split_counts": dict(sorted(split_counts.items())),
        "family_split_counts": dict(sorted(family_split.items())),
        "fine_split_counts": dict(sorted(fine_split.items())),
        "groups": len(group_split),
    }
    output_hash = _file_sha256(output_path)
    with connection:
        _merge_anchor_candidates(connection, candidates, request.anchors_per_class)
        connection.execute(
            "INSERT INTO completed_shards VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                shard_index,
                relative,
                shard["sha256"],
                shard["row_count"],
                output_relative,
                output_path.stat().st_size,
                output_hash,
                canonical_json(report),
            ),
        )
    return report


def _completed_indices(connection: sqlite3.Connection, request: SplitBuildRequest) -> set[int]:
    completed: set[int] = set()
    for index, relative, size, digest in connection.execute(
        "SELECT shard_index, output_file, output_size, output_sha256 FROM completed_shards"
    ):
        path = request.output_directory / cast(str, relative)
        if not path.is_file() or path.stat().st_size != cast(int, size):
            raise DataContractError("Completed Phase 3B assignment shard is missing or changed")
        if _file_sha256(path) != cast(str, digest):
            raise DataContractError("Completed Phase 3B assignment checksum mismatch")
        completed.add(cast(int, index))
    return completed


def _aggregate_reports(
    connection: sqlite3.Connection,
) -> tuple[Counter[str], Counter[str], Counter[str], int]:
    split_counts: Counter[str] = Counter()
    family_split: Counter[str] = Counter()
    fine_split: Counter[str] = Counter()
    rows = 0
    for (payload,) in connection.execute(
        "SELECT report_json FROM completed_shards ORDER BY shard_index"
    ):
        report = cast(dict[str, object], json.loads(cast(str, payload)))
        rows += cast(int, report["row_count"])
        split_counts.update(cast(dict[str, int], report["split_counts"]))
        family_split.update(cast(dict[str, int], report["family_split_counts"]))
        fine_split.update(cast(dict[str, int], report["fine_split_counts"]))
    return split_counts, family_split, fine_split, rows


def _nested_counts(values: Mapping[str, int]) -> dict[str, dict[str, int]]:
    nested: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for key, count in values.items():
        label, split = key.split("\0", maxsplit=1)
        nested[label][split] = count
    return {
        label: {split: counts.get(split, 0) for split in _SPLITS}
        for label, counts in sorted(nested.items())
    }


def _write_anchors(
    request: SplitBuildRequest,
    plan: _Plan,
    connection: sqlite3.Connection,
    family_counts: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise DataContractError("Phase 3B requires the project ML dependencies") from error
    rows: list[dict[str, object]] = []
    for family in sorted(family_counts):
        if family_counts[family]["train"] < request.anchors_per_class:
            raise DataContractError(f"Training split lacks anchor capacity for {family}")
        candidates = connection.execute(
            "SELECT score, record_id, canonical_fingerprint, source_file, source_row "
            "FROM anchor_candidates WHERE family=? ORDER BY score, record_id",
            (family,),
        ).fetchall()
        if len(candidates) != request.anchors_per_class:
            raise DataContractError(f"Anchor candidate count mismatch for {family}")
        for index, candidate in enumerate(candidates):
            score, record_id, fingerprint, source_file, source_row = candidate
            subset = "calibration" if index < request.calibration_anchors_per_class else "scoring"
            rows.append(
                {
                    "record_id": record_id,
                    "canonical_fingerprint": fingerprint,
                    "family_label": family,
                    "subset": subset,
                    "source_file": source_file,
                    "source_row": source_row,
                    "selection_score": score,
                }
            )
    anchor_path = request.output_directory / "anchors.parquet"
    schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("canonical_fingerprint", pa.string(), nullable=False),
            pa.field("family_label", pa.string(), nullable=False),
            pa.field("subset", pa.string(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
            pa.field("selection_score", pa.string(), nullable=False),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), anchor_path, compression="zstd")
    ids = sorted(cast(str, row["record_id"]) for row in rows)
    id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    stable = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "canonical_corpus_id": plan.report["canonical_corpus_id"],
        "plan_id": plan.report["plan_id"],
        "anchor_seed": cast(dict[str, int], plan.report["derived_seeds"])["anchor"],
        "records": len(rows),
        "record_ids_sha256": id_hash,
        "subset_counts": dict(Counter(cast(str, row["subset"]) for row in rows)),
        "family_counts": dict(Counter(cast(str, row["family_label"]) for row in rows)),
        "file": {
            "path": anchor_path.name,
            "byte_size": anchor_path.stat().st_size,
            "sha256": _file_sha256(anchor_path),
        },
        "gates": {
            "calibration_scoring_disjoint": len(ids) == len(set(ids)),
            "all_anchors_selected_from_train": True,
            "exact_class_balance": True,
        },
    }
    manifest = {"anchor_id": run_id(stable, length=64), **stable}
    manifest_path = request.output_directory / "anchor_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, {cast(str, row["record_id"]): row for row in rows}


def _finalize(
    request: SplitBuildRequest,
    plan: _Plan,
    manifest: Mapping[str, object],
    connection: sqlite3.Connection,
) -> dict[str, object]:
    split_counts, family_flat, fine_flat, records = _aggregate_reports(connection)
    expected = sum(
        cast(int, shard["row_count"]) for shard in cast(list[dict[str, object]], manifest["shards"])
    )
    if records != expected:
        raise DataContractError(
            f"Phase 3B coverage mismatch: expected={expected}, actual={records}"
        )
    family_counts = _nested_counts(family_flat)
    if any(counts[split] < 1 for counts in family_counts.values() for split in _SPLITS):
        raise DataContractError("A primary family is absent from a Phase 3B split")
    expected_families = {
        str(family): cast(int, count)
        for family, count in cast(dict[str, object], manifest["family_label_counts"]).items()
    }
    actual_families = {family: sum(counts.values()) for family, counts in family_counts.items()}
    if actual_families != expected_families:
        raise DataContractError("Phase 3B family counts do not match the canonical manifest")
    anchors, anchor_rows = _write_anchors(request, plan, connection, family_counts)
    files = [
        {
            "path": relative,
            "byte_size": size,
            "sha256": digest,
            "row_count": rows,
        }
        for relative, size, digest, rows in connection.execute(
            "SELECT output_file, output_size, output_sha256, input_rows "
            "FROM completed_shards ORDER BY shard_index"
        )
    ]
    stable = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "canonical_corpus_id": plan.report["canonical_corpus_id"],
        "plan_id": plan.report["plan_id"],
        "group_policy": plan.report["group_policy"],
        "base_seed": request.base_seed,
        "derived_seeds": plan.report["derived_seeds"],
        "target_basis_points": request.split_basis_points,
        "records": records,
        "split_counts": {split: split_counts[split] for split in _SPLITS},
        "realized_split_fractions": {split: split_counts[split] / records for split in _SPLITS},
        "family_split_counts": family_counts,
        "fine_split_counts": _nested_counts(fine_flat),
        "assignment_files": files,
        "anchor_manifest": {
            "anchor_id": anchors["anchor_id"],
            "path": "anchor_manifest.json",
            "records": anchors["records"],
        },
        "federated_train_records_after_anchor_removal": split_counts["train"] - len(anchor_rows),
        "gates": {
            "assignment_cardinality_matches_canonical": records == expected,
            "primary_family_cardinality_matches_canonical": True,
            "every_primary_family_in_every_split": True,
            "group_disjoint_by_construction": True,
            "fingerprint_disjoint_by_canonical_deduplication": True,
            "anchors_disjoint_and_train_only": all(
                cast(dict[str, bool], anchors["gates"]).values()
            ),
            "preprocessing_fitted": False,
            "client_partition_created": False,
        },
        "claim_boundary": plan.report["claim_boundary"],
    }
    report = {
        "split_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        **stable,
    }
    json_path = request.output_directory / f"{request.dataset.value}_phase3b_split.json"
    md_path = request.output_directory / f"{request.dataset.value}_phase3b_split.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Phase 3B split manifest: {request.dataset.value}",
        "",
        f"- Split ID: `{report['split_id']}`",
        f"- Canonical corpus ID: `{report['canonical_corpus_id']}`",
        f"- Plan ID: `{report['plan_id']}`",
        f"- Records assigned: {records:,}",
        f"- Train / validation / test: {split_counts['train']:,} / "
        f"{split_counts['validation']:,} / {split_counts['test']:,}",
        f"- Anchor records: {anchors['records']:,}",
        "- Preprocessing fitted: no",
        "- Client partitions created: no",
        "",
        "## Scientific boundary",
        "",
        cast(str, report["claim_boundary"]),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_phase3b_split(
    request: SplitBuildRequest, *, progress_every: int = 1
) -> dict[str, object]:
    """Materialize resumable record assignments and disjoint anchor manifests."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    manifest = _load_json(request.canonical_manifest, "canonical manifest")
    shards = cast(list[dict[str, object]], manifest["shards"])
    identity = canonical_json(
        {
            "schema_version": _SCHEMA_VERSION,
            "plan_id": plan.report["plan_id"],
            "shards": [
                {
                    "file": shard["file"],
                    "sha256": shard["sha256"],
                    "row_count": shard["row_count"],
                    "byte_size": shard["byte_size"],
                }
                for shard in shards
            ],
        }
    )
    connection = _open_work_database(request, identity)
    try:
        completed = _completed_indices(connection, request)
        for index, shard in enumerate(shards):
            if index in completed:
                continue
            _scan_shard(request, plan, connection, index, shard)
            if (index + 1) % progress_every == 0 or index + 1 == len(shards):
                print(f"phase3b_scan={index + 1}/{len(shards)} file={shard['file']}", flush=True)
        return _finalize(request, plan, manifest, connection)
    finally:
        connection.close()
