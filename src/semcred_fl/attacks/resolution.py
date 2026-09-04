"""Resumable train-only Phase 6B feature and trigger resolution."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from semcred_fl.attacks.artifacts import (
    Phase6BResolution,
    ResolvedCoordinate,
    TriggerCoordinate,
)
from semcred_fl.attacks.contracts import DatasetAttackProfile, phase6a_contract
from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    CorpusFile,
    load_preprocessed_corpus,
    load_training_class_counts,
)
from semcred_fl.identity import canonical_json, run_id, to_primitive

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Phase6BResolutionRequest:
    """Inputs and bounded-memory controls for one dataset resolution."""

    dataset: DatasetName
    preprocessing_report: Path
    client_partition_manifest: Path
    output_directory: Path
    work_database: Path
    batch_size: int = 100_000

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("Phase 6B batch_size must be positive")


@dataclass(frozen=True, slots=True)
class _Candidate:
    coordinate: str
    index: int
    source_feature: str
    lower: float
    upper: float


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DataContractError("Phase 6B resolution requires NumPy and PyArrow") from error


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(dataset: DatasetName) -> DatasetAttackProfile:
    for profile in phase6a_contract().profiles:
        if profile.dataset is dataset:
            return profile
    raise DataContractError("Phase 6A has no profile for the selected dataset")


def _preprocessor_path(report_path: Path, report: Mapping[str, object]) -> Path:
    reference = report.get("preprocessor")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise DataContractError("Phase 3C preprocessor reference is missing")
    path = (report_path.parent / cast(str, reference["path"])).resolve()
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise DataContractError("Phase 3C preprocessor is missing or checksum-invalid")
    return path


def _candidates(preprocessor: Mapping[str, object], np: Any) -> tuple[_Candidate, ...]:
    coordinates = preprocessor.get("coordinates")
    continuous = preprocessor.get("continuous_features")
    if not isinstance(coordinates, list) or not isinstance(continuous, dict):
        raise DataContractError("Phase 3C coordinate metadata is incomplete")
    result: list[_Candidate] = []
    for index, raw in enumerate(coordinates):
        if not isinstance(raw, dict) or raw.get("kind") != "continuous_standardized":
            continue
        coordinate = raw.get("coordinate")
        source_feature = raw.get("source_feature")
        statistics = continuous.get(source_feature) if isinstance(source_feature, str) else None
        if (
            not isinstance(coordinate, str)
            or coordinate != f"x{index:04d}"
            or not isinstance(source_feature, str)
            or not isinstance(statistics, dict)
        ):
            raise DataContractError("Phase 3C continuous coordinate metadata is malformed")
        lower = statistics.get("clip_lower")
        upper = statistics.get("clip_upper")
        mean = statistics.get("mean_after_clipping")
        scale = statistics.get("scale")
        if not all(isinstance(value, int | float) for value in (lower, upper, mean, scale)):
            raise DataContractError("Phase 3C continuous statistics are malformed")
        lower_value = float(cast(float | int, lower))
        upper_value = float(cast(float | int, upper))
        mean_value = float(cast(float | int, mean))
        scale_value = float(cast(float | int, scale))
        if upper_value <= lower_value or scale_value <= 0.0:
            continue
        standardized_lower = float(np.float32((lower_value - mean_value) / scale_value))
        standardized_upper = float(np.float32((upper_value - mean_value) / scale_value))
        if standardized_upper <= standardized_lower:
            continue
        result.append(
            _Candidate(
                coordinate,
                index,
                source_feature,
                standardized_lower,
                standardized_upper,
            )
        )
    if len(result) < phase6a_contract().backdoor.candidate_pool_size:
        raise DataContractError("Too few nonconstant continuous coordinates for Phase 6B")
    return tuple(result)


def _open_database(path: Path, identity: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS median_shards(
            shard_index INTEGER PRIMARY KEY,
            source_rows INTEGER NOT NULL,
            target_rows INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trigger_shards(
            shard_index INTEGER PRIMARY KEY,
            training_rows INTEGER NOT NULL
        );
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError("Phase 6B resume database identity mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _completed_prefix(
    connection: sqlite3.Connection, table: str, count_columns: tuple[str, ...]
) -> tuple[int, ...]:
    selected = ", ".join(("shard_index", *count_columns))
    rows = connection.execute(f"SELECT {selected} FROM {table} ORDER BY shard_index").fetchall()
    if [row[0] for row in rows] != list(range(len(rows))):
        raise DataContractError(f"Phase 6B {table} resume state is not a contiguous prefix")
    return tuple(sum(int(value) for value in row[1:]) for row in rows)


def _open_memmap(path: Path, rows: int, columns: int, np: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = rows * columns * 4
    if path.exists() and path.stat().st_size != expected_size:
        raise DataContractError(f"Phase 6B work matrix has the wrong size: {path.name}")
    mode = "r+" if path.exists() else "w+"
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(rows, columns))


def _scanner(
    file: CorpusFile,
    columns: list[str],
    filter_expression: Any,
    batch_size: int,
    ds: Any,
) -> Any:
    if _sha256(file.path) != file.sha256:
        raise DataContractError(f"Phase 3C transformed checksum mismatch: {file.path.name}")
    dataset = ds.dataset(str(file.path), format="parquet")
    return dataset.scanner(columns=columns, filter=filter_expression, batch_size=batch_size)


def _fill_median_matrices(
    *,
    files: tuple[CorpusFile, ...],
    candidates: tuple[_Candidate, ...],
    profile: DatasetAttackProfile,
    source_matrix: Any,
    target_matrix: Any,
    connection: sqlite3.Connection,
    batch_size: int,
    np: Any,
    ds: Any,
) -> None:
    completed = _completed_prefix(connection, "median_shards", ("source_rows", "target_rows"))
    rows = connection.execute(
        "SELECT source_rows, target_rows FROM median_shards ORDER BY shard_index"
    ).fetchall()
    source_offset = sum(int(row[0]) for row in rows)
    target_offset = sum(int(row[1]) for row in rows)
    columns = ["family_id", *(item.coordinate for item in candidates)]
    family_filter = ds.field("family_id").isin([profile.source_family_id, profile.target_family_id])
    train_filter = (ds.field("partition") == "train") & family_filter
    for shard_index in range(len(completed), len(files)):
        source_rows = target_rows = 0
        scanner = _scanner(files[shard_index], columns, train_filter, batch_size, ds)
        for batch in scanner.to_batches():
            families = batch.column(0).to_numpy(zero_copy_only=False)
            matrix = np.column_stack(
                [
                    batch.column(column_index).to_numpy(zero_copy_only=False)
                    for column_index in range(1, len(columns))
                ]
            ).astype(np.float32, copy=False)
            source_indices = np.flatnonzero(families == profile.source_family_id)
            target_indices = np.flatnonzero(families == profile.target_family_id)
            if source_indices.size:
                end = source_offset + source_rows + int(source_indices.size)
                source_matrix[source_offset + source_rows : end] = matrix[source_indices]
                source_rows += int(source_indices.size)
            if target_indices.size:
                end = target_offset + target_rows + int(target_indices.size)
                target_matrix[target_offset + target_rows : end] = matrix[target_indices]
                target_rows += int(target_indices.size)
        source_matrix.flush()
        target_matrix.flush()
        with connection:
            connection.execute(
                "INSERT INTO median_shards VALUES (?, ?, ?)",
                (shard_index, source_rows, target_rows),
            )
        source_offset += source_rows
        target_offset += target_rows
        print(f"phase6b_medians={shard_index + 1}/{len(files)}", flush=True)


def _rank_coordinates(
    candidates: tuple[_Candidate, ...], source_matrix: Any, target_matrix: Any, np: Any
) -> tuple[ResolvedCoordinate, ...]:
    resolved: list[ResolvedCoordinate] = []
    for column_index, candidate in enumerate(candidates):
        source_median = float(np.median(source_matrix[:, column_index]))
        target_median = float(np.median(target_matrix[:, column_index]))
        resolved.append(
            ResolvedCoordinate(
                coordinate=candidate.coordinate,
                index=candidate.index,
                source_feature=candidate.source_feature,
                lower_bound=candidate.lower,
                upper_bound=candidate.upper,
                source_median=source_median,
                benign_median=target_median,
                absolute_median_gap=abs(source_median - target_median),
            )
        )
    return tuple(sorted(resolved, key=lambda item: (-item.absolute_median_gap, item.coordinate)))


def _fill_trigger_matrix(
    *,
    files: tuple[CorpusFile, ...],
    pool: tuple[ResolvedCoordinate, ...],
    training_matrix: Any,
    connection: sqlite3.Connection,
    batch_size: int,
    np: Any,
    ds: Any,
) -> None:
    completed = _completed_prefix(connection, "trigger_shards", ("training_rows",))
    offset_row = connection.execute(
        "SELECT COALESCE(SUM(training_rows), 0) FROM trigger_shards"
    ).fetchone()
    offset = int(offset_row[0])
    columns = [item.coordinate for item in pool]
    train_filter = ds.field("partition") == "train"
    for shard_index in range(len(completed), len(files)):
        shard_rows = 0
        scanner = _scanner(files[shard_index], columns, train_filter, batch_size, ds)
        for batch in scanner.to_batches():
            matrix = np.column_stack(
                [
                    batch.column(column_index).to_numpy(zero_copy_only=False)
                    for column_index in range(len(columns))
                ]
            ).astype(np.float32, copy=False)
            end = offset + shard_rows + matrix.shape[0]
            training_matrix[offset + shard_rows : end] = matrix
            shard_rows += int(matrix.shape[0])
        training_matrix.flush()
        with connection:
            connection.execute(
                "INSERT INTO trigger_shards VALUES (?, ?)", (shard_index, shard_rows)
            )
        offset += shard_rows
        print(f"phase6b_trigger={shard_index + 1}/{len(files)}", flush=True)


def _zero_collision_trigger(
    pool: tuple[ResolvedCoordinate, ...], training_matrix: Any, np: Any
) -> tuple[TriggerCoordinate, ...]:
    column_by_name = {item.coordinate: index for index, item in enumerate(pool)}
    masks: dict[tuple[str, str], Any] = {}
    for item in pool:
        column = training_matrix[:, column_by_name[item.coordinate]]
        masks[(item.coordinate, "lower")] = np.packbits(
            column == np.float32(item.lower_bound), bitorder="little"
        )
        masks[(item.coordinate, "upper")] = np.packbits(
            column == np.float32(item.upper_bound), bitorder="little"
        )
    ordered = tuple(sorted(pool, key=lambda item: item.coordinate))
    policy = phase6a_contract().backdoor
    for size in range(policy.minimum_coordinates, policy.maximum_coordinates + 1):
        for coordinates in itertools.combinations(ordered, size):
            for endpoints in itertools.product(("lower", "upper"), repeat=size):
                combined = masks[(coordinates[0].coordinate, endpoints[0])].copy()
                for coordinate, endpoint in zip(coordinates[1:], endpoints[1:], strict=True):
                    np.bitwise_and(combined, masks[(coordinate.coordinate, endpoint)], out=combined)
                if not bool(np.any(combined)):
                    return tuple(
                        TriggerCoordinate(
                            coordinate=coordinate.coordinate,
                            index=coordinate.index,
                            source_feature=coordinate.source_feature,
                            endpoint=endpoint,
                            value=(
                                coordinate.lower_bound
                                if endpoint == "lower"
                                else coordinate.upper_bound
                            ),
                        )
                        for coordinate, endpoint in zip(coordinates, endpoints, strict=True)
                    )
    raise DataContractError("No zero-collision trigger exists within the frozen search space")


def _write_report(
    request: Phase6BResolutionRequest,
    resolution: Phase6BResolution,
    resolution_input_id: str,
) -> dict[str, object]:
    primitive = cast(dict[str, object], to_primitive(resolution))
    report: dict[str, object] = {
        "resolution_id": resolution.resolution_id,
        "resolution_input_id": resolution_input_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **primitive,
        "gates": {
            "source_and_benign_medians_exact_over_federated_train": True,
            "feature_coordinates_nonconstant_and_train_ranked": True,
            "trigger_from_top_twelve_coordinates": True,
            "trigger_has_zero_federated_training_collisions": True,
            "scanner_filter_excludes_validation_test_and_anchors": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    request.output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{request.dataset.value}_phase6b_resolution"
    json_path = request.output_directory / f"{stem}.json"
    partial = json_path.with_suffix(".json.partial")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(json_path)
    markdown = [
        f"# Phase 6B attack resolution: {request.dataset.value}",
        "",
        f"- Resolution ID: `{resolution.resolution_id}`",
        "- Clean-label coordinates: "
        + ", ".join(item.coordinate for item in resolution.feature_coordinates),
        "- Trigger: "
        + ", ".join(
            f"{item.coordinate}={item.endpoint}" for item in resolution.trigger_coordinates
        ),
        "- Trigger training collisions: 0",
        "- Resolution population: federated training after anchor removal",
        "- Validation, test, and anchor records accessed: no",
        "- Performance claims allowed: no",
        "",
    ]
    (request.output_directory / f"{stem}.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def build_phase6b_resolution(request: Phase6BResolutionRequest) -> dict[str, object]:
    """Resolve exact train-only medians and a zero-collision endpoint trigger."""

    final_path = request.output_directory / f"{request.dataset.value}_phase6b_resolution.json"
    np = _dependency("numpy")
    ds = _dependency("pyarrow.dataset")
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    if corpus.dataset != request.dataset.value:
        raise DataContractError("Phase 6B dataset differs from the Phase 3C corpus")
    counts = load_training_class_counts(request.client_partition_manifest, corpus)
    client_manifest = _load_json(
        request.client_partition_manifest, "Phase 3D client-partition manifest"
    )
    client_partition_id = client_manifest.get("client_partition_id")
    if not isinstance(client_partition_id, str):
        raise DataContractError("Phase 3D client-partition identity is missing")
    profile = _profile(request.dataset)
    if corpus.preprocessor_id != profile.preprocessor_id or corpus.split_id != profile.split_id:
        raise DataContractError("Phase 6A profile differs from the selected Phase 3 artifacts")
    if counts[profile.source_family_id] != profile.observed_federated_training_records:
        raise DataContractError("Phase 6A source-family count differs from Phase 3D")
    report = _load_json(request.preprocessing_report, "Phase 3C preprocessing report")
    preprocessor = _load_json(
        _preprocessor_path(request.preprocessing_report, report), "Phase 3C preprocessor"
    )
    candidates = _candidates(preprocessor, np)
    phase6a_id = run_id(phase6a_contract(), length=64)
    stable = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "6B_train_only_attack_resolution",
        "dataset": request.dataset.value,
        "phase6a_contract_id": phase6a_id,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "split_id": corpus.split_id,
        "anchor_id": corpus.anchor_id,
        "client_partition_id": client_partition_id,
        "candidate_coordinates": [to_primitive(item) for item in candidates],
        "transformed_files": [
            {"sha256": item.sha256, "row_count": item.row_count} for item in corpus.files
        ],
        "scanner_filter": "partition == train",
    }
    identity = hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()
    if final_path.exists():
        completed = _load_json(final_path, "completed Phase 6B resolution")
        if completed.get("resolution_input_id") != identity:
            raise DataContractError("Completed Phase 6B resolution belongs to other inputs")
        return completed
    connection = _open_database(request.work_database, identity)
    work = request.output_directory / "_work"
    source_count = counts[profile.source_family_id]
    target_count = counts[profile.target_family_id]
    source_matrix = _open_memmap(work / "source.float32", source_count, len(candidates), np)
    target_matrix = _open_memmap(work / "benign.float32", target_count, len(candidates), np)
    try:
        _fill_median_matrices(
            files=corpus.files,
            candidates=candidates,
            profile=profile,
            source_matrix=source_matrix,
            target_matrix=target_matrix,
            connection=connection,
            batch_size=request.batch_size,
            np=np,
            ds=ds,
        )
        scanned = connection.execute(
            "SELECT COALESCE(SUM(source_rows), 0), COALESCE(SUM(target_rows), 0) FROM median_shards"
        ).fetchone()
        if (int(scanned[0]), int(scanned[1])) != (source_count, target_count):
            raise DataContractError("Phase 6B source/benign scan cardinality mismatch")
        ranked = _rank_coordinates(candidates, source_matrix, target_matrix, np)
        pool = ranked[: phase6a_contract().backdoor.candidate_pool_size]
        training_matrix = _open_memmap(
            work / "trigger.float32", corpus.partition_counts["train"], len(pool), np
        )
        _fill_trigger_matrix(
            files=corpus.files,
            pool=pool,
            training_matrix=training_matrix,
            connection=connection,
            batch_size=request.batch_size,
            np=np,
            ds=ds,
        )
        trigger_count = connection.execute(
            "SELECT COALESCE(SUM(training_rows), 0) FROM trigger_shards"
        ).fetchone()
        if int(trigger_count[0]) != corpus.partition_counts["train"]:
            raise DataContractError("Phase 6B trigger scan cardinality mismatch")
        trigger = _zero_collision_trigger(pool, training_matrix, np)
        resolution = Phase6BResolution(
            dataset=request.dataset.value,
            phase6a_contract_id=phase6a_id,
            preprocessed_corpus_id=corpus.preprocessed_corpus_id,
            preprocessor_id=corpus.preprocessor_id,
            split_id=corpus.split_id,
            anchor_id=corpus.anchor_id,
            client_partition_id=client_partition_id,
            source_family=profile.source_family,
            source_family_id=profile.source_family_id,
            target_family=profile.target_family,
            target_family_id=profile.target_family_id,
            source_training_records=source_count,
            target_training_records=target_count,
            federated_training_records=corpus.partition_counts["train"],
            ranked_coordinates=ranked,
            feature_coordinates=ranked[: phase6a_contract().feature_poisoning.feature_count],
            trigger_coordinates=trigger,
            trigger_training_collisions=0,
        )
        return _write_report(request, resolution, identity)
    finally:
        connection.close()
