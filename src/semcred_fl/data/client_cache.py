"""Resumable schedule-bound Phase 3D client cache for real-data FL runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    ClientPartitionProfile,
    PreprocessedCorpus,
    TrainingBatch,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.seeding import derive_seed

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ClientCacheBuildRequest:
    """Frozen source artifacts and schedule scope for one client cache."""

    preprocessing_report: Path
    client_partition_manifest: Path
    profile: str
    selected_client_indices: tuple[int, ...]
    schedule_id: str
    output_directory: Path
    work_database: Path
    batch_size: int = 100_000

    def __post_init__(self) -> None:
        if (
            not self.profile
            or len(self.schedule_id) != 64
            or not self.selected_client_indices
            or tuple(sorted(set(self.selected_client_indices))) != self.selected_client_indices
            or self.batch_size < 1
        ):
            raise ValueError("Client-cache request is invalid")


@dataclass(frozen=True, slots=True)
class ClientCacheFile:
    """One checksum-bound cache shard with client-grouped row groups."""

    path: Path
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SelectedClientCache:
    """Validated cache metadata used by local client training."""

    manifest_path: Path
    cache_id: str
    dataset: str
    preprocessed_corpus_id: str
    client_partition_id: str
    profile: str
    schedule_id: str
    input_dimension: int
    num_classes: int
    selected_client_indices: tuple[int, ...]
    client_totals: tuple[int, ...]
    family_client_counts: tuple[tuple[int, ...], ...]
    files: tuple[ClientCacheFile, ...]

    def client_total(self, client_index: int) -> int:
        """Return the declared cache cardinality of one selected client."""

        try:
            position = self.selected_client_indices.index(client_index)
        except ValueError as error:
            raise ValueError("Client is outside this schedule-bound cache") from error
        return self.client_totals[position]

    def class_counts(self, client_index: int) -> tuple[int, ...]:
        """Return one cached client's exact class counts in class-ID order."""

        try:
            position = self.selected_client_indices.index(client_index)
        except ValueError as error:
            raise ValueError("Client is outside this schedule-bound cache") from error
        return tuple(row[position] for row in self.family_client_counts)


@dataclass(frozen=True, slots=True)
class ClientCacheBatch:
    """One cached local batch retaining record IDs for later attack selection."""

    record_ids: tuple[str, ...]
    features: Tensor
    targets: Tensor

    def training_batch(self) -> TrainingBatch:
        """Project to the Phase 5 local-training batch contract."""

        return TrainingBatch(self.features, self.targets)


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DataContractError("Client-cache construction requires NumPy and PyArrow") from error


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _plan(
    request: ClientCacheBuildRequest,
) -> tuple[PreprocessedCorpus, ClientPartitionProfile, str]:
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(
        request.client_partition_manifest, corpus, request.profile
    )
    if any(index < 0 or index >= profile.num_clients for index in request.selected_client_indices):
        raise DataContractError("Client-cache schedule contains an out-of-range client")
    stable = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "phase_6b_schedule_bound_client_cache",
        "dataset": corpus.dataset,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "split_id": corpus.split_id,
        "anchor_id": corpus.anchor_id,
        "client_partition_id": profile.client_partition_id,
        "profile": profile.profile,
        "assignment_column": profile.assignment_column,
        "schedule_id": request.schedule_id,
        "selected_client_indices": list(request.selected_client_indices),
        "transformed_files": [item.sha256 for item in corpus.files],
        "assignment_files": [item.sha256 for item in profile.files],
        "cache_format": "parquet_client_grouped_row_groups",
    }
    return corpus, profile, run_id(stable, length=64)


def _open_database(path: Path, identity: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cache_shards(
            shard_index INTEGER PRIMARY KEY,
            output_file TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError("Client-cache resume database identity mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _completed_prefix(
    connection: sqlite3.Connection, output_directory: Path
) -> tuple[dict[str, object], ...]:
    rows = connection.execute(
        "SELECT shard_index, output_file, byte_size, sha256, row_count, report_json "
        "FROM cache_shards ORDER BY shard_index"
    ).fetchall()
    if [row[0] for row in rows] != list(range(len(rows))):
        raise DataContractError("Client-cache resume state is not a contiguous shard prefix")
    reports: list[dict[str, object]] = []
    for _, relative, size, digest, count, payload in rows:
        path = output_directory / cast(str, relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(size)
            or _sha256(path) != cast(str, digest)
        ):
            raise DataContractError("Completed client-cache shard is missing or changed")
        report = cast(dict[str, object], json.loads(cast(str, payload)))
        if report.get("row_count") != int(count):
            raise DataContractError("Client-cache resume report is inconsistent")
        reports.append(report)
    return tuple(reports)


def _assignment_filter(column: str, selected: tuple[int, ...]) -> list[tuple[str, str, object]]:
    if len(selected) == 1:
        return [(column, "=", selected[0])]
    return [(column, "in", list(selected))]


def _empty_counts(selected: tuple[int, ...], num_classes: int) -> dict[int, list[int]]:
    return {client: [0] * num_classes for client in selected}


def _build_shard(
    *,
    request: ClientCacheBuildRequest,
    corpus: PreprocessedCorpus,
    profile: ClientPartitionProfile,
    connection: sqlite3.Connection,
    shard_index: int,
    np: Any,
    pa: Any,
    ds: Any,
    pq: Any,
) -> dict[str, object]:
    corpus_file = corpus.files[shard_index]
    assignment_file = profile.files[shard_index]
    if _sha256(corpus_file.path) != corpus_file.sha256:
        raise DataContractError("Client-cache transformed input checksum mismatch")
    assignments = pq.read_table(
        assignment_file.path,
        columns=["record_id", "family_id", profile.assignment_column],
        filters=_assignment_filter(profile.assignment_column, request.selected_client_indices),
    )
    assignment_values = assignments.to_pydict()
    selected: dict[str, tuple[int, int]] = {}
    for record_id, family_id, client_id in zip(
        assignment_values["record_id"],
        assignment_values["family_id"],
        assignment_values[profile.assignment_column],
        strict=True,
    ):
        key = str(record_id)
        if key in selected:
            raise DataContractError("Client-cache assignment contains a duplicate record ID")
        selected[key] = (int(family_id), int(client_id))
    per_client_tables: dict[int, list[Any]] = {
        client: [] for client in request.selected_client_indices
    }
    counts = _empty_counts(request.selected_client_indices, corpus.num_classes)
    transformed_columns = [
        "record_id",
        "partition",
        "family_id",
        *corpus.feature_columns,
    ]
    transformed_dataset = ds.dataset(str(corpus_file.path), format="parquet")
    scanner = transformed_dataset.scanner(
        columns=transformed_columns,
        filter=ds.field("partition") == "train",
        batch_size=request.batch_size,
    )
    matched = 0
    for batch in scanner.to_batches():
        values = batch.to_pydict()
        positions = [
            index
            for index, record_id in enumerate(values["record_id"])
            if str(record_id) in selected
        ]
        if not positions:
            continue
        features = np.column_stack(
            [
                np.asarray(values[column], dtype=np.float32)[positions]
                for column in corpus.feature_columns
            ]
        )
        if not np.isfinite(features).all():
            raise DataContractError("Non-finite feature reached the client cache")
        targets = np.asarray(values["family_id"], dtype=np.int64)[positions]
        partitions = np.asarray(values["partition"], dtype=object)[positions]
        record_ids = [str(values["record_id"][position]) for position in positions]
        client_ids = np.empty(len(positions), dtype=np.int16)
        for local_index, (_position, record_id, target) in enumerate(
            zip(positions, record_ids, targets, strict=True)
        ):
            expected_family, client_id = selected[record_id]
            if partitions[local_index] != "train" or expected_family != int(target):
                raise DataContractError("Client cache crossed a partition or family boundary")
            client_ids[local_index] = client_id
        for client_id in request.selected_client_indices:
            local = np.flatnonzero(client_ids == client_id)
            if local.size == 0:
                continue
            local_targets = targets[local]
            arrays: list[Any] = [
                pa.array([record_ids[int(index)] for index in local], type=pa.string()),
                pa.array(client_ids[local], type=pa.int16()),
                pa.array(local_targets, type=pa.int16()),
            ]
            arrays.extend(
                pa.array(features[local, column], type=pa.float32())
                for column in range(corpus.input_dimension)
            )
            table = pa.Table.from_arrays(
                arrays,
                names=["record_id", "client_id", "family_id", *corpus.feature_columns],
            )
            per_client_tables[client_id].append(table)
            bincount = np.bincount(local_targets, minlength=corpus.num_classes)
            for class_id, count in enumerate(bincount):
                counts[client_id][class_id] += int(count)
            matched += int(local.size)
    if matched != len(selected):
        raise DataContractError("Client cache did not match every selected assignment record")
    relative = f"shards/{shard_index:04d}-clients.parquet"
    output_path = request.output_directory / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".parquet.partial")
    schema = pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("client_id", pa.int16()),
            pa.field("family_id", pa.int16()),
            *[pa.field(coordinate, pa.float32()) for coordinate in corpus.feature_columns],
        ]
    )
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    try:
        for client_id in request.selected_client_indices:
            tables = per_client_tables[client_id]
            if not tables:
                continue
            client_table = pa.concat_tables(tables)
            writer.write_table(client_table, row_group_size=client_table.num_rows)
    finally:
        writer.close()
    partial.replace(output_path)
    digest = _sha256(output_path)
    report: dict[str, object] = {
        "row_count": matched,
        "client_family_counts": {str(key): value for key, value in counts.items()},
    }
    with connection:
        connection.execute(
            "INSERT INTO cache_shards VALUES (?, ?, ?, ?, ?, ?)",
            (
                shard_index,
                relative,
                output_path.stat().st_size,
                digest,
                matched,
                canonical_json(report),
            ),
        )
    return report


def _add_counts(target: dict[int, list[int]], source: Mapping[str, object]) -> None:
    for client_text, raw_counts in source.items():
        if not isinstance(raw_counts, list):
            raise DataContractError("Client-cache shard counts are malformed")
        client = int(client_text)
        if client not in target or len(raw_counts) != len(target[client]):
            raise DataContractError("Client-cache shard class system is malformed")
        for class_id, count in enumerate(raw_counts):
            if not isinstance(count, int) or count < 0:
                raise DataContractError("Client-cache shard count is invalid")
            target[client][class_id] += count


def _finalize(
    request: ClientCacheBuildRequest,
    corpus: PreprocessedCorpus,
    profile: ClientPartitionProfile,
    cache_input_id: str,
    connection: sqlite3.Connection,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    counts = _empty_counts(request.selected_client_indices, corpus.num_classes)
    total = 0
    rows = connection.execute(
        "SELECT output_file, byte_size, sha256, row_count, report_json "
        "FROM cache_shards ORDER BY shard_index"
    ).fetchall()
    if len(rows) != len(corpus.files):
        raise DataContractError("Client cache cannot finalize with missing source shards")
    for relative, size, digest, row_count, payload in rows:
        report = cast(dict[str, object], json.loads(cast(str, payload)))
        raw_counts = report.get("client_family_counts")
        if not isinstance(raw_counts, dict):
            raise DataContractError("Client-cache shard report is incomplete")
        _add_counts(counts, cast(dict[str, object], raw_counts))
        total += int(row_count)
        files.append(
            {
                "path": relative,
                "byte_size": int(size),
                "sha256": digest,
                "row_count": int(row_count),
            }
        )
    expected_counts = {
        client: list(profile.class_counts(client)) for client in request.selected_client_indices
    }
    if counts != expected_counts:
        raise DataContractError("Client-cache family counts differ from Phase 3D")
    selected_totals = [sum(counts[client]) for client in request.selected_client_indices]
    if total != sum(selected_totals):
        raise DataContractError("Client-cache total does not conserve selected clients")
    family_matrix = [
        [counts[client][class_id] for client in request.selected_client_indices]
        for class_id in range(corpus.num_classes)
    ]
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "phase_6b_schedule_bound_client_cache",
        "cache_input_id": cache_input_id,
        "dataset": corpus.dataset,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "split_id": corpus.split_id,
        "anchor_id": corpus.anchor_id,
        "client_partition_id": profile.client_partition_id,
        "profile": profile.profile,
        "schedule_id": request.schedule_id,
        "input_dimension": corpus.input_dimension,
        "num_classes": corpus.num_classes,
        "selected_client_indices": list(request.selected_client_indices),
        "client_totals": selected_totals,
        "family_client_counts": family_matrix,
        "records": total,
        "files": files,
        "gates": {
            "only_selected_schedule_clients_cached": True,
            "record_ids_unique_by_phase3_contract": True,
            "family_counts_match_phase3d": True,
            "validation_test_and_anchors_excluded": True,
            "features_finite": True,
            "model_training_performed": False,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    result: dict[str, object] = {
        "cache_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(UTC).isoformat(),
        **stable,
    }
    _write_json(request.output_directory / "client_cache.json", result)
    return result


def build_selected_client_cache(request: ClientCacheBuildRequest) -> dict[str, object]:
    """Build or resume one schedule-bound client-grouped Parquet cache."""

    corpus, profile, cache_input_id = _plan(request)
    final_path = request.output_directory / "client_cache.json"
    if final_path.exists():
        completed = _load_json(final_path, "completed client-cache manifest")
        if completed.get("cache_input_id") != cache_input_id:
            raise DataContractError("Completed client cache belongs to other inputs")
        return completed
    np = _dependency("numpy")
    pa = _dependency("pyarrow")
    ds = _dependency("pyarrow.dataset")
    pq = _dependency("pyarrow.parquet")
    connection = _open_database(request.work_database, cache_input_id)
    try:
        completed_shards = _completed_prefix(connection, request.output_directory)
        for shard_index in range(len(completed_shards), len(corpus.files)):
            _build_shard(
                request=request,
                corpus=corpus,
                profile=profile,
                connection=connection,
                shard_index=shard_index,
                np=np,
                pa=pa,
                ds=ds,
                pq=pq,
            )
            print(f"phase6b_cache={shard_index + 1}/{len(corpus.files)}", flush=True)
        return _finalize(request, corpus, profile, cache_input_id, connection)
    finally:
        connection.close()


def load_selected_client_cache(manifest_path: Path) -> SelectedClientCache:
    """Load and validate one completed schedule-bound client cache."""

    manifest = _load_json(manifest_path, "selected-client cache manifest")
    stable_manifest = {
        key: value for key, value in manifest.items() if key not in {"cache_id", "created_at_utc"}
    }
    if manifest.get("cache_id") != run_id(stable_manifest, length=64):
        raise DataContractError("Selected-client cache identity is invalid")
    required_true = (
        "only_selected_schedule_clients_cached",
        "record_ids_unique_by_phase3_contract",
        "family_counts_match_phase3d",
        "validation_test_and_anchors_excluded",
        "features_finite",
    )
    gates = manifest.get("gates")
    if (
        not isinstance(gates, dict)
        or not all(gates.get(name) is True for name in required_true)
        or gates.get("model_training_performed") is not False
        or gates.get("test_partition_accessed") is not False
        or gates.get("performance_claims_allowed") is not False
    ):
        raise DataContractError("Selected-client cache gates did not pass")
    string_fields = (
        "cache_id",
        "dataset",
        "preprocessed_corpus_id",
        "client_partition_id",
        "profile",
        "schedule_id",
    )
    if any(not isinstance(manifest.get(name), str) for name in string_fields):
        raise DataContractError("Selected-client cache identities are incomplete")
    input_dimension = manifest.get("input_dimension")
    num_classes = manifest.get("num_classes")
    clients = manifest.get("selected_client_indices")
    totals = manifest.get("client_totals")
    matrix = manifest.get("family_client_counts")
    if (
        not isinstance(input_dimension, int)
        or input_dimension < 1
        or not isinstance(num_classes, int)
        or num_classes < 2
        or not isinstance(clients, list)
        or not clients
        or not all(isinstance(value, int) for value in clients)
        or not isinstance(totals, list)
        or len(totals) != len(clients)
        or not all(isinstance(value, int) and value > 0 for value in totals)
        or not isinstance(matrix, list)
        or len(matrix) != num_classes
    ):
        raise DataContractError("Selected-client cache dimensions are malformed")
    typed_matrix: list[tuple[int, ...]] = []
    for row in matrix:
        if (
            not isinstance(row, list)
            or len(row) != len(clients)
            or not all(isinstance(value, int) and value >= 0 for value in row)
        ):
            raise DataContractError("Selected-client cache class matrix is malformed")
        typed_matrix.append(tuple(cast(list[int], row)))
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise DataContractError("Selected-client cache file inventory is missing")
    files: list[ClientCacheFile] = []
    for item in inventory:
        if not isinstance(item, dict):
            raise DataContractError("Selected-client cache file entry is malformed")
        relative = item.get("path")
        row_count = item.get("row_count")
        size = item.get("byte_size")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(row_count, int)
            or row_count < 0
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
        ):
            raise DataContractError("Selected-client cache file entry is invalid")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
            raise DataContractError("Selected-client cache file is missing or changed")
        files.append(ClientCacheFile(path, row_count, size, digest))
    return SelectedClientCache(
        manifest_path=manifest_path.resolve(),
        cache_id=cast(str, manifest["cache_id"]),
        dataset=cast(str, manifest["dataset"]),
        preprocessed_corpus_id=cast(str, manifest["preprocessed_corpus_id"]),
        client_partition_id=cast(str, manifest["client_partition_id"]),
        profile=cast(str, manifest["profile"]),
        schedule_id=cast(str, manifest["schedule_id"]),
        input_dimension=input_dimension,
        num_classes=num_classes,
        selected_client_indices=tuple(cast(list[int], clients)),
        client_totals=tuple(cast(list[int], totals)),
        family_client_counts=tuple(typed_matrix),
        files=tuple(files),
    )


def iter_cached_client_batches(
    cache: SelectedClientCache,
    client_index: int,
    *,
    batch_size: int,
    shuffle_seed: int,
) -> Iterator[ClientCacheBatch]:
    """Stream one cached client's complete population with deterministic batch shuffles."""

    if batch_size < 1 or shuffle_seed < 0:
        raise ValueError("Cached client batch controls are invalid")
    expected_total = cache.client_total(client_index)
    expected_counts = cache.class_counts(client_index)
    np = _dependency("numpy")
    pq = _dependency("pyarrow.parquet")
    emitted = 0
    counts = [0] * cache.num_classes
    columns = [
        "record_id",
        "family_id",
        *(f"x{index:04d}" for index in range(cache.input_dimension)),
    ]
    for file_index, file in enumerate(cache.files):
        table = pq.read_table(
            file.path,
            columns=columns,
            filters=[("client_id", "=", client_index)],
        )
        for batch_index, batch in enumerate(table.to_batches(max_chunksize=batch_size)):
            record_ids = tuple(str(value) for value in batch.column(0).to_pylist())
            targets = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            features = np.column_stack(
                [
                    batch.column(column).to_numpy(zero_copy_only=False)
                    for column in range(2, len(columns))
                ]
            ).astype(np.float32, copy=False)
            if len(targets) > 1:
                seed = derive_seed(
                    shuffle_seed,
                    f"phase6b-cache:{cache.profile}:{client_index}:file={file_index}:batch={batch_index}",
                )
                order = np.random.default_rng(seed).permutation(len(targets))
                record_ids = tuple(record_ids[int(index)] for index in order)
                targets = targets[order]
                features = features[order]
            feature_tensor = torch.from_numpy(np.ascontiguousarray(features))
            target_tensor = torch.from_numpy(np.ascontiguousarray(targets))
            if not torch.isfinite(feature_tensor).all().item():
                raise DataContractError("Non-finite feature reached cached client training")
            bincount = np.bincount(targets, minlength=cache.num_classes)
            for class_id, count in enumerate(bincount):
                counts[class_id] += int(count)
            emitted += len(targets)
            yield ClientCacheBatch(record_ids, feature_tensor, target_tensor)
    if emitted != expected_total or tuple(counts) != expected_counts:
        raise DataContractError("Cached client stream differs from its immutable manifest")
