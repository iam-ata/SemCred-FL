"""Validated streaming access to Phase 3C transformed records for model training."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from semcred_fl.data.contracts import DataContractError
from semcred_fl.seeding import derive_seed

_ALLOWED_PARTITIONS = frozenset(
    {"train", "validation", "test", "anchor_calibration", "anchor_scoring"}
)


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DataContractError("Model training requires the project ML dependencies") from error


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


@dataclass(frozen=True, slots=True)
class CorpusFile:
    """One manifest-bound Phase 3C Parquet shard."""

    path: Path
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PreprocessedCorpus:
    """Validated metadata needed to stream a Phase 3C corpus."""

    report_path: Path
    dataset: str
    preprocessed_corpus_id: str
    preprocessor_id: str
    split_id: str
    anchor_id: str
    input_dimension: int
    family_label_to_id: dict[str, int]
    partition_counts: dict[str, int]
    files: tuple[CorpusFile, ...]

    @property
    def num_classes(self) -> int:
        """Return the size of the primary-family coordinate system."""

        return len(self.family_label_to_id)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the stable transformed feature coordinate names."""

        return tuple(f"x{index:04d}" for index in range(self.input_dimension))


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One feature/target tensor pair."""

    features: Tensor
    targets: Tensor

    @property
    def records(self) -> int:
        """Return the number of records in this batch."""

        return int(self.targets.shape[0])


@dataclass(frozen=True, slots=True)
class AssignmentFile:
    """One checksum-bound Phase 3D assignment shard."""

    path: Path
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ClientPartitionProfile:
    """Validated Phase 3D profile needed for local client streaming."""

    manifest_path: Path
    client_partition_id: str
    profile: str
    assignment_column: str
    num_clients: int
    client_totals: tuple[int, ...]
    family_client_counts: tuple[tuple[int, ...], ...]
    files: tuple[AssignmentFile, ...]

    def class_counts(self, client_index: int) -> tuple[int, ...]:
        """Return exact local family counts in model class-ID order."""

        if not 0 <= client_index < self.num_clients:
            raise ValueError("client_index is outside the declared population")
        return tuple(row[client_index] for row in self.family_client_counts)


def _validated_family_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or len(value) < 2:
        raise DataContractError("Primary-family map is missing or too small")
    result: dict[str, int] = {}
    for label, class_id in value.items():
        if not isinstance(label, str) or not isinstance(class_id, int):
            raise DataContractError("Primary-family map is malformed")
        result[label] = class_id
    if sorted(result.values()) != list(range(len(result))):
        raise DataContractError("Primary-family IDs must be contiguous from zero")
    return result


def load_preprocessed_corpus(report_path: Path) -> PreprocessedCorpus:
    """Load and validate one completed Phase 3C report and its inventory."""

    report = _load_json(report_path, "Phase 3C preprocessing report")
    required_strings = (
        "dataset",
        "preprocessed_corpus_id",
        "split_id",
        "anchor_id",
    )
    if any(not isinstance(report.get(name), str) for name in required_strings):
        raise DataContractError("Phase 3C report identities are incomplete")
    gates = report.get("gates")
    required_gates = (
        "fit_cardinality_matches_federated_training_pool",
        "fit_excludes_validation_test_and_anchors",
        "no_nonfinite_model_input",
        "partition_cardinality_matches_phase3b",
        "preprocessing_fitted",
        "stable_feature_order_and_dimension",
        "transformed_cardinality_matches_canonical",
    )
    if not isinstance(gates, dict) or not all(gates.get(name) is True for name in required_gates):
        raise DataContractError("Required Phase 3C gates did not pass")

    preprocessor_reference = report.get("preprocessor")
    if not isinstance(preprocessor_reference, dict):
        raise DataContractError("Phase 3C preprocessor reference is missing")
    reference_path = preprocessor_reference.get("path")
    reference_hash = preprocessor_reference.get("sha256")
    if not isinstance(reference_path, str) or not isinstance(reference_hash, str):
        raise DataContractError("Phase 3C preprocessor inventory is malformed")
    preprocessor_path = (report_path.parent / reference_path).resolve()
    if not preprocessor_path.is_file() or _sha256(preprocessor_path) != reference_hash:
        raise DataContractError("Phase 3C preprocessor file is missing or checksum-invalid")
    preprocessor = _load_json(preprocessor_path, "Phase 3C preprocessor")
    preprocessor_id = preprocessor.get("preprocessor_id")
    dimension = preprocessor.get("model_input_dimension")
    if (
        not isinstance(preprocessor_id, str)
        or preprocessor_reference.get("preprocessor_id") != preprocessor_id
        or not isinstance(dimension, int)
        or dimension < 1
    ):
        raise DataContractError("Phase 3C preprocessor identity or dimension is invalid")
    for identity in ("dataset", "split_id", "anchor_id", "preprocessed_corpus_id"):
        if identity == "preprocessed_corpus_id":
            continue
        if preprocessor.get(identity) != report.get(identity):
            raise DataContractError(f"Phase 3C preprocessor {identity} differs from its report")

    family_map = _validated_family_map(preprocessor.get("family_label_to_id"))
    counts_value = report.get("partition_counts")
    if not isinstance(counts_value, dict):
        raise DataContractError("Phase 3C partition counts are missing")
    partition_counts: dict[str, int] = {}
    for name in _ALLOWED_PARTITIONS:
        count = counts_value.get(name)
        if not isinstance(count, int) or count < 0:
            raise DataContractError("Phase 3C partition counts are malformed")
        partition_counts[name] = count

    inventory = report.get("transformed_files")
    if not isinstance(inventory, list) or not inventory:
        raise DataContractError("Phase 3C transformed inventory is missing")
    files: list[CorpusFile] = []
    total_rows = 0
    for item in inventory:
        if not isinstance(item, dict):
            raise DataContractError("Phase 3C transformed inventory entry is malformed")
        relative_path = item.get("path")
        row_count = item.get("row_count")
        byte_size = item.get("byte_size")
        digest = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not isinstance(row_count, int)
            or not isinstance(byte_size, int)
            or not isinstance(digest, str)
            or row_count < 0
            or byte_size < 1
        ):
            raise DataContractError("Phase 3C transformed inventory entry is invalid")
        path = (report_path.parent / relative_path).resolve()
        if not path.is_file() or path.stat().st_size != byte_size:
            raise DataContractError("Phase 3C transformed shard is missing or has the wrong size")
        files.append(CorpusFile(path, row_count, byte_size, digest))
        total_rows += row_count
    records = report.get("records")
    if not isinstance(records, int) or total_rows != records:
        raise DataContractError("Phase 3C transformed inventory cardinality is inconsistent")

    return PreprocessedCorpus(
        report_path=report_path.resolve(),
        dataset=cast(str, report["dataset"]),
        preprocessed_corpus_id=cast(str, report["preprocessed_corpus_id"]),
        preprocessor_id=preprocessor_id,
        split_id=cast(str, report["split_id"]),
        anchor_id=cast(str, report["anchor_id"]),
        input_dimension=dimension,
        family_label_to_id=family_map,
        partition_counts=partition_counts,
        files=tuple(files),
    )


def load_training_class_counts(
    client_partition_manifest: Path, corpus: PreprocessedCorpus
) -> tuple[int, ...]:
    """Load exact non-anchor training family counts from a completed Phase 3D manifest."""

    manifest = _load_json(client_partition_manifest, "Phase 3D client-partition manifest")
    if (
        manifest.get("dataset") != corpus.dataset
        or manifest.get("preprocessed_corpus_id") != corpus.preprocessed_corpus_id
        or manifest.get("preprocessor_id") != corpus.preprocessor_id
        or manifest.get("split_id") != corpus.split_id
        or manifest.get("anchor_id") != corpus.anchor_id
    ):
        raise DataContractError("Phase 3D manifest does not bind the selected Phase 3C corpus")
    gates = manifest.get("gates")
    required_true = (
        "all_client_ids_in_range",
        "client_partition_created",
        "every_training_record_assigned_once_per_profile",
        "family_cardinality_conserved",
        "minimum_client_size_satisfied",
        "no_record_duplicated_within_a_profile",
        "only_federated_training_records_assigned",
        "quota_matrices_match_plan",
        "validation_test_and_anchors_excluded",
    )
    if (
        not isinstance(gates, dict)
        or not all(gates.get(name) is True for name in required_true)
        or gates.get("model_training_performed") is not False
        or gates.get("transformed_features_duplicated") is not False
    ):
        raise DataContractError("Required Phase 3D gates did not pass")
    counts = manifest.get("family_training_counts")
    if not isinstance(counts, dict):
        raise DataContractError("Phase 3D family training counts are missing")
    ordered = [0] * corpus.num_classes
    for label, class_id in corpus.family_label_to_id.items():
        count = counts.get(label)
        if not isinstance(count, int) or count < 0:
            raise DataContractError("Phase 3D family training counts are malformed")
        ordered[class_id] = count
    training_records = manifest.get("training_records")
    if (
        not isinstance(training_records, int)
        or sum(ordered) != training_records
        or training_records != corpus.partition_counts["train"]
    ):
        raise DataContractError("Phase 3D family counts do not conserve the training population")
    return tuple(ordered)


def load_client_partition_profile(
    client_partition_manifest: Path,
    corpus: PreprocessedCorpus,
    profile: str,
    *,
    verify_assignment_hashes: bool = True,
) -> ClientPartitionProfile:
    """Load one exact Phase 3D profile and validate its assignment inventory."""

    load_training_class_counts(client_partition_manifest, corpus)
    manifest = _load_json(client_partition_manifest, "Phase 3D client-partition manifest")
    partition_id = manifest.get("client_partition_id")
    num_clients = manifest.get("num_clients")
    columns = manifest.get("assignment_columns")
    profiles = manifest.get("profiles")
    inventory = manifest.get("assignment_files")
    if not isinstance(partition_id, str) or not partition_id:
        raise DataContractError("Phase 3D client-partition identity is missing")
    if not isinstance(num_clients, int) or num_clients < 1:
        raise DataContractError("Phase 3D client population is invalid")
    if not isinstance(columns, dict) or not isinstance(columns.get(profile), str):
        raise DataContractError("Requested Phase 3D profile has no assignment column")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise DataContractError("Requested Phase 3D profile is missing")
    profile_record = cast(dict[str, object], profiles[profile])
    totals = profile_record.get("client_totals")
    matrices = profile_record.get("family_client_counts")
    if (
        not isinstance(totals, list)
        or len(totals) != num_clients
        or not all(isinstance(value, int) and value > 0 for value in totals)
    ):
        raise DataContractError("Phase 3D client totals are malformed")
    if not isinstance(matrices, dict):
        raise DataContractError("Phase 3D profile family matrix is missing")
    family_matrix: list[tuple[int, ...]] = [()] * corpus.num_classes
    for label, class_id in corpus.family_label_to_id.items():
        row = matrices.get(label)
        if (
            not isinstance(row, list)
            or len(row) != num_clients
            or not all(isinstance(value, int) and value >= 0 for value in row)
        ):
            raise DataContractError("Phase 3D profile family matrix is malformed")
        family_matrix[class_id] = tuple(cast(list[int], row))
    typed_totals = tuple(cast(list[int], totals))
    if any(
        sum(row[client_index] for row in family_matrix) != typed_totals[client_index]
        for client_index in range(num_clients)
    ):
        raise DataContractError("Phase 3D profile family matrix does not conserve client totals")
    if not isinstance(inventory, list) or len(inventory) != len(corpus.files):
        raise DataContractError("Phase 3D assignment inventory does not align with Phase 3C shards")
    files: list[AssignmentFile] = []
    for item in inventory:
        if not isinstance(item, dict):
            raise DataContractError("Phase 3D assignment inventory entry is malformed")
        relative_path = item.get("path")
        row_count = item.get("row_count")
        byte_size = item.get("byte_size")
        digest = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not isinstance(row_count, int)
            or row_count < 0
            or not isinstance(byte_size, int)
            or byte_size < 1
            or not isinstance(digest, str)
        ):
            raise DataContractError("Phase 3D assignment inventory entry is invalid")
        path = (client_partition_manifest.parent / relative_path).resolve()
        if not path.is_file() or path.stat().st_size != byte_size:
            raise DataContractError("Phase 3D assignment shard is missing or has the wrong size")
        if verify_assignment_hashes and _sha256(path) != digest:
            raise DataContractError("Phase 3D assignment shard checksum is invalid")
        files.append(AssignmentFile(path, row_count, byte_size, digest))
    if sum(file.row_count for file in files) != sum(typed_totals):
        raise DataContractError("Phase 3D assignment inventory does not conserve profile records")
    return ClientPartitionProfile(
        manifest_path=client_partition_manifest.resolve(),
        client_partition_id=partition_id,
        profile=profile,
        assignment_column=cast(str, columns[profile]),
        num_clients=num_clients,
        client_totals=typed_totals,
        family_client_counts=tuple(family_matrix),
        files=tuple(files),
    )


def iter_client_batches(
    corpus: PreprocessedCorpus,
    profile: ClientPartitionProfile,
    client_index: int,
    *,
    batch_size: int,
    shuffle_seed: int,
) -> Iterator[TrainingBatch]:
    """Stream one Phase 3D client's complete non-anchor training population."""

    if not 0 <= client_index < profile.num_clients:
        raise ValueError("client_index is outside the declared population")
    if batch_size < 1 or shuffle_seed < 0:
        raise ValueError("batch_size and shuffle_seed must be positive")
    if len(corpus.files) != len(profile.files):
        raise DataContractError("Phase 3C and Phase 3D shard inventories differ")
    np = _dependency("numpy")
    pq = _dependency("pyarrow.parquet")
    emitted = 0
    actual_counts = [0] * corpus.num_classes
    transformed_columns = ("record_id", "partition", "family_id", *corpus.feature_columns)
    for file_index, (corpus_file, assignment_file) in enumerate(
        zip(corpus.files, profile.files, strict=True)
    ):
        assignments = pq.read_table(
            assignment_file.path,
            columns=["record_id", "family_id", profile.assignment_column],
            filters=[(profile.assignment_column, "=", client_index)],
        )
        selected_records = assignments.column("record_id").to_pylist()
        selected_families = assignments.column("family_id").to_pylist()
        if len(selected_records) != len(set(selected_records)):
            raise DataContractError("Phase 3D client assignment contains duplicate record IDs")
        selected = {
            str(record_id): int(family_id)
            for record_id, family_id in zip(selected_records, selected_families, strict=True)
        }
        if not selected:
            continue
        parquet = pq.ParquetFile(corpus_file.path)
        for batch_index, batch in enumerate(
            parquet.iter_batches(batch_size=batch_size, columns=list(transformed_columns))
        ):
            record_ids = batch.column(0).to_pylist()
            indices = np.fromiter(
                (index for index, record_id in enumerate(record_ids) if str(record_id) in selected),
                dtype=np.int64,
            )
            if indices.size == 0:
                continue
            partitions = batch.column(1).to_numpy(zero_copy_only=False)[indices]
            if np.any(partitions != "train"):
                raise DataContractError("Phase 3D client assignment reached a non-training record")
            targets = (
                batch.column(2).to_numpy(zero_copy_only=False)[indices].astype(np.int64, copy=False)
            )
            for position, target in zip(indices, targets, strict=True):
                if selected[str(record_ids[int(position)])] != int(target):
                    raise DataContractError("Phase 3C and Phase 3D family IDs differ")
            feature_values = np.column_stack(
                [
                    batch.column(column_index).to_numpy(zero_copy_only=False)[indices]
                    for column_index in range(3, len(transformed_columns))
                ]
            ).astype(np.float32, copy=False)
            if indices.size > 1:
                seed = derive_seed(
                    shuffle_seed,
                    f"phase5-client:{profile.profile}:{client_index}:file={file_index}:batch={batch_index}",
                )
                order = np.random.default_rng(seed).permutation(indices.size)
                feature_values = feature_values[order]
                targets = targets[order]
            features_tensor = torch.from_numpy(np.ascontiguousarray(feature_values))
            targets_tensor = torch.from_numpy(np.ascontiguousarray(targets))
            if not torch.isfinite(features_tensor).all().item():
                raise DataContractError("Non-finite transformed feature reached client training")
            counts = np.bincount(targets, minlength=corpus.num_classes)
            for class_id, count in enumerate(counts):
                actual_counts[class_id] += int(count)
            emitted += int(targets_tensor.shape[0])
            yield TrainingBatch(features_tensor, targets_tensor)
    if emitted != profile.client_totals[client_index]:
        raise DataContractError("Streamed client cardinality differs from Phase 3D")
    if tuple(actual_counts) != profile.class_counts(client_index):
        raise DataContractError("Streamed client family counts differ from Phase 3D")


def iter_partition_batches(
    corpus: PreprocessedCorpus,
    partition: str,
    *,
    batch_size: int,
    record_limit: int | None = None,
    shuffle_seed: int | None = None,
) -> Iterator[TrainingBatch]:
    """Stream one declared partition without loading the full corpus into memory."""

    if partition not in _ALLOWED_PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    if batch_size < 1 or (record_limit is not None and record_limit < 1):
        raise ValueError("batch_size and record_limit must be positive")
    np = _dependency("numpy")
    pq = _dependency("pyarrow.parquet")
    emitted = 0
    columns = ("partition", "family_id", *corpus.feature_columns)
    for file_index, item in enumerate(corpus.files):
        parquet = pq.ParquetFile(item.path)
        for batch_index, batch in enumerate(
            parquet.iter_batches(batch_size=batch_size, columns=list(columns))
        ):
            partition_values = batch.column(0).to_numpy(zero_copy_only=False)
            indices = np.flatnonzero(partition_values == partition)
            if indices.size == 0:
                continue
            if record_limit is not None:
                remaining = record_limit - emitted
                if remaining <= 0:
                    return
                indices = indices[:remaining]
            targets = batch.column(1).to_numpy(zero_copy_only=False)[indices]
            feature_values = np.column_stack(
                [
                    batch.column(column_index).to_numpy(zero_copy_only=False)[indices]
                    for column_index in range(2, len(columns))
                ]
            ).astype(np.float32, copy=False)
            targets = targets.astype(np.int64, copy=False)
            if shuffle_seed is not None and indices.size > 1:
                seed = derive_seed(
                    shuffle_seed,
                    f"phase4:{partition}:file={file_index}:batch={batch_index}",
                )
                order = np.random.default_rng(seed).permutation(indices.size)
                feature_values = feature_values[order]
                targets = targets[order]
            features_tensor = torch.from_numpy(np.ascontiguousarray(feature_values))
            targets_tensor = torch.from_numpy(np.ascontiguousarray(targets))
            if not torch.isfinite(features_tensor).all().item():
                raise DataContractError("Non-finite transformed feature reached model loading")
            emitted += int(targets_tensor.shape[0])
            yield TrainingBatch(features_tensor, targets_tensor)
            if record_limit is not None and emitted >= record_limit:
                return
    expected = corpus.partition_counts[partition]
    if record_limit is None and emitted != expected:
        raise DataContractError(
            f"Streamed {emitted} {partition} records but the manifest declares {expected}"
        )
