"""Resumable Phase 3D IID and Dirichlet client-partition manifests."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.seeding import derive_seed

_SCHEMA_VERSION: Final = 1
_ALGORITHM_VERSION: Final = "exact_family_quota_shuffled_sequence_v1"


@dataclass(frozen=True, slots=True)
class ClientPartitionBuildRequest:
    """Frozen inputs and controls for one Phase 3D client-partition build."""

    dataset: DatasetName
    preprocessing_report: Path
    split_manifest: Path
    anchor_manifest: Path
    output_directory: Path
    work_database: Path
    base_seed: int = 20260816
    num_clients: int = 50
    minimum_samples_per_client: int = 128
    dirichlet_alphas: tuple[float, ...] = (0.1, 0.3, 1.0)
    maximum_partition_attempts: int = 1_000
    parquet_batch_size: int = 50_000

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed cannot be negative")
        if self.num_clients < 2:
            raise ValueError("num_clients must be at least two")
        if self.minimum_samples_per_client < 1:
            raise ValueError("minimum_samples_per_client must be positive")
        if not self.dirichlet_alphas:
            raise ValueError("At least one Dirichlet alpha is required")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.dirichlet_alphas):
            raise ValueError("Dirichlet alphas must be finite and positive")
        if len(set(self.dirichlet_alphas)) != len(self.dirichlet_alphas):
            raise ValueError("Dirichlet alphas must be unique")
        if tuple(sorted(self.dirichlet_alphas)) != self.dirichlet_alphas:
            raise ValueError("Dirichlet alphas must be in ascending order")
        if self.maximum_partition_attempts < 1:
            raise ValueError("maximum_partition_attempts must be positive")
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class _Plan:
    report: dict[str, object]
    preprocessing: dict[str, object]
    preprocessor: dict[str, object]
    split: dict[str, object]
    anchor: dict[str, object]
    family_labels: tuple[str, ...]
    family_counts: tuple[int, ...]
    profiles: dict[str, dict[str, object]]


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DataContractError("Phase 3D requires the project ML dependencies") from error


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(base: Path, item: Mapping[str, object], description: str) -> Path:
    relative = item.get("path") if "path" in item else item.get("file")
    if not isinstance(relative, str):
        raise DataContractError(f"{description} has no path")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise DataContractError(f"{description} escapes its artifact directory") from error
    if not path.is_file() or path.stat().st_size != item.get("byte_size"):
        raise DataContractError(f"{description} is missing or changed: {relative}")
    return path


def _alpha_token(alpha: float) -> str:
    if alpha.is_integer():
        return f"{int(alpha)}_0"
    return format(alpha, ".12g").replace("-", "m").replace(".", "_")


def _profile_name(alpha: float) -> str:
    return f"dirichlet_{_alpha_token(alpha)}"


def _client_column(profile: str) -> str:
    return f"{profile}_client_id"


def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    weight_sum = sum(weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise DataContractError("Dirichlet draw has no finite positive mass")
    raw = [total * value / weight_sum for value in weights]
    quotas = [math.floor(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(range(len(weights)), key=lambda index: (-(raw[index] - quotas[index]), index))
    for index in order[:remainder]:
        quotas[index] += 1
    if sum(quotas) != total or any(value < 0 for value in quotas):
        raise DataContractError("Largest-remainder allocation failed conservation")
    return quotas


def _iid_matrix(family_counts: Sequence[int], num_clients: int, seed: int) -> list[list[int]]:
    matrix: list[list[int]] = []
    for family_id, count in enumerate(family_counts):
        base, remainder = divmod(count, num_clients)
        quotas = [base] * num_clients
        offset = derive_seed(seed, f"iid-remainder:{family_id}") % num_clients
        for position in range(remainder):
            quotas[(offset + position) % num_clients] += 1
        matrix.append(quotas)
    return matrix


def _dirichlet_matrix(
    family_counts: Sequence[int], num_clients: int, alpha: float, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        _largest_remainder(count, [rng.gammavariate(alpha, 1.0) for _ in range(num_clients)])
        for count in family_counts
    ]


def _client_totals(matrix: Sequence[Sequence[int]], num_clients: int) -> list[int]:
    return [sum(row[client_id] for row in matrix) for client_id in range(num_clients)]


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DataContractError("Cannot summarize an empty client population")
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _heterogeneity(
    matrix: Sequence[Sequence[int]], family_counts: Sequence[int], num_clients: int
) -> dict[str, object]:
    totals = _client_totals(matrix, num_clients)
    overall = sum(family_counts)
    global_distribution = [count / overall for count in family_counts]
    normalized_entropies: list[float] = []
    total_variations: list[float] = []
    observed_classes: list[float] = []
    entropy_scale = math.log(len(family_counts)) if len(family_counts) > 1 else 1.0
    for client_id, total in enumerate(totals):
        distribution = [row[client_id] / total for row in matrix]
        entropy = -sum(value * math.log(value) for value in distribution if value > 0.0)
        normalized_entropies.append(entropy / entropy_scale)
        total_variations.append(
            0.5
            * sum(
                abs(value - global_value)
                for value, global_value in zip(distribution, global_distribution, strict=True)
            )
        )
        observed_classes.append(float(sum(value > 0.0 for value in distribution)))
    return {
        "client_total_records": {
            "minimum": min(totals),
            "median": _quantile([float(value) for value in totals], 0.5),
            "maximum": max(totals),
            "mean": sum(totals) / num_clients,
        },
        "normalized_label_entropy": {
            "minimum": min(normalized_entropies),
            "median": _quantile(normalized_entropies, 0.5),
            "maximum": max(normalized_entropies),
            "mean": sum(normalized_entropies) / num_clients,
        },
        "total_variation_from_global_label_distribution": {
            "minimum": min(total_variations),
            "median": _quantile(total_variations, 0.5),
            "maximum": max(total_variations),
            "mean": sum(total_variations) / num_clients,
        },
        "observed_primary_families_per_client": {
            "minimum": int(min(observed_classes)),
            "median": _quantile(observed_classes, 0.5),
            "maximum": int(max(observed_classes)),
            "mean": sum(observed_classes) / num_clients,
        },
    }


def _profile(
    name: str,
    kind: str,
    alpha: float | None,
    attempt: int,
    seed: int,
    matrix: Sequence[Sequence[int]],
    family_labels: Sequence[str],
    family_counts: Sequence[int],
    num_clients: int,
) -> dict[str, object]:
    return {
        "profile": name,
        "kind": kind,
        "dirichlet_alpha": alpha,
        "accepted_attempt": attempt,
        "quota_seed": seed,
        "client_totals": _client_totals(matrix, num_clients),
        "family_client_counts": {
            label: list(row) for label, row in zip(family_labels, matrix, strict=True)
        },
        "realized_heterogeneity": _heterogeneity(matrix, family_counts, num_clients),
    }


def _profiles(
    request: ClientPartitionBuildRequest,
    family_labels: Sequence[str],
    family_counts: Sequence[int],
) -> dict[str, dict[str, object]]:
    iid_seed = derive_seed(request.base_seed, f"phase3d-quota:{request.dataset.value}:iid")
    iid = _iid_matrix(family_counts, request.num_clients, iid_seed)
    if min(_client_totals(iid, request.num_clients)) < request.minimum_samples_per_client:
        raise DataContractError("IID partition violates the minimum client size")
    result = {
        "iid": _profile(
            "iid",
            "iid_primary_family_balanced",
            None,
            0,
            iid_seed,
            iid,
            family_labels,
            family_counts,
            request.num_clients,
        )
    }
    for alpha in request.dirichlet_alphas:
        name = _profile_name(alpha)
        for attempt in range(request.maximum_partition_attempts):
            seed = derive_seed(
                request.base_seed,
                f"phase3d-quota:{request.dataset.value}:{name}:attempt:{attempt}",
            )
            matrix = _dirichlet_matrix(family_counts, request.num_clients, alpha, seed)
            if min(_client_totals(matrix, request.num_clients)) >= (
                request.minimum_samples_per_client
            ):
                result[name] = _profile(
                    name,
                    "dirichlet_primary_family_label_allocation",
                    alpha,
                    attempt,
                    seed,
                    matrix,
                    family_labels,
                    family_counts,
                    request.num_clients,
                )
                break
        else:
            raise DataContractError(
                f"No valid {name} allocation in {request.maximum_partition_attempts} attempts"
            )
    return result


def _anchor_family_counts(
    request: ClientPartitionBuildRequest, anchor: Mapping[str, object]
) -> dict[str, int]:
    pq = _dependency("pyarrow.parquet")

    file_info = anchor.get("file")
    if not isinstance(file_info, dict):
        raise DataContractError("Anchor file inventory is missing")
    path = _resolved_file(request.anchor_manifest.parent, file_info, "anchor Parquet")
    if _file_sha256(path) != file_info.get("sha256"):
        raise DataContractError("Anchor Parquet checksum mismatch")
    table = pq.read_table(path, columns=["record_id", "family_label", "subset"])
    values = table.to_pydict()
    record_ids = cast(list[str], values["record_id"])
    if len(record_ids) != len(set(record_ids)):
        raise DataContractError("Anchor record IDs are not unique")
    digest = hashlib.sha256("\n".join(sorted(record_ids)).encode()).hexdigest()
    if digest != anchor.get("record_ids_sha256"):
        raise DataContractError("Anchor record identity digest mismatch")
    family_counts: dict[str, int] = {}
    subset_counts: dict[str, int] = {}
    for family, subset in zip(values["family_label"], values["subset"], strict=True):
        family_counts[cast(str, family)] = family_counts.get(cast(str, family), 0) + 1
        subset_counts[cast(str, subset)] = subset_counts.get(cast(str, subset), 0) + 1
    if family_counts != anchor.get("family_counts") or subset_counts != anchor.get("subset_counts"):
        raise DataContractError("Anchor Parquet counts differ from its manifest")
    return family_counts


def _validated_input(
    request: ClientPartitionBuildRequest,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    tuple[str, ...],
    tuple[int, ...],
]:
    preprocessing = _load_json(request.preprocessing_report, "Phase 3C report")
    split = _load_json(request.split_manifest, "Phase 3B split manifest")
    anchor = _load_json(request.anchor_manifest, "Phase 3B anchor manifest")
    if (
        preprocessing.get("dataset") != request.dataset.value
        or split.get("dataset") != request.dataset.value
        or anchor.get("dataset") != request.dataset.value
    ):
        raise DataContractError("Phase 3D dataset identity mismatch")
    if split.get("base_seed") != request.base_seed:
        raise DataContractError("Phase 3D root seed differs from Phase 3B")
    if preprocessing.get("split_id") != split.get("split_id") or preprocessing.get(
        "anchor_id"
    ) != anchor.get("anchor_id"):
        raise DataContractError("Phase 3D split or anchor identity mismatch")
    gates = preprocessing.get("gates")
    required_gates = (
        "fit_excludes_validation_test_and_anchors",
        "fit_cardinality_matches_federated_training_pool",
        "transformed_cardinality_matches_canonical",
        "partition_cardinality_matches_phase3b",
        "stable_feature_order_and_dimension",
        "no_nonfinite_model_input",
        "preprocessing_fitted",
    )
    if not isinstance(gates, dict) or not all(gates.get(name) is True for name in required_gates):
        raise DataContractError("Required Phase 3C gates did not pass")
    if gates.get("client_partition_created") is not False:
        raise DataContractError("Phase 3C input already claims a client partition")
    preprocessor_info = preprocessing.get("preprocessor")
    if not isinstance(preprocessor_info, dict):
        raise DataContractError("Phase 3C preprocessor inventory is missing")
    preprocessor_path = _resolved_file(
        request.preprocessing_report.parent, preprocessor_info, "Phase 3C preprocessor"
    )
    if _file_sha256(preprocessor_path) != preprocessor_info.get("sha256"):
        raise DataContractError("Phase 3C preprocessor checksum mismatch")
    preprocessor = _load_json(preprocessor_path, "Phase 3C preprocessor")
    if preprocessor.get("preprocessor_id") != preprocessor_info.get("preprocessor_id"):
        raise DataContractError("Phase 3C preprocessor identity mismatch")
    label_to_id = preprocessor.get("family_label_to_id")
    if not isinstance(label_to_id, dict) or not all(
        isinstance(label, str) and isinstance(identifier, int)
        for label, identifier in label_to_id.items()
    ):
        raise DataContractError("Phase 3C family label map is invalid")
    family_labels = tuple(
        label for label, _ in sorted(label_to_id.items(), key=lambda item: cast(int, item[1]))
    )
    if sorted(cast(int, value) for value in label_to_id.values()) != list(
        range(len(family_labels))
    ):
        raise DataContractError("Phase 3C family IDs are not contiguous")
    split_family_counts = split.get("family_split_counts")
    if not isinstance(split_family_counts, dict) or set(split_family_counts) != set(family_labels):
        raise DataContractError("Phase 3B family counts differ from the Phase 3C label map")
    anchor_family_counts = _anchor_family_counts(request, anchor)
    counts: list[int] = []
    for label in family_labels:
        split_counts = split_family_counts[label]
        if not isinstance(split_counts, dict) or not isinstance(split_counts.get("train"), int):
            raise DataContractError("Phase 3B family training count is invalid")
        count = cast(int, split_counts["train"]) - anchor_family_counts.get(label, 0)
        if count < 1:
            raise DataContractError("A primary family has no federated training records")
        counts.append(count)
    partitions = preprocessing.get("partition_counts")
    if not isinstance(partitions, dict) or partitions.get("train") != sum(counts):
        raise DataContractError("Phase 3D family counts do not cover the Phase 3C training pool")
    if request.minimum_samples_per_client * request.num_clients > sum(counts):
        raise DataContractError("Minimum client-size requirement exceeds the training pool")
    transformed_files = preprocessing.get("transformed_files")
    if not isinstance(transformed_files, list) or not transformed_files:
        raise DataContractError("Phase 3C transformed-file inventory is missing")
    return (
        preprocessing,
        preprocessor,
        split,
        anchor,
        family_labels,
        tuple(counts),
    )


def _plan(request: ClientPartitionBuildRequest) -> _Plan:
    preprocessing, preprocessor, split, anchor, family_labels, family_counts = _validated_input(
        request
    )
    profiles = _profiles(request, family_labels, family_counts)
    transformed_files = cast(list[dict[str, object]], preprocessing["transformed_files"])
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "algorithm_version": _ALGORITHM_VERSION,
        "dataset": request.dataset.value,
        "base_seed": request.base_seed,
        "preprocessed_corpus_id": preprocessing.get("preprocessed_corpus_id"),
        "preprocessor_id": preprocessor.get("preprocessor_id"),
        "split_id": split.get("split_id"),
        "anchor_id": anchor.get("anchor_id"),
        "training_records": sum(family_counts),
        "family_label_to_id": preprocessor.get("family_label_to_id"),
        "family_training_counts": dict(zip(family_labels, family_counts, strict=True)),
        "num_clients": request.num_clients,
        "minimum_samples_per_client": request.minimum_samples_per_client,
        "maximum_partition_attempts": request.maximum_partition_attempts,
        "profiles": profiles,
        "assignment_columns": {name: _client_column(name) for name in profiles},
        "input_shards": len(transformed_files),
        "output_layout": "one_training_record_per_row_all_profiles",
        "validation_test_and_anchors_excluded": True,
        "transformed_features_duplicated": False,
        "client_assignment_created": False,
        "model_training_performed": False,
    }
    report: dict[str, object] = {"plan_id": run_id(stable, length=64), **stable}
    return _Plan(
        report,
        preprocessing,
        preprocessor,
        split,
        anchor,
        family_labels,
        family_counts,
        profiles,
    )


def _write_plan(report: Mapping[str, object], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = cast(str, report["dataset"])
    stem = f"{dataset}_phase3d_plan"
    (output_directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profiles = cast(dict[str, dict[str, object]], report["profiles"])
    lines = [
        f"# Phase 3D client-partition plan: {dataset}",
        "",
        f"- Plan ID: `{report['plan_id']}`",
        f"- Training records: {cast(int, report['training_records']):,}",
        f"- Clients: {report['num_clients']}",
        f"- Minimum records per client: {report['minimum_samples_per_client']}",
        "- Validation, test, and anchors excluded: yes",
        "- Model training performed: no",
        "",
        "| Profile | Accepted attempt | Minimum client records | Mean normalized entropy |",
        "|---|---:|---:|---:|",
    ]
    for name, profile in profiles.items():
        heterogeneity = cast(dict[str, dict[str, object]], profile["realized_heterogeneity"])
        lines.append(
            f"| `{name}` | {profile['accepted_attempt']} | "
            f"{heterogeneity['client_total_records']['minimum']} | "
            f"{cast(float, heterogeneity['normalized_label_entropy']['mean']):.6f} |"
        )
    lines.append("")
    (output_directory / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def plan_phase3d_clients(request: ClientPartitionBuildRequest) -> dict[str, object]:
    """Write a Phase 3D plan without scanning transformed feature shards."""

    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    return plan.report


def _open_database(request: ClientPartitionBuildRequest, identity: str) -> sqlite3.Connection:
    request.work_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(request.work_database)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assignment_shards(
            shard_index INTEGER PRIMARY KEY,
            output_file TEXT NOT NULL,
            output_size INTEGER NOT NULL,
            output_sha256 TEXT NOT NULL,
            training_rows INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError("Phase 3D resume database identity mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _profile_matrix(profile: Mapping[str, object], family_labels: Sequence[str]) -> list[list[int]]:
    by_label = cast(dict[str, list[int]], profile["family_client_counts"])
    return [by_label[label] for label in family_labels]


def _assignment_sequences(
    request: ClientPartitionBuildRequest, plan: _Plan
) -> dict[str, dict[int, object]]:
    np = _dependency("numpy")

    sequences: dict[str, dict[int, object]] = {}
    clients = np.arange(request.num_clients, dtype=np.int16)
    for profile_name, profile in plan.profiles.items():
        matrix = _profile_matrix(profile, plan.family_labels)
        profile_sequences: dict[int, object] = {}
        for family_id, (label, quotas) in enumerate(zip(plan.family_labels, matrix, strict=True)):
            values = np.repeat(clients, np.asarray(quotas, dtype=np.int64))
            expected = plan.family_counts[family_id]
            if len(values) != expected:
                raise DataContractError("Phase 3D assignment sequence has wrong length")
            seed = derive_seed(
                request.base_seed,
                f"phase3d-order:{request.dataset.value}:{profile_name}:{label}",
            )
            np.random.default_rng(seed).shuffle(values)
            profile_sequences[family_id] = values
        sequences[profile_name] = profile_sequences
    return sequences


def _completed_reports(
    connection: sqlite3.Connection, request: ClientPartitionBuildRequest
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for index, relative, size, digest, payload in connection.execute(
        "SELECT shard_index, output_file, output_size, output_sha256, report_json "
        "FROM assignment_shards ORDER BY shard_index"
    ):
        path = request.output_directory / cast(str, relative)
        if not path.is_file() or path.stat().st_size != cast(int, size):
            raise DataContractError(f"Completed Phase 3D file is missing or changed: {relative}")
        if _file_sha256(path) != cast(str, digest):
            raise DataContractError(f"Completed Phase 3D checksum mismatch: {relative}")
        result[cast(int, index)] = cast(dict[str, object], json.loads(cast(str, payload)))
    return result


def _empty_matrix(families: int, clients: int) -> list[list[int]]:
    return [[0] * clients for _ in range(families)]


def _add_matrix(target: list[list[int]], source: Sequence[Sequence[int]]) -> None:
    for target_row, source_row in zip(target, source, strict=True):
        for index, value in enumerate(source_row):
            target_row[index] += value


def _advance_offsets(offsets: list[int], family_counts: Sequence[int]) -> None:
    for family_id, value in enumerate(family_counts):
        offsets[family_id] += value


def _assignment_schema(profiles: Sequence[str]) -> object:
    pa = _dependency("pyarrow")

    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("family_id", pa.int16(), nullable=False),
            *(
                pa.field(_client_column(profile), pa.int16(), nullable=False)
                for profile in profiles
            ),
        ]
    )


def _assignment_shard(
    request: ClientPartitionBuildRequest,
    plan: _Plan,
    sequences: Mapping[str, Mapping[int, object]],
    offsets: list[int],
    connection: sqlite3.Connection,
    index: int,
) -> dict[str, object]:
    np = _dependency("numpy")
    pa = _dependency("pyarrow")
    pq = _dependency("pyarrow.parquet")

    input_item = cast(list[dict[str, object]], plan.preprocessing["transformed_files"])[index]
    input_path = _resolved_file(
        request.preprocessing_report.parent, input_item, "Phase 3C transformed shard"
    )
    if _file_sha256(input_path) != input_item.get("sha256"):
        raise DataContractError("Phase 3C transformed-shard checksum mismatch")
    profile_names = tuple(plan.profiles)
    schema = _assignment_schema(profile_names)
    output_directory = request.output_directory / "assignments"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_relative = f"assignments/{index:04d}-clients.parquet"
    output_path = request.output_directory / output_relative
    partial = output_path.with_suffix(".parquet.partial")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, schema, compression="zstd", use_dictionary=True)
    family_counts = [0] * len(plan.family_labels)
    profile_counts = {
        profile: _empty_matrix(len(plan.family_labels), request.num_clients)
        for profile in profile_names
    }
    training_rows = 0
    try:
        parquet = pq.ParquetFile(input_path)
        for batch in parquet.iter_batches(
            batch_size=request.parquet_batch_size,
            columns=["record_id", "partition", "family_id"],
        ):
            values = batch.to_pydict()
            partitions = np.asarray(values["partition"], dtype=object)
            selected = np.flatnonzero(partitions == "train")
            if not len(selected):
                continue
            record_ids = np.asarray(values["record_id"], dtype=object)[selected].tolist()
            family_ids = np.asarray(values["family_id"], dtype=np.int16)[selected]
            if np.any(family_ids < 0) or np.any(family_ids >= len(plan.family_labels)):
                raise DataContractError("Phase 3C family ID is outside the frozen label map")
            per_family_positions = {
                family_id: np.flatnonzero(family_ids == family_id)
                for family_id in range(len(plan.family_labels))
            }
            output_clients: dict[str, object] = {}
            for profile in profile_names:
                assigned = np.empty(len(selected), dtype=np.int16)
                for family_id, positions in per_family_positions.items():
                    count = len(positions)
                    if not count:
                        continue
                    start = offsets[family_id]
                    end = start + count
                    sequence = cast(Any, sequences[profile][family_id])
                    if end > len(sequence):
                        raise DataContractError("Phase 3D assignment sequence was over-consumed")
                    assigned[positions] = sequence[start:end]
                    bincount = np.bincount(assigned[positions], minlength=request.num_clients)
                    profile_counts[profile][family_id] = [
                        left + int(right)
                        for left, right in zip(
                            profile_counts[profile][family_id], bincount, strict=True
                        )
                    ]
                output_clients[profile] = assigned
            batch_family_counts = np.bincount(family_ids, minlength=len(plan.family_labels))
            for family_id, count in enumerate(batch_family_counts):
                family_counts[family_id] += int(count)
            _advance_offsets(offsets, [int(value) for value in batch_family_counts])
            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(record_ids, type=pa.string()),
                        pa.array(family_ids, type=pa.int16()),
                        *(
                            pa.array(output_clients[profile], type=pa.int16())
                            for profile in profile_names
                        ),
                    ],
                    schema=schema,
                )
            )
            training_rows += len(selected)
    finally:
        writer.close()
    if training_rows != sum(family_counts):
        partial.unlink(missing_ok=True)
        raise DataContractError("Phase 3D shard family counts do not conserve rows")
    partial.replace(output_path)
    report: dict[str, object] = {
        "training_rows": training_rows,
        "family_counts": family_counts,
        "profile_family_client_counts": profile_counts,
    }
    digest = _file_sha256(output_path)
    with connection:
        connection.execute(
            "INSERT INTO assignment_shards VALUES (?, ?, ?, ?, ?, ?)",
            (
                index,
                output_relative,
                output_path.stat().st_size,
                digest,
                training_rows,
                canonical_json(report),
            ),
        )
    return report


def _finalize(
    request: ClientPartitionBuildRequest,
    plan: _Plan,
    connection: sqlite3.Connection,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    rows = 0
    family_counts = [0] * len(plan.family_labels)
    actual = {
        profile: _empty_matrix(len(plan.family_labels), request.num_clients)
        for profile in plan.profiles
    }
    for relative, size, digest, output_rows, payload in connection.execute(
        "SELECT output_file, output_size, output_sha256, training_rows, report_json "
        "FROM assignment_shards ORDER BY shard_index"
    ):
        report = cast(dict[str, object], json.loads(cast(str, payload)))
        rows += cast(int, output_rows)
        shard_families = cast(list[int], report["family_counts"])
        for family_id, count in enumerate(shard_families):
            family_counts[family_id] += count
        shard_profiles = cast(dict[str, list[list[int]]], report["profile_family_client_counts"])
        for profile, matrix in shard_profiles.items():
            _add_matrix(actual[profile], matrix)
        files.append(
            {
                "path": relative,
                "byte_size": size,
                "sha256": digest,
                "row_count": output_rows,
            }
        )
    if rows != sum(plan.family_counts) or tuple(family_counts) != plan.family_counts:
        raise DataContractError("Phase 3D final family cardinality mismatch")
    for profile, profile_report in plan.profiles.items():
        expected = _profile_matrix(profile_report, plan.family_labels)
        if actual[profile] != expected:
            raise DataContractError(f"Phase 3D quota matrix mismatch for {profile}")
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "phase3d_plan_id": plan.report["plan_id"],
        "preprocessed_corpus_id": plan.report["preprocessed_corpus_id"],
        "preprocessor_id": plan.report["preprocessor_id"],
        "split_id": plan.report["split_id"],
        "anchor_id": plan.report["anchor_id"],
        "algorithm_version": _ALGORITHM_VERSION,
        "training_records": rows,
        "num_clients": request.num_clients,
        "minimum_samples_per_client": request.minimum_samples_per_client,
        "family_label_to_id": plan.report["family_label_to_id"],
        "family_training_counts": plan.report["family_training_counts"],
        "profiles": plan.profiles,
        "assignment_columns": plan.report["assignment_columns"],
        "assignment_files": files,
        "gates": {
            "only_federated_training_records_assigned": True,
            "validation_test_and_anchors_excluded": True,
            "every_training_record_assigned_once_per_profile": True,
            "no_record_duplicated_within_a_profile": True,
            "family_cardinality_conserved": True,
            "quota_matrices_match_plan": True,
            "minimum_client_size_satisfied": True,
            "all_client_ids_in_range": True,
            "transformed_features_duplicated": False,
            "client_partition_created": True,
            "model_training_performed": False,
        },
    }
    result: dict[str, object] = {
        "client_partition_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        **stable,
    }
    json_path = request.output_directory / f"{request.dataset.value}_phase3d_clients.json"
    md_path = request.output_directory / f"{request.dataset.value}_phase3d_clients.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Phase 3D client partitions: {request.dataset.value}",
        "",
        f"- Client partition ID: `{result['client_partition_id']}`",
        f"- Training records: {rows:,}",
        f"- Clients: {request.num_clients}",
        f"- Assignment files: {len(files)}",
        "- Validation, test, and anchors excluded: yes",
        "- Model training performed: no",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def build_phase3d_clients(
    request: ClientPartitionBuildRequest, *, progress_every: int = 1
) -> dict[str, object]:
    """Build exact IID and Dirichlet client assignments with shard-level resume."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    transformed_files = cast(list[dict[str, object]], plan.preprocessing["transformed_files"])
    identity = canonical_json(
        {
            "schema_version": _SCHEMA_VERSION,
            "plan_id": plan.report["plan_id"],
            "transformed_files": transformed_files,
        }
    )
    connection = _open_database(request, identity)
    try:
        sequences = _assignment_sequences(request, plan)
        completed = _completed_reports(connection, request)
        offsets = [0] * len(plan.family_labels)
        for index in range(len(transformed_files)):
            if index in completed:
                shard_report = completed[index]
                _advance_offsets(offsets, cast(list[int], shard_report["family_counts"]))
            else:
                _assignment_shard(request, plan, sequences, offsets, connection, index)
            if (index + 1) % progress_every == 0 or index + 1 == len(transformed_files):
                print(f"phase3d_assign={index + 1}/{len(transformed_files)}", flush=True)
        if tuple(offsets) != plan.family_counts:
            raise DataContractError("Phase 3D did not consume each family assignment sequence")
        return _finalize(request, plan, connection)
    finally:
        connection.close()
