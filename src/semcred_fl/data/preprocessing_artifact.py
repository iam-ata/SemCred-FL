"""Resumable leakage-safe Phase 3C preprocessing artifact builder."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from semcred_fl.config import DatasetName
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.preprocessing import (
    ABSENT_CATEGORY,
    CIC_PREPARATION_POLICY,
    EDGE_PREPARATION_POLICY,
    UNKNOWN_CATEGORY,
    DatasetPreparationPolicy,
)
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.seeding import derive_seed

_SCHEMA_VERSION: Final = 1
_PARTITIONS: Final = (
    "train",
    "validation",
    "test",
    "anchor_calibration",
    "anchor_scoring",
)


@dataclass(frozen=True, slots=True)
class PreprocessingBuildRequest:
    """Frozen inputs and controls for one Phase 3C preprocessing build."""

    dataset: DatasetName
    canonical_manifest: Path
    split_manifest: Path
    anchor_manifest: Path
    output_directory: Path
    work_database: Path
    base_seed: int = 20260816
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    quantile_sample_target: int = 1_000_000
    standard_deviation_epsilon: float = 1e-12
    parquet_batch_size: int = 50_000

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed cannot be negative")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
        if self.quantile_sample_target < 2:
            raise ValueError("quantile_sample_target must be at least two")
        if self.standard_deviation_epsilon <= 0.0:
            raise ValueError("standard_deviation_epsilon must be positive")
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class _Plan:
    report: dict[str, object]
    canonical: dict[str, object]
    split: dict[str, object]
    anchor: dict[str, object]
    policy: DatasetPreparationPolicy
    sample_seed: int
    sample_threshold: int


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


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DataContractError("Phase 3C requires the project ML dependencies") from error


def _policy(dataset: DatasetName) -> DatasetPreparationPolicy:
    return (
        CIC_PREPARATION_POLICY if dataset is DatasetName.CIC_IOT_2023 else EDGE_PREPARATION_POLICY
    )


def _validated_inventory(
    request: PreprocessingBuildRequest,
    canonical: Mapping[str, object],
    split: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    canonical_shards = canonical.get("shards")
    assignment_files = split.get("assignment_files")
    if not isinstance(canonical_shards, list) or not isinstance(assignment_files, list):
        raise DataContractError("Phase 3C input inventories are missing")
    if not canonical_shards or len(canonical_shards) != len(assignment_files):
        raise DataContractError("Canonical and assignment inventories do not align")
    for canonical_item, assignment_item in zip(canonical_shards, assignment_files, strict=True):
        if not isinstance(canonical_item, dict) or not isinstance(assignment_item, dict):
            raise DataContractError("Phase 3C inventory entry is malformed")
        if canonical_item.get("row_count") != assignment_item.get("row_count"):
            raise DataContractError("Canonical and assignment shard row counts differ")
    return cast(list[dict[str, object]], canonical_shards), cast(
        list[dict[str, object]], assignment_files
    )


def _plan(request: PreprocessingBuildRequest) -> _Plan:
    canonical = _load_json(request.canonical_manifest, "canonical manifest")
    split = _load_json(request.split_manifest, "Phase 3B split manifest")
    anchor = _load_json(request.anchor_manifest, "Phase 3B anchor manifest")
    provenance = canonical.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("dataset") != request.dataset.value:
        raise DataContractError("Canonical manifest dataset mismatch")
    if (
        split.get("dataset") != request.dataset.value
        or anchor.get("dataset") != request.dataset.value
    ):
        raise DataContractError("Phase 3B dataset mismatch")
    if split.get("base_seed") != request.base_seed:
        raise DataContractError("Phase 3C root seed differs from the Phase 3B split")
    corpus_id = canonical.get("corpus_id")
    if (
        split.get("canonical_corpus_id") != corpus_id
        or anchor.get("canonical_corpus_id") != corpus_id
    ):
        raise DataContractError("Phase 3C canonical corpus identity mismatch")
    anchor_reference = split.get("anchor_manifest")
    if not isinstance(anchor_reference, dict) or anchor_reference.get("anchor_id") != anchor.get(
        "anchor_id"
    ):
        raise DataContractError("Split and anchor manifest identities differ")
    gates = split.get("gates")
    anchor_gates = anchor.get("gates")
    required_split_gates = (
        "assignment_cardinality_matches_canonical",
        "primary_family_cardinality_matches_canonical",
        "every_primary_family_in_every_split",
        "group_disjoint_by_construction",
        "fingerprint_disjoint_by_canonical_deduplication",
        "anchors_disjoint_and_train_only",
    )
    if not isinstance(gates, dict) or not all(
        gates.get(name) is True for name in required_split_gates
    ):
        raise DataContractError("Required Phase 3B split gates did not pass")
    if not isinstance(anchor_gates, dict) or not all(
        value is True for value in anchor_gates.values()
    ):
        raise DataContractError("Required Phase 3B anchor gates did not pass")
    canonical_shards, assignment_files = _validated_inventory(request, canonical, split)
    train_records = split.get("federated_train_records_after_anchor_removal")
    if not isinstance(train_records, int) or train_records < 2:
        raise DataContractError("Federated training count is invalid")
    sample_seed = derive_seed(request.base_seed, f"phase3c-quantiles:{request.dataset.value}")
    threshold = (
        1 << 64
        if request.quantile_sample_target >= train_records
        else math.ceil(request.quantile_sample_target * (1 << 64) / train_records)
    )
    policy = _policy(request.dataset)
    if provenance.get("preparation_policy_id") != policy.policy_id:
        raise DataContractError("Canonical preparation policy mismatch")
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "canonical_corpus_id": corpus_id,
        "split_id": split.get("split_id"),
        "anchor_id": anchor.get("anchor_id"),
        "preparation_policy_id": policy.policy_id,
        "base_seed": request.base_seed,
        "derived_sample_seed": sample_seed,
        "quantile_method": "deterministic_sha256_threshold_sample_numpy_linear",
        "lower_quantile": request.lower_quantile,
        "upper_quantile": request.upper_quantile,
        "quantile_sample_target": request.quantile_sample_target,
        "sample_threshold_uint64": threshold,
        "standard_deviation": "population_after_clipping",
        "standard_deviation_epsilon": request.standard_deviation_epsilon,
        "continuous_features": list(policy.features.continuous_columns),
        "categorical_features": list(policy.features.categorical_columns),
        "canonical_shards": len(canonical_shards),
        "assignment_shards": len(assignment_files),
        "federated_training_records": train_records,
        "fit_excludes_anchors": True,
        "preprocessing_fitted": False,
        "transformed_records_created": False,
        "client_partition_created": False,
    }
    report: dict[str, object] = {"plan_id": run_id(stable, length=64), **stable}
    return _Plan(report, canonical, split, anchor, policy, sample_seed, threshold)


def _write_plan(report: Mapping[str, object], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['dataset']}_phase3c_plan"
    (output_directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / f"{stem}.md").write_text(
        "\n".join(
            (
                f"# Phase 3C preprocessing plan: {report['dataset']}",
                "",
                f"- Plan ID: `{report['plan_id']}`",
                f"- Split ID: `{report['split_id']}`",
                f"- Quantiles: {report['lower_quantile']} / {report['upper_quantile']}",
                f"- Quantile sample target: {report['quantile_sample_target']:,}",
                "- Fit population: federated training after anchor removal",
                "- Preprocessing fitted: no",
                "- Transformed records created: no",
                "",
            )
        ),
        encoding="utf-8",
    )


def plan_phase3c_preprocessing(request: PreprocessingBuildRequest) -> dict[str, object]:
    """Write a Phase 3C plan without scanning canonical records."""

    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    return plan.report


def _open_database(request: PreprocessingBuildRequest, identity: str) -> sqlite3.Connection:
    request.work_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(request.work_database)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sample_shards(
            shard_index INTEGER PRIMARY KEY, output_file TEXT NOT NULL,
            output_size INTEGER NOT NULL, output_sha256 TEXT NOT NULL,
            sampled_rows INTEGER NOT NULL, report_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS statistic_shards(
            shard_index INTEGER PRIMARY KEY, training_rows INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transformed_shards(
            shard_index INTEGER PRIMARY KEY, output_file TEXT NOT NULL,
            output_size INTEGER NOT NULL, output_sha256 TEXT NOT NULL,
            output_rows INTEGER NOT NULL, report_json TEXT NOT NULL
        );
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is not None and cast(str, row[0]) != identity:
        connection.close()
        raise DataContractError("Phase 3C resume database identity mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)", (identity,)
    )
    connection.commit()
    return connection


def _resolved_file(base: Path, item: Mapping[str, object], description: str) -> Path:
    relative = item.get("file") if "file" in item else item.get("path")
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


def _aligned_batches(
    canonical_path: Path,
    assignment_path: Path,
    canonical_columns: Sequence[str],
    assignment_columns: Sequence[str],
    batch_size: int,
) -> Iterator[tuple[Any, Any]]:
    pq = _dependency("pyarrow.parquet")

    canonical = pq.ParquetFile(canonical_path)
    assignment = pq.ParquetFile(assignment_path)
    if canonical.metadata.num_rows != assignment.metadata.num_rows:
        raise DataContractError("Canonical and assignment Parquet row counts differ")
    canonical_iterator = iter(
        canonical.iter_batches(batch_size=batch_size, columns=canonical_columns)
    )
    assignment_iterator = iter(
        assignment.iter_batches(batch_size=batch_size, columns=assignment_columns)
    )
    canonical_batch = assignment_batch = None
    canonical_offset = assignment_offset = 0
    while True:
        if canonical_batch is None or canonical_offset == canonical_batch.num_rows:
            canonical_batch = next(canonical_iterator, None)
            canonical_offset = 0
        if assignment_batch is None or assignment_offset == assignment_batch.num_rows:
            assignment_batch = next(assignment_iterator, None)
            assignment_offset = 0
        if canonical_batch is None or assignment_batch is None:
            if canonical_batch is not None or assignment_batch is not None:
                raise DataContractError("Canonical and assignment batch streams differ")
            break
        length = min(
            canonical_batch.num_rows - canonical_offset,
            assignment_batch.num_rows - assignment_offset,
        )
        yield (
            canonical_batch.slice(canonical_offset, length),
            assignment_batch.slice(assignment_offset, length),
        )
        canonical_offset += length
        assignment_offset += length


def _anchor_subsets(request: PreprocessingBuildRequest, plan: _Plan) -> dict[str, str]:
    pq = _dependency("pyarrow.parquet")

    file_info = plan.anchor.get("file")
    if not isinstance(file_info, dict):
        raise DataContractError("Anchor file inventory is missing")
    path = _resolved_file(request.anchor_manifest.parent, file_info, "anchor Parquet")
    if _file_sha256(path) != file_info.get("sha256"):
        raise DataContractError("Anchor Parquet checksum mismatch")
    table = pq.read_table(path, columns=["record_id", "subset"])
    values = table.to_pydict()
    result = dict(zip(values["record_id"], values["subset"], strict=True))
    if len(result) != table.num_rows or set(result.values()) != {"calibration", "scoring"}:
        raise DataContractError("Anchor record identities or subsets are invalid")
    expected_record_ids_sha256 = plan.anchor.get("record_ids_sha256")
    actual_record_ids_sha256 = hashlib.sha256("\n".join(sorted(result)).encode("utf-8")).hexdigest()
    if expected_record_ids_sha256 != actual_record_ids_sha256:
        raise DataContractError("Anchor record identity digest mismatch")
    return cast(dict[str, str], result)


def _sample_selected(record_id: str, seed: int, threshold: int) -> bool:
    if threshold >= 1 << 64:
        return True
    digest = hashlib.sha256(f"{seed}\0quantile-sample\0{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < threshold


def _input_pair(
    request: PreprocessingBuildRequest, plan: _Plan, index: int
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    canonical_items = cast(list[dict[str, object]], plan.canonical["shards"])
    assignment_items = cast(list[dict[str, object]], plan.split["assignment_files"])
    canonical_item = canonical_items[index]
    assignment_item = assignment_items[index]
    canonical_path = _resolved_file(
        request.canonical_manifest.parent, canonical_item, "canonical shard"
    )
    assignment_path = _resolved_file(
        request.split_manifest.parent, assignment_item, "assignment shard"
    )
    return canonical_path, assignment_path, canonical_item, assignment_item


def _verify_pair_hashes(
    canonical_path: Path,
    assignment_path: Path,
    canonical_item: Mapping[str, object],
    assignment_item: Mapping[str, object],
) -> None:
    if _file_sha256(canonical_path) != canonical_item.get("sha256"):
        raise DataContractError("Canonical shard checksum mismatch")
    if _file_sha256(assignment_path) != assignment_item.get("sha256"):
        raise DataContractError("Assignment shard checksum mismatch")


def _sample_shard(
    request: PreprocessingBuildRequest,
    plan: _Plan,
    anchors: Mapping[str, str],
    connection: sqlite3.Connection,
    index: int,
) -> None:
    np = _dependency("numpy")
    pa = _dependency("pyarrow")
    pq = _dependency("pyarrow.parquet")

    canonical_path, assignment_path, canonical_item, assignment_item = _input_pair(
        request, plan, index
    )
    _verify_pair_hashes(canonical_path, assignment_path, canonical_item, assignment_item)
    features = list(plan.policy.features.continuous_columns)
    arrays: list[object] = []
    training_rows = sampled_rows = 0
    for canonical_batch, assignment_batch in _aligned_batches(
        canonical_path,
        assignment_path,
        ["record_id", *features],
        ["record_id", "split"],
        request.parquet_batch_size,
    ):
        canonical_values = canonical_batch.to_pydict()
        assignment_values = assignment_batch.to_pydict()
        if canonical_values["record_id"] != assignment_values["record_id"]:
            raise DataContractError("Canonical and assignment record order differs")
        selected: list[int] = []
        for position, (record_id, split) in enumerate(
            zip(canonical_values["record_id"], assignment_values["split"], strict=True)
        ):
            if split == "train" and record_id not in anchors:
                training_rows += 1
                if _sample_selected(record_id, plan.sample_seed, plan.sample_threshold):
                    selected.append(position)
        if selected:
            matrix = np.column_stack(
                [
                    np.asarray(canonical_values[feature], dtype=np.float64)[selected]
                    for feature in features
                ]
            )
            if not np.isfinite(matrix).all():
                raise DataContractError("Non-finite canonical feature reached Phase 3C")
            arrays.append(matrix)
            sampled_rows += len(selected)
    matrix = np.concatenate(arrays, axis=0) if arrays else np.empty((0, len(features)))
    sample_directory = request.output_directory / "_work" / "quantile_samples"
    sample_directory.mkdir(parents=True, exist_ok=True)
    output_relative = f"_work/quantile_samples/{index:04d}.parquet"
    output_path = request.output_directory / output_relative
    partial = output_path.with_suffix(".parquet.partial")
    table = pa.Table.from_arrays(
        [
            pa.array(matrix[:, feature_index], type=pa.float64())
            for feature_index in range(len(features))
        ],
        names=features,
    )
    pq.write_table(table, partial, compression="zstd")
    partial.replace(output_path)
    report = {"training_rows": training_rows, "sampled_rows": sampled_rows}
    digest = _file_sha256(output_path)
    with connection:
        connection.execute(
            "INSERT INTO sample_shards VALUES (?, ?, ?, ?, ?, ?)",
            (
                index,
                output_relative,
                output_path.stat().st_size,
                digest,
                sampled_rows,
                canonical_json(report),
            ),
        )


def _completed_files(
    connection: sqlite3.Connection,
    request: PreprocessingBuildRequest,
    table: str,
) -> set[int]:
    if table not in {"sample_shards", "transformed_shards"}:
        raise ValueError("Unsupported completed-file table")
    completed: set[int] = set()
    for index, relative, size, digest in connection.execute(
        f"SELECT shard_index, output_file, output_size, output_sha256 FROM {table}"
    ):
        path = request.output_directory / cast(str, relative)
        if not path.is_file() or path.stat().st_size != cast(int, size):
            raise DataContractError(f"Completed Phase 3C file is missing or changed: {relative}")
        if _file_sha256(path) != cast(str, digest):
            raise DataContractError(f"Completed Phase 3C checksum mismatch: {relative}")
        completed.add(cast(int, index))
    return completed


def _fit_quantiles(
    request: PreprocessingBuildRequest, plan: _Plan, connection: sqlite3.Connection
) -> dict[str, dict[str, float]]:
    np = _dependency("numpy")
    pq = _dependency("pyarrow.parquet")

    existing = connection.execute("SELECT value FROM metadata WHERE key='quantiles'").fetchone()
    if existing is not None:
        return cast(dict[str, dict[str, float]], json.loads(cast(str, existing[0])))
    features = plan.policy.features.continuous_columns
    rows = cast(
        int,
        connection.execute("SELECT COALESCE(SUM(sampled_rows), 0) FROM sample_shards").fetchone()[
            0
        ],
    )
    training_rows = 0
    for (report_json,) in connection.execute(
        "SELECT report_json FROM sample_shards ORDER BY shard_index"
    ):
        report = json.loads(cast(str, report_json))
        training_rows += cast(int, report["training_rows"])
    if training_rows != plan.split.get("federated_train_records_after_anchor_removal"):
        raise DataContractError("Phase 3C training population differs from Phase 3B")
    if rows < 2:
        raise DataContractError("Deterministic quantile sample contains fewer than two rows")
    sample_paths = [
        request.output_directory / cast(str, row[0])
        for row in connection.execute("SELECT output_file FROM sample_shards ORDER BY shard_index")
    ]
    result: dict[str, dict[str, float]] = {}
    for feature in features:
        values = np.concatenate(
            [pq.read_table(path, columns=[feature]).column(0).to_numpy() for path in sample_paths]
        )
        lower, upper = np.quantile(
            values, [request.lower_quantile, request.upper_quantile], method="linear"
        )
        if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
            raise DataContractError(f"Non-finite fitted quantile for {feature}")
        result[feature] = {"lower": float(lower), "upper": float(upper)}
    with connection:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('quantiles', ?)",
            (canonical_json(result),),
        )
    return result


def _merge_moments(
    count_a: int,
    mean_a: Sequence[float],
    m2_a: Sequence[float],
    count_b: int,
    mean_b: Sequence[float],
    m2_b: Sequence[float],
) -> tuple[int, list[float], list[float]]:
    if count_a == 0:
        return count_b, list(mean_b), list(m2_b)
    if count_b == 0:
        return count_a, list(mean_a), list(m2_a)
    total = count_a + count_b
    means: list[float] = []
    m2_values: list[float] = []
    for left_mean, left_m2, right_mean, right_m2 in zip(mean_a, m2_a, mean_b, m2_b, strict=True):
        delta = right_mean - left_mean
        means.append(left_mean + delta * count_b / total)
        m2_values.append(left_m2 + right_m2 + delta * delta * count_a * count_b / total)
    return total, means, m2_values


def _statistic_shard(
    request: PreprocessingBuildRequest,
    plan: _Plan,
    anchors: Mapping[str, str],
    quantiles: Mapping[str, Mapping[str, float]],
    connection: sqlite3.Connection,
    index: int,
) -> None:
    np = _dependency("numpy")

    canonical_path, assignment_path, _, _ = _input_pair(request, plan, index)
    continuous = list(plan.policy.features.continuous_columns)
    categorical = list(plan.policy.features.categorical_columns)
    lower = np.asarray([quantiles[name]["lower"] for name in continuous], dtype=np.float64)
    upper = np.asarray([quantiles[name]["upper"] for name in continuous], dtype=np.float64)
    count = 0
    mean = [0.0] * len(continuous)
    m2 = [0.0] * len(continuous)
    below = [0] * len(continuous)
    above = [0] * len(continuous)
    categories = {name: set[str]() for name in categorical}
    for canonical_batch, assignment_batch in _aligned_batches(
        canonical_path,
        assignment_path,
        ["record_id", *continuous, *categorical],
        ["record_id", "split"],
        request.parquet_batch_size,
    ):
        canonical_values = canonical_batch.to_pydict()
        assignment_values = assignment_batch.to_pydict()
        if canonical_values["record_id"] != assignment_values["record_id"]:
            raise DataContractError("Canonical and assignment record order differs")
        eligible = [
            position
            for position, (record_id, split) in enumerate(
                zip(canonical_values["record_id"], assignment_values["split"], strict=True)
            )
            if split == "train" and record_id not in anchors
        ]
        if not eligible:
            continue
        matrix = np.column_stack(
            [np.asarray(canonical_values[name], dtype=np.float64)[eligible] for name in continuous]
        )
        if not np.isfinite(matrix).all():
            raise DataContractError("Non-finite canonical feature reached Phase 3C")
        below = [
            left + int(value)
            for left, value in zip(below, (matrix < lower).sum(axis=0), strict=True)
        ]
        above = [
            left + int(value)
            for left, value in zip(above, (matrix > upper).sum(axis=0), strict=True)
        ]
        clipped = np.clip(matrix, lower, upper)
        batch_count = clipped.shape[0]
        batch_mean = clipped.mean(axis=0).tolist()
        batch_m2 = ((clipped - np.asarray(batch_mean)) ** 2).sum(axis=0).tolist()
        count, mean, m2 = _merge_moments(count, mean, m2, batch_count, batch_mean, batch_m2)
        for name in categorical:
            categories[name].update(
                cast(str, canonical_values[name][position]) for position in eligible
            )
    report = {
        "training_rows": count,
        "mean": mean,
        "m2": m2,
        "below_clip": below,
        "above_clip": above,
        "categories": {name: sorted(values) for name, values in categories.items()},
    }
    with connection:
        connection.execute(
            "INSERT INTO statistic_shards VALUES (?, ?, ?)",
            (index, count, canonical_json(report)),
        )


def _ordered_vocabulary(
    policy: DatasetPreparationPolicy, observed: Mapping[str, set[str]]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for rule in policy.features.category_rules:
        domain: list[str] = []
        if rule.zero_is_absent:
            domain.append(ABSENT_CATEGORY)
        domain.extend(rule.allowed_text_tokens)
        domain.extend(str(value) for value in rule.allowed_numeric_codes)
        unknown = sorted(observed[rule.column] - set(domain))
        if unknown:
            raise DataContractError(f"Canonical category escaped the frozen domain: {rule.column}")
        result[rule.column] = [value for value in domain if value in observed[rule.column]]
        result[rule.column].append(UNKNOWN_CATEGORY)
    return result


def _feature_coordinates(
    policy: DatasetPreparationPolicy, vocabulary: Mapping[str, Sequence[str]]
) -> list[dict[str, object]]:
    continuous = set(policy.features.continuous_columns)
    coordinates: list[dict[str, object]] = []
    for source_feature in policy.features.prepared_feature_columns:
        if source_feature in continuous:
            coordinates.append(
                {
                    "coordinate": f"x{len(coordinates):04d}",
                    "source_feature": source_feature,
                    "kind": "continuous_standardized",
                    "category": None,
                    "semantic_group": source_feature,
                }
            )
        else:
            for category in vocabulary[source_feature]:
                coordinates.append(
                    {
                        "coordinate": f"x{len(coordinates):04d}",
                        "source_feature": source_feature,
                        "kind": "one_hot",
                        "category": category,
                        "semantic_group": source_feature,
                    }
                )
    return coordinates


def _fit_preprocessor(
    request: PreprocessingBuildRequest,
    plan: _Plan,
    quantiles: Mapping[str, Mapping[str, float]],
    connection: sqlite3.Connection,
) -> dict[str, object]:
    existing = connection.execute("SELECT value FROM metadata WHERE key='preprocessor'").fetchone()
    if existing is not None:
        return cast(dict[str, object], json.loads(cast(str, existing[0])))
    continuous = list(plan.policy.features.continuous_columns)
    count = 0
    mean = [0.0] * len(continuous)
    m2 = [0.0] * len(continuous)
    below = [0] * len(continuous)
    above = [0] * len(continuous)
    observed = {name: set[str]() for name in plan.policy.features.categorical_columns}
    for (payload,) in connection.execute(
        "SELECT report_json FROM statistic_shards ORDER BY shard_index"
    ):
        report = cast(dict[str, object], json.loads(cast(str, payload)))
        count, mean, m2 = _merge_moments(
            count,
            mean,
            m2,
            cast(int, report["training_rows"]),
            cast(list[float], report["mean"]),
            cast(list[float], report["m2"]),
        )
        below = [
            left + right
            for left, right in zip(below, cast(list[int], report["below_clip"]), strict=True)
        ]
        above = [
            left + right
            for left, right in zip(above, cast(list[int], report["above_clip"]), strict=True)
        ]
        for name, values in cast(dict[str, list[str]], report["categories"]).items():
            observed[name].update(values)
    expected = cast(int, plan.split["federated_train_records_after_anchor_removal"])
    if count != expected:
        raise DataContractError(
            f"Phase 3C fit coverage mismatch: expected={expected}, actual={count}"
        )
    standard_deviation = [math.sqrt(max(value / count, 0.0)) for value in m2]
    scale = [
        value if value > request.standard_deviation_epsilon else 1.0 for value in standard_deviation
    ]
    vocabulary = _ordered_vocabulary(plan.policy, observed)
    coordinates = _feature_coordinates(plan.policy, vocabulary)
    family_labels = sorted(cast(dict[str, object], plan.split["family_split_counts"]))
    fine_labels = sorted(cast(dict[str, object], plan.split["fine_split_counts"]))
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "phase3c_plan_id": plan.report["plan_id"],
        "canonical_corpus_id": plan.report["canonical_corpus_id"],
        "split_id": plan.report["split_id"],
        "anchor_id": plan.report["anchor_id"],
        "preparation_policy_id": plan.policy.policy_id,
        "fit_partition": "train_after_anchor_removal",
        "fit_records": count,
        "quantile_method": plan.report["quantile_method"],
        "quantile_sample_records": cast(
            int,
            connection.execute("SELECT SUM(sampled_rows) FROM sample_shards").fetchone()[0],
        ),
        "lower_quantile": request.lower_quantile,
        "upper_quantile": request.upper_quantile,
        "continuous_features": {
            name: {
                "clip_lower": quantiles[name]["lower"],
                "clip_upper": quantiles[name]["upper"],
                "mean_after_clipping": mean[index],
                "population_standard_deviation_after_clipping": standard_deviation[index],
                "scale": scale[index],
                "below_clip_count": below[index],
                "above_clip_count": above[index],
            }
            for index, name in enumerate(continuous)
        },
        "categorical_vocabulary": vocabulary,
        "coordinates": coordinates,
        "model_input_dimension": len(coordinates),
        "family_label_to_id": {label: index for index, label in enumerate(family_labels)},
        "fine_label_to_id": {label: index for index, label in enumerate(fine_labels)},
        "partition_values": list(_PARTITIONS),
    }
    artifact: dict[str, object] = {"preprocessor_id": run_id(stable, length=64), **stable}
    path = request.output_directory / "preprocessor.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with connection:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('preprocessor', ?)",
            (canonical_json(artifact),),
        )
    return artifact


def _transformed_schema(preprocessor: Mapping[str, object]) -> object:
    pa = _dependency("pyarrow")

    coordinates = cast(list[dict[str, object]], preprocessor["coordinates"])
    return pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("family_id", pa.int16(), nullable=False),
            pa.field("fine_id", pa.int16(), nullable=False),
            pa.field("fine_label_ambiguous", pa.bool_(), nullable=False),
            *(
                pa.field(cast(str, item["coordinate"]), pa.float32(), nullable=False)
                for item in coordinates
            ),
        ]
    )


def _transform_shard(
    request: PreprocessingBuildRequest,
    plan: _Plan,
    anchors: Mapping[str, str],
    preprocessor: Mapping[str, object],
    connection: sqlite3.Connection,
    index: int,
) -> None:
    np = _dependency("numpy")
    pa = _dependency("pyarrow")
    pq = _dependency("pyarrow.parquet")

    canonical_path, assignment_path, _, _ = _input_pair(request, plan, index)
    continuous = list(plan.policy.features.continuous_columns)
    categorical = list(plan.policy.features.categorical_columns)
    continuous_artifact = cast(dict[str, dict[str, float]], preprocessor["continuous_features"])
    lower = np.asarray([continuous_artifact[name]["clip_lower"] for name in continuous])
    upper = np.asarray([continuous_artifact[name]["clip_upper"] for name in continuous])
    mean = np.asarray([continuous_artifact[name]["mean_after_clipping"] for name in continuous])
    scale = np.asarray([continuous_artifact[name]["scale"] for name in continuous])
    vocabulary = cast(dict[str, list[str]], preprocessor["categorical_vocabulary"])
    family_to_id = cast(dict[str, int], preprocessor["family_label_to_id"])
    fine_to_id = cast(dict[str, int], preprocessor["fine_label_to_id"])
    schema = _transformed_schema(preprocessor)
    output_directory = request.output_directory / "shards"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_relative = f"shards/{index:04d}-preprocessed.parquet"
    output_path = request.output_directory / output_relative
    partial = output_path.with_suffix(".parquet.partial")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, schema, compression="zstd", use_dictionary=True)
    rows = 0
    partition_counts: Counter[str] = Counter()
    try:
        for canonical_batch, assignment_batch in _aligned_batches(
            canonical_path,
            assignment_path,
            ["record_id", *continuous, *categorical],
            ["record_id", "split", "family_label", "fine_label", "fine_label_ambiguous"],
            request.parquet_batch_size,
        ):
            canonical_values = canonical_batch.to_pydict()
            assignment_values = assignment_batch.to_pydict()
            record_ids = cast(list[str], canonical_values["record_id"])
            if record_ids != assignment_values["record_id"]:
                raise DataContractError("Canonical and assignment record order differs")
            matrix = np.column_stack(
                [np.asarray(canonical_values[name], dtype=np.float64) for name in continuous]
            )
            transformed = ((np.clip(matrix, lower, upper) - mean) / scale).astype(np.float32)
            if not np.isfinite(transformed).all():
                raise DataContractError("Non-finite standardized feature was produced")
            feature_arrays: list[object] = [
                pa.array(transformed[:, position], type=pa.float32())
                for position in range(len(continuous))
            ]
            continuous_by_name = dict(zip(continuous, feature_arrays, strict=True))
            categorical_by_name: dict[str, list[object]] = {}
            for name in categorical:
                values = np.asarray(canonical_values[name], dtype=object)
                known = vocabulary[name][:-1]
                categorical_by_name[name] = [
                    pa.array(values == category, type=pa.float32()) for category in known
                ]
                categorical_by_name[name].append(
                    pa.array(~np.isin(values, known), type=pa.float32())
                )
            output_features: list[object] = []
            categorical_offsets = dict.fromkeys(categorical, 0)
            for coordinate in cast(list[dict[str, object]], preprocessor["coordinates"]):
                source = cast(str, coordinate["source_feature"])
                if coordinate["kind"] == "continuous_standardized":
                    output_features.append(continuous_by_name[source])
                else:
                    offset = categorical_offsets[source]
                    output_features.append(categorical_by_name[source][offset])
                    categorical_offsets[source] += 1
            partitions = []
            for record_id, split in zip(record_ids, assignment_values["split"], strict=True):
                subset = anchors.get(record_id)
                partition = f"anchor_{subset}" if subset is not None else split
                partitions.append(partition)
                partition_counts[partition] += 1
            arrays = [
                pa.array(record_ids, type=pa.string()),
                pa.array(partitions, type=pa.string()),
                pa.array(
                    [family_to_id[value] for value in assignment_values["family_label"]],
                    type=pa.int16(),
                ),
                pa.array(
                    [fine_to_id[value] for value in assignment_values["fine_label"]],
                    type=pa.int16(),
                ),
                pa.array(assignment_values["fine_label_ambiguous"], type=pa.bool_()),
                *output_features,
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            rows += len(record_ids)
    finally:
        writer.close()
    expected = cast(list[dict[str, object]], plan.canonical["shards"])[index]["row_count"]
    if rows != expected:
        partial.unlink(missing_ok=True)
        raise DataContractError("Transformed shard cardinality mismatch")
    partial.replace(output_path)
    report = {"rows": rows, "partition_counts": dict(sorted(partition_counts.items()))}
    digest = _file_sha256(output_path)
    with connection:
        connection.execute(
            "INSERT INTO transformed_shards VALUES (?, ?, ?, ?, ?, ?)",
            (
                index,
                output_relative,
                output_path.stat().st_size,
                digest,
                rows,
                canonical_json(report),
            ),
        )


def _finalize(
    request: PreprocessingBuildRequest,
    plan: _Plan,
    preprocessor: Mapping[str, object],
    connection: sqlite3.Connection,
) -> dict[str, object]:
    rows = 0
    partitions: Counter[str] = Counter()
    files: list[dict[str, object]] = []
    for relative, size, digest, output_rows, payload in connection.execute(
        "SELECT output_file, output_size, output_sha256, output_rows, report_json "
        "FROM transformed_shards ORDER BY shard_index"
    ):
        rows += cast(int, output_rows)
        shard_report = cast(dict[str, object], json.loads(cast(str, payload)))
        partitions.update(cast(dict[str, int], shard_report["partition_counts"]))
        files.append(
            {
                "path": relative,
                "byte_size": size,
                "sha256": digest,
                "row_count": output_rows,
            }
        )
    expected_rows = cast(int, plan.split["records"])
    expected_partitions = {
        "train": cast(int, plan.split["federated_train_records_after_anchor_removal"]),
        "validation": cast(dict[str, int], plan.split["split_counts"])["validation"],
        "test": cast(dict[str, int], plan.split["split_counts"])["test"],
        "anchor_calibration": cast(dict[str, int], plan.anchor["subset_counts"])["calibration"],
        "anchor_scoring": cast(dict[str, int], plan.anchor["subset_counts"])["scoring"],
    }
    if rows != expected_rows or dict(partitions) != expected_partitions:
        raise DataContractError("Final Phase 3C partition cardinality mismatch")
    preprocessor_path = request.output_directory / "preprocessor.json"
    stable: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset": request.dataset.value,
        "phase3c_plan_id": plan.report["plan_id"],
        "canonical_corpus_id": plan.report["canonical_corpus_id"],
        "split_id": plan.report["split_id"],
        "anchor_id": plan.report["anchor_id"],
        "preprocessor": {
            "preprocessor_id": preprocessor["preprocessor_id"],
            "path": preprocessor_path.name,
            "byte_size": preprocessor_path.stat().st_size,
            "sha256": _file_sha256(preprocessor_path),
            "model_input_dimension": preprocessor["model_input_dimension"],
        },
        "records": rows,
        "partition_counts": expected_partitions,
        "transformed_files": files,
        "gates": {
            "fit_excludes_validation_test_and_anchors": True,
            "fit_cardinality_matches_federated_training_pool": True,
            "transformed_cardinality_matches_canonical": True,
            "partition_cardinality_matches_phase3b": True,
            "stable_feature_order_and_dimension": True,
            "no_nonfinite_model_input": True,
            "preprocessing_fitted": True,
            "client_partition_created": False,
        },
    }
    report: dict[str, object] = {
        "preprocessed_corpus_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        **stable,
    }
    json_path = request.output_directory / f"{request.dataset.value}_phase3c_preprocessed.json"
    md_path = request.output_directory / f"{request.dataset.value}_phase3c_preprocessed.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        "\n".join(
            (
                f"# Phase 3C preprocessed corpus: {request.dataset.value}",
                "",
                f"- Preprocessed corpus ID: `{report['preprocessed_corpus_id']}`",
                f"- Preprocessor ID: `{preprocessor['preprocessor_id']}`",
                f"- Records: {rows:,}",
                f"- Model input dimension: {preprocessor['model_input_dimension']}",
                f"- Quantile sample records: {preprocessor['quantile_sample_records']:,}",
                "- Fitted only on training after anchor removal: yes",
                "- Client partitions created: no",
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


def build_phase3c_preprocessing(
    request: PreprocessingBuildRequest, *, progress_every: int = 1
) -> dict[str, object]:
    """Fit and apply a resumable Phase 3C preprocessing artifact."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    plan = _plan(request)
    _write_plan(plan.report, request.output_directory)
    canonical_shards = cast(list[dict[str, object]], plan.canonical["shards"])
    assignment_files = cast(list[dict[str, object]], plan.split["assignment_files"])
    identity = canonical_json(
        {
            "schema_version": _SCHEMA_VERSION,
            "plan_id": plan.report["plan_id"],
            "canonical_shards": canonical_shards,
            "assignment_files": assignment_files,
        }
    )
    connection = _open_database(request, identity)
    try:
        anchors = _anchor_subsets(request, plan)
        completed_samples = _completed_files(connection, request, "sample_shards")
        for index in range(len(canonical_shards)):
            if index not in completed_samples:
                _sample_shard(request, plan, anchors, connection, index)
            if (index + 1) % progress_every == 0 or index + 1 == len(canonical_shards):
                print(f"phase3c_sample={index + 1}/{len(canonical_shards)}", flush=True)
        quantiles = _fit_quantiles(request, plan, connection)
        completed_statistics = {
            cast(int, row[0])
            for row in connection.execute("SELECT shard_index FROM statistic_shards")
        }
        for index in range(len(canonical_shards)):
            if index not in completed_statistics:
                _statistic_shard(request, plan, anchors, quantiles, connection, index)
            if (index + 1) % progress_every == 0 or index + 1 == len(canonical_shards):
                print(f"phase3c_statistics={index + 1}/{len(canonical_shards)}", flush=True)
        preprocessor = _fit_preprocessor(request, plan, quantiles, connection)
        completed_transforms = _completed_files(connection, request, "transformed_shards")
        for index in range(len(canonical_shards)):
            if index not in completed_transforms:
                _transform_shard(request, plan, anchors, preprocessor, connection, index)
            if (index + 1) % progress_every == 0 or index + 1 == len(canonical_shards):
                print(f"phase3c_transform={index + 1}/{len(canonical_shards)}", flush=True)
        return _finalize(request, plan, preprocessor, connection)
    finally:
        connection.close()
