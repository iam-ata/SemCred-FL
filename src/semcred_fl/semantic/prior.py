"""Anchor-derived relevance, normalization, and prior fusion for Phase 7."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from semcred_fl.data.training_corpus import PreprocessedCorpus
from semcred_fl.identity import run_id
from semcred_fl.semantic.contracts import SemanticPriorError


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise SemanticPriorError("Phase 7 requires the project ML dependencies") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticPriorError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticPriorError(f"{description} must be an object")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class FeatureGroupLayout:
    """Exact mapping from semantic groups to transformed model coordinates."""

    dataset: str
    preprocessor_id: str
    group_names: tuple[str, ...]
    coordinate_indices: tuple[tuple[int, ...], ...]
    coordinate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dataset or not self.preprocessor_id or len(self.preprocessor_id) < 8:
            raise SemanticPriorError("Feature-group layout identity is incomplete")
        if not self.group_names or len(self.group_names) != len(set(self.group_names)):
            raise SemanticPriorError("Semantic group names must be nonempty and unique")
        if len(self.group_names) != len(self.coordinate_indices):
            raise SemanticPriorError("Semantic groups and coordinate-index rows differ")
        flattened = tuple(index for row in self.coordinate_indices for index in row)
        if any(not row for row in self.coordinate_indices):
            raise SemanticPriorError("Every semantic group requires a model coordinate")
        if sorted(flattened) != list(range(len(self.coordinate_names))):
            raise SemanticPriorError("Semantic groups must partition every model coordinate once")

    @property
    def layout_id(self) -> str:
        """Return the deterministic identity of the feature-group layout."""

        return run_id(self, length=64)


def load_feature_group_layout(corpus: PreprocessedCorpus) -> FeatureGroupLayout:
    """Load the Phase 3C semantic-group partition in model-coordinate order."""

    report = _json(corpus.report_path, "Phase 3C report")
    reference = report.get("preprocessor")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise SemanticPriorError("Phase 3C report has no preprocessor reference")
    preprocessor_path = (corpus.report_path.parent / cast(str, reference["path"])).resolve()
    preprocessor = _json(preprocessor_path, "Phase 3C preprocessor")
    if (
        preprocessor.get("dataset") != corpus.dataset
        or preprocessor.get("preprocessor_id") != corpus.preprocessor_id
        or preprocessor.get("model_input_dimension") != corpus.input_dimension
    ):
        raise SemanticPriorError("Phase 3C preprocessor differs from the selected corpus")
    coordinates_value = preprocessor.get("coordinates")
    if not isinstance(coordinates_value, list) or len(coordinates_value) != corpus.input_dimension:
        raise SemanticPriorError("Phase 3C preprocessor has no complete coordinate layout")
    group_order: list[str] = []
    grouped: dict[str, list[int]] = {}
    coordinate_names: list[str] = []
    for index, item_value in enumerate(coordinates_value):
        if not isinstance(item_value, dict):
            raise SemanticPriorError("Preprocessor coordinate entry is malformed")
        item = cast(dict[str, object], item_value)
        coordinate = item.get("coordinate")
        group = item.get("semantic_group")
        if coordinate != f"x{index:04d}" or not isinstance(group, str) or not group:
            raise SemanticPriorError("Preprocessor coordinate order or semantic group is invalid")
        coordinate_names.append(coordinate)
        if group not in grouped:
            group_order.append(group)
            grouped[group] = []
        grouped[group].append(index)
    return FeatureGroupLayout(
        dataset=corpus.dataset,
        preprocessor_id=corpus.preprocessor_id,
        group_names=tuple(group_order),
        coordinate_indices=tuple(tuple(grouped[name]) for name in group_order),
        coordinate_names=tuple(coordinate_names),
    )


@dataclass(frozen=True, slots=True)
class AnchorDataPrior:
    """Class-specific empirical relevance computed only from calibration anchors."""

    dataset: str
    preprocessed_corpus_id: str
    preprocessor_id: str
    anchor_id: str
    anchor_manifest_sha256: str
    anchor_file_sha256: str
    calibration_record_ids_sha256: str
    feature_group_layout_id: str
    class_names: tuple[str, ...]
    feature_group_names: tuple[str, ...]
    class_counts: tuple[int, ...]
    scores: tuple[tuple[float, ...], ...]
    epsilon: float
    partition: str = "anchor_calibration"
    validation_partition_accessed: bool = False
    test_partition_accessed: bool = False
    scoring_anchor_accessed: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.preprocessed_corpus_id,
            self.preprocessor_id,
            self.anchor_id,
            self.anchor_manifest_sha256,
            self.anchor_file_sha256,
            self.calibration_record_ids_sha256,
            self.feature_group_layout_id,
        )
        if any(len(value) != 64 for value in identities):
            raise SemanticPriorError("Anchor-data prior requires full SHA-256 identities")
        if self.partition != "anchor_calibration" or self.epsilon <= 0.0:
            raise SemanticPriorError("Anchor-data prior partition or epsilon is invalid")
        if len(self.class_counts) != len(self.class_names) or any(
            count < 1 for count in self.class_counts
        ):
            raise SemanticPriorError("Anchor-data prior class counts are invalid")
        if len(self.scores) != len(self.class_names):
            raise SemanticPriorError("Anchor-data prior class dimensions differ")
        for row in self.scores:
            if len(row) != len(self.feature_group_names) or any(
                not math.isfinite(value) or value < 0.0 for value in row
            ):
                raise SemanticPriorError("Anchor-data prior scores are invalid")
        if (
            self.validation_partition_accessed
            or self.test_partition_accessed
            or self.scoring_anchor_accessed
        ):
            raise SemanticPriorError("Anchor-data prior crossed its calibration-only boundary")

    @property
    def anchor_prior_id(self) -> str:
        """Return the deterministic identity of the empirical prior."""

        return run_id(self, length=64)


def _anchor_manifest(corpus: PreprocessedCorpus, path: Path) -> tuple[dict[str, object], Path, str]:
    manifest = _json(path, "Phase 3B anchor manifest")
    if (
        manifest.get("dataset") != corpus.dataset
        or manifest.get("anchor_id") != corpus.anchor_id
        or manifest.get("canonical_corpus_id") is None
    ):
        raise SemanticPriorError("Anchor manifest differs from the selected corpus")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or not all(value is True for value in gates.values()):
        raise SemanticPriorError("Required Phase 3B anchor gates did not pass")
    file_value = manifest.get("file")
    if not isinstance(file_value, dict):
        raise SemanticPriorError("Anchor manifest has no file inventory")
    relative = file_value.get("path")
    digest = file_value.get("sha256")
    size = file_value.get("byte_size")
    if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
        raise SemanticPriorError("Anchor file inventory is malformed")
    anchor_file = (path.parent / relative).resolve()
    if (
        not anchor_file.is_file()
        or anchor_file.stat().st_size != size
        or _sha256(anchor_file) != digest
    ):
        raise SemanticPriorError("Anchor file is missing or checksum-invalid")
    return manifest, anchor_file, _sha256(path)


def compute_anchor_data_prior(
    corpus: PreprocessedCorpus,
    anchor_manifest_path: Path,
    *,
    epsilon: float = 1e-12,
) -> AnchorDataPrior:
    """Compute the manuscript RMS standardized-mean prior from calibration anchors."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    np = _dependency("numpy")
    pq = _dependency("pyarrow.parquet")
    layout = load_feature_group_layout(corpus)
    manifest, anchor_file, manifest_sha = _anchor_manifest(corpus, anchor_manifest_path)
    raw = pq.read_table(
        anchor_file,
        columns=["record_id", "family_label", "subset"],
        filters=[("subset", "=", "calibration")],
    )
    record_ids = tuple(str(value) for value in raw.column("record_id").to_pylist())
    family_labels = tuple(str(value) for value in raw.column("family_label").to_pylist())
    if not record_ids or len(record_ids) != len(set(record_ids)):
        raise SemanticPriorError("Calibration anchor record IDs are empty or duplicated")
    expected_labels = dict(zip(record_ids, family_labels, strict=True))
    unknown_labels = set(family_labels) - set(corpus.family_label_to_id)
    if unknown_labels:
        raise SemanticPriorError("Calibration anchors contain an unknown primary family")
    expected_ids = set(record_ids)
    observed_ids: set[str] = set()
    feature_parts: list[Any] = []
    target_parts: list[Any] = []
    columns = ["record_id", "partition", "family_id", *layout.coordinate_names]
    for corpus_file in corpus.files:
        table = pq.read_table(
            corpus_file.path,
            columns=columns,
            filters=[("partition", "=", "anchor_calibration")],
        )
        if table.num_rows == 0:
            continue
        if _sha256(corpus_file.path) != corpus_file.sha256:
            raise SemanticPriorError("Calibration-anchor shard checksum is invalid")
        ids = tuple(str(value) for value in table.column("record_id").to_pylist())
        if set(ids) - expected_ids or observed_ids.intersection(ids):
            raise SemanticPriorError("Transformed calibration anchors differ from their manifest")
        targets = (
            table.column("family_id").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
        for record_id, class_id in zip(ids, targets, strict=True):
            if corpus.family_label_to_id[expected_labels[record_id]] != int(class_id):
                raise SemanticPriorError("Anchor family label differs after preprocessing")
        features = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in layout.coordinate_names]
        ).astype(np.float64, copy=False)
        if not np.isfinite(features).all():
            raise SemanticPriorError("Non-finite calibration feature reached Phase 7")
        observed_ids.update(ids)
        feature_parts.append(features)
        target_parts.append(targets)
    if observed_ids != expected_ids:
        raise SemanticPriorError("Not every manifested calibration anchor was transformed")
    matrix = np.concatenate(feature_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    standard_deviation = np.std(matrix, axis=0, ddof=0)
    ordered_classes = tuple(
        label for label, _ in sorted(corpus.family_label_to_id.items(), key=lambda item: item[1])
    )
    class_counts: list[int] = []
    score_rows: list[tuple[float, ...]] = []
    for class_id, _ in enumerate(ordered_classes):
        positive = targets == class_id
        negative = ~positive
        count = int(np.count_nonzero(positive))
        if count < 1 or not np.any(negative):
            raise SemanticPriorError("Calibration anchors do not support one-vs-rest relevance")
        class_counts.append(count)
        difference = np.mean(matrix[positive], axis=0) - np.mean(matrix[negative], axis=0)
        standardized = difference / (standard_deviation + epsilon)
        group_scores = tuple(
            float(np.sqrt(np.mean(np.square(standardized[list(indices)]))))
            for indices in layout.coordinate_indices
        )
        score_rows.append(group_scores)
    expected_count = corpus.partition_counts["anchor_calibration"]
    if matrix.shape != (expected_count, corpus.input_dimension):
        raise SemanticPriorError("Calibration-anchor cardinality or dimension is inconsistent")
    id_digest = hashlib.sha256("\n".join(sorted(record_ids)).encode("utf-8")).hexdigest()
    return AnchorDataPrior(
        dataset=corpus.dataset,
        preprocessed_corpus_id=corpus.preprocessed_corpus_id,
        preprocessor_id=corpus.preprocessor_id,
        anchor_id=corpus.anchor_id,
        anchor_manifest_sha256=manifest_sha,
        anchor_file_sha256=_sha256(anchor_file),
        calibration_record_ids_sha256=id_digest,
        feature_group_layout_id=layout.layout_id,
        class_names=ordered_classes,
        feature_group_names=layout.group_names,
        class_counts=tuple(class_counts),
        scores=tuple(score_rows),
        epsilon=epsilon,
    )


def normalize_prior(values: tuple[float, ...], *, epsilon: float) -> tuple[float, ...]:
    """Apply the manuscript L1-plus-epsilon normalization to one relevance vector."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Prior values must be finite, nonnegative, and nonempty")
    denominator = sum(values) + epsilon
    return tuple(value / denominator for value in values)


def fuse_priors(
    llm_prior: tuple[float, ...],
    data_prior: tuple[float, ...],
    *,
    alpha_l: float,
) -> tuple[float, ...]:
    """Fuse normalized prior endpoints while preserving exact alpha endpoints."""

    if len(llm_prior) != len(data_prior) or not llm_prior:
        raise ValueError("Prior endpoints must have the same nonzero dimension")
    if not math.isfinite(alpha_l) or not 0.0 <= alpha_l <= 1.0:
        raise ValueError("alpha_l must be finite and in [0,1]")
    values = (*llm_prior, *data_prior)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Prior endpoints must be finite and nonnegative")
    if alpha_l == 0.0:
        return data_prior
    if alpha_l == 1.0:
        return llm_prior
    return tuple(
        alpha_l * llm_value + (1.0 - alpha_l) * data_value
        for llm_value, data_value in zip(llm_prior, data_prior, strict=True)
    )
