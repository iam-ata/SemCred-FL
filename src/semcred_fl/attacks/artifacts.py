"""Typed immutable artifacts produced by the Phase 6B train-only calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from semcred_fl.identity import canonical_json, run_id


class AttackArtifactError(ValueError):
    """Raised when a Phase 6B artifact is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ResolvedCoordinate:
    """One Phase 3C continuous coordinate and its train-derived statistics."""

    coordinate: str
    index: int
    source_feature: str
    lower_bound: float
    upper_bound: float
    source_median: float
    benign_median: float
    absolute_median_gap: float

    def __post_init__(self) -> None:
        values = (
            self.lower_bound,
            self.upper_bound,
            self.source_median,
            self.benign_median,
            self.absolute_median_gap,
        )
        if not self.coordinate or self.index < 0 or not self.source_feature:
            raise AttackArtifactError("Resolved coordinate identity is invalid")
        if any(not math.isfinite(value) for value in values):
            raise AttackArtifactError("Resolved coordinate values must be finite")
        if self.lower_bound >= self.upper_bound:
            raise AttackArtifactError("Resolved coordinate must have a positive clipping range")
        if not math.isclose(
            self.absolute_median_gap,
            abs(self.source_median - self.benign_median),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AttackArtifactError("Resolved coordinate median gap is inconsistent")


@dataclass(frozen=True, slots=True)
class TriggerCoordinate:
    """One coordinate/value pair in the resolved zero-collision trigger."""

    coordinate: str
    index: int
    source_feature: str
    endpoint: str
    value: float

    def __post_init__(self) -> None:
        if self.endpoint not in {"lower", "upper"}:
            raise AttackArtifactError("Trigger endpoint must be lower or upper")
        if not self.coordinate or self.index < 0 or not self.source_feature:
            raise AttackArtifactError("Trigger coordinate identity is invalid")
        if not math.isfinite(self.value):
            raise AttackArtifactError("Trigger value must be finite")


@dataclass(frozen=True, slots=True)
class Phase6BResolution:
    """Feature-poisoning and backdoor values resolved without evaluation data."""

    dataset: str
    phase6a_contract_id: str
    preprocessed_corpus_id: str
    preprocessor_id: str
    split_id: str
    anchor_id: str
    client_partition_id: str
    source_family: str
    source_family_id: int
    target_family: str
    target_family_id: int
    source_training_records: int
    target_training_records: int
    federated_training_records: int
    ranked_coordinates: tuple[ResolvedCoordinate, ...]
    feature_coordinates: tuple[ResolvedCoordinate, ...]
    trigger_coordinates: tuple[TriggerCoordinate, ...]
    trigger_training_collisions: int
    resolution_partition: str = "federated_train_after_anchor_removal"
    test_partition_accessed: bool = False
    validation_partition_accessed: bool = False
    anchor_records_accessed: bool = False
    performance_claims_allowed: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.phase6a_contract_id,
            self.preprocessed_corpus_id,
            self.preprocessor_id,
            self.split_id,
            self.anchor_id,
            self.client_partition_id,
        )
        if any(len(value) != 64 for value in identities):
            raise AttackArtifactError("Phase 6B inputs require full SHA-256 identities")
        if self.source_family_id < 1 or self.target_family != "benign":
            raise AttackArtifactError("Phase 6B targeted-family mapping is invalid")
        if self.source_training_records < 1 or self.target_training_records < 1:
            raise AttackArtifactError("Phase 6B source and benign populations must be nonempty")
        if self.federated_training_records < (
            self.source_training_records + self.target_training_records
        ):
            raise AttackArtifactError("Phase 6B training cardinalities are inconsistent")
        if len(self.feature_coordinates) != 3:
            raise AttackArtifactError("Clean-label poisoning requires exactly three coordinates")
        if not 3 <= len(self.trigger_coordinates) <= 6:
            raise AttackArtifactError("Backdoor trigger requires three to six coordinates")
        ranked_names = tuple(item.coordinate for item in self.ranked_coordinates)
        if len(ranked_names) != len(set(ranked_names)):
            raise AttackArtifactError("Ranked Phase 6B coordinates must be unique")
        if tuple(item.coordinate for item in self.feature_coordinates) != ranked_names[:3]:
            raise AttackArtifactError("Clean-label coordinates must be the three highest ranked")
        if any(item.coordinate not in set(ranked_names[:12]) for item in self.trigger_coordinates):
            raise AttackArtifactError("Trigger coordinates must come from the top-twelve pool")
        if self.trigger_training_collisions != 0:
            raise AttackArtifactError("The resolved trigger must have zero training collisions")
        if self.resolution_partition != "federated_train_after_anchor_removal":
            raise AttackArtifactError("Phase 6B resolution is training-only")
        if (
            self.test_partition_accessed
            or self.validation_partition_accessed
            or self.anchor_records_accessed
            or self.performance_claims_allowed
        ):
            raise AttackArtifactError("Phase 6B resolution crossed its evidence boundary")

    @property
    def resolution_id(self) -> str:
        """Return the deterministic identity of this complete resolution."""

        return run_id(self, length=64)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise AttackArtifactError("A norm reference cannot use an empty population")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


@dataclass(frozen=True, slots=True)
class BenignNormReference:
    """Frozen benign-only update-norm interval for one dataset and client profile."""

    dataset: str
    profile: str
    preprocessed_corpus_id: str
    client_partition_id: str
    detector_config_id: str
    local_training_config_id: str
    development_run_ids: tuple[str, ...]
    update_norms_sha256: str
    update_count: int
    lower_quantile: float
    upper_quantile: float
    lower_norm: float
    upper_norm: float
    reference_population: str = "benign_only_development_runs_same_dataset_model_and_partition"
    current_round_benign_updates_accessed: bool = False
    test_partition_accessed: bool = False
    anchor_records_accessed: bool = False
    performance_claims_allowed: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.preprocessed_corpus_id,
            self.client_partition_id,
            self.detector_config_id,
            self.local_training_config_id,
            self.update_norms_sha256,
        )
        if any(len(value) != 64 for value in identities):
            raise AttackArtifactError("Norm-reference identities must be full SHA-256 values")
        if self.update_count != len(self.development_run_ids) or self.update_count < 4:
            raise AttackArtifactError("Norm reference requires at least four identified updates")
        if not 0.0 < self.lower_quantile < self.upper_quantile < 1.0:
            raise AttackArtifactError("Norm-reference quantiles are invalid")
        if not 0.0 <= self.lower_norm <= self.upper_norm:
            raise AttackArtifactError("Norm-reference interval is invalid")
        if (
            self.current_round_benign_updates_accessed
            or self.test_partition_accessed
            or self.anchor_records_accessed
            or self.performance_claims_allowed
        ):
            raise AttackArtifactError("Norm reference crossed its evidence boundary")

    @property
    def reference_id(self) -> str:
        """Return the deterministic identity of this frozen norm reference."""

        return run_id(self, length=64)


def benign_norm_reference(
    *,
    dataset: str,
    profile: str,
    preprocessed_corpus_id: str,
    client_partition_id: str,
    detector_config_id: str,
    local_training_config_id: str,
    development_run_ids: Sequence[str],
    update_norms: Sequence[float],
) -> BenignNormReference:
    """Build an immutable interquartile reference from prior benign-only updates."""

    if len(development_run_ids) != len(update_norms):
        raise AttackArtifactError("Every benign update norm requires one development-run ID")
    if len(set(development_run_ids)) != len(development_run_ids):
        raise AttackArtifactError("Benign development-run IDs must be unique")
    if any(len(value) != 64 for value in development_run_ids):
        raise AttackArtifactError("Development-run IDs must be full SHA-256 values")
    if any(not math.isfinite(value) or value < 0.0 for value in update_norms):
        raise AttackArtifactError("Benign update norms must be finite and nonnegative")
    ordered_pairs = sorted(zip(development_run_ids, update_norms, strict=True))
    ordered_ids = tuple(item[0] for item in ordered_pairs)
    ordered_norms = tuple(float(item[1]) for item in ordered_pairs)
    sorted_norms = sorted(ordered_norms)
    digest = hashlib.sha256(canonical_json(list(ordered_norms)).encode("utf-8")).hexdigest()
    return BenignNormReference(
        dataset=dataset,
        profile=profile,
        preprocessed_corpus_id=preprocessed_corpus_id,
        client_partition_id=client_partition_id,
        detector_config_id=detector_config_id,
        local_training_config_id=local_training_config_id,
        development_run_ids=ordered_ids,
        update_norms_sha256=digest,
        update_count=len(ordered_norms),
        lower_quantile=0.25,
        upper_quantile=0.75,
        lower_norm=_linear_quantile(sorted_norms, 0.25),
        upper_norm=_linear_quantile(sorted_norms, 0.75),
    )


def _artifact_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttackArtifactError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise AttackArtifactError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _artifact_string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise AttackArtifactError(f"Attack artifact field {key} is missing")
    return value


def _artifact_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise AttackArtifactError(f"Attack artifact field {key} is invalid")
    return value


def _artifact_float(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float):
        raise AttackArtifactError(f"Attack artifact field {key} is invalid")
    return float(value)


def _artifact_bool(record: dict[str, object], key: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise AttackArtifactError(f"Attack artifact field {key} is invalid")
    return value


def _resolved_coordinate(value: object) -> ResolvedCoordinate:
    if not isinstance(value, dict):
        raise AttackArtifactError("Resolved-coordinate record is malformed")
    record = cast(dict[str, object], value)
    return ResolvedCoordinate(
        coordinate=_artifact_string(record, "coordinate"),
        index=_artifact_int(record, "index"),
        source_feature=_artifact_string(record, "source_feature"),
        lower_bound=_artifact_float(record, "lower_bound"),
        upper_bound=_artifact_float(record, "upper_bound"),
        source_median=_artifact_float(record, "source_median"),
        benign_median=_artifact_float(record, "benign_median"),
        absolute_median_gap=_artifact_float(record, "absolute_median_gap"),
    )


def _trigger_coordinate(value: object) -> TriggerCoordinate:
    if not isinstance(value, dict):
        raise AttackArtifactError("Trigger-coordinate record is malformed")
    record = cast(dict[str, object], value)
    return TriggerCoordinate(
        coordinate=_artifact_string(record, "coordinate"),
        index=_artifact_int(record, "index"),
        source_feature=_artifact_string(record, "source_feature"),
        endpoint=_artifact_string(record, "endpoint"),
        value=_artifact_float(record, "value"),
    )


def load_phase6b_resolution(path: Path) -> Phase6BResolution:
    """Load and independently validate one completed Phase 6B.1 resolution."""

    report = _artifact_json(path, "Phase 6B.1 resolution")
    gates = report.get("gates")
    required_true = (
        "source_and_benign_medians_exact_over_federated_train",
        "feature_coordinates_nonconstant_and_train_ranked",
        "trigger_from_top_twelve_coordinates",
        "trigger_has_zero_federated_training_collisions",
        "scanner_filter_excludes_validation_test_and_anchors",
    )
    if (
        not isinstance(gates, dict)
        or not all(gates.get(name) is True for name in required_true)
        or gates.get("test_partition_accessed") is not False
        or gates.get("performance_claims_allowed") is not False
    ):
        raise AttackArtifactError("Phase 6B.1 resolution gates did not pass")
    ranked_value = report.get("ranked_coordinates")
    feature_value = report.get("feature_coordinates")
    trigger_value = report.get("trigger_coordinates")
    if (
        not isinstance(ranked_value, list)
        or not isinstance(feature_value, list)
        or not isinstance(trigger_value, list)
    ):
        raise AttackArtifactError("Phase 6B.1 coordinate inventories are missing")
    resolution = Phase6BResolution(
        dataset=_artifact_string(report, "dataset"),
        phase6a_contract_id=_artifact_string(report, "phase6a_contract_id"),
        preprocessed_corpus_id=_artifact_string(report, "preprocessed_corpus_id"),
        preprocessor_id=_artifact_string(report, "preprocessor_id"),
        split_id=_artifact_string(report, "split_id"),
        anchor_id=_artifact_string(report, "anchor_id"),
        client_partition_id=_artifact_string(report, "client_partition_id"),
        source_family=_artifact_string(report, "source_family"),
        source_family_id=_artifact_int(report, "source_family_id"),
        target_family=_artifact_string(report, "target_family"),
        target_family_id=_artifact_int(report, "target_family_id"),
        source_training_records=_artifact_int(report, "source_training_records"),
        target_training_records=_artifact_int(report, "target_training_records"),
        federated_training_records=_artifact_int(report, "federated_training_records"),
        ranked_coordinates=tuple(_resolved_coordinate(value) for value in ranked_value),
        feature_coordinates=tuple(_resolved_coordinate(value) for value in feature_value),
        trigger_coordinates=tuple(_trigger_coordinate(value) for value in trigger_value),
        trigger_training_collisions=_artifact_int(report, "trigger_training_collisions"),
        resolution_partition=_artifact_string(report, "resolution_partition"),
        test_partition_accessed=_artifact_bool(report, "test_partition_accessed"),
        validation_partition_accessed=_artifact_bool(report, "validation_partition_accessed"),
        anchor_records_accessed=_artifact_bool(report, "anchor_records_accessed"),
        performance_claims_allowed=_artifact_bool(report, "performance_claims_allowed"),
    )
    if report.get("resolution_id") != resolution.resolution_id:
        raise AttackArtifactError("Phase 6B.1 resolution identity is invalid")
    return resolution


def load_benign_norm_reference(path: Path) -> BenignNormReference:
    """Load and validate the immutable reference nested in a Phase 6B.2 artifact."""

    artifact = _artifact_json(path, "Phase 6B.2 norm reference")
    calibration_id = _artifact_string(artifact, "calibration_id")
    stable_artifact = {
        key: value
        for key, value in artifact.items()
        if key not in {"calibration_id", "created_at_utc"}
    }
    if len(calibration_id) != 64 or run_id(stable_artifact, length=64) != calibration_id:
        raise AttackArtifactError("Phase 6B.2 calibration identity is invalid")
    gates = artifact.get("gates")
    required_true = (
        "all_scheduled_updates_accepted",
        "all_development_clients_honest",
        "cache_bound_to_schedule",
        "same_dataset_model_partition_and_training_contract",
        "reference_uses_prior_benign_development_updates_only",
    )
    required_false = (
        "current_round_benign_updates_accessed",
        "validation_partition_accessed",
        "test_partition_accessed",
        "anchor_records_accessed",
        "performance_claims_allowed",
    )
    if (
        not isinstance(gates, dict)
        or not all(gates.get(name) is True for name in required_true)
        or not all(gates.get(name) is False for name in required_false)
    ):
        raise AttackArtifactError("Phase 6B.2 calibration gates did not pass")
    reference_value = artifact.get("reference")
    if not isinstance(reference_value, dict):
        raise AttackArtifactError("Phase 6B.2 reference record is missing")
    record = cast(dict[str, object], reference_value)
    development_value = record.get("development_run_ids")
    if not isinstance(development_value, list) or not all(
        isinstance(value, str) for value in development_value
    ):
        raise AttackArtifactError("Phase 6B.2 development identities are malformed")
    reference = BenignNormReference(
        dataset=_artifact_string(record, "dataset"),
        profile=_artifact_string(record, "profile"),
        preprocessed_corpus_id=_artifact_string(record, "preprocessed_corpus_id"),
        client_partition_id=_artifact_string(record, "client_partition_id"),
        detector_config_id=_artifact_string(record, "detector_config_id"),
        local_training_config_id=_artifact_string(record, "local_training_config_id"),
        development_run_ids=tuple(cast(list[str], development_value)),
        update_norms_sha256=_artifact_string(record, "update_norms_sha256"),
        update_count=_artifact_int(record, "update_count"),
        lower_quantile=_artifact_float(record, "lower_quantile"),
        upper_quantile=_artifact_float(record, "upper_quantile"),
        lower_norm=_artifact_float(record, "lower_norm"),
        upper_norm=_artifact_float(record, "upper_norm"),
        reference_population=_artifact_string(record, "reference_population"),
        current_round_benign_updates_accessed=_artifact_bool(
            record, "current_round_benign_updates_accessed"
        ),
        test_partition_accessed=_artifact_bool(record, "test_partition_accessed"),
        anchor_records_accessed=_artifact_bool(record, "anchor_records_accessed"),
        performance_claims_allowed=_artifact_bool(record, "performance_claims_allowed"),
    )
    if record.get("reference_id") != reference.reference_id:
        raise AttackArtifactError("Phase 6B.2 reference identity is invalid")
    return reference
