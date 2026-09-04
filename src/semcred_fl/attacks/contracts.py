"""Frozen Phase 6A attack definitions.

This module defines experimental contracts only.  It deliberately does not expose
poisoning transformations; those are implemented after the train-only feature and
norm-reference artifacts have been resolved and hashed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semcred_fl.config import DatasetName


class AttackContractError(ValueError):
    """Raised when an attack definition violates the frozen threat model."""


def _validate_fraction_grid(values: tuple[float, ...], *, name: str) -> None:
    if not values:
        raise AttackContractError(f"{name} cannot be empty")
    if tuple(sorted(set(values))) != values:
        raise AttackContractError(f"{name} must be strictly increasing and unique")
    if any(value <= 0.0 or value >= 0.5 for value in values):
        raise AttackContractError(f"{name} must remain within the malicious-minority range")


@dataclass(frozen=True, slots=True)
class DatasetAttackProfile:
    """Dataset-specific source/target labels and frozen Phase 3 identities."""

    dataset: DatasetName
    split_id: str
    preprocessor_id: str
    source_family: str
    source_family_id: int
    target_family: str
    target_family_id: int
    source_selection_rule: str
    minimum_training_records: int
    observed_federated_training_records: int

    def __post_init__(self) -> None:
        if not self.split_id or not self.preprocessor_id:
            raise AttackContractError("Dataset profiles require frozen split/preprocessor IDs")
        if not self.source_family or self.source_family == self.target_family:
            raise AttackContractError("Source and target attack families must be distinct")
        if self.target_family != "benign" or self.target_family_id != 0:
            raise AttackContractError("The targeted IDS objective is fixed to the benign class")
        if self.source_family_id < 1:
            raise AttackContractError("The source family must be non-benign")
        if self.minimum_training_records < 1:
            raise AttackContractError("The source-family training floor must be positive")
        if self.observed_federated_training_records < self.minimum_training_records:
            raise AttackContractError("The frozen source family does not meet the training floor")


@dataclass(frozen=True, slots=True)
class DataPoisoningPolicy:
    """Common row-selection and label-flip policy."""

    poison_fractions: tuple[float, ...] = (0.1, 0.3, 0.5)
    no_op_fraction: float = 0.0
    selection_scope: str = "eligible_source_rows_on_each_malicious_client"
    selection_rule: str = "lowest_sha256_rank_without_replacement"
    selection_key_fields: tuple[str, ...] = (
        "run_seed",
        "attack_id",
        "client_id",
        "record_id",
    )
    selected_count_rule: str = "floor(poison_fraction*eligible_rows)"
    allowed_partition: str = "federated_train_after_anchor_removal"

    def __post_init__(self) -> None:
        if self.poison_fractions != (0.1, 0.3, 0.5):
            raise AttackContractError("The manuscript freezes poison fractions at 0.1, 0.3, 0.5")
        if self.no_op_fraction != 0.0:
            raise AttackContractError("The clean no-op fraction must be exactly zero")
        required = {"run_seed", "attack_id", "client_id", "record_id"}
        if set(self.selection_key_fields) != required:
            raise AttackContractError(
                "Record selection must be run-, attack-, client-, and row-specific"
            )
        if self.allowed_partition != "federated_train_after_anchor_removal":
            raise AttackContractError(
                "Attacks must not access anchors, validation, or test records"
            )


@dataclass(frozen=True, slots=True)
class FeaturePoisoningPolicy:
    """Train-only resolution rule for label-preserving feature poisoning."""

    coordinate_system: str = "phase3c_continuous_standardized"
    feature_count: int = 3
    candidate_rule: str = "nonconstant_continuous_coordinates_with_positive_clip_range"
    ranking_rule: str = "descending_absolute_source_vs_benign_standardized_median_gap"
    transformation: str = "x_prime=x+strength*(benign_median-x)"
    strengths: tuple[float, ...] = (0.5, 1.0)
    projection: str = "clip_to_phase3c_training_0.001_0.999_bounds"
    preserve_labels: bool = True
    resolution_partition: str = "federated_train_after_anchor_removal"

    def __post_init__(self) -> None:
        if self.feature_count < 1:
            raise AttackContractError("Clean-label poisoning requires at least one feature")
        if any(value <= 0.0 or value > 1.0 for value in self.strengths):
            raise AttackContractError("Clean-label strengths must be in (0, 1]")
        if not self.preserve_labels:
            raise AttackContractError("Clean-label poisoning must preserve labels")
        if self.resolution_partition != "federated_train_after_anchor_removal":
            raise AttackContractError("Feature resolution is training-only")


@dataclass(frozen=True, slots=True)
class TriggerPolicy:
    """Fixed train-resolved tabular backdoor trigger contract."""

    coordinate_system: str = "phase3c_continuous_standardized"
    minimum_coordinates: int = 3
    maximum_coordinates: int = 6
    candidate_pool_size: int = 12
    value_rule: str = "rarest_phase3c_clip_endpoint_tuple_on_federated_train"
    tie_break_rule: str = "coordinate_name_then_lower_before_upper"
    training_collision_requirement: int = 0
    poisoned_label_rule: str = "source_family_to_benign"
    triggered_test_source: str = "clean_test_source_family_rows"
    natural_test_collision_policy: str = "report_and_exclude_from_asr_denominator"

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_coordinates <= self.maximum_coordinates:
            raise AttackContractError("Trigger coordinate bounds are invalid")
        if self.candidate_pool_size < self.maximum_coordinates:
            raise AttackContractError("Trigger candidate pool is too small")
        if self.training_collision_requirement != 0:
            raise AttackContractError("The frozen trigger requires zero clean-training collisions")
        if self.poisoned_label_rule != "source_family_to_benign":
            raise AttackContractError("Backdoor training labels must match the benign target")


@dataclass(frozen=True, slots=True)
class CoordinationPolicy:
    """Capabilities of a reproducible malicious-client coalition."""

    mode: str = "shared_objective_and_parameters"
    synchronized_active_rounds: bool = True
    shared_fields: tuple[str, ...] = (
        "attack_family",
        "source_family",
        "target_family",
        "poison_fraction",
        "resolved_feature_or_trigger_artifact_id",
    )
    client_specific_record_selection: bool = True
    may_share_malicious_updates: bool = True
    may_observe_current_benign_updates: bool = False
    may_access_anchor_records: bool = False

    def __post_init__(self) -> None:
        if not self.synchronized_active_rounds or not self.client_specific_record_selection:
            raise AttackContractError(
                "Coordinated attacks require synchronized, non-duplicated actions"
            )
        if self.may_observe_current_benign_updates or self.may_access_anchor_records:
            raise AttackContractError("The coalition exceeds the manuscript threat model")


@dataclass(frozen=True, slots=True)
class DirectModelPolicy:
    """Exact direct update-manipulation formulas."""

    scaling_factors: tuple[float, ...] = (5.0, 10.0)
    scaling_formula: str = "delta_prime=scale*delta"
    reversal_factors: tuple[float, ...] = (1.0, 5.0, 10.0)
    reversal_formula: str = "delta_prime=-scale*delta"
    replacement_formula: str = (
        "delta_prime=(participants_per_round/coalition_size)*(theta_star-theta_t)"
    )
    replacement_target_model: str = "local_model_trained_on_frozen_backdoor_objective"
    declared_sample_count_manipulation: bool = False

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.scaling_factors + self.reversal_factors):
            raise AttackContractError("Direct-model scale factors must be positive")
        if self.declared_sample_count_manipulation:
            raise AttackContractError("Phase 6 does not combine update and sample-count attacks")


@dataclass(frozen=True, slots=True)
class AdaptiveNormPolicy:
    """Prior-reference norm camouflage without current benign-update access."""

    base_objective: str = "targeted_label_flip_source_family_to_benign"
    reference_population: str = "benign_only_development_runs_same_dataset_model_and_partition"
    reference_quantiles: tuple[float, float] = (0.25, 0.75)
    projection: str = "radially_clip_l2_norm_to_frozen_reference_interval"
    reference_artifact_required: bool = True
    current_round_benign_updates_visible: bool = False
    anchor_records_visible: bool = False

    def __post_init__(self) -> None:
        lower, upper = self.reference_quantiles
        if not 0.0 < lower < upper < 1.0:
            raise AttackContractError("Adaptive norm reference quantiles are invalid")
        if not self.reference_artifact_required:
            raise AttackContractError("Adaptive norm attacks require a frozen reference artifact")
        if self.current_round_benign_updates_visible or self.anchor_records_visible:
            raise AttackContractError("Adaptive attackers cannot see benign updates or anchors")


@dataclass(frozen=True, slots=True)
class AttackMatrix:
    """Development and confirmatory attack-factor grids."""

    malicious_client_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    heterogeneity_profiles: tuple[str, ...] = (
        "iid",
        "dirichlet_0_1",
        "dirichlet_0_3",
        "dirichlet_1_0",
    )
    pilot_seed_count: int = 5
    confirmatory_minimum_seed_count: int = 10
    seed_derivation: str = "derive_seed(root_seed,'attack/<attack_id>/<replicate_index>')"

    def __post_init__(self) -> None:
        _validate_fraction_grid(self.malicious_client_ratios, name="malicious_client_ratios")
        if self.pilot_seed_count != 5:
            raise AttackContractError("The reduced development matrix uses five seeds")
        if self.confirmatory_minimum_seed_count < 10:
            raise AttackContractError("Confirmatory paired analysis requires at least ten seeds")
        if len(set(self.heterogeneity_profiles)) != len(self.heterogeneity_profiles):
            raise AttackContractError("Heterogeneity profiles must be unique")


@dataclass(frozen=True, slots=True)
class Phase6AContract:
    """Complete immutable Phase 6A attack-definition object."""

    schema_version: int = 1
    phase: str = "6A"
    status: str = "contract_frozen_resolution_artifacts_pending"
    base_seed: int = 20260816
    profiles: tuple[DatasetAttackProfile, ...] = field(default_factory=tuple)
    data_poisoning: DataPoisoningPolicy = field(default_factory=DataPoisoningPolicy)
    feature_poisoning: FeaturePoisoningPolicy = field(default_factory=FeaturePoisoningPolicy)
    backdoor: TriggerPolicy = field(default_factory=TriggerPolicy)
    coordination: CoordinationPolicy = field(default_factory=CoordinationPolicy)
    direct_model: DirectModelPolicy = field(default_factory=DirectModelPolicy)
    adaptive_norm: AdaptiveNormPolicy = field(default_factory=AdaptiveNormPolicy)
    matrix: AttackMatrix = field(default_factory=AttackMatrix)
    test_partition_accessed: bool = False
    performance_claims_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.phase != "6A":
            raise AttackContractError("Unsupported Phase 6A schema")
        if self.base_seed < 0:
            raise AttackContractError("The base seed cannot be negative")
        if {profile.dataset for profile in self.profiles} != set(DatasetName):
            raise AttackContractError("Exactly one profile is required for each supported dataset")
        if self.test_partition_accessed or self.performance_claims_allowed:
            raise AttackContractError(
                "Phase 6A cannot access test data or support performance claims"
            )


def phase6a_contract() -> Phase6AContract:
    """Return the frozen contract derived from the completed Phase 3 artifacts."""

    return Phase6AContract(
        profiles=(
            DatasetAttackProfile(
                dataset=DatasetName.EDGE_IIOTSET,
                split_id="b6616d2030f682906d205b0a105234a78a86ec995588fd4bbfe100ba2b6ca886",
                preprocessor_id=(
                    "c175c802c8c30049f0ddfcba72b44c1c020e52e7472b691b5488dc9d842bc7ff"
                ),
                source_family="injection",
                source_family_id=3,
                target_family="benign",
                target_family_id=0,
                source_selection_rule=(
                    "least_frequent_non_benign_primary_family_with_at_least_10000_"
                    "federated_training_records"
                ),
                minimum_training_records=10_000,
                observed_federated_training_records=46_519,
            ),
            DatasetAttackProfile(
                dataset=DatasetName.CIC_IOT_2023,
                split_id="590fb686703ef03c30314499629f13d8b083f24f45348f3cd289198abcac99ca",
                preprocessor_id=(
                    "e39b411b302da0c97ef766c0d8c9e36caa4c718e65f702f40e78705fde497e8a"
                ),
                source_family="web_based",
                source_family_id=7,
                target_family="benign",
                target_family_id=0,
                source_selection_rule=(
                    "least_frequent_non_benign_primary_family_with_at_least_10000_"
                    "federated_training_records"
                ),
                minimum_training_records=10_000,
                observed_federated_training_records=17_060,
            ),
        )
    )
