"""Typed experiment configuration and scientific validity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import floor, isfinite


class ConfigurationError(ValueError):
    """Raised when a configuration violates an experiment invariant."""


class DatasetName(StrEnum):
    """Datasets supported by the frozen primary protocol."""

    EDGE_IIOTSET = "edge_iiotset"
    CIC_IOT_2023 = "cic_iot_2023"


class StudyPhase(StrEnum):
    """Distinguish exploratory pilot runs from confirmatory runs."""

    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"


class ParticipationPolicy(StrEnum):
    """Round-construction policies used by the threat experiments."""

    STRATIFIED_MINORITY = "stratified_minority"
    RANDOM_AVAILABILITY_STRESS = "random_availability_stress"


class BaselineProtocol(StrEnum):
    """Communication protocols used by baseline implementations."""

    PARAMETER_UPDATE = "parameter_update"
    PUBLIC_DATA_LOGITS = "public_data_logits"


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    """Disjoint trusted subsets used for calibration and scoring."""

    calibration_records_per_class: int = 32
    scoring_records_per_class: int = 32
    require_disjoint: bool = True

    def __post_init__(self) -> None:
        if self.calibration_records_per_class < 1:
            raise ConfigurationError("Calibration anchors per class must be positive")
        if self.scoring_records_per_class < 1:
            raise ConfigurationError("Scoring anchors per class must be positive")
        if not self.require_disjoint:
            raise ConfigurationError("Calibration and scoring anchor sets must be disjoint")


@dataclass(frozen=True, slots=True)
class StatisticalConfig:
    """Repetition and paired-inference configuration."""

    phase: StudyPhase = StudyPhase.PILOT
    seed_count: int = 5
    paired_test: str = "wilcoxon_exact_two_sided"
    multiplicity_correction: str = "holm"
    power_analysis_id: str | None = None

    def __post_init__(self) -> None:
        if self.seed_count < 2:
            raise ConfigurationError("At least two paired seeds are required")
        if self.phase is StudyPhase.CONFIRMATORY:
            if self.seed_count < 10:
                raise ConfigurationError(
                    "Confirmatory exact Wilcoxon-Holm analysis requires at least ten seeds"
                )
            if not self.power_analysis_id:
                raise ConfigurationError(
                    "Confirmatory runs require a documented power-analysis identifier"
                )


@dataclass(frozen=True, slots=True)
class FederationConfig:
    """Client population, participation, and Byzantine-bound configuration."""

    total_clients: int = 50
    clients_per_round: int = 20
    malicious_ratio: float = 0.4
    participation_policy: ParticipationPolicy = ParticipationPolicy.STRATIFIED_MINORITY
    multi_krum_enabled: bool = True
    multi_krum_f: int = 8

    def __post_init__(self) -> None:
        if self.total_clients < 2:
            raise ConfigurationError("At least two clients are required")
        if not 1 <= self.clients_per_round <= self.total_clients:
            raise ConfigurationError("clients_per_round must be within the client population")
        if not 0.0 <= self.malicious_ratio < 0.5:
            raise ConfigurationError("The declared threat model requires a malicious minority")
        if self.multi_krum_enabled:
            if self.multi_krum_f < 0:
                raise ConfigurationError("The Multi-Krum Byzantine bound cannot be negative")
            if self.clients_per_round < 2 * self.multi_krum_f + 3:
                raise ConfigurationError("Multi-Krum requires n >= 2f + 3")

    @property
    def malicious_clients_per_round(self) -> int:
        """Return the configured malicious-client count for stratified rounds."""

        return floor(self.clients_per_round * self.malicious_ratio)

    @property
    def minority_is_enforced(self) -> bool:
        """Return whether every constructed round enforces the threat assumption."""

        return self.participation_policy is ParticipationPolicy.STRATIFIED_MINORITY


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    """Credibility-weighted aggregation controls."""

    burn_in_rounds: int = 3
    semantic_fusion_coefficient: float = 0.5
    disagreement_penalty: float = 1.0
    trust_decrease_rate: float = 0.6
    trust_recovery_rate: float = 0.1
    sample_count_exponent: float = 0.0
    sample_cap_median_multiplier: float = 2.0
    trust_exponent: float = 2.0
    threshold_multiplier: float = 2.5
    minimum_trust: float = 0.1
    minimum_retained_ratio: float = 0.5
    clipping_multiplier: float = 2.5
    history_decay: float = 0.8

    def __post_init__(self) -> None:
        numeric = (
            self.semantic_fusion_coefficient,
            self.disagreement_penalty,
            self.trust_decrease_rate,
            self.trust_recovery_rate,
            self.sample_count_exponent,
            self.sample_cap_median_multiplier,
            self.trust_exponent,
            self.threshold_multiplier,
            self.minimum_trust,
            self.minimum_retained_ratio,
            self.clipping_multiplier,
            self.history_decay,
        )
        if any(not isfinite(value) for value in numeric):
            raise ConfigurationError("Aggregation controls must be finite")
        if self.burn_in_rounds < 0:
            raise ConfigurationError("burn_in_rounds cannot be negative")
        if not 0.0 <= self.semantic_fusion_coefficient <= 1.0:
            raise ConfigurationError("semantic_fusion_coefficient must be in [0,1]")
        if self.disagreement_penalty < 0.0:
            raise ConfigurationError("disagreement_penalty cannot be negative")
        if not 0.0 < self.trust_recovery_rate < self.trust_decrease_rate <= 1.0:
            raise ConfigurationError("Trust recovery must be slower than trust decrease")
        if not 0.0 <= self.sample_count_exponent <= 1.0:
            raise ConfigurationError("sample_count_exponent must be in [0,1]")
        if self.sample_cap_median_multiplier <= 0.0:
            raise ConfigurationError("The sample-count cap multiplier must be positive")
        if self.trust_exponent < 1.0:
            raise ConfigurationError("trust_exponent must be at least one")
        if self.threshold_multiplier < 0.0:
            raise ConfigurationError("threshold_multiplier cannot be negative")
        if not 0.0 <= self.minimum_trust <= 1.0:
            raise ConfigurationError("minimum_trust must be in [0,1]")
        if not 0.0 < self.minimum_retained_ratio <= 1.0:
            raise ConfigurationError("minimum_retained_ratio must be in (0,1]")
        if self.clipping_multiplier <= 0.0:
            raise ConfigurationError("clipping_multiplier must be positive")
        if not 0.0 <= self.history_decay < 1.0:
            raise ConfigurationError("history_decay must be in [0,1)")

    @property
    def declared_counts_affect_weights(self) -> bool:
        """Return whether declared sample counts affect aggregation weights."""

        return self.sample_count_exponent > 0


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    """Offline prior and online semantic-scoring controls."""

    attribution_interval: int = 1
    reuse_stale_attribution: bool = False
    immutable_prior_artifact_required: bool = True

    def __post_init__(self) -> None:
        if self.attribution_interval < 1:
            raise ConfigurationError("attribution_interval must be at least one round")
        if self.reuse_stale_attribution:
            raise ConfigurationError("Skipped semantic rounds must not reuse stale attribution")
        if not self.immutable_prior_artifact_required:
            raise ConfigurationError(
                "Every run must reference an immutable semantic-prior artifact"
            )


@dataclass(frozen=True, slots=True)
class CredibilityConfig:
    """Phase 8 robust-normalization and credibility coefficients."""

    beta_loss: float = 0.5
    beta_attribution: float = 0.5
    beta_center: float = 0.25
    beta_norm: float = 0.25
    beta_direction: float = 0.25
    beta_history: float = 0.25
    z_max: float = 20.0
    epsilon: float = 1e-12
    negligible_norm: float = 1e-12
    minimum_history_clients: int = 3

    def __post_init__(self) -> None:
        semantic = (self.beta_loss, self.beta_attribution)
        numerical = (
            self.beta_center,
            self.beta_norm,
            self.beta_direction,
            self.beta_history,
        )
        if any(not isfinite(value) or value < 0.0 for value in semantic + numerical):
            raise ConfigurationError("Credibility coefficients must be finite and nonnegative")
        if sum(semantic) <= 0.0 or sum(numerical) <= 0.0:
            raise ConfigurationError(
                "Semantic and numerical coefficient groups must each be nonzero"
            )
        if not isfinite(self.z_max) or self.z_max <= 0.0:
            raise ConfigurationError("z_max must be finite and positive")
        if not isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ConfigurationError("epsilon must be finite and positive")
        if not isfinite(self.negligible_norm) or self.negligible_norm < 0.0:
            raise ConfigurationError("negligible_norm must be finite and nonnegative")
        if self.minimum_history_clients < 3:
            raise ConfigurationError("Numerical history requires at least three valid participants")


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """One baseline and its native or explicitly surrogate protocol."""

    name: str
    protocol: BaselineProtocol
    is_surrogate: bool = False
    public_dataset_id: str | None = None

    def __post_init__(self) -> None:
        normalized = self.name.casefold().replace("_", "-")
        if normalized == "fedkd-ids" and not self.is_surrogate:
            if self.protocol is not BaselineProtocol.PUBLIC_DATA_LOGITS:
                raise ConfigurationError("Native FedKD-IDS must exchange public-data logits")
            if not self.public_dataset_id:
                raise ConfigurationError("Native FedKD-IDS requires a public-dataset artifact")
        if normalized == "fedkd-ids" and self.is_surrogate:
            if self.protocol is not BaselineProtocol.PARAMETER_UPDATE:
                raise ConfigurationError(
                    "The FedKD-IDS surrogate is the parameter-update adaptation"
                )


def _default_baselines() -> tuple[BaselineConfig, ...]:
    return (
        BaselineConfig("fedavg", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig("coordinate-median", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig("trimmed-mean", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig("multi-krum", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig("agat-fl", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig("fed-lsae", BaselineProtocol.PARAMETER_UPDATE),
        BaselineConfig(
            "fedkd-ids",
            BaselineProtocol.PUBLIC_DATA_LOGITS,
            public_dataset_id="pending-dataset-artifact",
        ),
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level immutable configuration for one experiment family."""

    dataset: DatasetName = DatasetName.EDGE_IIOTSET
    base_seed: int = 20260816
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    statistics: StatisticalConfig = field(default_factory=StatisticalConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    credibility: CredibilityConfig = field(default_factory=CredibilityConfig)
    baselines: tuple[BaselineConfig, ...] = field(default_factory=_default_baselines)

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ConfigurationError("base_seed cannot be negative")
        names = [baseline.name.casefold() for baseline in self.baselines]
        if len(names) != len(set(names)):
            raise ConfigurationError("Baseline names must be unique")
