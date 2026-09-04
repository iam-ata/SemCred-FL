"""Tests for experiment invariants."""

from __future__ import annotations

from dataclasses import replace

import pytest

from semcred_fl.config import (
    AggregationConfig,
    AnchorConfig,
    BaselineConfig,
    BaselineProtocol,
    ConfigurationError,
    CredibilityConfig,
    ExperimentConfig,
    FederationConfig,
    ParticipationPolicy,
    SemanticConfig,
    StatisticalConfig,
    StudyPhase,
)


def test_default_is_a_valid_pilot() -> None:
    config = ExperimentConfig()
    assert config.statistics.phase is StudyPhase.PILOT
    assert config.statistics.seed_count == 5
    assert config.federation.total_clients == 50
    assert config.federation.clients_per_round == 20
    assert config.federation.malicious_clients_per_round == 8
    assert config.federation.minority_is_enforced
    assert not config.aggregation.declared_counts_affect_weights


def test_confirmatory_plan_requires_seed_floor_and_power_record() -> None:
    with pytest.raises(ConfigurationError, match="at least ten seeds"):
        StatisticalConfig(phase=StudyPhase.CONFIRMATORY, seed_count=9, power_analysis_id="p1")
    with pytest.raises(ConfigurationError, match="power-analysis"):
        StatisticalConfig(phase=StudyPhase.CONFIRMATORY, seed_count=10)
    plan = StatisticalConfig(
        phase=StudyPhase.CONFIRMATORY,
        seed_count=12,
        power_analysis_id="pilot-2026-08",
    )
    assert plan.seed_count == 12


@pytest.mark.parametrize("count", [0, -1])
def test_anchor_counts_must_be_positive(count: int) -> None:
    with pytest.raises(ConfigurationError, match="Calibration"):
        AnchorConfig(calibration_records_per_class=count)
    with pytest.raises(ConfigurationError, match="Scoring"):
        AnchorConfig(scoring_records_per_class=count)


def test_anchor_sets_must_be_disjoint() -> None:
    with pytest.raises(ConfigurationError, match="disjoint"):
        AnchorConfig(require_disjoint=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_clients": 1}, "two clients"),
        ({"clients_per_round": 51}, "within"),
        ({"malicious_ratio": 0.5}, "minority"),
        ({"multi_krum_f": -1}, "cannot be negative"),
        ({"clients_per_round": 10, "multi_krum_f": 4}, "n >= 2f"),
    ],
)
def test_federation_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        FederationConfig(**kwargs)  # type: ignore[arg-type]


def test_random_availability_is_labeled_as_assumption_stress() -> None:
    config = FederationConfig(participation_policy=ParticipationPolicy.RANDOM_AVAILABILITY_STRESS)
    assert not config.minority_is_enforced


def test_multi_krum_can_be_disabled_for_small_rounds() -> None:
    config = FederationConfig(clients_per_round=10, multi_krum_enabled=False)
    assert not config.multi_krum_enabled


@pytest.mark.parametrize(
    "kwargs",
    [
        {"burn_in_rounds": -1},
        {"semantic_fusion_coefficient": 1.1},
        {"disagreement_penalty": -0.1},
        {"trust_recovery_rate": 0.6},
        {"trust_decrease_rate": 0.0},
        {"sample_count_exponent": -0.1},
        {"sample_count_exponent": 1.1},
        {"sample_cap_median_multiplier": 0.0},
        {"trust_exponent": 0.9},
        {"threshold_multiplier": -1.0},
        {"minimum_trust": 1.1},
        {"minimum_retained_ratio": 0.0},
        {"clipping_multiplier": 0.0},
        {"history_decay": 1.0},
    ],
)
def test_aggregation_requires_nonnegative_or_positive_controls(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ConfigurationError):
        AggregationConfig(**kwargs)


def test_nonzero_sample_exponent_is_explicit() -> None:
    config = AggregationConfig(sample_count_exponent=0.5)
    assert config.declared_counts_affect_weights


def test_phase9_defaults_match_confirmatory_manuscript() -> None:
    config = AggregationConfig()
    assert config.burn_in_rounds == 3
    assert config.semantic_fusion_coefficient == 0.5
    assert config.disagreement_penalty == 1.0
    assert config.trust_decrease_rate == 0.6
    assert config.trust_recovery_rate == 0.1
    assert config.sample_count_exponent == 0.0
    assert config.sample_cap_median_multiplier == 2.0
    assert config.trust_exponent == 2.0
    assert config.minimum_retained_ratio == 0.5
    assert config.history_decay == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attribution_interval": 0},
        {"reuse_stale_attribution": True},
        {"immutable_prior_artifact_required": False},
    ],
)
def test_semantic_configuration_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SemanticConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta_loss": -0.1},
        {"beta_loss": 0.0, "beta_attribution": 0.0},
        {"beta_center": 0.0, "beta_norm": 0.0, "beta_direction": 0.0, "beta_history": 0.0},
        {"z_max": 0.0},
        {"epsilon": 0.0},
        {"negligible_norm": -1.0},
        {"minimum_history_clients": 2},
    ],
)
def test_credibility_configuration_rejects_invalid_controls(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        CredibilityConfig(**kwargs)  # type: ignore[arg-type]


def test_default_credibility_coefficients_match_frozen_protocol() -> None:
    config = CredibilityConfig()
    assert config.beta_loss == config.beta_attribution == 0.5
    assert (
        config.beta_center
        == config.beta_norm
        == config.beta_direction
        == config.beta_history
        == 0.25
    )
    assert config.minimum_history_clients == 3


def test_fedkd_native_protocol_requires_logits_and_public_data() -> None:
    with pytest.raises(ConfigurationError, match="public-data logits"):
        BaselineConfig("fedkd-ids", BaselineProtocol.PARAMETER_UPDATE)
    with pytest.raises(ConfigurationError, match="public-dataset"):
        BaselineConfig("fedkd_ids", BaselineProtocol.PUBLIC_DATA_LOGITS)
    native = BaselineConfig(
        "fedkd-ids",
        BaselineProtocol.PUBLIC_DATA_LOGITS,
        public_dataset_id="nbaiot-public-v1",
    )
    assert not native.is_surrogate


def test_fedkd_surrogate_is_labeled_and_uses_parameter_updates() -> None:
    with pytest.raises(ConfigurationError, match="parameter-update"):
        BaselineConfig(
            "fedkd-ids",
            BaselineProtocol.PUBLIC_DATA_LOGITS,
            is_surrogate=True,
        )
    surrogate = BaselineConfig(
        "fedkd-ids",
        BaselineProtocol.PARAMETER_UPDATE,
        is_surrogate=True,
    )
    assert surrogate.is_surrogate


def test_experiment_rejects_negative_seed_and_duplicate_baselines() -> None:
    with pytest.raises(ConfigurationError, match="base_seed"):
        ExperimentConfig(base_seed=-1)
    duplicate = BaselineConfig("fedavg", BaselineProtocol.PARAMETER_UPDATE)
    with pytest.raises(ConfigurationError, match="unique"):
        ExperimentConfig(baselines=(duplicate, replace(duplicate)))
