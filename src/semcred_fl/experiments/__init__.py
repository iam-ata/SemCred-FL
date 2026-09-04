"""Experiment orchestration introduced at the Phase 12 boundary."""

from semcred_fl.experiments.phase12_attack import (
    Phase12AttackConfig,
    Phase12AttackRequest,
    run_phase12_attack_comparison,
)
from semcred_fl.experiments.phase12_baselines import (
    Phase12BaselineConfig,
    Phase12BaselineRequest,
    run_phase12_reduced_baselines,
)
from semcred_fl.experiments.phase12_clean import (
    Phase12CleanConfig,
    Phase12CleanRequest,
    run_phase12_clean_comparison,
)
from semcred_fl.experiments.phase12_clean_matrix import (
    CONFIRMATORY_SEEDS,
    Phase12CleanConditionConfig,
    Phase12CleanConditionRequest,
    Phase12CleanMatrixRequest,
    inspect_phase12_clean_matrix,
    run_phase12_clean_condition,
    run_phase12_clean_matrix,
)
from semcred_fl.experiments.phase12_completion import (
    audit_phase12_clean_campaign,
    finalize_phase12_clean_campaign,
)
from semcred_fl.experiments.phase12_smoke import Phase12SmokeConfig, run_phase12_smoke

__all__ = [
    "CONFIRMATORY_SEEDS",
    "Phase12AttackConfig",
    "Phase12AttackRequest",
    "Phase12BaselineConfig",
    "Phase12BaselineRequest",
    "Phase12CleanConfig",
    "Phase12CleanConditionConfig",
    "Phase12CleanConditionRequest",
    "Phase12CleanMatrixRequest",
    "Phase12CleanRequest",
    "Phase12SmokeConfig",
    "audit_phase12_clean_campaign",
    "finalize_phase12_clean_campaign",
    "inspect_phase12_clean_matrix",
    "run_phase12_attack_comparison",
    "run_phase12_reduced_baselines",
    "run_phase12_clean_comparison",
    "run_phase12_clean_condition",
    "run_phase12_clean_matrix",
    "run_phase12_smoke",
]
