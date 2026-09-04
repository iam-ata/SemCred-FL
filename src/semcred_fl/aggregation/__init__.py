"""SemCred-FL fusion, trust, eligibility, clipping, and aggregation."""

from semcred_fl.aggregation.burn_in import BurnInRecord, aggregate_burn_in
from semcred_fl.aggregation.clipping import ClippingResult, clip_eligible_updates, clip_update
from semcred_fl.aggregation.eligibility import (
    EligibilityResult,
    adaptive_trust_threshold,
    minimum_retained_count,
    select_burn_in_history_clients,
    select_eligible_clients,
)
from semcred_fl.aggregation.semcred import (
    AggregationRecord,
    aggregate_semcred_updates,
    weighted_clipped_sum,
)
from semcred_fl.aggregation.trust import asymmetric_trust
from semcred_fl.aggregation.weighting import WeightingResult, trust_aware_weights

__all__ = [
    "AggregationRecord",
    "BurnInRecord",
    "ClippingResult",
    "EligibilityResult",
    "WeightingResult",
    "adaptive_trust_threshold",
    "aggregate_burn_in",
    "aggregate_semcred_updates",
    "asymmetric_trust",
    "clip_eligible_updates",
    "clip_update",
    "minimum_retained_count",
    "select_burn_in_history_clients",
    "select_eligible_clients",
    "trust_aware_weights",
    "weighted_clipped_sum",
]
