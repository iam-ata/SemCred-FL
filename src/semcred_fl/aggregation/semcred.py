"""Post-burn-in SemCred-FL eligibility, clipping, weighting, and aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from semcred_fl.aggregation.clipping import ClippingResult, clip_eligible_updates
from semcred_fl.aggregation.eligibility import EligibilityResult, select_eligible_clients
from semcred_fl.aggregation.weighting import WeightingResult, trust_aware_weights
from semcred_fl.config import AggregationConfig
from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class AggregationRecord:
    """Auditable post-burn-in SemCred-FL aggregation result."""

    threshold: float
    minimum_count: int
    eligible_client_ids: tuple[str, ...]
    retention_added_client_ids: tuple[str, ...]
    clipping_bound: float
    sample_count_cap: float | None
    weights: dict[str, float]
    clipped_updates: dict[str, Tensor]
    aggregate_delta: Tensor

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Aggregation threshold must be finite and in [0,1]")
        if not self.eligible_client_ids or len(self.eligible_client_ids) < self.minimum_count:
            raise ValueError("Aggregation record violates minimum retention")
        if set(self.eligible_client_ids) != set(self.weights):
            raise ValueError("Eligible clients and aggregation weights differ")
        if set(self.eligible_client_ids) != set(self.clipped_updates):
            raise ValueError("Eligible clients and clipped updates differ")
        if self.aggregate_delta.ndim != 1 or not torch.isfinite(self.aggregate_delta).all().item():
            raise ValueError("Aggregate update must be a finite flat vector")


def _update_map(updates: tuple[DefenseUpdate, ...]) -> dict[str, DefenseUpdate]:
    if not updates or len({update.client_id for update in updates}) != len(updates):
        raise ValueError("SemCred aggregation requires unique accepted updates")
    shape = updates[0].delta.shape
    if len(shape) != 1 or any(update.delta.shape != shape for update in updates):
        raise ValueError("SemCred updates must share one flat-vector shape")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("SemCred aggregation received a non-finite update")
    return {update.client_id: update for update in updates}


def weighted_clipped_sum(clipping: ClippingResult, weighting: WeightingResult) -> Tensor:
    """Return the exact weighted sum of clipped eligible updates."""

    if set(clipping.clipped_updates) != set(weighting.weights):
        raise ValueError("Clipping and weighting client inventories differ")
    first = next(iter(clipping.clipped_updates.values()))
    aggregate = torch.zeros_like(first)
    for client_id in sorted(weighting.weights):
        aggregate = aggregate + clipping.clipped_updates[client_id] * weighting.weights[client_id]
    if not torch.isfinite(aggregate).all().item():
        raise RuntimeError("SemCred weighted aggregation produced a non-finite update")
    return aggregate


def aggregate_semcred_updates(
    updates: tuple[DefenseUpdate, ...],
    trusts: Mapping[str, float],
    *,
    config: AggregationConfig | None = None,
    epsilon: float = 1e-12,
) -> AggregationRecord:
    """Apply post-burn-in eligibility, clipping, and trust-aware weighting."""

    controls = config or AggregationConfig()
    update_map = _update_map(updates)
    if set(update_map) != set(trusts):
        raise ValueError("Current trust population differs from accepted updates")
    eligibility = select_eligible_clients(
        trusts,
        minimum_trust=controls.minimum_trust,
        threshold_multiplier=controls.threshold_multiplier,
        retained_ratio=controls.minimum_retained_ratio,
    )
    clipping = clip_eligible_updates(
        update_map,
        eligibility.eligible_client_ids,
        clipping_multiplier=controls.clipping_multiplier,
        epsilon=epsilon,
    )
    weighting = trust_aware_weights(
        update_map,
        trusts,
        eligibility.eligible_client_ids,
        sample_count_exponent=controls.sample_count_exponent,
        sample_cap_median_multiplier=controls.sample_cap_median_multiplier,
        trust_exponent=controls.trust_exponent,
        epsilon=epsilon,
    )
    return _aggregation_record(eligibility, clipping, weighting)


def _aggregation_record(
    eligibility: EligibilityResult,
    clipping: ClippingResult,
    weighting: WeightingResult,
) -> AggregationRecord:
    return AggregationRecord(
        threshold=eligibility.threshold,
        minimum_count=eligibility.minimum_count,
        eligible_client_ids=eligibility.eligible_client_ids,
        retention_added_client_ids=eligibility.retention_added_client_ids,
        clipping_bound=clipping.bound,
        sample_count_cap=weighting.sample_count_cap,
        weights=weighting.weights,
        clipped_updates=clipping.clipped_updates,
        aggregate_delta=weighted_clipped_sum(clipping, weighting),
    )
