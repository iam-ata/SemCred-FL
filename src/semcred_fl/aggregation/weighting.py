"""Declared-count capping and normalized trust-aware aggregation weights."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from semcred_fl.credibility.robust_stats import median
from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class WeightingResult:
    """Sample-count cap, effective counts, and normalized eligible weights."""

    sample_count_cap: float | None
    capped_sample_counts: dict[str, float]
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.sample_count_cap is not None and (
            not math.isfinite(self.sample_count_cap) or self.sample_count_cap <= 0.0
        ):
            raise ValueError("Enabled sample-count cap must be finite and positive")
        if not self.weights or set(self.weights) != set(self.capped_sample_counts):
            raise ValueError("Weight and sample-count inventories differ")
        if any(not math.isfinite(value) or value < 0.0 for value in self.weights.values()):
            raise ValueError("Aggregation weights must be finite and nonnegative")
        if not math.isclose(sum(self.weights.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Aggregation weights must sum to one")


def trust_aware_weights(
    updates: Mapping[str, DefenseUpdate],
    trusts: Mapping[str, float],
    eligible_client_ids: tuple[str, ...],
    *,
    sample_count_exponent: float = 0.0,
    sample_cap_median_multiplier: float = 2.0,
    trust_exponent: float = 2.0,
    epsilon: float = 1e-12,
) -> WeightingResult:
    """Compute capped-count and persistent-trust weights for eligible clients."""

    if not eligible_client_ids or len(set(eligible_client_ids)) != len(eligible_client_ids):
        raise ValueError("Weighting requires unique eligible clients")
    if any(
        client_id not in updates or client_id not in trusts for client_id in eligible_client_ids
    ):
        raise ValueError("An eligible client lacks an update or current trust")
    if not math.isfinite(sample_count_exponent) or not 0.0 <= sample_count_exponent <= 1.0:
        raise ValueError("sample_count_exponent must be finite and in [0,1]")
    if not math.isfinite(sample_cap_median_multiplier) or sample_cap_median_multiplier <= 0.0:
        raise ValueError("sample_cap_median_multiplier must be finite and positive")
    if not math.isfinite(trust_exponent) or trust_exponent < 1.0:
        raise ValueError("trust_exponent must be finite and at least one")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    counts = {
        client_id: float(updates[client_id].declared_sample_count)
        for client_id in eligible_client_ids
    }
    if any(value < 1.0 for value in counts.values()):
        raise ValueError("Eligible declared sample counts must be positive")
    trust_values = {client_id: trusts[client_id] for client_id in eligible_client_ids}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in trust_values.values()):
        raise ValueError("Eligible trust values must be finite and in [0,1]")

    count_cap: float | None = None
    if sample_count_exponent > 0.0:
        count_cap = sample_cap_median_multiplier * median(tuple(counts.values()))
    capped = {
        client_id: min(count, count_cap) if count_cap is not None else count
        for client_id, count in counts.items()
    }
    numerators = {
        client_id: capped[client_id] ** sample_count_exponent
        * (trust_values[client_id] + epsilon) ** trust_exponent
        for client_id in eligible_client_ids
    }
    denominator = sum(numerators.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Aggregation-weight denominator must be finite and positive")
    weights = {
        client_id: numerators[client_id] / denominator for client_id in sorted(eligible_client_ids)
    }
    return WeightingResult(count_cap, dict(sorted(capped.items())), weights)
