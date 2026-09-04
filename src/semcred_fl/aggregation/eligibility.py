"""Adaptive trust threshold and deterministic minimum retention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from semcred_fl.credibility.robust_stats import median, median_absolute_deviation


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """Thresholded population after deterministic minimum retention."""

    threshold: float
    minimum_count: int
    threshold_client_ids: tuple[str, ...]
    eligible_client_ids: tuple[str, ...]
    retention_added_client_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Eligibility threshold must be finite and in [0,1]")
        if self.minimum_count < 1 or len(self.eligible_client_ids) < self.minimum_count:
            raise ValueError("Eligibility result violates minimum retention")


def minimum_retained_count(population_size: int, retained_ratio: float = 0.5) -> int:
    """Return max(1, ceil(q_min times the participating population))."""

    if population_size < 1:
        raise ValueError("Minimum retention requires a nonempty population")
    if not math.isfinite(retained_ratio) or not 0.0 < retained_ratio <= 1.0:
        raise ValueError("retained_ratio must be finite and in (0,1]")
    return max(1, math.ceil(retained_ratio * population_size))


def adaptive_trust_threshold(
    trusts: Mapping[str, float],
    *,
    minimum_trust: float = 0.1,
    threshold_multiplier: float = 2.5,
) -> float:
    """Compute the manuscript median-minus-scaled-MAD trust threshold."""

    if not trusts or any(not client_id for client_id in trusts):
        raise ValueError("Trust threshold requires participating client IDs")
    values = tuple(trusts[client_id] for client_id in sorted(trusts))
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Participating trust values must be finite and in [0,1]")
    if not math.isfinite(minimum_trust) or not 0.0 <= minimum_trust <= 1.0:
        raise ValueError("minimum_trust must be finite and in [0,1]")
    if not math.isfinite(threshold_multiplier) or threshold_multiplier < 0.0:
        raise ValueError("threshold_multiplier must be finite and nonnegative")
    candidate = median(values) - 1.4826 * threshold_multiplier * median_absolute_deviation(values)
    return min(1.0, max(minimum_trust, candidate))


def select_eligible_clients(
    trusts: Mapping[str, float],
    *,
    minimum_trust: float = 0.1,
    threshold_multiplier: float = 2.5,
    retained_ratio: float = 0.5,
) -> EligibilityResult:
    """Apply adaptive thresholding and deterministic trust-ranked retention."""

    threshold = adaptive_trust_threshold(
        trusts,
        minimum_trust=minimum_trust,
        threshold_multiplier=threshold_multiplier,
    )
    minimum_count = minimum_retained_count(len(trusts), retained_ratio)
    threshold_ids = tuple(
        sorted(client_id for client_id, trust in trusts.items() if trust >= threshold)
    )
    selected = set(threshold_ids)
    excluded = sorted(
        (client_id for client_id in trusts if client_id not in selected),
        key=lambda client_id: (-trusts[client_id], client_id),
    )
    added: list[str] = []
    for client_id in excluded:
        if len(selected) >= minimum_count:
            break
        selected.add(client_id)
        added.append(client_id)
    return EligibilityResult(
        threshold=threshold,
        minimum_count=minimum_count,
        threshold_client_ids=threshold_ids,
        eligible_client_ids=tuple(sorted(selected)),
        retention_added_client_ids=tuple(added),
    )


def select_burn_in_history_clients(
    credibility: Mapping[str, float],
    *,
    retained_ratio: float = 0.5,
) -> EligibilityResult:
    """Select at-or-above-median credibility clients for burn-in history admission."""

    if not credibility or any(not client_id for client_id in credibility):
        raise ValueError("Burn-in selection requires participating client IDs")
    values = tuple(credibility[client_id] for client_id in sorted(credibility))
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Burn-in credibility must be finite and in [0,1]")
    threshold = median(values)
    minimum_count = minimum_retained_count(len(credibility), retained_ratio)
    threshold_ids = tuple(
        sorted(client_id for client_id, score in credibility.items() if score >= threshold)
    )
    selected = set(threshold_ids)
    excluded = sorted(
        (client_id for client_id in credibility if client_id not in selected),
        key=lambda client_id: (-credibility[client_id], client_id),
    )
    added: list[str] = []
    for client_id in excluded:
        if len(selected) >= minimum_count:
            break
        selected.add(client_id)
        added.append(client_id)
    return EligibilityResult(
        threshold=threshold,
        minimum_count=minimum_count,
        threshold_client_ids=threshold_ids,
        eligible_client_ids=tuple(sorted(selected)),
        retention_added_client_ids=tuple(added),
    )
