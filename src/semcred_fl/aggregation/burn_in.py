"""Coordinate-median burn-in and controlled history admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from semcred_fl.aggregation.clipping import ClippingResult, clip_eligible_updates
from semcred_fl.aggregation.eligibility import EligibilityResult, select_burn_in_history_clients
from semcred_fl.credibility.numerical_score import coordinate_median
from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class BurnInRecord:
    """Median global update and controlled clipped-history admission."""

    aggregate_delta: Tensor
    history_eligibility: EligibilityResult
    history_clipping: ClippingResult

    @property
    def admitted_client_ids(self) -> tuple[str, ...]:
        """Return clients whose clipped burn-in updates may enter history."""

        return self.history_eligibility.eligible_client_ids


def aggregate_burn_in(
    updates: tuple[DefenseUpdate, ...],
    fused_scores: Mapping[str, float],
    *,
    retained_ratio: float = 0.5,
    clipping_multiplier: float = 2.5,
    epsilon: float = 1e-12,
) -> BurnInRecord:
    """Aggregate all updates by coordinate median and gate history by credibility."""

    if not updates or len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Burn-in requires unique accepted updates")
    update_map = {update.client_id: update for update in updates}
    if set(update_map) != set(fused_scores):
        raise ValueError("Burn-in credibility population differs from accepted updates")
    eligibility = select_burn_in_history_clients(
        fused_scores,
        retained_ratio=retained_ratio,
    )
    clipping = clip_eligible_updates(
        update_map,
        eligibility.eligible_client_ids,
        clipping_multiplier=clipping_multiplier,
        epsilon=epsilon,
    )
    aggregate = coordinate_median(tuple(update.delta for update in updates))
    return BurnInRecord(aggregate, eligibility, clipping)
