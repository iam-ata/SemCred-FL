"""Evidence-bounded AGAT-FL aggregation core.

This module mirrors the normalized trust-weighted sum present in the authors'
Zenodo code artifact. It is not a complete AGAT-FL reproduction: the archive
does not provide a federated loop or a training procedure for the TrustGAT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from semcred_fl.baselines.fedavg import AggregationResult
from semcred_fl.federation.updates import DefenseUpdate


def agat_trust_aggregate(
    updates: tuple[DefenseUpdate, ...],
    trust_scores: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> AggregationResult:
    """Aggregate updates using externally computed normalized AGAT trust scores."""

    if not updates:
        raise ValueError("AGAT-FL trust aggregation requires at least one update")
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("AGAT-FL client IDs must be unique")
    client_ids = {update.client_id for update in updates}
    if set(trust_scores) != client_ids:
        raise ValueError("AGAT-FL trust and update populations differ")
    shape = updates[0].delta.shape
    if any(update.delta.shape != shape for update in updates):
        raise ValueError("AGAT-FL updates have inconsistent shapes")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("AGAT-FL received a non-finite update")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("AGAT-FL tolerance must be finite and positive")

    weights = {client_id: float(trust_scores[client_id]) for client_id in sorted(client_ids)}
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("AGAT-FL trust scores must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("AGAT-FL trust scores must already sum to one")

    aggregate = torch.zeros_like(updates[0].delta)
    for update in updates:
        aggregate = aggregate + update.delta * weights[update.client_id]
    return AggregationResult("agat_fl_trust_core", aggregate, weights)
