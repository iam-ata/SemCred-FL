"""Coordinate-wise median aggregation for Phase 5 and later burn-in use."""

from __future__ import annotations

import torch

from semcred_fl.baselines.fedavg import AggregationResult
from semcred_fl.federation.updates import DefenseUpdate


def coordinate_median(updates: tuple[DefenseUpdate, ...]) -> AggregationResult:
    """Return the usual midpoint median independently in every coordinate."""

    if not updates:
        raise ValueError("Coordinate median requires at least one update")
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Coordinate-median client IDs must be unique")
    shape = updates[0].delta.shape
    if any(update.delta.shape != shape for update in updates):
        raise ValueError("Coordinate-median updates have inconsistent shapes")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("Coordinate median received a non-finite update")
    stacked = torch.stack(tuple(update.delta for update in updates))
    ordered, _ = torch.sort(stacked, dim=0)
    count = ordered.shape[0]
    if count % 2:
        median = ordered[count // 2]
    else:
        median = (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    return AggregationResult("coordinate_median", median, {})
