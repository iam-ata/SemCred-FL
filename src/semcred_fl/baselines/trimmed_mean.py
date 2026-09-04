"""Coordinate-wise trimmed-mean aggregation."""

from __future__ import annotations

import torch

from semcred_fl.baselines.fedavg import AggregationResult
from semcred_fl.federation.updates import DefenseUpdate


def trimmed_mean(
    updates: tuple[DefenseUpdate, ...],
    *,
    trim_count: int,
) -> AggregationResult:
    """Discard ``trim_count`` values from each tail in every coordinate."""

    if not updates:
        raise ValueError("Trimmed mean requires at least one update")
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Trimmed-mean client IDs must be unique")
    shape = updates[0].delta.shape
    if any(update.delta.shape != shape for update in updates):
        raise ValueError("Trimmed-mean updates have inconsistent shapes")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("Trimmed mean received a non-finite update")
    if isinstance(trim_count, bool) or trim_count < 0:
        raise ValueError("Trimmed-mean trim count must be a nonnegative integer")
    if 2 * trim_count >= len(updates):
        raise ValueError("Trimmed mean requires more than twice the trim count")

    ordered, _ = torch.sort(torch.stack(tuple(update.delta for update in updates)), dim=0)
    retained = ordered[trim_count : len(updates) - trim_count]
    return AggregationResult("trimmed_mean", retained.mean(dim=0), {})
