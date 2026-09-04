"""Deterministic Multi-Krum over validated flat client updates."""

from __future__ import annotations

import math

import torch

from semcred_fl.baselines.fedavg import AggregationResult
from semcred_fl.federation.updates import DefenseUpdate


def multi_krum(
    updates: tuple[DefenseUpdate, ...],
    *,
    assumed_byzantine: int,
    selected_count: int | None = None,
) -> AggregationResult:
    """Average the lowest-scoring Multi-Krum candidates.

    Each candidate score is the sum of its ``n-f-2`` nearest squared
    Euclidean distances. The default selection count is ``n-f``.
    """

    if not updates:
        raise ValueError("Multi-Krum requires at least one update")
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Multi-Krum client IDs must be unique")
    shape = updates[0].delta.shape
    if any(update.delta.shape != shape for update in updates):
        raise ValueError("Multi-Krum updates have inconsistent shapes")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("Multi-Krum received a non-finite update")
    if isinstance(assumed_byzantine, bool) or assumed_byzantine < 0:
        raise ValueError("Multi-Krum Byzantine count must be a nonnegative integer")

    count = len(updates)
    if count < 2 * assumed_byzantine + 3:
        raise ValueError("Multi-Krum requires n >= 2f + 3")
    retained_count = count - assumed_byzantine if selected_count is None else selected_count
    if (
        isinstance(retained_count, bool)
        or retained_count < 1
        or retained_count > count - assumed_byzantine
    ):
        raise ValueError("Multi-Krum selected count must be in [1, n-f]")

    ordered_updates = tuple(sorted(updates, key=lambda item: item.client_id))
    stacked = torch.stack(tuple(update.delta for update in ordered_updates))
    distances = torch.cdist(stacked, stacked, p=2).square()
    neighbor_count = count - assumed_byzantine - 2
    candidates: list[tuple[float, str, int]] = []
    for index, update in enumerate(ordered_updates):
        other_distances = torch.cat((distances[index, :index], distances[index, index + 1 :]))
        nearest = torch.topk(other_distances, k=neighbor_count, largest=False).values
        score = float(nearest.sum().item())
        if not math.isfinite(score):
            raise ValueError("Multi-Krum produced a non-finite candidate score")
        candidates.append((score, update.client_id, index))

    selected = sorted(candidates)[:retained_count]
    selected_ids = {client_id for _, client_id, _ in selected}
    weight = 1.0 / retained_count
    aggregate = torch.stack(tuple(stacked[index] for _, _, index in selected)).mean(dim=0)
    weights = {
        update.client_id: weight for update in ordered_updates if update.client_id in selected_ids
    }
    return AggregationResult("multi_krum", aggregate, weights)
