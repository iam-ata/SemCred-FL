"""Sample-count-weighted FedAvg over validated client updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """One aggregate update and its auditable client weights."""

    method: str
    delta: Tensor
    weights: dict[str, float]


def fedavg(updates: tuple[DefenseUpdate, ...]) -> AggregationResult:
    """Return the declared-sample-count-weighted mean update."""

    if not updates:
        raise ValueError("FedAvg requires at least one update")
    if len({update.client_id for update in updates}) != len(updates):
        raise ValueError("FedAvg client IDs must be unique")
    shape = updates[0].delta.shape
    if any(update.delta.shape != shape for update in updates):
        raise ValueError("FedAvg updates have inconsistent shapes")
    if any(update.declared_sample_count < 1 for update in updates):
        raise ValueError("FedAvg sample counts must be positive")
    if any(not torch.isfinite(update.delta).all().item() for update in updates):
        raise ValueError("FedAvg received a non-finite update")
    total = sum(update.declared_sample_count for update in updates)
    weights = {
        update.client_id: update.declared_sample_count / total
        for update in sorted(updates, key=lambda item: item.client_id)
    }
    aggregate = torch.zeros_like(updates[0].delta)
    for update in updates:
        aggregate = aggregate + update.delta * weights[update.client_id]
    return AggregationResult("fedavg", aggregate, weights)
