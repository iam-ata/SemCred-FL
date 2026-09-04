"""Median-based norm clipping for eligible defense updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from semcred_fl.credibility.robust_stats import median
from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class ClippingResult:
    """Clipped eligible updates and their audited norms."""

    bound: float
    clipped_updates: dict[str, Tensor]
    original_norms: dict[str, float]
    clipped_norms: dict[str, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.bound) or self.bound < 0.0:
            raise ValueError("Clipping bound must be finite and nonnegative")
        keys = set(self.clipped_updates)
        if not keys or keys != set(self.original_norms) or keys != set(self.clipped_norms):
            raise ValueError("Clipping result inventories differ")
        tolerance = max(1e-7, self.bound * 1e-6)
        if any(value > self.bound + tolerance for value in self.clipped_norms.values()):
            raise ValueError("A clipped update exceeds the clipping bound")


def clip_update(delta: Tensor, bound: float, *, epsilon: float = 1e-12) -> Tensor:
    """Clip one flat update without changing its direction when the scale is positive."""

    if delta.ndim != 1 or delta.numel() < 1 or not torch.isfinite(delta).all().item():
        raise ValueError("Clipping requires a finite nonempty flat update")
    if not math.isfinite(bound) or bound < 0.0:
        raise ValueError("Clipping bound must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    norm = float(torch.linalg.vector_norm(delta).item())
    scale = min(1.0, bound / (norm + epsilon))
    result = (delta * scale).detach().clone()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("Clipping produced a non-finite update")
    return result


def clip_eligible_updates(
    updates: Mapping[str, DefenseUpdate],
    eligible_client_ids: tuple[str, ...],
    *,
    clipping_multiplier: float = 2.5,
    epsilon: float = 1e-12,
) -> ClippingResult:
    """Compute an eligible-set median bound and clip exactly that set."""

    if not eligible_client_ids or len(set(eligible_client_ids)) != len(eligible_client_ids):
        raise ValueError("Clipping requires unique eligible clients")
    if any(client_id not in updates for client_id in eligible_client_ids):
        raise ValueError("An eligible client has no accepted defense update")
    if not math.isfinite(clipping_multiplier) or clipping_multiplier <= 0.0:
        raise ValueError("clipping_multiplier must be finite and positive")
    original_norms = {
        client_id: float(torch.linalg.vector_norm(updates[client_id].delta).item())
        for client_id in eligible_client_ids
    }
    if any(not math.isfinite(value) for value in original_norms.values()):
        raise ValueError("Eligible update norm is non-finite")
    bound = clipping_multiplier * median(tuple(original_norms.values()))
    clipped = {
        client_id: clip_update(updates[client_id].delta, bound, epsilon=epsilon)
        for client_id in eligible_client_ids
    }
    clipped_norms = {
        client_id: float(torch.linalg.vector_norm(delta).item())
        for client_id, delta in clipped.items()
    }
    return ClippingResult(bound, clipped, original_norms, clipped_norms)
