"""Server-side algorithmic core of Fed-LSAE."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from semcred_fl.baselines.fedavg import AggregationResult, fedavg
from semcred_fl.federation.updates import DefenseUpdate


def rbf_kernel(representation: Tensor, *, bandwidth: float) -> Tensor:
    """Return an RBF Gram matrix for a sample-by-feature representation."""

    if representation.ndim != 2 or representation.shape[0] < 2:
        raise ValueError("RBF representation must contain at least two samples")
    if not torch.isfinite(representation).all().item():
        raise ValueError("RBF representation must be finite")
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("RBF bandwidth must be finite and positive")
    squared_distances = torch.cdist(representation, representation, p=2).square()
    return torch.exp(-squared_distances / (2.0 * bandwidth**2))


def centered_kernel_alignment(
    first_kernel: Tensor,
    second_kernel: Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Compute normalized HSIC alignment between two kernel matrices."""

    if (
        first_kernel.ndim != 2
        or second_kernel.ndim != 2
        or first_kernel.shape != second_kernel.shape
        or first_kernel.shape[0] != first_kernel.shape[1]
        or first_kernel.shape[0] < 2
    ):
        raise ValueError("CKA requires equal square kernel matrices")
    if (
        not torch.isfinite(first_kernel).all().item()
        or not torch.isfinite(second_kernel).all().item()
    ):
        raise ValueError("CKA kernels must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("CKA epsilon must be finite and positive")

    count = first_kernel.shape[0]
    identity = torch.eye(count, dtype=first_kernel.dtype, device=first_kernel.device)
    centering = identity - torch.full_like(identity, 1.0 / count)
    first_centered = centering @ first_kernel @ centering
    second_centered = centering @ second_kernel @ centering
    numerator = torch.sum(first_centered * second_centered)
    denominator = torch.sqrt(
        torch.sum(first_centered.square()) * torch.sum(second_centered.square())
    )
    denominator_value = float(denominator.item())
    if denominator_value <= epsilon:
        raise ValueError("CKA is undefined for a degenerate centered kernel")
    score = float((numerator / denominator).item())
    if not math.isfinite(score):
        raise ValueError("CKA produced a non-finite score")
    return min(1.0, max(0.0, score))


def rbf_cka(first: Tensor, second: Tensor, *, bandwidth: float) -> float:
    """Compute RBF-CKA between equal sample populations."""

    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("RBF-CKA representations must share a sample population")
    return centered_kernel_alignment(
        rbf_kernel(first, bandwidth=bandwidth),
        rbf_kernel(second, bandwidth=bandwidth),
    )


def _cluster_score(values: tuple[float, ...]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values)


def fed_lsae_partition(cka_scores: Mapping[str, float]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Solve deterministic one-dimensional two-means and return benign/suspect IDs."""

    if len(cka_scores) < 3:
        raise ValueError("Fed-LSAE requires at least three CKA scores")
    if any(not client_id for client_id in cka_scores):
        raise ValueError("Fed-LSAE client IDs must be nonempty")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in cka_scores.values()):
        raise ValueError("Fed-LSAE CKA scores must be finite and in [0,1]")

    ordered = tuple(sorted(cka_scores.items(), key=lambda item: (item[1], item[0])))
    candidates: list[tuple[float, int]] = []
    for split in range(1, len(ordered)):
        if ordered[split - 1][1] == ordered[split][1]:
            continue
        left = tuple(value for _, value in ordered[:split])
        right = tuple(value for _, value in ordered[split:])
        candidates.append((_cluster_score(left) + _cluster_score(right), split))
    if not candidates:
        raise ValueError("Fed-LSAE cannot form two distinct CKA clusters")
    candidates.sort()
    best_score, split = candidates[0]
    if len(candidates) > 1 and math.isclose(
        best_score, candidates[1][0], rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("Fed-LSAE two-means optimum is ambiguous")

    left_ids = tuple(sorted(client_id for client_id, _ in ordered[:split]))
    right_ids = tuple(sorted(client_id for client_id, _ in ordered[split:]))
    if len(left_ids) == len(right_ids):
        raise ValueError("Fed-LSAE paper does not resolve equal-size cluster labels")
    benign = left_ids if len(left_ids) > len(right_ids) else right_ids
    suspected = right_ids if benign is left_ids else left_ids
    return benign, suspected


def fed_lsae_aggregate(
    updates: tuple[DefenseUpdate, ...],
    cka_scores: Mapping[str, float],
) -> AggregationResult:
    """Filter the smaller CKA cluster and apply sample-weighted FedAvg."""

    if {update.client_id for update in updates} != set(cka_scores):
        raise ValueError("Fed-LSAE update and CKA populations differ")
    benign_ids, _ = fed_lsae_partition(cka_scores)
    benign_set = set(benign_ids)
    selected = tuple(update for update in updates if update.client_id in benign_set)
    result = fedavg(selected)
    return AggregationResult("fed_lsae_selection_core", result.delta, result.weights)
