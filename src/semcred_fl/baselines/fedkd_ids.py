"""Native public-data/logit aggregation core of FedKD-IDS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

ABSTAIN_LABEL = -1


@dataclass(frozen=True, slots=True)
class FedKdIdsAggregationResult:
    """Auditable global labels and server-verifier diagnostics."""

    global_labels: Tensor
    eligible_client_ids: tuple[str, ...]
    hamming_distances: dict[str, dict[str, float]]
    harmonic_means: dict[str, float]
    threshold: float


def apply_local_verifier_mask(detector_labels: Tensor, verifier_labels: Tensor) -> Tensor:
    """Replace detector labels with -1 where the local verifier rejects a sample."""

    if detector_labels.ndim != 1 or verifier_labels.shape != detector_labels.shape:
        raise ValueError("FedKD-IDS local label vectors must be equal and one-dimensional")
    if detector_labels.dtype == torch.bool or verifier_labels.dtype == torch.bool:
        raise ValueError("FedKD-IDS labels must use integer class IDs")
    if detector_labels.is_floating_point() or verifier_labels.is_floating_point():
        raise ValueError("FedKD-IDS labels must use integer class IDs")
    if not torch.all((verifier_labels == 0) | (verifier_labels == 1)).item():
        raise ValueError("FedKD-IDS verifier labels must be binary")
    masked = detector_labels.clone()
    masked[verifier_labels == 1] = ABSTAIN_LABEL
    return masked


def _harmonic_mean(distances: tuple[float, ...]) -> float:
    if any(distance == 0.0 for distance in distances):
        return 0.0
    return len(distances) / sum(1.0 / distance for distance in distances)


def aggregate_public_labels(
    local_labels: Mapping[str, Tensor],
    *,
    class_count: int,
    threshold_multiplier: float = 0.5,
) -> FedKdIdsAggregationResult:
    """Apply the FedKD-IDS global verifier and hard-vote public labels."""

    if len(local_labels) < 2:
        raise ValueError("FedKD-IDS requires at least two collaborators")
    if class_count < 2:
        raise ValueError("FedKD-IDS class count must be at least two")
    if not math.isfinite(threshold_multiplier) or threshold_multiplier <= 0.0:
        raise ValueError("FedKD-IDS threshold multiplier must be finite and positive")
    client_ids = tuple(sorted(local_labels))
    if any(not client_id for client_id in client_ids):
        raise ValueError("FedKD-IDS collaborator IDs must be nonempty")
    first = local_labels[client_ids[0]]
    if first.ndim != 1 or first.numel() < 1:
        raise ValueError("FedKD-IDS public-label vectors must be nonempty and flat")
    for labels in local_labels.values():
        if labels.shape != first.shape or labels.dtype == torch.bool or labels.is_floating_point():
            raise ValueError("FedKD-IDS public-label vectors have inconsistent contracts")
        valid = (labels == ABSTAIN_LABEL) | ((labels >= 0) & (labels < class_count))
        if not torch.all(valid).item():
            raise ValueError("FedKD-IDS public labels contain an invalid class ID")

    pairwise: dict[str, dict[str, float]] = {client_id: {} for client_id in client_ids}
    for position, first_id in enumerate(client_ids):
        for second_id in client_ids[position + 1 :]:
            distance = float(
                torch.mean(
                    (local_labels[first_id] != local_labels[second_id]).to(torch.float64)
                ).item()
            )
            pairwise[first_id][second_id] = distance
            pairwise[second_id][first_id] = distance
    harmonic_means = {
        client_id: _harmonic_mean(tuple(pairwise[client_id].values())) for client_id in client_ids
    }
    threshold = threshold_multiplier * sum(harmonic_means.values()) / len(harmonic_means)
    eligible = tuple(client_id for client_id in client_ids if harmonic_means[client_id] < threshold)
    if not eligible:
        raise ValueError("FedKD-IDS global verifier retained no collaborator")

    global_values: list[int] = []
    for sample_index in range(first.numel()):
        votes = torch.zeros(class_count, dtype=torch.int64)
        for client_id in eligible:
            label = int(local_labels[client_id][sample_index].item())
            if label != ABSTAIN_LABEL:
                votes[label] += 1
        maximum = int(votes.max().item())
        if maximum == 0:
            raise ValueError("FedKD-IDS received no eligible vote for a public sample")
        winners = torch.nonzero(votes == maximum, as_tuple=False).flatten()
        if winners.numel() != 1:
            raise ValueError("FedKD-IDS paper does not specify a hard-vote tie break")
        global_values.append(int(winners[0].item()))

    return FedKdIdsAggregationResult(
        global_labels=torch.tensor(global_values, dtype=first.dtype),
        eligible_client_ids=eligible,
        hamming_distances=pairwise,
        harmonic_means=harmonic_means,
        threshold=threshold,
    )
