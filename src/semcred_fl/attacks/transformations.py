"""Deterministic Phase 6 data- and model-poisoning transformations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from semcred_fl.attacks.artifacts import Phase6BResolution


@dataclass(frozen=True, slots=True)
class PoisonedBatch:
    """One transformed local batch plus simulation-only audit fields."""

    features: Tensor
    targets: Tensor
    selected_indices: tuple[int, ...]
    attack_id: str
    poison_fraction: float


@dataclass(frozen=True, slots=True)
class TriggeredEvaluationBatch:
    """Triggered source-family test rows and natural-collision accounting."""

    features: Tensor
    source_targets: Tensor
    original_indices: tuple[int, ...]
    source_records: int
    natural_collisions: int

    @property
    def asr_denominator(self) -> int:
        """Return the number of newly triggered records eligible for ASR."""

        return self.source_records - self.natural_collisions


def _validate_batch(features: Tensor, targets: Tensor, record_ids: Sequence[str]) -> None:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("Attack batch shapes are inconsistent")
    if features.shape[0] != len(record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Attack record IDs must be unique and batch-aligned")
    if not torch.isfinite(features).all().item():
        raise ValueError("Attack input features must be finite")


def selected_source_indices(
    *,
    targets: Tensor,
    record_ids: Sequence[str],
    source_family_id: int,
    poison_fraction: float,
    run_seed: int,
    attack_id: str,
    client_id: str,
) -> tuple[int, ...]:
    """Select the lowest deterministic SHA-256 ranks without replacement."""

    if not 0.0 <= poison_fraction <= 1.0 or run_seed < 0 or not attack_id or not client_id:
        raise ValueError("Poison-selection controls are invalid")
    if targets.ndim != 1 or len(record_ids) != targets.shape[0]:
        raise ValueError("Poison-selection inputs are not aligned")
    eligible = [
        index for index, value in enumerate(targets.tolist()) if int(value) == source_family_id
    ]
    selected_count = math.floor(poison_fraction * len(eligible))
    if selected_count == 0:
        return ()

    def rank(index: int) -> tuple[bytes, int]:
        payload = f"{run_seed}\0{attack_id}\0{client_id}\0{record_ids[index]}".encode()
        return hashlib.sha256(payload).digest(), index

    return tuple(sorted(sorted(eligible, key=rank)[:selected_count]))


def selected_record_indices(
    record_ids: Sequence[str], selected_record_ids: Sequence[str]
) -> tuple[int, ...]:
    """Locate a precomputed client-population selection within one streamed batch."""

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Batch record IDs must be unique")
    if len(set(selected_record_ids)) != len(selected_record_ids):
        raise ValueError("Precomputed poison record IDs must be unique")
    selected = set(selected_record_ids)
    return tuple(index for index, record_id in enumerate(record_ids) if record_id in selected)


def targeted_label_flip_selected(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    selected_record_ids: Sequence[str],
    source_family_id: int,
    target_family_id: int,
    poison_fraction: float,
    attack_id: str,
) -> PoisonedBatch:
    """Relabel an immutable client-population selection appearing in this batch."""

    _validate_batch(features, targets, record_ids)
    selected = selected_record_indices(record_ids, selected_record_ids)
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    rows = torch.tensor(selected, dtype=torch.long, device=targets.device)
    if torch.any(targets[rows] != source_family_id).item():
        raise ValueError("Precomputed label-flip selection contains a non-source row")
    poisoned_targets = targets.clone()
    poisoned_targets[rows] = target_family_id
    return PoisonedBatch(features, poisoned_targets, selected, attack_id, poison_fraction)


def clean_label_poison_selected(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    selected_record_ids: Sequence[str],
    resolution: Phase6BResolution,
    poison_fraction: float,
    strength: float,
    attack_id: str,
) -> PoisonedBatch:
    """Apply clean-label movement to a precomputed client-population selection."""

    _validate_batch(features, targets, record_ids)
    if strength not in {0.5, 1.0}:
        raise ValueError("Clean-label poisoning strength must be 0.5 or 1.0")
    selected = selected_record_indices(record_ids, selected_record_ids)
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    rows = torch.tensor(selected, dtype=torch.long, device=features.device)
    if torch.any(targets[rows] != resolution.source_family_id).item():
        raise ValueError("Precomputed clean-label selection contains a non-source row")
    poisoned_features = features.clone()
    for coordinate in resolution.feature_coordinates:
        current = poisoned_features[rows, coordinate.index]
        moved = current + strength * (coordinate.benign_median - current)
        poisoned_features[rows, coordinate.index] = torch.clamp(
            moved, min=coordinate.lower_bound, max=coordinate.upper_bound
        )
    return PoisonedBatch(poisoned_features, targets, selected, attack_id, poison_fraction)


def backdoor_poison_selected(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    selected_record_ids: Sequence[str],
    resolution: Phase6BResolution,
    poison_fraction: float,
    attack_id: str,
) -> PoisonedBatch:
    """Insert the frozen trigger into a precomputed client-population selection."""

    _validate_batch(features, targets, record_ids)
    selected = selected_record_indices(record_ids, selected_record_ids)
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    rows = torch.tensor(selected, dtype=torch.long, device=features.device)
    if torch.any(targets[rows] != resolution.source_family_id).item():
        raise ValueError("Precomputed backdoor selection contains a non-source row")
    poisoned_features = features.clone()
    poisoned_targets = targets.clone()
    for coordinate in resolution.trigger_coordinates:
        poisoned_features[rows, coordinate.index] = coordinate.value
    poisoned_targets[rows] = resolution.target_family_id
    return PoisonedBatch(poisoned_features, poisoned_targets, selected, attack_id, poison_fraction)


def targeted_label_flip(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    source_family_id: int,
    target_family_id: int,
    poison_fraction: float,
    run_seed: int,
    attack_id: str,
    client_id: str,
) -> PoisonedBatch:
    """Relabel the deterministic source-family subset while preserving features."""

    _validate_batch(features, targets, record_ids)
    selected = selected_source_indices(
        targets=targets,
        record_ids=record_ids,
        source_family_id=source_family_id,
        poison_fraction=poison_fraction,
        run_seed=run_seed,
        attack_id=attack_id,
        client_id=client_id,
    )
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    poisoned_targets = targets.clone()
    poisoned_targets[list(selected)] = target_family_id
    return PoisonedBatch(features, poisoned_targets, selected, attack_id, poison_fraction)


def clean_label_poison(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    resolution: Phase6BResolution,
    poison_fraction: float,
    strength: float,
    run_seed: int,
    attack_id: str,
    client_id: str,
) -> PoisonedBatch:
    """Move selected source rows toward benign medians without changing labels."""

    _validate_batch(features, targets, record_ids)
    if strength not in {0.5, 1.0}:
        raise ValueError("Clean-label poisoning strength must be 0.5 or 1.0")
    selected = selected_source_indices(
        targets=targets,
        record_ids=record_ids,
        source_family_id=resolution.source_family_id,
        poison_fraction=poison_fraction,
        run_seed=run_seed,
        attack_id=attack_id,
        client_id=client_id,
    )
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    poisoned_features = features.clone()
    rows = torch.tensor(selected, dtype=torch.long, device=features.device)
    for coordinate in resolution.feature_coordinates:
        current = poisoned_features[rows, coordinate.index]
        moved = current + strength * (coordinate.benign_median - current)
        poisoned_features[rows, coordinate.index] = torch.clamp(
            moved, min=coordinate.lower_bound, max=coordinate.upper_bound
        )
    return PoisonedBatch(poisoned_features, targets, selected, attack_id, poison_fraction)


def backdoor_poison(
    *,
    features: Tensor,
    targets: Tensor,
    record_ids: Sequence[str],
    resolution: Phase6BResolution,
    poison_fraction: float,
    run_seed: int,
    attack_id: str,
    client_id: str,
) -> PoisonedBatch:
    """Apply the frozen trigger and benign target to selected source rows."""

    _validate_batch(features, targets, record_ids)
    selected = selected_source_indices(
        targets=targets,
        record_ids=record_ids,
        source_family_id=resolution.source_family_id,
        poison_fraction=poison_fraction,
        run_seed=run_seed,
        attack_id=attack_id,
        client_id=client_id,
    )
    if not selected:
        return PoisonedBatch(features, targets, (), attack_id, poison_fraction)
    poisoned_features = features.clone()
    poisoned_targets = targets.clone()
    rows = torch.tensor(selected, dtype=torch.long, device=features.device)
    for coordinate in resolution.trigger_coordinates:
        poisoned_features[rows, coordinate.index] = coordinate.value
    poisoned_targets[rows] = resolution.target_family_id
    return PoisonedBatch(poisoned_features, poisoned_targets, selected, attack_id, poison_fraction)


def triggered_test_batch(
    *, features: Tensor, targets: Tensor, resolution: Phase6BResolution
) -> TriggeredEvaluationBatch:
    """Create triggered clean source-family test rows and exclude natural collisions."""

    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("Triggered-test batch shapes are inconsistent")
    source_indices = torch.nonzero(targets == resolution.source_family_id, as_tuple=False).flatten()
    if source_indices.numel() == 0:
        empty = features.new_empty((0, features.shape[1]))
        return TriggeredEvaluationBatch(empty, targets.new_empty((0,)), (), 0, 0)
    source_features = features[source_indices]
    collision_mask = torch.ones(source_features.shape[0], dtype=torch.bool, device=features.device)
    for coordinate in resolution.trigger_coordinates:
        collision_mask &= source_features[:, coordinate.index] == coordinate.value
    eligible = ~collision_mask
    eligible_indices = source_indices[eligible]
    triggered = features[eligible_indices].clone()
    for coordinate in resolution.trigger_coordinates:
        triggered[:, coordinate.index] = coordinate.value
    return TriggeredEvaluationBatch(
        features=triggered,
        source_targets=targets[eligible_indices].clone(),
        original_indices=tuple(int(value) for value in eligible_indices.tolist()),
        source_records=int(source_indices.numel()),
        natural_collisions=int(collision_mask.sum().item()),
    )


def scale_update(delta: Tensor, scale: float) -> Tensor:
    """Return ``scale * delta`` for the frozen scaling attack."""

    if scale <= 0.0 or not math.isfinite(scale) or not torch.isfinite(delta).all().item():
        raise ValueError("Update scaling inputs are invalid")
    return delta * scale


def reverse_update(delta: Tensor, scale: float) -> Tensor:
    """Return ``-scale * delta`` for the frozen sign-reversal attack."""

    return -scale_update(delta, scale)


def coalition_model_replacement(
    *,
    target_parameters: Tensor,
    global_parameters: Tensor,
    participants_per_round: int,
    coalition_size: int,
) -> Tensor:
    """Construct the FedAvg-oriented coalition model-replacement update."""

    if (
        target_parameters.shape != global_parameters.shape
        or participants_per_round < 1
        or not 1 <= coalition_size <= participants_per_round
        or not torch.isfinite(target_parameters).all().item()
        or not torch.isfinite(global_parameters).all().item()
    ):
        raise ValueError("Model-replacement inputs are invalid")
    factor = participants_per_round / coalition_size
    return factor * (target_parameters - global_parameters)


def constrain_update_norm(delta: Tensor, lower_norm: float, upper_norm: float) -> Tensor:
    """Radially project an update norm into a frozen benign interquartile interval."""

    if (
        not 0.0 <= lower_norm <= upper_norm
        or not math.isfinite(lower_norm)
        or not math.isfinite(upper_norm)
        or not torch.isfinite(delta).all().item()
    ):
        raise ValueError("Norm-constrained attack inputs are invalid")
    norm = float(torch.linalg.vector_norm(delta).item())
    if norm == 0.0:
        if lower_norm > 0.0:
            raise ValueError("A zero update has no direction for a positive lower norm")
        return delta
    target_norm = min(max(norm, lower_norm), upper_norm)
    if target_norm == norm:
        return delta
    return delta * (target_norm / norm)
