"""Client-update contracts and defense-safe validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ClientUpdate:
    """Simulation-side local-training output.

    ``is_malicious`` and ``attack_metadata`` are evaluation fields. Defense and
    aggregation code receive only the projected :class:`DefenseUpdate`.
    """

    round_id: int
    client_id: str
    delta: Tensor
    declared_sample_count: int
    mean_local_loss: float
    epoch_durations_seconds: tuple[float, ...]
    update_norm: float
    finite: bool
    missing_class_ids: tuple[int, ...] = ()
    is_malicious: bool | None = None
    attack_metadata: dict[str, object] | None = None

    def to_defense_update(self) -> DefenseUpdate:
        """Remove simulation-only attacker fields before defense processing."""

        return DefenseUpdate(
            round_id=self.round_id,
            client_id=self.client_id,
            delta=self.delta,
            declared_sample_count=self.declared_sample_count,
            mean_local_loss=self.mean_local_loss,
            update_norm=self.update_norm,
            finite=self.finite,
            missing_class_ids=self.missing_class_ids,
        )


@dataclass(frozen=True, slots=True)
class DefenseUpdate:
    """Client update visible to validation and aggregation code."""

    round_id: int
    client_id: str
    delta: Tensor
    declared_sample_count: int
    mean_local_loss: float
    update_norm: float
    finite: bool
    missing_class_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UpdateValidation:
    """Accepted defense updates and deterministic rejection reasons."""

    accepted: tuple[DefenseUpdate, ...]
    rejected: dict[str, str]


def validate_client_updates(
    updates: tuple[ClientUpdate, ...], *, expected_round: int, expected_values: int
) -> UpdateValidation:
    """Validate structural and finite-value gates without reading attacker labels."""

    if expected_round < 0 or expected_values < 1:
        raise ValueError("Expected round and vector cardinality are invalid")
    counts: dict[str, int] = {}
    for update in updates:
        counts[update.client_id] = counts.get(update.client_id, 0) + 1
    accepted: list[DefenseUpdate] = []
    rejected: dict[str, str] = {}
    for update in updates:
        client_id = update.client_id
        reason: str | None = None
        if not client_id:
            reason = "empty_client_id"
        elif counts[client_id] != 1:
            reason = "duplicate_client_id"
        elif update.round_id != expected_round:
            reason = "wrong_round"
        elif update.delta.ndim != 1 or update.delta.numel() != expected_values:
            reason = "wrong_update_shape"
        elif update.declared_sample_count < 1:
            reason = "invalid_declared_sample_count"
        elif not update.finite or not torch.isfinite(update.delta).all().item():
            reason = "nonfinite_update"
        elif not math.isfinite(update.mean_local_loss) or update.mean_local_loss < 0.0:
            reason = "invalid_local_loss"
        elif not math.isfinite(update.update_norm) or update.update_norm < 0.0:
            reason = "invalid_update_norm"
        elif not math.isclose(
            update.update_norm,
            float(torch.linalg.vector_norm(update.delta).item()),
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            reason = "update_norm_mismatch"
        if reason is None:
            accepted.append(update.to_defense_update())
        else:
            rejected[client_id or "<empty>"] = reason
    accepted.sort(key=lambda item: item.client_id)
    return UpdateValidation(tuple(accepted), dict(sorted(rejected.items())))
