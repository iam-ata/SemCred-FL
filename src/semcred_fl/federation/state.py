"""Persistent trust and accepted-update history for SemCred-FL clients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from semcred_fl.aggregation.trust import asymmetric_trust


@dataclass(frozen=True, slots=True)
class ClientState:
    """Defense state changed only by participation and accepted-history events."""

    trust: float
    accepted_history: Tensor | None
    first_seen_round: int
    last_seen_round: int
    accepted_rounds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.trust) or not 0.0 <= self.trust <= 1.0:
            raise ValueError("Client trust must be finite and in [0,1]")
        if self.first_seen_round < 0 or self.last_seen_round < self.first_seen_round:
            raise ValueError("Client observation rounds are invalid")
        if self.accepted_rounds != tuple(sorted(set(self.accepted_rounds))):
            raise ValueError("Accepted rounds must be unique and increasing")
        if any(
            round_id < self.first_seen_round or round_id > self.last_seen_round
            for round_id in self.accepted_rounds
        ):
            raise ValueError("Accepted history round lies outside the observation interval")
        if self.accepted_history is None and self.accepted_rounds:
            raise ValueError("Accepted rounds require an accepted-history vector")
        if self.accepted_history is not None and (
            self.accepted_history.ndim != 1
            or self.accepted_history.numel() < 1
            or not torch.isfinite(self.accepted_history).all().item()
        ):
            raise ValueError("Accepted history must be a finite nonempty flat vector")


def update_participating_trust(
    states: Mapping[str, ClientState],
    fused_scores: Mapping[str, float],
    *,
    round_id: int,
    decrease_rate: float = 0.6,
    recovery_rate: float = 0.1,
) -> dict[str, ClientState]:
    """Update participating clients while preserving every inactive state unchanged."""

    if round_id < 0 or not fused_scores or any(not client_id for client_id in fused_scores):
        raise ValueError("Trust update requires a valid round and participating clients")
    if not 0.0 < recovery_rate < decrease_rate <= 1.0:
        raise ValueError("Trust recovery must be slower than trust decrease")
    result = dict(states)
    for client_id in sorted(fused_scores):
        credibility = fused_scores[client_id]
        if not math.isfinite(credibility) or not 0.0 <= credibility <= 1.0:
            raise ValueError("Fused credibility must be finite and in [0,1]")
        previous = states.get(client_id)
        if previous is None:
            result[client_id] = ClientState(
                trust=credibility,
                accepted_history=None,
                first_seen_round=round_id,
                last_seen_round=round_id,
            )
            continue
        if round_id <= previous.last_seen_round:
            raise ValueError("A participating client requires a later round")
        trust = asymmetric_trust(
            previous.trust,
            credibility,
            decrease_rate=decrease_rate,
            recovery_rate=recovery_rate,
        )
        result[client_id] = ClientState(
            trust=min(1.0, max(0.0, trust)),
            accepted_history=previous.accepted_history,
            first_seen_round=previous.first_seen_round,
            last_seen_round=round_id,
            accepted_rounds=previous.accepted_rounds,
        )
    return result


def admit_clipped_histories(
    states: Mapping[str, ClientState],
    admitted_updates: Mapping[str, Tensor],
    *,
    round_id: int,
    history_decay: float = 0.8,
) -> dict[str, ClientState]:
    """Initialize or update history only for explicitly admitted clipped updates."""

    if round_id < 0 or not 0.0 <= history_decay < 1.0:
        raise ValueError("History round or decay is invalid")
    if any(not client_id for client_id in admitted_updates):
        raise ValueError("History admission requires nonempty client IDs")
    result = dict(states)
    for client_id in sorted(admitted_updates):
        state = states.get(client_id)
        if state is None or state.last_seen_round != round_id:
            raise ValueError("Only a client participating in this round may enter history")
        update = admitted_updates[client_id]
        if update.ndim != 1 or update.numel() < 1 or not torch.isfinite(update).all().item():
            raise ValueError("Admitted history update must be a finite nonempty flat vector")
        if round_id in state.accepted_rounds:
            raise ValueError("A client update cannot be admitted twice in one round")
        if state.accepted_history is None:
            history = update.detach().clone()
        else:
            if state.accepted_history.shape != update.shape:
                raise ValueError("Accepted history and clipped update shapes differ")
            history = (
                history_decay * state.accepted_history
                + (1.0 - history_decay) * update.to(state.accepted_history)
            ).detach()
        result[client_id] = ClientState(
            trust=state.trust,
            accepted_history=history,
            first_seen_round=state.first_seen_round,
            last_seen_round=state.last_seen_round,
            accepted_rounds=(*state.accepted_rounds, round_id),
        )
    return result
