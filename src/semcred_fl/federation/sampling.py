"""Pre-generated deterministic client-participation schedules."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from semcred_fl.identity import run_id
from semcred_fl.seeding import derive_seed


@dataclass(frozen=True, slots=True)
class ClientSchedule:
    """Immutable selected-client sequence shared across compared methods."""

    client_ids: tuple[str, ...]
    rounds: tuple[tuple[str, ...], ...]
    clients_per_round: int
    seed: int
    schedule_id: str

    def __post_init__(self) -> None:
        known = set(self.client_ids)
        if len(known) != len(self.client_ids) or not known:
            raise ValueError("Client schedule population must be nonempty and unique")
        if self.seed < 0 or not 1 <= self.clients_per_round <= len(known):
            raise ValueError("Client schedule controls are invalid")
        if not self.rounds:
            raise ValueError("Client schedule requires at least one round")
        for selected in self.rounds:
            if len(selected) != self.clients_per_round or len(set(selected)) != len(selected):
                raise ValueError("A schedule round has the wrong number of unique clients")
            if not set(selected).issubset(known):
                raise ValueError("A schedule round contains an unknown client")
        stable = self._stable_record()
        if run_id(stable, length=64) != self.schedule_id:
            raise ValueError("Client schedule identity is invalid")

    def _stable_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "client_ids": list(self.client_ids),
            "rounds": [list(selected) for selected in self.rounds],
            "clients_per_round": self.clients_per_round,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible schedule record."""

        return {"schedule_id": self.schedule_id, **self._stable_record()}

    def write(self, path: Path) -> None:
        """Atomically persist the immutable schedule."""

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial.replace(path)


def generate_client_schedule(
    client_ids: tuple[str, ...], *, rounds: int, clients_per_round: int, seed: int
) -> ClientSchedule:
    """Generate round-stratified selections without replacement within a round."""

    population = tuple(sorted(client_ids))
    if len(set(population)) != len(population) or not population:
        raise ValueError("Client IDs must be nonempty and unique")
    if rounds < 1 or seed < 0 or not 1 <= clients_per_round <= len(population):
        raise ValueError("Schedule generation controls are invalid")
    selections: list[tuple[str, ...]] = []
    for round_id in range(rounds):
        rng = random.Random(derive_seed(seed, f"phase5-client-sampling:{round_id}"))
        selections.append(tuple(sorted(rng.sample(population, clients_per_round))))
    stable: dict[str, object] = {
        "schema_version": 1,
        "client_ids": list(population),
        "rounds": [list(selected) for selected in selections],
        "clients_per_round": clients_per_round,
        "seed": seed,
    }
    return ClientSchedule(
        client_ids=population,
        rounds=tuple(selections),
        clients_per_round=clients_per_round,
        seed=seed,
        schedule_id=run_id(stable, length=64),
    )
