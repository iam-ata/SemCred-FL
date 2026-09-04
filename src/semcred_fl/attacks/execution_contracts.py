"""Frozen Phase 6C attack-execution and stratified-realization contracts."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from semcred_fl.federation.sampling import ClientSchedule
from semcred_fl.identity import run_id
from semcred_fl.seeding import derive_seed

_ATTACK_NAMES = frozenset(
    {
        "clean",
        "label_flip",
        "clean_label",
        "backdoor",
        "scaling",
        "sign_reversal",
        "model_replacement",
        "adaptive_norm",
    }
)
_DATA_ATTACKS = frozenset(
    {"label_flip", "clean_label", "backdoor", "model_replacement", "adaptive_norm"}
)


@dataclass(frozen=True, slots=True)
class Phase6CExecutionConfig:
    """Shared model, training, and realization controls for Phase 6C."""

    schema_version: int = 1
    phase: str = "6C"
    base_seed: int = 20260816
    total_clients: int = 50
    clients_per_round: int = 20
    total_rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    proximal_mu: float = 0.0
    maximum_class_weight: float = 10.0
    hidden_dimensions: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    profiles: tuple[str, ...] = (
        "iid",
        "dirichlet_0_1",
        "dirichlet_0_3",
        "dirichlet_1_0",
    )
    malicious_client_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    poison_fractions: tuple[float, ...] = (0.1, 0.3, 0.5)
    clean_label_strengths: tuple[float, ...] = (0.5, 1.0)
    scaling_factors: tuple[float, ...] = (5.0, 10.0)
    reversal_factors: tuple[float, ...] = (1.0, 5.0, 10.0)
    attack_names: tuple[str, ...] = tuple(sorted(_ATTACK_NAMES))
    schedule_mode: str = "stratified_fixed_coalition_malicious_minority"
    test_partition_accessed: bool = False
    validation_partition_accessed: bool = False
    anchor_records_accessed: bool = False
    performance_claims_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.phase != "6C" or self.base_seed < 0:
            raise ValueError("Phase 6C execution identity is invalid")
        if not 1 <= self.clients_per_round <= self.total_clients or self.total_rounds < 1:
            raise ValueError("Phase 6C client or round controls are invalid")
        if self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("Phase 6C local-training controls are invalid")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("Phase 6C learning rate is invalid")
        if self.proximal_mu < 0.0 or not math.isfinite(self.proximal_mu):
            raise ValueError("Phase 6C proximal coefficient is invalid")
        if self.maximum_class_weight < 1.0:
            raise ValueError("Phase 6C class-weight cap is invalid")
        if not self.hidden_dimensions or any(value < 1 for value in self.hidden_dimensions):
            raise ValueError("Phase 6C detector dimensions are invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Phase 6C dropout is invalid")
        if set(self.attack_names) != _ATTACK_NAMES:
            raise ValueError("Phase 6C attack family inventory differs from Phase 6A")
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Phase 6C profile inventory is invalid")
        for ratio in self.malicious_client_ratios:
            total = ratio * self.total_clients
            active = ratio * self.clients_per_round
            if (
                not 0.0 < ratio < 0.5
                or not math.isclose(total, round(total), abs_tol=1e-12)
                or not math.isclose(active, round(active), abs_tol=1e-12)
                or round(active) * 2 >= self.clients_per_round
            ):
                raise ValueError("Phase 6C ratio cannot form an exact minority schedule")
        if tuple(sorted(set(self.poison_fractions))) != self.poison_fractions:
            raise ValueError("Phase 6C poison fractions are invalid")
        if self.schedule_mode != "stratified_fixed_coalition_malicious_minority":
            raise ValueError("Phase 6C schedule mode is invalid")
        if (
            self.test_partition_accessed
            or self.validation_partition_accessed
            or self.anchor_records_accessed
            or self.performance_claims_allowed
        ):
            raise ValueError("Phase 6C contract crossed its evidence boundary")

    @property
    def contract_id(self) -> str:
        """Return the deterministic full identity of the execution contract."""

        return run_id(self, length=64)


def phase6c_execution_config() -> Phase6CExecutionConfig:
    """Return the manuscript-aligned Phase 6C execution contract."""

    return Phase6CExecutionConfig()


@dataclass(frozen=True, slots=True)
class AttackRunSpec:
    """One resolved attack condition independent of the compared defense."""

    dataset: str
    profile: str
    attack_name: str
    malicious_ratio: float
    seed: int
    poison_fraction: float = 0.0
    strength: float | None = None
    scale: float | None = None
    aggregation: str = "fedavg"

    def validate(self, config: Phase6CExecutionConfig) -> None:
        """Fail when this condition is outside the frozen Phase 6 grids."""

        if (
            not self.dataset
            or self.profile not in config.profiles
            or self.attack_name not in _ATTACK_NAMES
            or self.malicious_ratio not in config.malicious_client_ratios
            or self.seed < 0
            or self.aggregation not in {"fedavg", "coordinate_median"}
        ):
            raise ValueError("Phase 6C attack condition identity is invalid")
        allowed_poison = (0.0, *config.poison_fractions)
        if self.attack_name in _DATA_ATTACKS:
            if self.poison_fraction not in allowed_poison:
                raise ValueError("Phase 6C data-attack poison fraction is invalid")
        elif self.poison_fraction != 0.0:
            raise ValueError("Direct and clean conditions cannot poison records")
        if self.attack_name == "clean_label":
            if self.strength not in config.clean_label_strengths or self.scale is not None:
                raise ValueError("Clean-label attack strength is invalid")
        elif self.strength is not None:
            raise ValueError("Only clean-label poisoning accepts a strength")
        if self.attack_name == "scaling":
            if self.scale not in config.scaling_factors:
                raise ValueError("Scaling factor is outside the frozen grid")
        elif self.attack_name == "sign_reversal":
            if self.scale not in config.reversal_factors:
                raise ValueError("Sign-reversal factor is outside the frozen grid")
        elif self.scale is not None:
            raise ValueError("This Phase 6C attack does not accept a direct scale")

    def stable_record(self, config: Phase6CExecutionConfig) -> dict[str, object]:
        """Return the defense-independent attack-condition identity record."""

        self.validate(config)
        return {
            "schema_version": 1,
            "phase": "6C-attack-condition",
            "phase6c_contract_id": config.contract_id,
            "dataset": self.dataset,
            "profile": self.profile,
            "attack_name": self.attack_name,
            "malicious_ratio": self.malicious_ratio,
            "seed": self.seed,
            "poison_fraction": self.poison_fraction,
            "strength": self.strength,
            "scale": self.scale,
            "aggregation": self.aggregation,
        }

    def attack_id(self, config: Phase6CExecutionConfig) -> str:
        """Return this full attack-condition identity."""

        return run_id(self.stable_record(config), length=64)

    @property
    def requires_data_poisoning(self) -> bool:
        """Return whether malicious local batches use selected source rows."""

        return self.attack_name in _DATA_ATTACKS


@dataclass(frozen=True, slots=True)
class AttackRealization:
    """Fixed malicious coalition and stratified schedule for one ratio and seed."""

    malicious_client_ids: tuple[str, ...]
    active_malicious_per_round: int
    schedule: ClientSchedule
    realization_id: str


def _ranked_clients(client_ids: tuple[str, ...], *, seed: int, ratio: float) -> list[str]:
    def rank(client_id: str) -> tuple[bytes, str]:
        payload = f"{seed}\0phase6c-coalition\0{ratio:.12g}\0{client_id}".encode()
        return hashlib.sha256(payload).digest(), client_id

    return sorted(client_ids, key=rank)


def build_attack_realization(
    config: Phase6CExecutionConfig, spec: AttackRunSpec
) -> AttackRealization:
    """Build a fixed coalition and exact malicious-minority schedule."""

    spec.validate(config)
    client_ids = tuple(f"client-{index:02d}" for index in range(config.total_clients))
    malicious_count = round(spec.malicious_ratio * config.total_clients)
    active_malicious = round(spec.malicious_ratio * config.clients_per_round)
    malicious = tuple(
        sorted(
            _ranked_clients(client_ids, seed=spec.seed, ratio=spec.malicious_ratio)[
                :malicious_count
            ]
        )
    )
    honest = tuple(sorted(set(client_ids) - set(malicious)))
    rounds: list[tuple[str, ...]] = []
    for round_id in range(config.total_rounds):
        rng = random.Random(
            derive_seed(
                spec.seed,
                f"phase6c-stratified:{spec.malicious_ratio:.12g}:round={round_id}",
            )
        )
        selected_malicious = rng.sample(list(malicious), active_malicious)
        selected_honest = rng.sample(list(honest), config.clients_per_round - active_malicious)
        rounds.append(tuple(sorted((*selected_malicious, *selected_honest))))
    schedule_stable: dict[str, object] = {
        "schema_version": 1,
        "client_ids": list(client_ids),
        "rounds": [list(selected) for selected in rounds],
        "clients_per_round": config.clients_per_round,
        "seed": spec.seed,
    }
    schedule = ClientSchedule(
        client_ids=client_ids,
        rounds=tuple(rounds),
        clients_per_round=config.clients_per_round,
        seed=spec.seed,
        schedule_id=run_id(schedule_stable, length=64),
    )
    stable = {
        "schema_version": 1,
        "phase": "6C-stratified-attack-realization",
        "phase6c_contract_id": config.contract_id,
        "seed": spec.seed,
        "malicious_ratio": spec.malicious_ratio,
        "malicious_client_ids": list(malicious),
        "active_malicious_per_round": active_malicious,
        "schedule_id": schedule.schedule_id,
    }
    return AttackRealization(
        malicious_client_ids=malicious,
        active_malicious_per_round=active_malicious,
        schedule=schedule,
        realization_id=run_id(stable, length=64),
    )
