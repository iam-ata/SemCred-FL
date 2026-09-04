"""Deterministic synchronous Phase 5 federated round engine."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from semcred_fl.aggregation.burn_in import aggregate_burn_in
from semcred_fl.aggregation.semcred import aggregate_semcred_updates
from semcred_fl.baselines.coordinate_median import coordinate_median
from semcred_fl.baselines.fedavg import AggregationResult, fedavg
from semcred_fl.baselines.multi_krum import multi_krum
from semcred_fl.baselines.trimmed_mean import trimmed_mean
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.fusion import FusedCredibility, fuse_credibility
from semcred_fl.federation.checkpoint import (
    load_federated_checkpoint,
    save_federated_checkpoint,
)
from semcred_fl.federation.parameters import (
    ParameterManifest,
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.federation.sampling import ClientSchedule
from semcred_fl.federation.state import (
    ClientState,
    admit_clipped_histories,
    update_participating_trust,
)
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP

_SCHEMA_VERSION = 1


class UpdateProvider(Protocol):
    """Produces selected-client updates from one immutable broadcast vector."""

    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        """Return simulation-side client updates for one synchronous round."""


@dataclass(frozen=True, slots=True)
class SemCredRoundResult:
    """One complete state transition of the Phase 9 server aggregation path."""

    round_id: int
    burn_in: bool
    fused_credibility: dict[str, FusedCredibility]
    participating_trust: dict[str, float]
    threshold: float
    eligible_client_ids: tuple[str, ...]
    history_admitted_client_ids: tuple[str, ...]
    clipping_bound: float
    sample_count_cap: float | None
    weights: dict[str, float]
    aggregate_delta: Tensor
    client_states: dict[str, ClientState]

    def __post_init__(self) -> None:
        if self.round_id < 0 or not self.eligible_client_ids:
            raise ValueError("SemCred round identity or eligible population is invalid")
        if self.aggregate_delta.ndim != 1 or not torch.isfinite(self.aggregate_delta).all().item():
            raise ValueError("SemCred round aggregate must be a finite flat vector")
        if set(self.fused_credibility) != set(self.participating_trust):
            raise ValueError("Credibility and participating-trust populations differ")


def process_semcred_round(
    *,
    round_id: int,
    updates: tuple[DefenseUpdate, ...],
    semantic_scores: Mapping[str, float | None],
    numerical_scores: Mapping[str, float | None],
    client_states: Mapping[str, ClientState],
    config: AggregationConfig | None = None,
    epsilon: float = 1e-12,
) -> SemCredRoundResult:
    """Fuse scores, update trust, aggregate, and admit clipped history for one round."""

    if round_id < 0 or not updates or len({update.client_id for update in updates}) != len(updates):
        raise ValueError("SemCred processing requires one valid unique-update population")
    client_ids = {update.client_id for update in updates}
    if client_ids != set(semantic_scores) or client_ids != set(numerical_scores):
        raise ValueError("Phase 8 score populations differ from accepted defense updates")
    controls = config or AggregationConfig()
    fused = {
        client_id: fuse_credibility(
            semantic_scores[client_id],
            numerical_scores[client_id],
            semantic_coefficient=controls.semantic_fusion_coefficient,
            disagreement_penalty=controls.disagreement_penalty,
            epsilon=epsilon,
        )
        for client_id in sorted(client_ids)
    }
    fused_scores = {client_id: record.score for client_id, record in fused.items()}
    trust_states = update_participating_trust(
        client_states,
        fused_scores,
        round_id=round_id,
        decrease_rate=controls.trust_decrease_rate,
        recovery_rate=controls.trust_recovery_rate,
    )
    participating_trust = {
        client_id: trust_states[client_id].trust for client_id in sorted(client_ids)
    }

    burn_in = round_id < controls.burn_in_rounds
    if burn_in:
        aggregation = aggregate_burn_in(
            updates,
            fused_scores,
            retained_ratio=controls.minimum_retained_ratio,
            clipping_multiplier=controls.clipping_multiplier,
            epsilon=epsilon,
        )
        eligible = aggregation.admitted_client_ids
        admitted = aggregation.history_clipping.clipped_updates
        threshold = aggregation.history_eligibility.threshold
        clipping_bound = aggregation.history_clipping.bound
        sample_count_cap = None
        weights: dict[str, float] = {}
        aggregate_delta = aggregation.aggregate_delta
    else:
        aggregation_record = aggregate_semcred_updates(
            updates,
            participating_trust,
            config=controls,
            epsilon=epsilon,
        )
        eligible = aggregation_record.eligible_client_ids
        admitted = aggregation_record.clipped_updates
        threshold = aggregation_record.threshold
        clipping_bound = aggregation_record.clipping_bound
        sample_count_cap = aggregation_record.sample_count_cap
        weights = aggregation_record.weights
        aggregate_delta = aggregation_record.aggregate_delta
    updated_states = admit_clipped_histories(
        trust_states,
        admitted,
        round_id=round_id,
        history_decay=controls.history_decay,
    )
    return SemCredRoundResult(
        round_id=round_id,
        burn_in=burn_in,
        fused_credibility=fused,
        participating_trust=participating_trust,
        threshold=threshold,
        eligible_client_ids=eligible,
        history_admitted_client_ids=tuple(sorted(admitted)),
        clipping_bound=clipping_bound,
        sample_count_cap=sample_count_cap,
        weights=weights,
        aggregate_delta=aggregate_delta,
        client_states=updated_states,
    )


@dataclass(frozen=True, slots=True)
class FederatedRunConfig:
    """Frozen controls for the basic and standard-baseline FL engine."""

    output_directory: Path
    total_rounds: int
    aggregation: str = "fedavg"
    seed: int = 20260816
    scientific_context_id: str | None = None
    trim_count: int = 0
    krum_byzantine_count: int = 0
    krum_selected_count: int | None = None

    def __post_init__(self) -> None:
        if self.total_rounds < 1 or self.seed < 0:
            raise ValueError("Federated rounds and seed are invalid")
        if self.aggregation not in {
            "fedavg",
            "coordinate_median",
            "trimmed_mean",
            "multi_krum",
        }:
            raise ValueError("Unknown federated aggregation method")
        if self.scientific_context_id is not None and len(self.scientific_context_id) != 64:
            raise ValueError("Federated scientific context must be a full SHA-256 identity")
        if isinstance(self.trim_count, bool) or self.trim_count < 0:
            raise ValueError("Trimmed-mean trim count must be a nonnegative integer")
        if isinstance(self.krum_byzantine_count, bool) or self.krum_byzantine_count < 0:
            raise ValueError("Multi-Krum Byzantine count must be a nonnegative integer")
        if self.krum_selected_count is not None and (
            isinstance(self.krum_selected_count, bool) or self.krum_selected_count < 1
        ):
            raise ValueError("Multi-Krum selected count must be positive when provided")
        if self.aggregation != "trimmed_mean" and self.trim_count != 0:
            raise ValueError("Trim count is valid only for trimmed-mean aggregation")
        if self.aggregation != "multi_krum" and (
            self.krum_byzantine_count != 0 or self.krum_selected_count is not None
        ):
            raise ValueError("Multi-Krum controls are valid only for Multi-Krum aggregation")

    def stable_record(self) -> dict[str, object]:
        """Return run controls that determine scientific behavior."""

        record: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "phase": (
                "phase_10_standard_baselines"
                if self.aggregation in {"trimmed_mean", "multi_krum"}
                else "phase_5_basic_fl_engine"
            ),
            "total_rounds": self.total_rounds,
            "aggregation": self.aggregation,
            "seed": self.seed,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
        if self.scientific_context_id is not None:
            record["scientific_context_id"] = self.scientific_context_id
        if self.aggregation == "trimmed_mean":
            record["trim_count"] = self.trim_count
        if self.aggregation == "multi_krum":
            record["krum_byzantine_count"] = self.krum_byzantine_count
            record["krum_selected_count"] = self.krum_selected_count
        return record


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _record_sha256(record: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _validate_round_log(path: Path) -> tuple[int, str | None]:
    if not path.exists():
        return 0, None
    expected_previous: str | None = None
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Federated round log contains invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError("Federated round log record is not a dictionary")
            record = cast(dict[str, object], value)
            recorded_hash = record.pop("record_sha256", None)
            if (
                record.get("round_id") != count
                or record.get("previous_record_sha256") != expected_previous
            ):
                raise ValueError("Federated round log ordering or hash chain is invalid")
            calculated = _record_sha256(record)
            if recorded_hash != calculated:
                raise ValueError("Federated round log record hash is invalid")
            expected_previous = calculated
            count += 1
    return count, expected_previous


def _aggregate(config: FederatedRunConfig, updates: tuple[DefenseUpdate, ...]) -> AggregationResult:
    if config.aggregation == "fedavg":
        return fedavg(updates)
    if config.aggregation == "coordinate_median":
        return coordinate_median(updates)
    if config.aggregation == "trimmed_mean":
        return trimmed_mean(updates, trim_count=config.trim_count)
    if config.aggregation == "multi_krum":
        return multi_krum(
            updates,
            assumed_byzantine=config.krum_byzantine_count,
            selected_count=config.krum_selected_count,
        )
    raise ValueError("Unknown aggregation method")


def _safe_updates(
    received: tuple[ClientUpdate, ...],
    selected: tuple[str, ...],
    *,
    round_id: int,
    expected_values: int,
) -> tuple[tuple[DefenseUpdate, ...], dict[str, str]]:
    validation = validate_client_updates(
        received, expected_round=round_id, expected_values=expected_values
    )
    rejected = dict(validation.rejected)
    selected_set = set(selected)
    accepted: list[DefenseUpdate] = []
    for update in validation.accepted:
        if update.client_id not in selected_set:
            rejected[update.client_id] = "unselected_client"
        else:
            accepted.append(update)
    present = {update.client_id for update in received}
    for client_id in selected:
        if client_id not in present:
            rejected[client_id] = "missing_update"
    return tuple(accepted), dict(sorted(rejected.items()))


def _resolved_record(
    config: FederatedRunConfig,
    schedule: ClientSchedule,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    initial_parameters_sha256: str,
    initial_model_device: str,
) -> dict[str, object]:
    return {
        **config.stable_record(),
        "schedule_id": schedule.schedule_id,
        "detector": detector.to_dict(),
        "parameter_manifest": manifest.to_dict(),
        "initial_parameters_sha256": initial_parameters_sha256,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "initial_model_device": initial_model_device,
        },
    }


def run_federated_simulation(
    *,
    initial_model: IntrusionMLP,
    schedule: ClientSchedule,
    update_provider: UpdateProvider,
    config: FederatedRunConfig,
    max_rounds_this_call: int | None = None,
) -> dict[str, object]:
    """Run or exactly resume a deterministic synchronous Phase 5 simulation."""

    if len(schedule.rounds) != config.total_rounds:
        raise ValueError("Schedule round count differs from the run configuration")
    if schedule.seed != config.seed:
        raise ValueError("Schedule seed differs from the federated root seed")
    if max_rounds_this_call is not None and max_rounds_this_call < 1:
        raise ValueError("max_rounds_this_call must be positive")
    manifest = parameter_manifest(initial_model)
    initial_vector = flatten_parameters(initial_model, manifest).detach().cpu()
    resolved = _resolved_record(
        config,
        schedule,
        initial_model.config,
        manifest,
        _tensor_sha256(initial_vector),
        str(next(initial_model.parameters()).device),
    )
    identity = run_id(resolved, length=64)
    run_directory = config.output_directory.resolve() / identity
    final_path = run_directory / "federated_run.json"
    if final_path.exists():
        return cast(dict[str, object], json.loads(final_path.read_text(encoding="utf-8")))
    run_directory.mkdir(parents=True, exist_ok=True)
    runtime_path = run_directory / "runtime.json"
    if not runtime_path.exists():
        _write_json(
            runtime_path,
            {
                "started_at_utc": datetime.now(UTC).isoformat(),
                "completed_at_utc": None,
            },
        )
    resolved_path = run_directory / "resolved_config.json"
    expected_resolved = {"run_id": identity, **resolved}
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != expected_resolved:
            raise ValueError("Existing Phase 5 resolved configuration differs")
    else:
        _write_json(resolved_path, expected_resolved)
        schedule.write(run_directory / "client_schedule.json")

    rounds_path = run_directory / "rounds.jsonl"
    timing_path = run_directory / "timings.jsonl"
    log_rounds, log_last_hash = _validate_round_log(rounds_path)
    latest_path = run_directory / "latest_checkpoint.json"
    model = initial_model
    next_round = 0
    last_record_hash: str | None = None
    latest_checkpoint: dict[str, object] | None = None
    if latest_path.exists():
        pointer_raw = json.loads(latest_path.read_text(encoding="utf-8"))
        if not isinstance(pointer_raw, dict):
            raise ValueError("Latest-checkpoint pointer is malformed")
        pointer = cast(dict[str, object], pointer_raw)
        relative_path = pointer.get("path")
        digest = pointer.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise ValueError("Latest-checkpoint pointer is incomplete")
        loaded = load_federated_checkpoint(
            run_directory / relative_path, expected_sha256=digest, device="cpu"
        )
        if loaded.run_id != identity or loaded.schedule_id != schedule.schedule_id:
            raise ValueError("Federated checkpoint is bound to another run or schedule")
        model = loaded.model
        manifest = loaded.manifest
        next_round = loaded.next_round
        last_record_hash = loaded.last_record_sha256
        latest_checkpoint = pointer
    elif log_rounds:
        raise ValueError("Federated round log exists without a resume checkpoint")
    if log_rounds != next_round or log_last_hash != last_record_hash:
        raise ValueError("Federated checkpoint cursor differs from the round log")

    stop_round = config.total_rounds
    if max_rounds_this_call is not None:
        stop_round = min(stop_round, next_round + max_rounds_this_call)
    for round_id in range(next_round, stop_round):
        selected = schedule.rounds[round_id]
        broadcast = flatten_parameters(model, manifest).detach().cpu()
        received = tuple(update_provider(round_id, selected, broadcast.clone(), manifest))
        accepted, rejected = _safe_updates(
            received,
            selected,
            round_id=round_id,
            expected_values=manifest.total_values,
        )
        if accepted:
            aggregate = _aggregate(config, accepted)
            if not torch.isfinite(aggregate.delta).all().item():
                raise RuntimeError("Aggregation produced a non-finite update")
            apply_flat_delta(model, aggregate.delta, manifest)
            failed = False
        else:
            aggregate = AggregationResult(
                config.aggregation,
                torch.zeros(manifest.total_values, dtype=broadcast.dtype),
                {},
            )
            failed = True
        update_records = [
            {
                "client_id": update.client_id,
                "declared_sample_count": update.declared_sample_count,
                "mean_local_loss": update.mean_local_loss,
                "update_norm": update.update_norm,
                "missing_class_ids": list(update.missing_class_ids),
            }
            for update in accepted
        ]
        core: dict[str, object] = {
            "round_id": round_id,
            "previous_record_sha256": last_record_hash,
            "selected_client_ids": list(selected),
            "accepted_client_ids": [update.client_id for update in accepted],
            "rejected_updates": rejected,
            "aggregation": {
                "method": aggregate.method,
                "weights": aggregate.weights,
                "delta_norm": float(torch.linalg.vector_norm(aggregate.delta).item()),
                "delta_sha256": _tensor_sha256(aggregate.delta),
            },
            "client_updates": update_records,
            "round_failed": failed,
            "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        }
        record_hash = _record_sha256(core)
        _append_json_line(rounds_path, {**core, "record_sha256": record_hash})
        for update in received:
            _append_json_line(
                timing_path,
                {
                    "round_id": round_id,
                    "client_id": update.client_id,
                    "epoch_durations_seconds": list(update.epoch_durations_seconds),
                },
            )
        checkpoint_path = run_directory / "checkpoints" / f"round-{round_id + 1:04d}.pt"
        checkpoint_hash = save_federated_checkpoint(
            checkpoint_path,
            model,
            manifest,
            next_round=round_id + 1,
            run_id=identity,
            schedule_id=schedule.schedule_id,
            last_record_sha256=record_hash,
        )
        latest_checkpoint = {
            "path": checkpoint_path.relative_to(run_directory).as_posix(),
            "sha256": checkpoint_hash,
            "next_round": round_id + 1,
            "last_record_sha256": record_hash,
        }
        _write_json(latest_path, latest_checkpoint)
        last_record_hash = record_hash
        next_round = round_id + 1

    if next_round < config.total_rounds:
        return {
            "run_id": identity,
            "status": "interrupted",
            "next_round": next_round,
            "latest_checkpoint": latest_checkpoint,
        }
    if latest_checkpoint is None or last_record_hash is None:
        raise RuntimeError("Federated simulation completed without a checkpoint")
    final_vector = flatten_parameters(model, manifest)
    if not torch.isfinite(final_vector).all().item() or not math.isfinite(
        float(torch.linalg.vector_norm(final_vector).item())
    ):
        raise RuntimeError("Final federated model is non-finite")
    report: dict[str, object] = {
        "run_id": identity,
        **resolved,
        "status": "complete",
        "rounds_completed": next_round,
        "last_record_sha256": last_record_hash,
        "final_parameters_sha256": _tensor_sha256(final_vector),
        "latest_checkpoint": latest_checkpoint,
        "gates": {
            "all_rounds_checkpointed": True,
            "canonical_round_log_hash_chained": True,
            "defense_inputs_exclude_attack_metadata": True,
            "resume_cursor_matches_round_log": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    _write_json(final_path, report)
    runtime = cast(dict[str, object], json.loads(runtime_path.read_text(encoding="utf-8")))
    runtime["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(runtime_path, runtime)
    return report
