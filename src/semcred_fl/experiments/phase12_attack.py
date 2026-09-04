"""Resumable Phase 12 Step 3 attacked non-IID comparison."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from semcred_fl.attacks.artifacts import Phase6BResolution, load_phase6b_resolution
from semcred_fl.attacks.execution import (
    ClientPoisonSelection,
    Phase6CAttackRequest,
    _effective_class_counts,
    _selection_manifest,
)
from semcred_fl.attacks.execution_contracts import (
    AttackRealization,
    AttackRunSpec,
    Phase6CExecutionConfig,
    build_attack_realization,
)
from semcred_fl.baselines.fedavg import fedavg
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.numerical_score import evaluate_numerical_updates
from semcred_fl.credibility.semantic_score import (
    evaluate_semantic_updates,
    load_scoring_anchor_batch,
)
from semcred_fl.data.client_cache import (
    ClientCacheBuildRequest,
    SelectedClientCache,
    build_selected_client_cache,
    iter_cached_client_batches,
    load_selected_client_cache,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    PreprocessedCorpus,
    TrainingBatch,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.evaluation.metrics import (
    eligibility_metrics,
    malicious_client_ranking,
    trust_separation,
)
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.experiments.phase12_clean import (
    _evaluate_validation,
    _load_method_checkpoint,
    _load_torch,
    _save_method_checkpoint,
    _save_torch,
    _tensor_sha256,
    _write_json,
    _write_json_immutable,
)
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import (
    ParameterManifest,
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.federation.sampling import ClientSchedule
from semcred_fl.federation.server import process_semcred_round
from semcred_fl.federation.state import ClientState
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.semantic.pipeline import SemanticPriorBundle, load_semantic_prior_bundle
from semcred_fl.semantic.prior import FeatureGroupLayout, load_feature_group_layout

_METHODS = ("fedavg", "semcred_fl")
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Phase12AttackConfig:
    """Frozen non-claim controls for the Step 3 attacked pilot."""

    seed: int = 20260816
    profile: str = "dirichlet_0_3"
    attack_name: str = "label_flip"
    malicious_ratio: float = 0.2
    poison_fraction: float = 0.3
    total_clients: int = 50
    clients_per_round: int = 10
    total_rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    proximal_mu: float = 0.0
    maximum_class_weight: float = 10.0
    hidden_dimensions: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    burn_in_rounds: int = 1
    attribution_interval: int = 1
    validation_batch_size: int = 8192
    cache_scan_batch_size: int = 100_000
    prefer_fused_adam: bool = True
    materialize_client_batches: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.profile or self.attack_name != "label_flip":
            raise ValueError("Step 3 identity requires label flip and a nonnegative seed")
        if self.profile == "iid":
            raise ValueError("Step 3 requires a non-IID client profile")
        if self.total_clients < 3 or not 3 <= self.clients_per_round <= self.total_clients:
            raise ValueError("Step 3 client population or participation is invalid")
        malicious_total = self.malicious_ratio * self.total_clients
        malicious_active = self.malicious_ratio * self.clients_per_round
        if (
            not 0.0 < self.malicious_ratio < 0.5
            or not math.isclose(malicious_total, round(malicious_total), abs_tol=1e-12)
            or not math.isclose(malicious_active, round(malicious_active), abs_tol=1e-12)
            or round(malicious_active) * 2 >= self.clients_per_round
        ):
            raise ValueError("Step 3 malicious ratio cannot form an exact minority schedule")
        if not 0.0 < self.poison_fraction <= 1.0:
            raise ValueError("Step 3 poison fraction must be in (0,1]")
        if self.total_rounds < 2 or self.local_epochs < 1:
            raise ValueError("Step 3 round and local-epoch counts are invalid")
        if self.batch_size < 1 or self.validation_batch_size < 1 or self.cache_scan_batch_size < 1:
            raise ValueError("Step 3 batch sizes must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Step 3 learning rate must be finite and positive")
        if not math.isfinite(self.proximal_mu) or self.proximal_mu < 0.0:
            raise ValueError("Step 3 proximal coefficient must be finite and nonnegative")
        if not math.isfinite(self.maximum_class_weight) or self.maximum_class_weight < 1.0:
            raise ValueError("Step 3 class-weight cap is invalid")
        if not self.hidden_dimensions or any(width < 1 for width in self.hidden_dimensions):
            raise ValueError("Step 3 hidden dimensions must be positive")
        if (
            not 0.0 <= self.dropout < 1.0
            or not 0 <= self.burn_in_rounds < self.total_rounds
            or self.attribution_interval < 1
        ):
            raise ValueError("Step 3 dropout, burn-in, or attribution interval is invalid")

    def stable_record(self) -> dict[str, object]:
        """Return every frozen scientific control."""

        return {
            "schema": "semcred_phase12_step3_config_v1",
            "seed": self.seed,
            "dataset": "edge_iiotset",
            "profile": self.profile,
            "attack_name": self.attack_name,
            "malicious_ratio": self.malicious_ratio,
            "poison_fraction": self.poison_fraction,
            "total_clients": self.total_clients,
            "clients_per_round": self.clients_per_round,
            "total_rounds": self.total_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "proximal_mu": self.proximal_mu,
            "maximum_class_weight": self.maximum_class_weight,
            "hidden_dimensions": list(self.hidden_dimensions),
            "dropout": self.dropout,
            "burn_in_rounds": self.burn_in_rounds,
            "attribution_interval": self.attribution_interval,
            "validation_batch_size": self.validation_batch_size,
            "bounded_pilot_only": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class Phase12AttackRequest:
    """Artifact bindings and execution controls for Step 3."""

    preprocessing_report: Path
    client_partition_manifest: Path
    resolution_path: Path
    semantic_prior_path: Path
    output_directory: Path
    cache_directory: Path
    cache_work_database: Path
    device: str = "auto"
    max_rounds_this_call: int | None = None
    config: Phase12AttackConfig = field(default_factory=Phase12AttackConfig)

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Step 3 device must be auto, cpu, or cuda")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Step 3 maximum rounds per call must be positive")


@dataclass(frozen=True, slots=True)
class _ExecutionProfile:
    device: str
    optimizer_backend: str
    batch_delivery: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _resolve_execution(request: Phase12AttackRequest) -> _ExecutionProfile:
    if request.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Step 3 but is unavailable")
    device = request.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "fused" if request.config.prefer_fused_adam and device == "cuda" else "standard"
    delivery = (
        "materialized_attacked_batches_single_device_transfer_v1"
        if request.config.materialize_client_batches
        else "streamed_attacked_batches_v1"
    )
    return _ExecutionProfile(device, backend, delivery)


def _phase6c_config(config: Phase12AttackConfig) -> Phase6CExecutionConfig:
    return Phase6CExecutionConfig(
        base_seed=config.seed,
        total_clients=config.total_clients,
        clients_per_round=config.clients_per_round,
        total_rounds=config.total_rounds,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        proximal_mu=config.proximal_mu,
        maximum_class_weight=config.maximum_class_weight,
        hidden_dimensions=config.hidden_dimensions,
        dropout=config.dropout,
        profiles=(config.profile,),
        malicious_client_ratios=(config.malicious_ratio,),
        poison_fractions=(config.poison_fraction,),
    )


def _attack_spec(config: Phase12AttackConfig) -> AttackRunSpec:
    return AttackRunSpec(
        dataset="edge_iiotset",
        profile=config.profile,
        attack_name=config.attack_name,
        malicious_ratio=config.malicious_ratio,
        seed=config.seed,
        poison_fraction=config.poison_fraction,
    )


def _client_index(client_id: str) -> int:
    suffix = client_id.removeprefix("client-")
    if not client_id.startswith("client-") or not suffix.isdigit():
        raise DataContractError("Step 3 client ID is malformed")
    return int(suffix)


def _scheduled_client_indices(schedule: ClientSchedule) -> tuple[int, ...]:
    return tuple(
        sorted({_client_index(client_id) for selected in schedule.rounds for client_id in selected})
    )


class _AttackedBatchSource:
    def __init__(
        self,
        *,
        cache: SelectedClientCache,
        client_index: int,
        batch_size: int,
        device: str,
        malicious: bool,
        selected_record_ids: Sequence[str],
        resolution: Phase6BResolution,
        materialize: bool,
    ) -> None:
        self._cache = cache
        self._client_index = client_index
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._malicious = malicious
        self._selected = frozenset(selected_record_ids)
        if len(self._selected) != len(selected_record_ids):
            raise DataContractError("Step 3 poison selection contains duplicate record IDs")
        self._resolution = resolution
        self._materialize = materialize

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        batches: list[TrainingBatch] = []
        seen = 0
        for batch in iter_cached_client_batches(
            self._cache,
            self._client_index,
            batch_size=self._batch_size,
            shuffle_seed=seed,
        ):
            targets = batch.targets
            if self._malicious and self._selected:
                positions = tuple(
                    index
                    for index, record_id in enumerate(batch.record_ids)
                    if record_id in self._selected
                )
                if positions:
                    rows = torch.tensor(positions, dtype=torch.long)
                    if torch.any(targets[rows] != self._resolution.source_family_id).item():
                        raise DataContractError("Step 3 poison selection contains a non-source row")
                    targets = targets.clone()
                    targets[rows] = self._resolution.target_family_id
                    seen += len(positions)
            training = TrainingBatch(batch.features, targets)
            if not self._materialize:
                yield training
            else:
                batches.append(training)
        if self._malicious and seen != len(self._selected):
            raise DataContractError("Step 3 poison selection was not fully delivered")
        if not self._materialize:
            return
        if not batches:
            raise DataContractError("Selected Step 3 client produced no cached batches")
        sizes = tuple(batch.targets.shape[0] for batch in batches)
        features = torch.cat(tuple(batch.features for batch in batches), dim=0).to(self._device)
        targets = torch.cat(tuple(batch.targets for batch in batches), dim=0).to(self._device)
        offset = 0
        for size in sizes:
            end = offset + int(size)
            yield TrainingBatch(features[offset:end], targets[offset:end])
            offset = end


def _local_config(
    request: Phase12AttackRequest, execution: _ExecutionProfile
) -> LocalTrainingConfig:
    config = request.config
    return LocalTrainingConfig(
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        proximal_mu=config.proximal_mu,
        maximum_class_weight=config.maximum_class_weight,
        seed=config.seed,
        device=execution.device,
        optimizer_backend=execution.optimizer_backend,
        prevalidated_batches=True,
    )


def _load_cached_update(
    path: Path, *, expected_input_id: str, expected_values: int
) -> ClientUpdate:
    payload = _load_torch(path)
    if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise DataContractError("Cached Step 3 update version is invalid")
    if payload.get("update_input_id") != expected_input_id:
        raise DataContractError("Cached Step 3 update belongs to different inputs")
    delta = payload.get("delta")
    if not isinstance(delta, Tensor) or delta.ndim != 1 or delta.numel() != expected_values:
        raise DataContractError("Cached Step 3 update tensor is malformed")
    if payload.get("delta_sha256") != _tensor_sha256(delta):
        raise DataContractError("Cached Step 3 update identity differs")
    missing = payload.get("missing_class_ids")
    durations = payload.get("epoch_durations_seconds")
    malicious = payload.get("is_malicious")
    if not isinstance(missing, list) or not all(
        not isinstance(value, bool) and isinstance(value, int) for value in missing
    ):
        raise DataContractError("Cached Step 3 missing-class record is malformed")
    if not isinstance(durations, list) or not all(
        not isinstance(value, bool) and isinstance(value, int | float) for value in durations
    ):
        raise DataContractError("Cached Step 3 timing record is malformed")
    if not isinstance(malicious, bool):
        raise DataContractError("Cached Step 3 simulation label is malformed")
    update = ClientUpdate(
        round_id=cast(int, payload["round_id"]),
        client_id=cast(str, payload["client_id"]),
        delta=delta,
        declared_sample_count=cast(int, payload["declared_sample_count"]),
        mean_local_loss=float(payload["mean_local_loss"]),
        epoch_durations_seconds=tuple(float(value) for value in durations),
        update_norm=float(payload["update_norm"]),
        finite=True,
        missing_class_ids=tuple(cast(list[int], missing)),
        is_malicious=malicious,
        attack_metadata={
            "attack_id": cast(str, payload["attack_id"]),
            "selection_id": payload.get("selection_id"),
            "selected_poison_records": cast(int, payload["selected_poison_records"]),
            "transformation": "label_flip" if malicious else "none",
        },
    )
    if not torch.isfinite(delta).all().item() or not math.isclose(
        update.update_norm,
        float(torch.linalg.vector_norm(delta).item()),
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):
        raise DataContractError("Cached Step 3 update failed finite/norm validation")
    return update


def _train_or_load_update(
    *,
    request: Phase12AttackRequest,
    comparison_input_id: str,
    cache: SelectedClientCache,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    local_config: LocalTrainingConfig,
    execution: _ExecutionProfile,
    resolution: Phase6BResolution,
    realization: AttackRealization,
    attack_id: str,
    selections: Mapping[str, ClientPoisonSelection],
    round_id: int,
    client_id: str,
    global_parameters: Tensor,
) -> tuple[ClientUpdate, str, bool]:
    client_index = _client_index(client_id)
    malicious = client_id in set(realization.malicious_client_ids)
    selection = selections.get(client_id)
    selected_ids = selection.selected_record_ids if selection is not None else ()
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step3_update_input_v1",
        "comparison_input_id": comparison_input_id,
        "cache_id": cache.cache_id,
        "attack_id": attack_id,
        "realization_id": realization.realization_id,
        "selection_id": selection.selection_id if selection is not None else None,
        "malicious": malicious,
        "round_id": round_id,
        "client_id": client_id,
        "global_parameters_sha256": _tensor_sha256(global_parameters),
        "detector": detector.to_dict(),
        "local_training": local_config.to_dict(),
        "batch_delivery": execution.batch_delivery,
    }
    update_input_id = run_id(stable, length=64)
    path = request.output_directory / "shared_updates" / f"{update_input_id}.pt"
    if path.exists():
        return (
            _load_cached_update(
                path,
                expected_input_id=update_input_id,
                expected_values=manifest.total_values,
            ),
            update_input_id,
            True,
        )
    class_counts = _effective_class_counts(
        cache,
        client_index,
        _attack_spec(request.config),
        resolution,
        len(selected_ids),
        malicious,
    )
    update = train_local_client(
        detector=detector,
        manifest=manifest,
        global_parameters=global_parameters,
        round_id=round_id,
        client_id=client_id,
        declared_sample_count=cache.client_total(client_index),
        class_counts=class_counts,
        batch_source=_AttackedBatchSource(
            cache=cache,
            client_index=client_index,
            batch_size=request.config.batch_size,
            device=execution.device,
            malicious=malicious,
            selected_record_ids=selected_ids,
            resolution=resolution,
            materialize=request.config.materialize_client_batches,
        ),
        config=local_config,
        is_malicious=malicious,
    )
    update = replace(
        update,
        attack_metadata={
            "attack_id": attack_id,
            "selection_id": selection.selection_id if selection is not None else None,
            "selected_poison_records": len(selected_ids),
            "transformation": "label_flip" if malicious else "none",
        },
    )
    payload: dict[str, object] = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "update_input_id": update_input_id,
        "update_input": stable,
        "round_id": update.round_id,
        "client_id": update.client_id,
        "delta": update.delta,
        "delta_sha256": _tensor_sha256(update.delta),
        "declared_sample_count": update.declared_sample_count,
        "mean_local_loss": update.mean_local_loss,
        "epoch_durations_seconds": list(update.epoch_durations_seconds),
        "update_norm": update.update_norm,
        "missing_class_ids": list(update.missing_class_ids),
        "is_malicious": malicious,
        "attack_id": attack_id,
        "selection_id": selection.selection_id if selection is not None else None,
        "selected_poison_records": len(selected_ids),
    }
    _save_torch(path, payload)
    return update, update_input_id, False


def _method_run_id(comparison_input_id: str, method: str) -> str:
    return run_id(
        {
            "schema": "semcred_phase12_step3_method_run_v1",
            "comparison_input_id": comparison_input_id,
            "method": method,
        },
        length=64,
    )


def _round_record(
    *,
    method: str,
    round_id: int,
    previous_record_id: str | None,
    selected: Sequence[str],
    accepted: tuple[DefenseUpdate, ...],
    original_updates: Sequence[ClientUpdate],
    update_ids: Sequence[str],
    aggregate_delta: Tensor,
    model: IntrusionMLP,
    manifest: ParameterManifest,
    semcred: dict[str, object] | None,
) -> tuple[str, dict[str, object]]:
    post_decision_audit = {
        "active_malicious_client_ids": [
            update.client_id for update in original_updates if update.is_malicious is True
        ],
        "client_updates": [
            {
                "client_id": update.client_id,
                "is_malicious": update.is_malicious,
                "selected_poison_records": (update.attack_metadata or {}).get(
                    "selected_poison_records", 0
                ),
                "selection_id": (update.attack_metadata or {}).get("selection_id"),
                "submitted_delta_sha256": _tensor_sha256(update.delta),
            }
            for update in original_updates
        ],
        "joined_after_defense_decision": True,
    }
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step3_round_v1",
        "method": method,
        "round_id": round_id,
        "previous_round_record_id": previous_record_id,
        "selected_client_ids": list(selected),
        "accepted_client_ids": [update.client_id for update in accepted],
        "update_input_ids": list(update_ids),
        "aggregate_delta_sha256": _tensor_sha256(aggregate_delta),
        "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "semcred": semcred,
        "post_decision_attack_audit": post_decision_audit,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    record_id = run_id(stable, length=64)
    return record_id, {"round_record_id": record_id, **stable}


def _client_assessment(
    method_directory: Path,
    *,
    total_rounds: int,
    malicious_client_ids: Sequence[str],
    states: Mapping[str, ClientState],
) -> dict[str, object]:
    malicious_set = set(malicious_client_ids)
    labels: list[bool] = []
    suspiciousness: list[float] = []
    eligible: list[bool] = []
    for round_id in range(total_rounds):
        path = method_directory / "rounds" / f"round-{round_id:04d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not isinstance(record.get("semcred"), dict):
            raise DataContractError("Step 3 SemCred round evidence is malformed")
        semcred = cast(dict[str, object], record["semcred"])
        selected = record.get("selected_client_ids")
        trust = semcred.get("participating_trust")
        eligible_ids = semcred.get("eligible_client_ids")
        if (
            not isinstance(selected, list)
            or not all(isinstance(value, str) for value in selected)
            or not isinstance(trust, dict)
            or set(trust) != set(selected)
            or not isinstance(eligible_ids, list)
        ):
            raise DataContractError("Step 3 post-decision client evidence is malformed")
        eligible_set = set(cast(list[str], eligible_ids))
        for client_id in cast(list[str], selected):
            value = trust[client_id]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DataContractError("Step 3 participating trust is malformed")
            labels.append(client_id in malicious_set)
            suspiciousness.append(1.0 - float(value))
            eligible.append(client_id in eligible_set)
    ranking = malicious_client_ranking(labels, suspiciousness)
    eligibility = eligibility_metrics(labels, eligible)
    observed_ids = sorted(states)
    separation = trust_separation(
        [client_id in malicious_set for client_id in observed_ids],
        [states[client_id].trust for client_id in observed_ids],
    )
    if (
        ranking.auroc is None
        or ranking.auprc is None
        or eligibility.benign_false_rejection.value is None
        or eligibility.malicious_false_acceptance.value is None
        or separation.mean_gap is None
        or separation.median_gap is None
    ):
        raise DataContractError("Step 3 client-assessment population is incomplete")
    return {
        "event_ranking": asdict(ranking),
        "event_eligibility": {
            "benign_false_rejection": eligibility.benign_false_rejection.to_dict(),
            "malicious_false_acceptance": eligibility.malicious_false_acceptance.to_dict(),
        },
        "final_observed_trust_separation": asdict(separation),
        "participation_events": len(labels),
        "observed_clients": len(observed_ids),
        "gates": {
            "simulation_truth_joined_after_defense_decision": True,
            "attack_metadata_excluded_from_defense_inputs": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }


def _run_method(
    *,
    request: Phase12AttackRequest,
    method: str,
    comparison_input_id: str,
    corpus: PreprocessedCorpus,
    cache: SelectedClientCache,
    schedule: ClientSchedule,
    detector: DetectorConfig,
    execution: _ExecutionProfile,
    anchors: TrainingBatch,
    layout: FeatureGroupLayout,
    prior: SemanticPriorBundle,
    resolution: Phase6BResolution,
    realization: AttackRealization,
    attack_id: str,
    selections: Mapping[str, ClientPoisonSelection],
) -> dict[str, object]:
    method_id = _method_run_id(comparison_input_id, method)
    method_directory = request.output_directory / "methods" / method
    final_path = method_directory / "method_report.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DataContractError("Completed Step 3 method report is malformed")
        stable_value = {
            key: item
            for key, item in value.items()
            if key not in {"method_result_id", "status", "completed_at_utc"}
        }
        if (
            value.get("method_run_id") != method_id
            or value.get("status") != "complete"
            or value.get("method_result_id") != run_id(stable_value, length=64)
        ):
            raise DataContractError("Completed Step 3 method report belongs to other inputs")
        return cast(dict[str, object], value)

    torch.manual_seed(request.config.seed)
    model = IntrusionMLP(detector)
    states: dict[str, ClientState] = {}
    next_round = 0
    previous_record_id: str | None = None
    pointer_path = method_directory / "latest_checkpoint.json"
    if pointer_path.exists():
        model, states, next_round, previous_record_id = _load_method_checkpoint(
            method_directory=method_directory,
            method=method,
            method_run_id=method_id,
            comparison_input_id=comparison_input_id,
            schedule=schedule,
        )
    manifest = parameter_manifest(model)
    local_config = _local_config(request, execution)
    stop_round = request.config.total_rounds
    if request.max_rounds_this_call is not None:
        stop_round = min(stop_round, next_round + request.max_rounds_this_call)

    for round_id in range(next_round, stop_round):
        selected = schedule.rounds[round_id]
        broadcast = flatten_parameters(model, manifest).detach().cpu()
        round_started = time.perf_counter()
        updates: list[ClientUpdate] = []
        update_ids: list[str] = []
        reused = 0
        for position, client_id in enumerate(selected, start=1):
            print(
                f"phase12_step3={method} round={round_id + 1}/{request.config.total_rounds} "
                f"client={position}/{len(selected)} id={client_id}",
                flush=True,
            )
            update, update_id, was_reused = _train_or_load_update(
                request=request,
                comparison_input_id=comparison_input_id,
                cache=cache,
                detector=detector,
                manifest=manifest,
                local_config=local_config,
                execution=execution,
                resolution=resolution,
                realization=realization,
                attack_id=attack_id,
                selections=selections,
                round_id=round_id,
                client_id=client_id,
                global_parameters=broadcast,
            )
            updates.append(update)
            update_ids.append(update_id)
            reused += int(was_reused)
        validation = validate_client_updates(
            tuple(updates),
            expected_round=round_id,
            expected_values=manifest.total_values,
        )
        if validation.rejected or len(validation.accepted) != len(selected):
            raise DataContractError("Step 3 attacked client updates failed validation")
        accepted = validation.accepted
        semcred_record: dict[str, object] | None = None
        if method == "fedavg":
            aggregate_delta = fedavg(accepted).delta
        else:
            semantic = evaluate_semantic_updates(
                model,
                accepted,
                manifest,
                anchors,
                layout,
                prior,
                attribution_enabled=round_id % request.config.attribution_interval == 0,
            )
            histories = {
                client_id: state.accepted_history
                for client_id, state in states.items()
                if state.accepted_history is not None
            }
            numerical = evaluate_numerical_updates(accepted, histories=histories)
            result = process_semcred_round(
                round_id=round_id,
                updates=accepted,
                semantic_scores={key: value.score for key, value in semantic.items()},
                numerical_scores={key: value.score for key, value in numerical.items()},
                client_states=states,
                config=AggregationConfig(burn_in_rounds=request.config.burn_in_rounds),
            )
            aggregate_delta = result.aggregate_delta
            states = result.client_states
            semcred_record = {
                "burn_in": result.burn_in,
                "attribution_enabled": round_id % request.config.attribution_interval == 0,
                "eligible_client_ids": list(result.eligible_client_ids),
                "history_admitted_client_ids": list(result.history_admitted_client_ids),
                "participating_trust": result.participating_trust,
                "semantic_scores": {key: semantic[key].score for key in sorted(semantic)},
                "numerical_scores": {key: numerical[key].score for key in sorted(numerical)},
            }
        apply_flat_delta(model, aggregate_delta, manifest)
        record_id, record = _round_record(
            method=method,
            round_id=round_id,
            previous_record_id=previous_record_id,
            selected=selected,
            accepted=accepted,
            original_updates=updates,
            update_ids=update_ids,
            aggregate_delta=aggregate_delta,
            model=model,
            manifest=manifest,
            semcred=semcred_record,
        )
        _write_json_immutable(method_directory / "rounds" / f"round-{round_id:04d}.json", record)
        pointer = _save_method_checkpoint(
            method_directory=method_directory,
            method=method,
            method_run_id=method_id,
            comparison_input_id=comparison_input_id,
            schedule=schedule,
            model=model,
            states=states,
            next_round=round_id + 1,
            last_round_record_id=record_id,
        )
        _write_json(pointer_path, pointer)
        _write_json(
            method_directory / "runtime" / f"round-{round_id:04d}.json",
            {
                "round_id": round_id,
                "duration_seconds": time.perf_counter() - round_started,
                "updates_trained": len(selected) - reused,
                "updates_reused": reused,
            },
        )
        previous_record_id = record_id
        next_round = round_id + 1

    if next_round < request.config.total_rounds:
        return {
            "method": method,
            "method_run_id": method_id,
            "status": "interrupted",
            "next_round": next_round,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
    validation_metrics = _evaluate_validation(
        model,
        corpus,
        batch_size=request.config.validation_batch_size,
        device=execution.device,
    )
    client_assessment = (
        _client_assessment(
            method_directory,
            total_rounds=request.config.total_rounds,
            malicious_client_ids=realization.malicious_client_ids,
            states=states,
        )
        if method == "semcred_fl"
        else None
    )
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step3_method_result_v1",
        "method": method,
        "method_run_id": method_id,
        "comparison_input_id": comparison_input_id,
        "schedule_id": schedule.schedule_id,
        "rounds_completed": next_round,
        "last_round_record_id": previous_record_id,
        "final_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "validation_metrics": validation_metrics.to_dict(),
        "client_assessment_metrics": client_assessment,
        "gates": {
            "attacked_training_only": True,
            "shared_initialization_and_schedule": True,
            "all_selected_client_records_used": True,
            "simulation_truth_excluded_from_defense_inputs": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    report = {
        **stable,
        "method_result_id": run_id(stable, length=64),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(final_path, report)
    return report


def _metric_records(
    method_reports: Sequence[dict[str, object]],
    *,
    seed: int,
    comparison_result_id: str,
) -> tuple[RunMetricRecord, ...]:
    records: list[RunMetricRecord] = []
    classification_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "worst_class_recall",
        "macro_false_positive_rate",
    )
    for report in method_reports:
        method = cast(str, report["method"])
        method_run_id = cast(str, report["method_run_id"])
        metrics = cast(dict[str, object], report["validation_metrics"])
        values = {name: metrics[name] for name in classification_names}
        if method == "semcred_fl":
            assessment = cast(dict[str, object], report["client_assessment_metrics"])
            ranking = cast(dict[str, object], assessment["event_ranking"])
            event_eligibility = cast(dict[str, object], assessment["event_eligibility"])
            benign_rejection = cast(dict[str, object], event_eligibility["benign_false_rejection"])
            malicious_acceptance = cast(
                dict[str, object], event_eligibility["malicious_false_acceptance"]
            )
            separation = cast(dict[str, object], assessment["final_observed_trust_separation"])
            values.update(
                {
                    "malicious_client_auroc": ranking["auroc"],
                    "malicious_client_auprc": ranking["auprc"],
                    "benign_false_rejection": benign_rejection["value"],
                    "malicious_false_acceptance": malicious_acceptance["value"],
                    "trust_mean_gap": separation["mean_gap"],
                    "trust_median_gap": separation["median_gap"],
                }
            )
        for metric_name, raw_value in values.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise DataContractError("Step 3 metric is malformed")
            records.append(
                RunMetricRecord(
                    run_id=method_run_id,
                    experiment_family_id="phase12_step3_attacked_pilot",
                    dataset="edge_iiotset",
                    method=method,
                    condition_id="dirichlet_0_3_label_flip_mr_0_2_pf_0_3_validation",
                    seed=seed,
                    metric_name=metric_name,
                    value=float(raw_value),
                    source_artifact_id=comparison_result_id,
                    test_partition_accessed=False,
                    performance_claims_allowed=False,
                )
            )
    return tuple(records)


def run_phase12_attack_comparison(request: Phase12AttackRequest) -> dict[str, object]:
    """Build or resume the Step 3 label-flip non-IID comparison."""

    execution = _resolve_execution(request)
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(
        request.client_partition_manifest,
        corpus,
        request.config.profile,
    )
    resolution = load_phase6b_resolution(request.resolution_path)
    prior = load_semantic_prior_bundle(request.semantic_prior_path)
    layout = load_feature_group_layout(corpus)
    anchors = load_scoring_anchor_batch(corpus)
    if corpus.dataset != "edge_iiotset":
        raise DataContractError("Phase 12 Step 3 is frozen to Edge-IIoTset")
    if profile.num_clients != request.config.total_clients:
        raise DataContractError("Step 3 client population differs from Phase 3D")
    if (
        resolution.dataset != corpus.dataset
        or resolution.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or resolution.preprocessor_id != corpus.preprocessor_id
        or resolution.client_partition_id != profile.client_partition_id
    ):
        raise DataContractError("Step 3 attack resolution differs from the selected corpus")
    if (
        prior.dataset != corpus.dataset
        or prior.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or prior.preprocessor_id != corpus.preprocessor_id
        or prior.anchor_id != corpus.anchor_id
        or prior.feature_group_names != layout.group_names
    ):
        raise DataContractError("Step 3 semantic prior differs from the selected corpus")

    attack_config = _phase6c_config(request.config)
    spec = _attack_spec(request.config)
    realization = build_attack_realization(attack_config, spec)
    schedule = realization.schedule
    scheduled_indices = _scheduled_client_indices(schedule)
    cache_scope_id = run_id(
        {
            "schema": "semcred_phase12_step3_schedule_cache_v1",
            "schedule_id": schedule.schedule_id,
            "selected_client_indices": list(scheduled_indices),
        },
        length=64,
    )
    build_selected_client_cache(
        ClientCacheBuildRequest(
            preprocessing_report=request.preprocessing_report,
            client_partition_manifest=request.client_partition_manifest,
            profile=request.config.profile,
            selected_client_indices=scheduled_indices,
            schedule_id=cache_scope_id,
            output_directory=request.cache_directory,
            work_database=request.cache_work_database,
            batch_size=request.config.cache_scan_batch_size,
        )
    )
    cache = load_selected_client_cache(request.cache_directory / "client_cache.json")
    if (
        cache.dataset != corpus.dataset
        or cache.profile != request.config.profile
        or cache.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or cache.client_partition_id != profile.client_partition_id
        or cache.schedule_id != cache_scope_id
        or cache.selected_client_indices != scheduled_indices
    ):
        raise DataContractError("Step 3 schedule-bound cache differs from frozen inputs")
    attack_id = spec.attack_id(attack_config)
    selection_request = Phase6CAttackRequest(
        preprocessing_report=request.preprocessing_report,
        client_partition_manifest=request.client_partition_manifest,
        resolution_path=request.resolution_path,
        output_directory=request.output_directory / "attack_contract",
        cache_directory=request.cache_directory,
        cache_work_database=request.cache_work_database,
        spec=spec,
        device=execution.device,
        cache_scan_batch_size=request.config.cache_scan_batch_size,
        config=attack_config,
    )
    selection_manifest_id, selections = _selection_manifest(
        request=selection_request,
        cache=cache,
        resolution=resolution,
        realization=realization,
        attack_id=attack_id,
    )
    detector = DetectorConfig(
        corpus.input_dimension,
        corpus.num_classes,
        request.config.hidden_dimensions,
        request.config.dropout,
    )
    torch.manual_seed(request.config.seed)
    initial_model = IntrusionMLP(detector)
    initial_manifest = parameter_manifest(initial_model)
    comparison_input: dict[str, object] = {
        **request.config.stable_record(),
        "phase6c_contract_id": attack_config.contract_id,
        "attack_id": attack_id,
        "realization_id": realization.realization_id,
        "selection_manifest_id": selection_manifest_id,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "client_partition_id": profile.client_partition_id,
        "resolution_id": resolution.resolution_id,
        "cache_id": cache.cache_id,
        "scheduled_client_indices": list(scheduled_indices),
        "semantic_prior_id": prior.bundle_id,
        "feature_group_layout_id": layout.layout_id,
        "schedule_id": schedule.schedule_id,
        "detector": detector.to_dict(),
        "initial_parameters_sha256": _tensor_sha256(
            flatten_parameters(initial_model, initial_manifest)
        ),
        "execution_profile": execution.to_dict(),
        "methods": list(_METHODS),
    }
    comparison_input_id = run_id(comparison_input, length=64)
    request.output_directory.mkdir(parents=True, exist_ok=True)
    resolved_path = request.output_directory / "resolved_config.json"
    resolved = {"comparison_input_id": comparison_input_id, **comparison_input}
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise DataContractError("Existing Step 3 resolved configuration differs")
    else:
        _write_json(resolved_path, resolved)
        schedule.write(request.output_directory / "client_schedule.json")

    completed_path = request.output_directory / "comparison_report.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise DataContractError("Completed Step 3 comparison report is malformed")
        completed_stable = {
            key: value
            for key, value in completed.items()
            if key not in {"comparison_result_id", "status", "reporting", "completed_at_utc"}
        }
        if (
            completed.get("comparison_input_id") != comparison_input_id
            or completed.get("status") != "complete"
            or completed.get("comparison_result_id") != run_id(completed_stable, length=64)
        ):
            raise DataContractError("Completed Step 3 comparison report is malformed")
        return cast(dict[str, object], completed)

    method_reports = tuple(
        _run_method(
            request=request,
            method=method,
            comparison_input_id=comparison_input_id,
            corpus=corpus,
            cache=cache,
            schedule=schedule,
            detector=detector,
            execution=execution,
            anchors=anchors,
            layout=layout,
            prior=prior,
            resolution=resolution,
            realization=realization,
            attack_id=attack_id,
            selections=selections,
        )
        for method in _METHODS
    )
    if any(report.get("status") != "complete" for report in method_reports):
        return {
            "status": "interrupted",
            "comparison_input_id": comparison_input_id,
            "attack_id": attack_id,
            "realization_id": realization.realization_id,
            "methods": list(method_reports),
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_step3_attack_comparison_v1",
        "comparison_input_id": comparison_input_id,
        "attack_id": attack_id,
        "realization_id": realization.realization_id,
        "method_results": [
            {
                "method": report["method"],
                "method_run_id": report["method_run_id"],
                "method_result_id": report["method_result_id"],
                "final_parameters_sha256": report["final_parameters_sha256"],
                "validation_metrics": report["validation_metrics"],
                "client_assessment_metrics": report["client_assessment_metrics"],
            }
            for report in method_reports
        ],
        "gates": {
            "bounded_pilot_only": True,
            "one_attack_one_non_iid_profile_one_seed": True,
            "shared_initialization_schedule_optimizer_and_attack": True,
            "simulation_truth_excluded_from_defense_inputs": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    comparison_result_id = run_id(stable_result, length=64)
    records = _metric_records(
        method_reports,
        seed=request.config.seed,
        comparison_result_id=comparison_result_id,
    )
    metrics_path = request.output_directory / "run_metrics.jsonl"
    metric_bytes = "".join(canonical_json(asdict(record)) + "\n" for record in records).encode()
    if metrics_path.exists() and metrics_path.read_bytes() != metric_bytes:
        raise DataContractError("Existing Step 3 run metrics differ")
    if not metrics_path.exists():
        metrics_path.write_bytes(metric_bytes)
    summary = write_summary_bundle(
        records,
        request.output_directory / "reporting",
        bootstrap_resamples=1_000,
        bootstrap_seed=request.config.seed,
    )
    rendered = render_summary_artifacts(
        request.output_directory / "reporting" / "metric_summaries.json",
        request.output_directory / "reporting",
        metric_name="macro_f1",
    )
    report = {
        **stable_result,
        "comparison_result_id": comparison_result_id,
        "status": "complete",
        "reporting": {
            "summary_bundle_id": summary["summary_bundle_id"],
            "macro_f1_artifact_id": rendered["artifact_id"],
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(completed_path, report)
    return report
