"""Resumable Phase 12 Step 2 clean Edge-IIoTset comparison."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from semcred_fl.baselines.fedavg import fedavg
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.numerical_score import evaluate_numerical_updates
from semcred_fl.credibility.semantic_score import (
    evaluate_semantic_updates,
    load_scoring_anchor_batch,
)
from semcred_fl.data.client_cache import (
    SelectedClientCache,
    iter_cached_client_batches,
    load_selected_client_cache,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    PreprocessedCorpus,
    TrainingBatch,
    iter_partition_batches,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import (
    ParameterManifest,
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.federation.sampling import ClientSchedule, generate_client_schedule
from semcred_fl.federation.server import process_semcred_round
from semcred_fl.federation.state import ClientState
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.modeling.metrics import ClassificationMetrics, ConfusionMatrixAccumulator
from semcred_fl.semantic.pipeline import SemanticPriorBundle, load_semantic_prior_bundle
from semcred_fl.semantic.prior import FeatureGroupLayout, load_feature_group_layout

_METHODS = ("fedavg", "semcred_fl")
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Phase12CleanConfig:
    """Frozen non-claim pilot controls for the clean comparison."""

    seed: int = 20260816
    profile: str = "iid"
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
    prefer_fused_adam: bool = True
    materialize_client_batches: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0 or self.profile != "iid":
            raise ValueError("Step 2 requires a nonnegative seed and the IID profile")
        if self.total_clients < 2 or not 2 <= self.clients_per_round <= self.total_clients:
            raise ValueError("Step 2 client population or participation is invalid")
        if self.total_rounds < 1 or self.local_epochs < 1:
            raise ValueError("Step 2 round and local-epoch counts must be positive")
        if self.batch_size < 1 or self.validation_batch_size < 1:
            raise ValueError("Step 2 batch sizes must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Step 2 learning rate must be finite and positive")
        if not math.isfinite(self.proximal_mu) or self.proximal_mu < 0.0:
            raise ValueError("Step 2 proximal coefficient must be finite and nonnegative")
        if not math.isfinite(self.maximum_class_weight) or self.maximum_class_weight < 1.0:
            raise ValueError("Step 2 maximum class weight is invalid")
        if not self.hidden_dimensions or any(width < 1 for width in self.hidden_dimensions):
            raise ValueError("Step 2 hidden dimensions must be positive")
        if (
            not 0.0 <= self.dropout < 1.0
            or not 0 <= self.burn_in_rounds < self.total_rounds
            or self.attribution_interval < 1
        ):
            raise ValueError("Step 2 dropout, burn-in, or attribution interval is invalid")

    def stable_record(self) -> dict[str, object]:
        """Return the scientific pilot controls without local artifact paths."""

        return {
            "schema": "semcred_phase12_step2_clean_config_v1",
            "dataset": "edge_iiotset",
            "profile": self.profile,
            "seed": self.seed,
            "total_clients": self.total_clients,
            "clients_per_round": self.clients_per_round,
            "total_rounds": self.total_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "proximal_mu": self.proximal_mu,
            "maximum_class_weight": self.maximum_class_weight,
            "hidden_dimensions": list(self.hidden_dimensions),
            "dropout": self.dropout,
            "burn_in_rounds": self.burn_in_rounds,
            "attribution_interval": self.attribution_interval,
            "validation_batch_size": self.validation_batch_size,
            "prefer_fused_adam": self.prefer_fused_adam,
            "materialize_client_batches": self.materialize_client_batches,
            "pilot_budget": True,
            "confirmatory": False,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class Phase12CleanRequest:
    """Artifact bindings and execution controls for Phase 12 Step 2."""

    preprocessing_report: Path
    client_partition_manifest: Path
    semantic_prior_path: Path
    client_cache_manifest: Path
    output_directory: Path
    device: str = "auto"
    max_rounds_this_call: int | None = None
    config: Phase12CleanConfig = Phase12CleanConfig()

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Step 2 device must be auto, cpu, or cuda")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Step 2 maximum rounds per call must be positive")


@dataclass(frozen=True, slots=True)
class _ExecutionProfile:
    device: str
    optimizer_backend: str
    batch_delivery: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _MaterializedClientBatchSource:
    """Preserve cache batch order while transferring one client once per epoch."""

    def __init__(
        self,
        cache: SelectedClientCache,
        client_index: int,
        batch_size: int,
        device: str,
        *,
        materialize: bool,
    ) -> None:
        self._cache = cache
        self._client_index = client_index
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._materialize = materialize

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        source = iter_cached_client_batches(
            self._cache,
            self._client_index,
            batch_size=self._batch_size,
            shuffle_seed=seed,
        )
        if not self._materialize:
            for batch in source:
                yield batch.training_batch()
            return
        batches = tuple(source)
        if not batches:
            raise DataContractError("Selected Step 2 client produced no cached batches")
        sizes = tuple(batch.targets.shape[0] for batch in batches)
        features = torch.cat(tuple(batch.features for batch in batches), dim=0).to(self._device)
        targets = torch.cat(tuple(batch.targets for batch in batches), dim=0).to(self._device)
        offset = 0
        for size in sizes:
            end = offset + int(size)
            yield TrainingBatch(features[offset:end], targets[offset:end])
            offset = end


def _resolve_execution(request: Phase12CleanRequest) -> _ExecutionProfile:
    if request.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Step 2 but is unavailable")
    device = request.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "fused" if request.config.prefer_fused_adam and device == "cuda" else "standard"
    delivery = (
        "materialized_preserve_cache_batches_single_device_transfer_v1"
        if request.config.materialize_client_batches
        else "streamed_cache_batches_v1"
    )
    return _ExecutionProfile(device, backend, delivery)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _write_json_immutable(path: Path, value: dict[str, object]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise DataContractError(f"Immutable Step 2 artifact differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(encoded)
    partial.replace(path)


def _save_torch(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), partial)
    partial.replace(path)
    return _sha256(path)


def _load_torch(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise DataContractError("Step 2 tensor artifact checksum differs")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise DataContractError("Step 2 tensor artifact must be a dictionary")
    return cast(dict[str, Any], value)


def _client_index(client_id: str) -> int:
    suffix = client_id.removeprefix("client-")
    if not client_id.startswith("client-") or not suffix.isdigit():
        raise DataContractError("Step 2 client ID is malformed")
    return int(suffix)


def _local_config(
    request: Phase12CleanRequest, execution: _ExecutionProfile
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


def _update_input(
    *,
    comparison_input_id: str,
    cache: SelectedClientCache,
    detector: DetectorConfig,
    local_config: LocalTrainingConfig,
    execution: _ExecutionProfile,
    round_id: int,
    client_id: str,
    global_parameters: Tensor,
) -> dict[str, object]:
    return {
        "schema": "semcred_phase12_step2_update_input_v1",
        "comparison_input_id": comparison_input_id,
        "cache_id": cache.cache_id,
        "round_id": round_id,
        "client_id": client_id,
        "global_parameters_sha256": _tensor_sha256(global_parameters),
        "detector": detector.to_dict(),
        "local_training": local_config.to_dict(),
        "batch_delivery": execution.batch_delivery,
    }


def _load_cached_update(
    path: Path,
    *,
    expected_input_id: str,
    expected_values: int,
) -> ClientUpdate:
    payload = _load_torch(path)
    if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise DataContractError("Cached Step 2 update version is invalid")
    if payload.get("update_input_id") != expected_input_id:
        raise DataContractError("Cached Step 2 update belongs to different inputs")
    delta = payload.get("delta")
    if not isinstance(delta, Tensor) or delta.ndim != 1 or delta.numel() != expected_values:
        raise DataContractError("Cached Step 2 update tensor is malformed")
    if payload.get("delta_sha256") != _tensor_sha256(delta):
        raise DataContractError("Cached Step 2 update identity differs from its tensor")
    scalar_fields = ("mean_local_loss", "update_norm")
    if any(not isinstance(payload.get(name), int | float) for name in scalar_fields):
        raise DataContractError("Cached Step 2 update metrics are malformed")
    missing = payload.get("missing_class_ids")
    durations = payload.get("epoch_durations_seconds")
    if not isinstance(missing, list) or not all(isinstance(value, int) for value in missing):
        raise DataContractError("Cached Step 2 missing-class record is malformed")
    if not isinstance(durations, list) or not all(
        isinstance(value, int | float) for value in durations
    ):
        raise DataContractError("Cached Step 2 timing record is malformed")
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
        is_malicious=False,
    )
    if not torch.isfinite(delta).all().item() or not math.isclose(
        update.update_norm,
        float(torch.linalg.vector_norm(delta).item()),
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):
        raise DataContractError("Cached Step 2 update failed finite/norm validation")
    return update


def _train_or_load_update(
    *,
    request: Phase12CleanRequest,
    comparison_input_id: str,
    cache: SelectedClientCache,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    local_config: LocalTrainingConfig,
    execution: _ExecutionProfile,
    round_id: int,
    client_id: str,
    global_parameters: Tensor,
) -> tuple[ClientUpdate, str, bool]:
    stable = _update_input(
        comparison_input_id=comparison_input_id,
        cache=cache,
        detector=detector,
        local_config=local_config,
        execution=execution,
        round_id=round_id,
        client_id=client_id,
        global_parameters=global_parameters,
    )
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
    client_index = _client_index(client_id)
    update = train_local_client(
        detector=detector,
        manifest=manifest,
        global_parameters=global_parameters,
        round_id=round_id,
        client_id=client_id,
        declared_sample_count=cache.client_total(client_index),
        class_counts=cache.class_counts(client_index),
        batch_source=_MaterializedClientBatchSource(
            cache,
            client_index,
            request.config.batch_size,
            execution.device,
            materialize=request.config.materialize_client_batches,
        ),
        config=local_config,
        is_malicious=False,
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
    }
    _save_torch(path, payload)
    return update, update_input_id, False


def _state_payload(states: Mapping[str, ClientState]) -> dict[str, object]:
    return {
        client_id: {
            "trust": state.trust,
            "accepted_history": state.accepted_history,
            "first_seen_round": state.first_seen_round,
            "last_seen_round": state.last_seen_round,
            "accepted_rounds": list(state.accepted_rounds),
        }
        for client_id, state in sorted(states.items())
    }


def _load_states(value: object) -> dict[str, ClientState]:
    if not isinstance(value, dict):
        raise DataContractError("Step 2 checkpoint client states are malformed")
    states: dict[str, ClientState] = {}
    for client_id, raw in value.items():
        if not isinstance(client_id, str) or not isinstance(raw, dict):
            raise DataContractError("Step 2 checkpoint client-state entry is malformed")
        accepted = raw.get("accepted_rounds")
        history = raw.get("accepted_history")
        if not isinstance(accepted, list) or not all(isinstance(item, int) for item in accepted):
            raise DataContractError("Step 2 checkpoint accepted rounds are malformed")
        if history is not None and not isinstance(history, Tensor):
            raise DataContractError("Step 2 checkpoint accepted history is malformed")
        states[client_id] = ClientState(
            trust=float(raw["trust"]),
            accepted_history=history,
            first_seen_round=cast(int, raw["first_seen_round"]),
            last_seen_round=cast(int, raw["last_seen_round"]),
            accepted_rounds=tuple(cast(list[int], accepted)),
        )
    return states


def _method_run_id(comparison_input_id: str, method: str) -> str:
    return run_id(
        {
            "schema": "semcred_phase12_step2_method_run_v1",
            "comparison_input_id": comparison_input_id,
            "method": method,
        },
        length=64,
    )


def _save_method_checkpoint(
    *,
    method_directory: Path,
    method: str,
    method_run_id: str,
    comparison_input_id: str,
    schedule: ClientSchedule,
    model: IntrusionMLP,
    states: Mapping[str, ClientState],
    next_round: int,
    last_round_record_id: str,
) -> dict[str, object]:
    checkpoint = method_directory / "checkpoints" / f"round-{next_round:04d}.pt"
    digest = _save_torch(
        checkpoint,
        {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "method": method,
            "method_run_id": method_run_id,
            "comparison_input_id": comparison_input_id,
            "schedule_id": schedule.schedule_id,
            "next_round": next_round,
            "last_round_record_id": last_round_record_id,
            "detector_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "client_states": _state_payload(states),
        },
    )
    return {
        "path": checkpoint.relative_to(method_directory).as_posix(),
        "sha256": digest,
        "next_round": next_round,
        "last_round_record_id": last_round_record_id,
    }


def _load_method_checkpoint(
    *,
    method_directory: Path,
    method: str,
    method_run_id: str,
    comparison_input_id: str,
    schedule: ClientSchedule,
) -> tuple[IntrusionMLP, dict[str, ClientState], int, str | None]:
    pointer_path = method_directory / "latest_checkpoint.json"
    if not pointer_path.exists():
        raise FileNotFoundError(pointer_path)
    pointer_value = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer_value, dict):
        raise DataContractError("Step 2 latest-checkpoint pointer is malformed")
    relative = pointer_value.get("path")
    digest = pointer_value.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise DataContractError("Step 2 latest-checkpoint pointer is incomplete")
    payload = _load_torch(method_directory / relative, digest)
    expected = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "method": method,
        "method_run_id": method_run_id,
        "comparison_input_id": comparison_input_id,
        "schedule_id": schedule.schedule_id,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise DataContractError("Step 2 checkpoint belongs to a different method run")
    detector_value = payload.get("detector_config")
    model_state = payload.get("model_state")
    next_round = payload.get("next_round")
    last_record = payload.get("last_round_record_id")
    if not isinstance(detector_value, dict) or not isinstance(model_state, dict):
        raise DataContractError("Step 2 checkpoint model payload is malformed")
    if not isinstance(next_round, int) or next_round < 1:
        raise DataContractError("Step 2 checkpoint cursor is invalid")
    if not isinstance(last_record, str):
        raise DataContractError("Step 2 checkpoint round identity is invalid")
    if (
        pointer_value.get("next_round") != next_round
        or pointer_value.get("last_round_record_id") != last_record
    ):
        raise DataContractError("Step 2 checkpoint cursor differs from its pointer")
    round_path = method_directory / "rounds" / f"round-{next_round - 1:04d}.json"
    round_value = json.loads(round_path.read_text(encoding="utf-8"))
    if not isinstance(round_value, dict):
        raise DataContractError("Step 2 latest round record is malformed")
    round_stable = {key: value for key, value in round_value.items() if key != "round_record_id"}
    if (
        round_value.get("round_record_id") != last_record
        or run_id(round_stable, length=64) != last_record
    ):
        raise DataContractError("Step 2 latest round identity is invalid")
    model = IntrusionMLP(DetectorConfig.from_dict(cast(dict[str, object], detector_value)))
    model.load_state_dict(cast(dict[str, Any], model_state), strict=True)
    return model, _load_states(payload.get("client_states")), next_round, last_record


def _evaluate_validation(
    model: IntrusionMLP,
    corpus: PreprocessedCorpus,
    *,
    batch_size: int,
    device: str,
) -> ClassificationMetrics:
    accumulator = ConfusionMatrixAccumulator(corpus.num_classes)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in iter_partition_batches(corpus, "validation", batch_size=batch_size):
            predictions = torch.argmax(model(batch.features.to(device)), dim=1).cpu().tolist()
            accumulator.update(batch.targets.tolist(), predictions)
    return accumulator.compute()


def _round_record(
    *,
    method: str,
    round_id: int,
    previous_record_id: str | None,
    selected: Sequence[str],
    accepted: tuple[DefenseUpdate, ...],
    update_ids: Sequence[str],
    aggregate_delta: Tensor,
    model: IntrusionMLP,
    manifest: ParameterManifest,
    semcred: dict[str, object] | None,
) -> tuple[str, dict[str, object]]:
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step2_round_v1",
        "method": method,
        "round_id": round_id,
        "previous_round_record_id": previous_record_id,
        "selected_client_ids": list(selected),
        "accepted_client_ids": [update.client_id for update in accepted],
        "update_input_ids": list(update_ids),
        "aggregate_delta_sha256": _tensor_sha256(aggregate_delta),
        "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "semcred": semcred,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    record_id = run_id(stable, length=64)
    return record_id, {"round_record_id": record_id, **stable}


def _run_method(
    *,
    request: Phase12CleanRequest,
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
) -> dict[str, object]:
    method_id = _method_run_id(comparison_input_id, method)
    method_directory = request.output_directory / "methods" / method
    final_path = method_directory / "method_report.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise DataContractError("Completed Step 2 method report is malformed")
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
            raise DataContractError("Completed Step 2 method report belongs to other inputs")
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
                f"phase12_step2={method} round={round_id + 1}/{request.config.total_rounds} "
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
            raise DataContractError("Step 2 clean client updates failed validation")
        accepted = validation.accepted
        semcred_record: dict[str, object] | None = None
        if method == "fedavg":
            aggregate = fedavg(accepted)
            aggregate_delta = aggregate.delta
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
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step2_method_result_v1",
        "method": method,
        "method_run_id": method_id,
        "comparison_input_id": comparison_input_id,
        "schedule_id": schedule.schedule_id,
        "rounds_completed": next_round,
        "last_round_record_id": previous_record_id,
        "final_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "validation_metrics": validation_metrics.to_dict(),
        "gates": {
            "clean_training_only": True,
            "shared_initialization_and_schedule": True,
            "all_selected_client_records_used": True,
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


def _comparison_metrics(
    method_reports: Sequence[dict[str, object]],
    *,
    seed: int,
    comparison_result_id: str,
) -> tuple[RunMetricRecord, ...]:
    records: list[RunMetricRecord] = []
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "worst_class_recall",
        "macro_false_positive_rate",
    )
    for report in method_reports:
        metrics = report.get("validation_metrics")
        method = report.get("method")
        method_run_id = report.get("method_run_id")
        if (
            not isinstance(metrics, dict)
            or not isinstance(method, str)
            or not isinstance(method_run_id, str)
        ):
            raise DataContractError("Step 2 method metrics are malformed")
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DataContractError("Step 2 validation metric is malformed")
            records.append(
                RunMetricRecord(
                    run_id=method_run_id,
                    experiment_family_id="phase12_step2_clean_pilot",
                    dataset="edge_iiotset",
                    method=method,
                    condition_id="iid_clean_validation",
                    seed=seed,
                    metric_name=metric_name,
                    value=float(value),
                    source_artifact_id=comparison_result_id,
                    test_partition_accessed=False,
                    performance_claims_allowed=False,
                )
            )
    return tuple(records)


def run_phase12_clean_comparison(request: Phase12CleanRequest) -> dict[str, object]:
    """Build or resume the Step 2 clean FedAvg/SemCred-FL pilot."""

    execution = _resolve_execution(request)
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(
        request.client_partition_manifest,
        corpus,
        request.config.profile,
    )
    cache = load_selected_client_cache(request.client_cache_manifest)
    prior = load_semantic_prior_bundle(request.semantic_prior_path)
    layout = load_feature_group_layout(corpus)
    anchors = load_scoring_anchor_batch(corpus)
    if corpus.dataset != "edge_iiotset":
        raise DataContractError("Phase 12 Step 2 is frozen to Edge-IIoTset")
    if profile.num_clients != request.config.total_clients:
        raise DataContractError("Step 2 client population differs from Phase 3D")
    if (
        cache.dataset != corpus.dataset
        or cache.profile != request.config.profile
        or cache.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or cache.client_partition_id != profile.client_partition_id
        or cache.selected_client_indices != tuple(range(request.config.total_clients))
    ):
        raise DataContractError("Step 2 client cache differs from the frozen data artifacts")
    if (
        prior.dataset != corpus.dataset
        or prior.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or prior.preprocessor_id != corpus.preprocessor_id
        or prior.anchor_id != corpus.anchor_id
        or prior.feature_group_names != layout.group_names
    ):
        raise DataContractError("Step 2 semantic prior differs from the selected corpus")

    client_ids = tuple(f"client-{index:02d}" for index in range(request.config.total_clients))
    schedule = generate_client_schedule(
        client_ids,
        rounds=request.config.total_rounds,
        clients_per_round=request.config.clients_per_round,
        seed=request.config.seed,
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
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "client_partition_id": profile.client_partition_id,
        "cache_id": cache.cache_id,
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
            raise DataContractError("Existing Step 2 resolved configuration differs")
    else:
        _write_json(resolved_path, resolved)
        schedule.write(request.output_directory / "client_schedule.json")

    completed_path = request.output_directory / "comparison_report.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise DataContractError("Completed Step 2 comparison report is malformed")
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
            raise DataContractError("Completed Step 2 comparison report is malformed")
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
        )
        for method in _METHODS
    )
    if any(report.get("status") != "complete" for report in method_reports):
        return {
            "status": "interrupted",
            "comparison_input_id": comparison_input_id,
            "methods": list(method_reports),
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }

    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_step2_clean_comparison_v1",
        "comparison_input_id": comparison_input_id,
        "method_results": [
            {
                "method": report["method"],
                "method_run_id": report["method_run_id"],
                "method_result_id": report["method_result_id"],
                "final_parameters_sha256": report["final_parameters_sha256"],
                "validation_metrics": report["validation_metrics"],
            }
            for report in method_reports
        ],
        "gates": {
            "bounded_pilot_only": True,
            "shared_initialization_schedule_and_optimizer": True,
            "all_selected_client_records_used": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    comparison_result_id = run_id(stable_result, length=64)
    records = _comparison_metrics(
        method_reports,
        seed=request.config.seed,
        comparison_result_id=comparison_result_id,
    )
    metrics_path = request.output_directory / "run_metrics.jsonl"
    metric_bytes = "".join(canonical_json(asdict(record)) + "\n" for record in records).encode()
    if metrics_path.exists() and metrics_path.read_bytes() != metric_bytes:
        raise DataContractError("Existing Step 2 run metrics differ")
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
    report: dict[str, object] = {
        **stable_result,
        "comparison_result_id": comparison_result_id,
        "status": "complete",
        "reporting": {
            "summary_bundle_id": summary["summary_bundle_id"],
            "macro_f1_artifact_id": rendered["artifact_id"],
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(request.output_directory / "comparison_report.json", report)
    return report
