"""Resumable Phase 12 Step 5 clean utility condition and matrix runners."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from semcred_fl.baselines.coordinate_median import coordinate_median
from semcred_fl.baselines.fedavg import fedavg
from semcred_fl.baselines.multi_krum import multi_krum
from semcred_fl.baselines.trimmed_mean import trimmed_mean
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.numerical_score import (
    NumericalCoefficients,
    evaluate_numerical_updates,
)
from semcred_fl.credibility.semantic_score import (
    SemanticCoefficients,
    evaluate_semantic_updates,
    load_scoring_anchor_batch,
)
from semcred_fl.data.client_cache import (
    ClientCacheBuildRequest,
    SelectedClientCache,
    build_selected_client_cache,
    load_selected_client_cache,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    PreprocessedCorpus,
    TrainingBatch,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.experiments.phase12_clean import (
    _evaluate_validation,
    _load_cached_update,
    _load_method_checkpoint,
    _MaterializedClientBatchSource,
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
from semcred_fl.federation.sampling import ClientSchedule, generate_client_schedule
from semcred_fl.federation.server import process_semcred_round
from semcred_fl.federation.state import ClientState
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.semantic.pipeline import SemanticPriorBundle, load_semantic_prior_bundle
from semcred_fl.semantic.prior import FeatureGroupLayout, load_feature_group_layout

DATASETS = ("edge_iiotset", "cic_iot_2023")
PROFILES = ("iid", "dirichlet_1_0", "dirichlet_0_3", "dirichlet_0_1")
METHODS = ("fedavg", "coordinate_median", "trimmed_mean", "multi_krum", "semcred_fl")
CONFIRMATORY_SEEDS = tuple(range(20260816, 20260826))
_STANDARD_METHODS = frozenset(METHODS[:-1])
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Phase12CleanConditionConfig:
    """Scientific and execution controls for one Step 5 clean condition."""

    dataset: str
    profile: str
    seed: int = 20260816
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
    burn_in_rounds: int = 3
    attribution_interval: int = 1
    semantic_fusion_coefficient: float = 0.5
    disagreement_penalty: float = 1.0
    trust_decrease_rate: float = 0.6
    trust_recovery_rate: float = 0.1
    sample_count_exponent: float = 0.0
    sample_cap_median_multiplier: float = 2.0
    trust_exponent: float = 2.0
    threshold_multiplier: float = 2.5
    minimum_trust: float = 0.1
    minimum_retained_ratio: float = 0.5
    clipping_multiplier: float = 2.5
    history_decay: float = 0.8
    semantic_loss_weight: float = 0.5
    semantic_attribution_weight: float = 0.5
    numerical_center_weight: float = 0.25
    numerical_norm_weight: float = 0.25
    numerical_direction_weight: float = 0.25
    numerical_history_weight: float = 0.25
    robust_z_max: float = 20.0
    numerical_minimum_history_clients: int = 3
    numerical_epsilon: float = 1e-12
    numerical_negligible_norm: float = 1e-12
    validation_batch_size: int = 8192
    cache_scan_batch_size: int = 100_000
    trim_count: int = 4
    krum_byzantine_count: int = 4
    krum_selected_count: int = 16
    prefer_fused_adam: bool = True
    materialize_client_batches: bool = True
    methods: tuple[str, ...] = METHODS

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS or self.profile not in PROFILES or self.seed < 0:
            raise ValueError("Step 5 dataset, profile, or seed is invalid")
        if (
            not self.methods
            or any(method not in METHODS for method in self.methods)
            or len(set(self.methods)) != len(self.methods)
        ):
            raise ValueError("Step 5 method selection is invalid")
        if self.total_clients < 3 or not 3 <= self.clients_per_round <= self.total_clients:
            raise ValueError("Step 5 client population or participation is invalid")
        if self.total_rounds < 2 or self.local_epochs < 1:
            raise ValueError("Step 5 round and local-epoch counts are invalid")
        if self.batch_size < 1 or self.validation_batch_size < 1 or self.cache_scan_batch_size < 1:
            raise ValueError("Step 5 batch sizes must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Step 5 learning rate must be finite and positive")
        if not math.isfinite(self.proximal_mu) or self.proximal_mu < 0.0:
            raise ValueError("Step 5 proximal coefficient must be finite and nonnegative")
        if not math.isfinite(self.maximum_class_weight) or self.maximum_class_weight < 1.0:
            raise ValueError("Step 5 class-weight cap is invalid")
        if not self.hidden_dimensions or any(width < 1 for width in self.hidden_dimensions):
            raise ValueError("Step 5 hidden dimensions must be positive")
        if (
            not 0.0 <= self.dropout < 1.0
            or not 0 <= self.burn_in_rounds < self.total_rounds
            or self.attribution_interval < 1
        ):
            raise ValueError("Step 5 dropout, burn-in, or attribution interval is invalid")
        self.aggregation_config()
        self.semantic_coefficients()
        self.numerical_coefficients()
        if (
            not math.isfinite(self.robust_z_max)
            or self.robust_z_max <= 0.0
            or self.numerical_minimum_history_clients < 3
            or not math.isfinite(self.numerical_epsilon)
            or self.numerical_epsilon <= 0.0
            or not math.isfinite(self.numerical_negligible_norm)
            or self.numerical_negligible_norm < 0.0
        ):
            raise ValueError("Step 5 robust-scoring controls are invalid")
        if (
            isinstance(self.trim_count, bool)
            or self.trim_count < 0
            or 2 * self.trim_count >= self.clients_per_round
        ):
            raise ValueError("Step 5 trimmed-mean count is inadmissible")
        if (
            isinstance(self.krum_byzantine_count, bool)
            or self.krum_byzantine_count < 0
            or self.clients_per_round < 2 * self.krum_byzantine_count + 3
        ):
            raise ValueError("Step 5 Multi-Krum Byzantine count is inadmissible")
        if (
            isinstance(self.krum_selected_count, bool)
            or not 1
            <= self.krum_selected_count
            <= self.clients_per_round - self.krum_byzantine_count
        ):
            raise ValueError("Step 5 Multi-Krum selected count is inadmissible")

    def aggregation_config(self) -> AggregationConfig:
        """Return the fully explicit SemCred-FL aggregation controls."""

        return AggregationConfig(
            burn_in_rounds=self.burn_in_rounds,
            semantic_fusion_coefficient=self.semantic_fusion_coefficient,
            disagreement_penalty=self.disagreement_penalty,
            trust_decrease_rate=self.trust_decrease_rate,
            trust_recovery_rate=self.trust_recovery_rate,
            sample_count_exponent=self.sample_count_exponent,
            sample_cap_median_multiplier=self.sample_cap_median_multiplier,
            trust_exponent=self.trust_exponent,
            threshold_multiplier=self.threshold_multiplier,
            minimum_trust=self.minimum_trust,
            minimum_retained_ratio=self.minimum_retained_ratio,
            clipping_multiplier=self.clipping_multiplier,
            history_decay=self.history_decay,
        )

    def semantic_coefficients(self) -> SemanticCoefficients:
        """Return the explicit semantic-score coefficients."""

        return SemanticCoefficients(
            loss=self.semantic_loss_weight,
            attribution=self.semantic_attribution_weight,
        )

    def numerical_coefficients(self) -> NumericalCoefficients:
        """Return the explicit numerical-score coefficients."""

        return NumericalCoefficients(
            center=self.numerical_center_weight,
            norm=self.numerical_norm_weight,
            direction=self.numerical_direction_weight,
            history=self.numerical_history_weight,
        )

    @property
    def full_budget(self) -> bool:
        """Return whether the manuscript's initial full training budget is active."""

        return (
            self.total_clients == 50
            and self.clients_per_round == 20
            and self.total_rounds == 100
            and self.local_epochs == 3
            and self.batch_size == 128
            and self.methods == METHODS
        )

    @property
    def evidence_scope(self) -> str:
        """Return the frozen dataset-level evidence scope."""

        if self.dataset == "edge_iiotset":
            return "validation_development_confirmatory_candidate"
        return "validation_secondary_license_provenance_blocked"

    def stable_record(self) -> dict[str, object]:
        """Return the content-addressed Step 5 scientific controls."""

        return {
            "schema": "semcred_phase12_step5_clean_condition_config_v1",
            "dataset": self.dataset,
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
            "aggregation": asdict(self.aggregation_config()),
            "semantic_coefficients": asdict(self.semantic_coefficients()),
            "numerical_coefficients": asdict(self.numerical_coefficients()),
            "robust_scoring": {
                "z_max": self.robust_z_max,
                "minimum_history_clients": self.numerical_minimum_history_clients,
                "epsilon": self.numerical_epsilon,
                "negligible_norm": self.numerical_negligible_norm,
            },
            "validation_batch_size": self.validation_batch_size,
            "trim_count": self.trim_count,
            "krum_byzantine_count": self.krum_byzantine_count,
            "krum_selected_count": self.krum_selected_count,
            "baseline_threat_budget": "primary_malicious_ratio_0_2",
            "methods": list(self.methods),
            "full_budget": self.full_budget,
            "evidence_scope": self.evidence_scope,
            "validation_partition_access": "completion_only",
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class Phase12CleanConditionRequest:
    """Artifact paths and runtime controls for one Step 5 condition."""

    preprocessing_report: Path
    client_partition_manifest: Path
    semantic_prior_path: Path
    output_directory: Path
    cache_root: Path
    device: str = "auto"
    max_rounds_this_call: int | None = None
    config: Phase12CleanConditionConfig = field(
        default_factory=lambda: Phase12CleanConditionConfig("edge_iiotset", "iid")
    )

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Step 5 device must be auto, cpu, or cuda")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Step 5 maximum rounds per call must be positive")


@dataclass(frozen=True, slots=True)
class Phase12CleanMatrixRequest:
    """Project registry and condition filters for the Step 5 matrix."""

    project_root: Path
    output_directory: Path
    cache_root: Path
    datasets: tuple[str, ...] = DATASETS
    profiles: tuple[str, ...] = PROFILES
    seeds: tuple[int, ...] = (20260816,)
    device: str = "auto"
    max_rounds_this_call: int | None = None
    max_conditions_this_call: int | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.datasets
            or any(value not in DATASETS for value in self.datasets)
            or len(set(self.datasets)) != len(self.datasets)
            or not self.profiles
            or any(value not in PROFILES for value in self.profiles)
            or len(set(self.profiles)) != len(self.profiles)
            or not self.seeds
            or any(value < 0 for value in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("Step 5 matrix filters are invalid")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Step 5 matrix device must be auto, cpu, or cuda")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Step 5 matrix round limit must be positive")
        if self.max_conditions_this_call is not None and self.max_conditions_this_call < 1:
            raise ValueError("Step 5 matrix condition limit must be positive")


@dataclass(frozen=True, slots=True)
class _ExecutionProfile:
    device: str
    optimizer_backend: str
    batch_delivery: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _execution(request: Phase12CleanConditionRequest) -> _ExecutionProfile:
    if request.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Step 5 but is unavailable")
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


def _client_index(client_id: str) -> int:
    suffix = client_id.removeprefix("client-")
    if not client_id.startswith("client-") or not suffix.isdigit():
        raise DataContractError("Step 5 client ID is malformed")
    return int(suffix)


def _scheduled_indices(schedule: ClientSchedule) -> tuple[int, ...]:
    return tuple(
        sorted({_client_index(client_id) for selected in schedule.rounds for client_id in selected})
    )


def _local_config(
    config: Phase12CleanConditionConfig, execution: _ExecutionProfile
) -> LocalTrainingConfig:
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


def _train_or_load_update(
    *,
    request: Phase12CleanConditionRequest,
    condition_input_id: str,
    cache: SelectedClientCache,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    local_config: LocalTrainingConfig,
    execution: _ExecutionProfile,
    round_id: int,
    client_id: str,
    global_parameters: Tensor,
) -> tuple[ClientUpdate, str, bool]:
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step5_update_input_v1",
        "condition_input_id": condition_input_id,
        "cache_id": cache.cache_id,
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
        update = _load_cached_update(
            path,
            expected_input_id=update_input_id,
            expected_values=manifest.total_values,
        )
        return update, update_input_id, True
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


def _method_run_id(condition_input_id: str, method: str) -> str:
    return run_id(
        {
            "schema": "semcred_phase12_step5_method_run_v1",
            "condition_input_id": condition_input_id,
            "method": method,
        },
        length=64,
    )


def _aggregate_standard(
    method: str,
    updates: tuple[DefenseUpdate, ...],
    config: Phase12CleanConditionConfig,
) -> tuple[Tensor, dict[str, object]]:
    if method == "fedavg":
        result = fedavg(updates)
        controls: dict[str, object] = {}
    elif method == "coordinate_median":
        result = coordinate_median(updates)
        controls = {}
    elif method == "trimmed_mean":
        result = trimmed_mean(updates, trim_count=config.trim_count)
        controls = {"trim_count": config.trim_count}
    elif method == "multi_krum":
        result = multi_krum(
            updates,
            assumed_byzantine=config.krum_byzantine_count,
            selected_count=config.krum_selected_count,
        )
        controls = {
            "assumed_byzantine": config.krum_byzantine_count,
            "selected_count": config.krum_selected_count,
        }
    else:
        raise ValueError(f"Unknown Step 5 standard method: {method}")
    return result.delta, {"controls": controls, "weights": result.weights}


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
    aggregation: dict[str, object],
) -> tuple[str, dict[str, object]]:
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step5_round_v1",
        "method": method,
        "round_id": round_id,
        "previous_round_record_id": previous_record_id,
        "selected_client_ids": list(selected),
        "accepted_client_ids": [update.client_id for update in accepted],
        "update_input_ids": list(update_ids),
        "aggregate_delta_sha256": _tensor_sha256(aggregate_delta),
        "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "aggregation": aggregation,
        "clean_training_only": True,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    record_id = run_id(stable, length=64)
    return record_id, {"round_record_id": record_id, **stable}


def _load_completed_method(path: Path, method_id: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read Step 5 method report: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError("Completed Step 5 method report is malformed")
    stable = {
        key: item
        for key, item in value.items()
        if key not in {"method_result_id", "status", "completed_at_utc"}
    }
    if (
        value.get("method_run_id") != method_id
        or value.get("status") != "complete"
        or value.get("method_result_id") != run_id(stable, length=64)
    ):
        raise DataContractError("Completed Step 5 method report identity is invalid")
    return cast(dict[str, object], value)


def _run_method(
    *,
    request: Phase12CleanConditionRequest,
    method: str,
    condition_input_id: str,
    corpus: PreprocessedCorpus,
    cache: SelectedClientCache,
    schedule: ClientSchedule,
    detector: DetectorConfig,
    execution: _ExecutionProfile,
    anchors: TrainingBatch,
    layout: FeatureGroupLayout,
    prior: SemanticPriorBundle,
) -> dict[str, object]:
    method_id = _method_run_id(condition_input_id, method)
    method_directory = request.output_directory / "methods" / method
    final_path = method_directory / "method_report.json"
    if final_path.exists():
        return _load_completed_method(final_path, method_id)

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
            comparison_input_id=condition_input_id,
            schedule=schedule,
        )
    manifest = parameter_manifest(model)
    local_config = _local_config(request.config, execution)
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
                f"phase12_step5={request.config.dataset}/{request.config.profile}/{method} "
                f"round={round_id + 1}/{request.config.total_rounds} "
                f"client={position}/{len(selected)} id={client_id}",
                flush=True,
            )
            update, update_id, was_reused = _train_or_load_update(
                request=request,
                condition_input_id=condition_input_id,
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
            raise DataContractError("Step 5 clean client updates failed validation")
        accepted = validation.accepted
        if method in _STANDARD_METHODS:
            aggregate_delta, aggregation = _aggregate_standard(method, accepted, request.config)
        elif method == "semcred_fl":
            semantic = evaluate_semantic_updates(
                model,
                accepted,
                manifest,
                anchors,
                layout,
                prior,
                attribution_enabled=round_id % request.config.attribution_interval == 0,
                coefficients=request.config.semantic_coefficients(),
                z_max=request.config.robust_z_max,
                epsilon=request.config.numerical_epsilon,
                negligible_norm=request.config.numerical_negligible_norm,
            )
            histories = {
                client_id: state.accepted_history
                for client_id, state in states.items()
                if state.accepted_history is not None
            }
            numerical = evaluate_numerical_updates(
                accepted,
                histories=histories,
                coefficients=request.config.numerical_coefficients(),
                z_max=request.config.robust_z_max,
                epsilon=request.config.numerical_epsilon,
                negligible_norm=request.config.numerical_negligible_norm,
                minimum_history_clients=request.config.numerical_minimum_history_clients,
            )
            result = process_semcred_round(
                round_id=round_id,
                updates=accepted,
                semantic_scores={key: value.score for key, value in semantic.items()},
                numerical_scores={key: value.score for key, value in numerical.items()},
                client_states=states,
                config=request.config.aggregation_config(),
                epsilon=request.config.numerical_epsilon,
            )
            aggregate_delta = result.aggregate_delta
            states = result.client_states
            aggregation = {
                "burn_in": result.burn_in,
                "attribution_enabled": round_id % request.config.attribution_interval == 0,
                "eligible_client_ids": list(result.eligible_client_ids),
                "history_admitted_client_ids": list(result.history_admitted_client_ids),
                "participating_trust": result.participating_trust,
                "semantic_scores": {key: semantic[key].score for key in sorted(semantic)},
                "numerical_scores": {key: numerical[key].score for key in sorted(numerical)},
            }
        else:
            raise ValueError(f"Unknown Step 5 method: {method}")
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
            aggregation=aggregation,
        )
        _write_json_immutable(method_directory / "rounds" / f"round-{round_id:04d}.json", record)
        pointer = _save_method_checkpoint(
            method_directory=method_directory,
            method=method,
            method_run_id=method_id,
            comparison_input_id=condition_input_id,
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

    metrics = _evaluate_validation(
        model,
        corpus,
        batch_size=request.config.validation_batch_size,
        device=execution.device,
    )
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step5_method_result_v1",
        "method": method,
        "method_run_id": method_id,
        "condition_input_id": condition_input_id,
        "schedule_id": schedule.schedule_id,
        "rounds_completed": next_round,
        "last_round_record_id": previous_record_id,
        "final_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "validation_metrics": metrics.to_dict(),
        "gates": {
            "clean_training_only": True,
            "shared_initialization_schedule_and_optimizer": True,
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


def _metric_records(
    reports: Sequence[dict[str, object]],
    *,
    config: Phase12CleanConditionConfig,
    condition_result_id: str,
) -> tuple[RunMetricRecord, ...]:
    names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "worst_class_recall",
        "macro_false_positive_rate",
    )
    records: list[RunMetricRecord] = []
    for report in reports:
        metrics = report.get("validation_metrics")
        method = report.get("method")
        method_run_id = report.get("method_run_id")
        if (
            not isinstance(metrics, dict)
            or not isinstance(method, str)
            or not isinstance(method_run_id, str)
        ):
            raise DataContractError("Step 5 method metrics are malformed")
        for name in names:
            raw = metrics.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise DataContractError("Step 5 validation metric is malformed")
            records.append(
                RunMetricRecord(
                    run_id=method_run_id,
                    experiment_family_id="phase12_step5_clean_utility",
                    dataset=config.dataset,
                    method=method,
                    condition_id=f"{config.profile}_clean_validation",
                    seed=config.seed,
                    metric_name=name,
                    value=float(raw),
                    source_artifact_id=condition_result_id,
                    test_partition_accessed=False,
                    performance_claims_allowed=False,
                )
            )
    return tuple(records)


def _load_completed_condition(path: Path, condition_input_id: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read Step 5 condition report: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError("Completed Step 5 condition report is malformed")
    stable = {
        key: item
        for key, item in value.items()
        if key not in {"condition_result_id", "status", "reporting", "completed_at_utc"}
    }
    if (
        value.get("condition_input_id") != condition_input_id
        or value.get("status") != "complete"
        or value.get("condition_result_id") != run_id(stable, length=64)
    ):
        raise DataContractError("Completed Step 5 condition identity is invalid")
    return cast(dict[str, object], value)


def run_phase12_clean_condition(request: Phase12CleanConditionRequest) -> dict[str, object]:
    """Build or resume one clean utility condition."""

    execution = _execution(request)
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(
        request.client_partition_manifest,
        corpus,
        request.config.profile,
    )
    prior = load_semantic_prior_bundle(request.semantic_prior_path)
    layout = load_feature_group_layout(corpus)
    anchors = load_scoring_anchor_batch(corpus)
    if corpus.dataset != request.config.dataset:
        raise DataContractError("Step 5 dataset differs from the selected corpus")
    if profile.num_clients != request.config.total_clients:
        raise DataContractError("Step 5 client population differs from Phase 3D")
    if (
        prior.dataset != corpus.dataset
        or prior.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or prior.preprocessor_id != corpus.preprocessor_id
        or prior.anchor_id != corpus.anchor_id
        or prior.feature_group_names != layout.group_names
    ):
        raise DataContractError("Step 5 semantic prior differs from the selected corpus")

    client_ids = tuple(f"client-{index:02d}" for index in range(request.config.total_clients))
    schedule = generate_client_schedule(
        client_ids,
        rounds=request.config.total_rounds,
        clients_per_round=request.config.clients_per_round,
        seed=request.config.seed,
    )
    indices = _scheduled_indices(schedule)
    cache_scope_id = run_id(
        {
            "schema": "semcred_phase12_step5_selected_client_cache_v1",
            "dataset": request.config.dataset,
            "profile": request.config.profile,
            "selected_client_indices": list(indices),
        },
        length=64,
    )
    cache_directory = (
        request.cache_root / request.config.dataset / request.config.profile / cache_scope_id
    )
    build_selected_client_cache(
        ClientCacheBuildRequest(
            preprocessing_report=request.preprocessing_report,
            client_partition_manifest=request.client_partition_manifest,
            profile=request.config.profile,
            selected_client_indices=indices,
            schedule_id=cache_scope_id,
            output_directory=cache_directory,
            work_database=cache_directory / "client_cache.sqlite3",
            batch_size=request.config.cache_scan_batch_size,
        )
    )
    cache = load_selected_client_cache(cache_directory / "client_cache.json")
    if (
        cache.dataset != corpus.dataset
        or cache.profile != request.config.profile
        or cache.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or cache.client_partition_id != profile.client_partition_id
        or cache.schedule_id != cache_scope_id
        or cache.selected_client_indices != indices
    ):
        raise DataContractError("Step 5 selected-client cache differs from frozen inputs")

    detector = DetectorConfig(
        corpus.input_dimension,
        corpus.num_classes,
        request.config.hidden_dimensions,
        request.config.dropout,
    )
    torch.manual_seed(request.config.seed)
    initial_model = IntrusionMLP(detector)
    manifest = parameter_manifest(initial_model)
    stable_input: dict[str, object] = {
        **request.config.stable_record(),
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "client_partition_id": profile.client_partition_id,
        "cache_id": cache.cache_id,
        "cached_client_indices": list(indices),
        "semantic_prior_id": prior.bundle_id,
        "feature_group_layout_id": layout.layout_id,
        "schedule_id": schedule.schedule_id,
        "detector": detector.to_dict(),
        "initial_parameters_sha256": _tensor_sha256(flatten_parameters(initial_model, manifest)),
        "execution_profile": execution.to_dict(),
    }
    condition_input_id = run_id(stable_input, length=64)
    request.output_directory.mkdir(parents=True, exist_ok=True)
    resolved = {"condition_input_id": condition_input_id, **stable_input}
    resolved_path = request.output_directory / "resolved_config.json"
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise DataContractError("Existing Step 5 resolved configuration differs")
    else:
        _write_json(resolved_path, resolved)
        schedule.write(request.output_directory / "client_schedule.json")

    final_path = request.output_directory / "condition_report.json"
    if final_path.exists():
        return _load_completed_condition(final_path, condition_input_id)

    method_reports = tuple(
        _run_method(
            request=request,
            method=method,
            condition_input_id=condition_input_id,
            corpus=corpus,
            cache=cache,
            schedule=schedule,
            detector=detector,
            execution=execution,
            anchors=anchors,
            layout=layout,
            prior=prior,
        )
        for method in request.config.methods
    )
    if any(report.get("status") != "complete" for report in method_reports):
        return {
            "status": "interrupted",
            "condition_input_id": condition_input_id,
            "dataset": request.config.dataset,
            "profile": request.config.profile,
            "seed": request.config.seed,
            "methods": list(method_reports),
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }

    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_step5_clean_condition_result_v1",
        "condition_input_id": condition_input_id,
        "dataset": request.config.dataset,
        "profile": request.config.profile,
        "seed": request.config.seed,
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
            "clean_training_only": True,
            "five_standardized_methods_complete": request.config.methods == METHODS,
            "requested_methods_complete": True,
            "shared_initialization_schedule_and_optimizer": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    condition_result_id = run_id(stable_result, length=64)
    records = _metric_records(
        method_reports,
        config=request.config,
        condition_result_id=condition_result_id,
    )
    metrics_path = request.output_directory / "run_metrics.jsonl"
    metric_bytes = "".join(canonical_json(asdict(record)) + "\n" for record in records).encode()
    if metrics_path.exists() and metrics_path.read_bytes() != metric_bytes:
        raise DataContractError("Existing Step 5 run metrics differ")
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
        "condition_result_id": condition_result_id,
        "status": "complete",
        "reporting": {
            "summary_bundle_id": summary["summary_bundle_id"],
            "macro_f1_artifact_id": rendered["artifact_id"],
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(final_path, report)
    return report


def _artifact_paths(project_root: Path, dataset: str) -> tuple[Path, Path, Path]:
    prefix = project_root / "artifacts"
    return (
        prefix / "preprocessing" / "phase_3c" / dataset / f"{dataset}_phase3c_preprocessed.json",
        prefix / "client_partitions" / "phase_3d" / dataset / f"{dataset}_phase3d_clients.json",
        prefix / "semantic_priors" / "phase_7" / dataset / "semantic_prior_bundle.json",
    )


def _condition_directory(root: Path, config: Phase12CleanConditionConfig) -> Path:
    return root / "conditions" / config.dataset / config.profile / f"seed_{config.seed}"


def _matrix_identity(
    request: Phase12CleanMatrixRequest,
) -> tuple[tuple[tuple[str, str, int], ...], dict[str, object], str]:
    """Return the ordered registry, stable input, and content identity."""

    normalized_overrides = cast(
        dict[str, object],
        json.loads(canonical_json(dict(request.config_overrides))),
    )
    registry = tuple(
        (dataset, profile, seed)
        for dataset in request.datasets
        for profile in request.profiles
        for seed in request.seeds
    )
    condition_configs = tuple(
        Phase12CleanConditionConfig(
            dataset=dataset,
            profile=profile,
            seed=seed,
            **request.config_overrides,
        )
        for dataset, profile, seed in registry
    )
    configured_methods = condition_configs[0].methods
    stable_input: dict[str, object] = {
        "schema": "semcred_phase12_step5_clean_matrix_input_v3",
        "registry": [
            {"dataset": dataset, "profile": profile, "seed": seed}
            for dataset, profile, seed in registry
        ],
        "condition_config_ids": [
            run_id(config.stable_record(), length=64) for config in condition_configs
        ],
        "config_overrides": normalized_overrides,
        "methods": list(configured_methods),
        "requested_device": request.device,
        "validation_partition_access": "completion_only",
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    return registry, stable_input, run_id(stable_input, length=64)


def inspect_phase12_clean_matrix(request: Phase12CleanMatrixRequest) -> dict[str, object]:
    """Inspect a clean matrix without creating files or loading dataset records."""

    registry, stable_input, matrix_input_id = _matrix_identity(request)
    missing: list[str] = []
    for dataset in request.datasets:
        for path in _artifact_paths(request.project_root, dataset):
            if not path.is_file():
                missing.append(str(path.resolve()))
    methods = stable_input["methods"]
    if not isinstance(methods, list):
        raise TypeError("Step 5 stable method selection is malformed")
    return {
        "status": "ready" if not missing else "blocked",
        "matrix_input_id": matrix_input_id,
        "datasets": list(request.datasets),
        "profiles": list(request.profiles),
        "seeds": list(request.seeds),
        "methods": methods,
        "conditions_total": len(registry),
        "method_runs_total": len(registry) * len(methods),
        "federated_rounds_per_method": request.config_overrides.get("total_rounds", 100),
        "local_epochs_per_client": request.config_overrides.get("local_epochs", 3),
        "missing_artifacts": sorted(missing),
        "campaign_directory": str((request.output_directory / matrix_input_id).resolve()),
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }


def run_phase12_clean_matrix(request: Phase12CleanMatrixRequest) -> dict[str, object]:
    """Build or resume the selected Step 5 clean utility matrix."""

    registry, stable_input, matrix_input_id = _matrix_identity(request)
    campaign_directory = request.output_directory / matrix_input_id
    campaign_directory.mkdir(parents=True, exist_ok=True)
    resolved_path = campaign_directory / "resolved_matrix.json"
    resolved = {"matrix_input_id": matrix_input_id, **stable_input}
    if resolved_path.exists():
        if json.loads(resolved_path.read_text(encoding="utf-8")) != resolved:
            raise DataContractError("Existing Step 5 matrix registry differs")
    else:
        _write_json(resolved_path, resolved)

    final_path = campaign_directory / "matrix_report.json"
    if final_path.exists():
        try:
            value = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataContractError(f"Cannot read Step 5 matrix report: {error}") from error
        if not isinstance(value, dict):
            raise DataContractError("Completed Step 5 matrix report is malformed")
        stable = {
            key: item
            for key, item in value.items()
            if key not in {"matrix_result_id", "status", "completed_at_utc"}
        }
        if (
            value.get("matrix_input_id") != matrix_input_id
            or value.get("status") != "complete"
            or value.get("matrix_result_id") != run_id(stable, length=64)
        ):
            raise DataContractError("Completed Step 5 matrix identity is invalid")
        return cast(dict[str, object], value)

    results: list[dict[str, object]] = []
    newly_completed = 0
    for dataset, profile, seed in registry:
        config = Phase12CleanConditionConfig(
            dataset=dataset,
            profile=profile,
            seed=seed,
            **request.config_overrides,
        )
        preprocessing, partitions, prior = _artifact_paths(request.project_root, dataset)
        condition_request = Phase12CleanConditionRequest(
            preprocessing_report=preprocessing,
            client_partition_manifest=partitions,
            semantic_prior_path=prior,
            output_directory=_condition_directory(
                request.output_directory / matrix_input_id,
                config,
            ),
            cache_root=request.cache_root,
            device=request.device,
            max_rounds_this_call=request.max_rounds_this_call,
            config=config,
        )
        existed = (condition_request.output_directory / "condition_report.json").exists()
        report = run_phase12_clean_condition(condition_request)
        if report.get("status") != "complete":
            return {
                "status": "interrupted",
                "matrix_input_id": matrix_input_id,
                "conditions_total": len(registry),
                "conditions_complete": len(results),
                "active_condition": report,
                "campaign_directory": str(campaign_directory.resolve()),
                "test_partition_accessed": False,
                "performance_claims_allowed": False,
            }
        results.append(report)
        newly_completed += int(not existed)
        if (
            request.max_conditions_this_call is not None
            and newly_completed >= request.max_conditions_this_call
            and len(results) < len(registry)
        ):
            return {
                "status": "interrupted",
                "matrix_input_id": matrix_input_id,
                "conditions_total": len(registry),
                "conditions_complete": len(results),
                "next_condition_index": len(results),
                "campaign_directory": str(campaign_directory.resolve()),
                "test_partition_accessed": False,
                "performance_claims_allowed": False,
            }

    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_step5_clean_matrix_result_v1",
        "matrix_input_id": matrix_input_id,
        "condition_results": [
            {
                "dataset": report["dataset"],
                "profile": report["profile"],
                "seed": report["seed"],
                "condition_input_id": report["condition_input_id"],
                "condition_result_id": report["condition_result_id"],
            }
            for report in results
        ],
        "gates": {
            "registry_complete": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
            "confirmatory_inference_allowed": False,
        },
    }
    report = {
        **stable_result,
        "matrix_result_id": run_id(stable_result, length=64),
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(final_path, report)
    return report
