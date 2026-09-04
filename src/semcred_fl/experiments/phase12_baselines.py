"""Resumable Phase 12 Step 4 reduced standard-baseline matrix."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
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
from semcred_fl.attacks.execution_contracts import AttackRealization, build_attack_realization
from semcred_fl.baselines.coordinate_median import coordinate_median
from semcred_fl.baselines.multi_krum import multi_krum
from semcred_fl.baselines.trimmed_mean import trimmed_mean
from semcred_fl.data.client_cache import SelectedClientCache, load_selected_client_cache
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import PreprocessedCorpus, load_preprocessed_corpus
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.experiments.phase12_attack import (
    Phase12AttackConfig,
    Phase12AttackRequest,
    _attack_spec,
    _AttackedBatchSource,
    _client_index,
    _load_cached_update,
    _local_config,
    _phase6c_config,
    _resolve_execution,
    run_phase12_attack_comparison,
)
from semcred_fl.experiments.phase12_clean import (
    _evaluate_validation,
    _load_method_checkpoint,
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
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP

_BASELINES = ("coordinate_median", "trimmed_mean", "multi_krum")
_ALL_METHODS = ("fedavg", "coordinate_median", "trimmed_mean", "multi_krum", "semcred_fl")
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Phase12BaselineConfig:
    """Frozen controls for the Step 4 reduced comparison."""

    attack: Phase12AttackConfig = field(default_factory=Phase12AttackConfig)
    trim_count: int = 2
    krum_byzantine_count: int = 2
    krum_selected_count: int = 8

    def __post_init__(self) -> None:
        count = self.attack.clients_per_round
        if isinstance(self.trim_count, bool) or not 0 <= self.trim_count * 2 < count:
            raise ValueError("Step 4 trim count is inadmissible")
        if (
            isinstance(self.krum_byzantine_count, bool)
            or self.krum_byzantine_count < 0
            or count < 2 * self.krum_byzantine_count + 3
        ):
            raise ValueError("Step 4 Multi-Krum Byzantine count is inadmissible")
        if (
            isinstance(self.krum_selected_count, bool)
            or not 1 <= self.krum_selected_count <= count - self.krum_byzantine_count
        ):
            raise ValueError("Step 4 Multi-Krum selected count is inadmissible")

    def stable_record(self) -> dict[str, object]:
        """Return the content-addressed Step 4 scientific controls."""

        return {
            "schema": "semcred_phase12_step4_config_v1",
            "source_attack_config": self.attack.stable_record(),
            "methods": list(_ALL_METHODS),
            "trim_count": self.trim_count,
            "krum_byzantine_count": self.krum_byzantine_count,
            "krum_selected_count": self.krum_selected_count,
            "bounded_pilot_only": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class Phase12BaselineRequest:
    """Paths and runtime controls for the Step 4 matrix."""

    step3_request: Phase12AttackRequest
    output_directory: Path
    max_rounds_this_call: int | None = None
    config: Phase12BaselineConfig = field(default_factory=Phase12BaselineConfig)

    def __post_init__(self) -> None:
        if self.step3_request.config != self.config.attack:
            raise ValueError("Step 4 must inherit the exact Step 3 attack configuration")
        if self.step3_request.max_rounds_this_call is not None:
            raise ValueError("Step 4 requires an unrestricted completed Step 3 request")
        if self.output_directory == self.step3_request.output_directory:
            raise ValueError("Step 4 output must not overwrite Step 3")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Step 4 maximum rounds per call must be positive")


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _method_run_id(matrix_input_id: str, method: str) -> str:
    return run_id(
        {
            "schema": "semcred_phase12_step4_method_run_v1",
            "matrix_input_id": matrix_input_id,
            "method": method,
        },
        length=64,
    )


def _train_or_load_update(
    *,
    request: Phase12BaselineRequest,
    update_namespace_id: str,
    cache: SelectedClientCache,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    local_config: LocalTrainingConfig,
    resolution: Phase6BResolution,
    realization: AttackRealization,
    attack_id: str,
    selections: Mapping[str, ClientPoisonSelection],
    round_id: int,
    client_id: str,
    global_parameters: Tensor,
) -> tuple[ClientUpdate, str, bool]:
    attack_request = request.step3_request
    execution = _resolve_execution(attack_request)
    client_index = _client_index(client_id)
    malicious = client_id in frozenset(realization.malicious_client_ids)
    selection = selections.get(client_id)
    selected_ids = selection.selected_record_ids if selection is not None else ()
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step3_update_input_v1",
        "comparison_input_id": update_namespace_id,
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
    candidates = (
        request.output_directory / "shared_updates" / f"{update_input_id}.pt",
        attack_request.output_directory / "shared_updates" / f"{update_input_id}.pt",
    )
    for path in candidates:
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
        _attack_spec(request.config.attack),
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
            batch_size=request.config.attack.batch_size,
            device=execution.device,
            malicious=malicious,
            selected_record_ids=selected_ids,
            resolution=resolution,
            materialize=request.config.attack.materialize_client_batches,
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
    _save_torch(candidates[0], payload)
    return update, update_input_id, False


def _aggregate(
    method: str,
    updates: tuple[DefenseUpdate, ...],
    config: Phase12BaselineConfig,
) -> Tensor:
    if method == "coordinate_median":
        return coordinate_median(updates).delta
    if method == "trimmed_mean":
        return trimmed_mean(updates, trim_count=config.trim_count).delta
    if method == "multi_krum":
        return multi_krum(
            updates,
            assumed_byzantine=config.krum_byzantine_count,
            selected_count=config.krum_selected_count,
        ).delta
    raise ValueError("Unknown Step 4 baseline")


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
) -> tuple[str, dict[str, object]]:
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step4_round_v1",
        "method": method,
        "round_id": round_id,
        "previous_round_record_id": previous_record_id,
        "selected_client_ids": list(selected),
        "accepted_client_ids": [update.client_id for update in accepted],
        "update_input_ids": list(update_ids),
        "aggregate_delta_sha256": _tensor_sha256(aggregate_delta),
        "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "post_decision_attack_audit": {
            "active_malicious_client_ids": [
                update.client_id for update in original_updates if update.is_malicious is True
            ],
            "joined_after_aggregation": True,
        },
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    record_id = run_id(stable, length=64)
    return record_id, {"round_record_id": record_id, **stable}


def _run_baseline(
    *,
    request: Phase12BaselineRequest,
    method: str,
    matrix_input_id: str,
    step3_input_id: str,
    corpus: PreprocessedCorpus,
    cache: SelectedClientCache,
    schedule: ClientSchedule,
    detector: DetectorConfig,
    resolution: Phase6BResolution,
    realization: AttackRealization,
    attack_id: str,
    selections: Mapping[str, ClientPoisonSelection],
) -> dict[str, object]:
    method_id = _method_run_id(matrix_input_id, method)
    method_directory = request.output_directory / "methods" / method
    final_path = method_directory / "method_report.json"
    if final_path.exists():
        value = _load_json(final_path, "completed Step 4 method report")
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
            raise DataContractError("Completed Step 4 method report belongs to other inputs")
        return value

    torch.manual_seed(request.config.attack.seed)
    model = IntrusionMLP(detector)
    next_round = 0
    previous_record_id: str | None = None
    pointer_path = method_directory / "latest_checkpoint.json"
    if pointer_path.exists():
        model, states, next_round, previous_record_id = _load_method_checkpoint(
            method_directory=method_directory,
            method=method,
            method_run_id=method_id,
            comparison_input_id=matrix_input_id,
            schedule=schedule,
        )
        if states:
            raise DataContractError("Standard Step 4 baselines cannot restore client trust state")
    manifest = parameter_manifest(model)
    execution = _resolve_execution(request.step3_request)
    local_config = _local_config(request.step3_request, execution)
    stop_round = request.config.attack.total_rounds
    if request.max_rounds_this_call is not None:
        stop_round = min(stop_round, next_round + request.max_rounds_this_call)

    for round_id in range(next_round, stop_round):
        selected = schedule.rounds[round_id]
        broadcast = flatten_parameters(model, manifest).detach().cpu()
        started = time.perf_counter()
        updates: list[ClientUpdate] = []
        update_ids: list[str] = []
        reused = 0
        for position, client_id in enumerate(selected, start=1):
            print(
                f"phase12_step4={method} round={round_id + 1}/"
                f"{request.config.attack.total_rounds} client={position}/{len(selected)} "
                f"id={client_id}",
                flush=True,
            )
            update, update_id, was_reused = _train_or_load_update(
                request=request,
                update_namespace_id=step3_input_id,
                cache=cache,
                detector=detector,
                manifest=manifest,
                local_config=local_config,
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
            raise DataContractError("Step 4 attacked client updates failed validation")
        accepted = validation.accepted
        aggregate_delta = _aggregate(method, accepted, request.config)
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
        )
        _write_json_immutable(method_directory / "rounds" / f"round-{round_id:04d}.json", record)
        pointer = _save_method_checkpoint(
            method_directory=method_directory,
            method=method,
            method_run_id=method_id,
            comparison_input_id=matrix_input_id,
            schedule=schedule,
            model=model,
            states={},
            next_round=round_id + 1,
            last_round_record_id=record_id,
        )
        _write_json(pointer_path, pointer)
        _write_json(
            method_directory / "runtime" / f"round-{round_id:04d}.json",
            {
                "round_id": round_id,
                "duration_seconds": time.perf_counter() - started,
                "updates_trained": len(selected) - reused,
                "updates_reused": reused,
            },
        )
        previous_record_id = record_id
        next_round = round_id + 1

    if next_round < request.config.attack.total_rounds:
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
        batch_size=request.config.attack.validation_batch_size,
        device=execution.device,
    )
    stable: dict[str, object] = {
        "schema": "semcred_phase12_step4_method_result_v1",
        "method": method,
        "method_run_id": method_id,
        "matrix_input_id": matrix_input_id,
        "source_step3_comparison_input_id": step3_input_id,
        "schedule_id": schedule.schedule_id,
        "rounds_completed": next_round,
        "last_round_record_id": previous_record_id,
        "final_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "validation_metrics": metrics.to_dict(),
        "gates": {
            "same_step3_condition": True,
            "simulation_truth_excluded_from_aggregation": True,
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
    seed: int,
    matrix_result_id: str,
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
        method = report.get("method")
        method_run_id = report.get("method_run_id")
        metrics = report.get("validation_metrics")
        if (
            not isinstance(method, str)
            or not isinstance(method_run_id, str)
            or not isinstance(metrics, dict)
        ):
            raise DataContractError("Step 4 method metrics are malformed")
        for name in names:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DataContractError("Step 4 validation metric is malformed")
            records.append(
                RunMetricRecord(
                    run_id=method_run_id,
                    experiment_family_id="phase12_step4_reduced_baseline_pilot",
                    dataset="edge_iiotset",
                    method=method,
                    condition_id="dirichlet_0_3_label_flip_mr_0_2_pf_0_3_validation",
                    seed=seed,
                    metric_name=name,
                    value=float(value),
                    source_artifact_id=matrix_result_id,
                    test_partition_accessed=False,
                    performance_claims_allowed=False,
                )
            )
    return tuple(records)


def run_phase12_reduced_baselines(request: Phase12BaselineRequest) -> dict[str, object]:
    """Build or resume Step 4 and combine it with the completed Step 3 controls."""

    step3_report = run_phase12_attack_comparison(request.step3_request)
    if step3_report.get("status") != "complete":
        raise DataContractError("Step 4 requires a completed Step 3 comparison")
    step3_input_id = cast(str, step3_report["comparison_input_id"])
    step3_result_id = cast(str, step3_report["comparison_result_id"])
    attack_config = request.config.attack
    corpus = load_preprocessed_corpus(request.step3_request.preprocessing_report)
    cache = load_selected_client_cache(request.step3_request.cache_directory / "client_cache.json")
    resolution_path = request.step3_request.resolution_path
    resolution = load_phase6b_resolution(resolution_path)
    phase6c_config = _phase6c_config(attack_config)
    spec = _attack_spec(attack_config)
    realization = build_attack_realization(phase6c_config, spec)
    schedule = realization.schedule
    attack_id = spec.attack_id(phase6c_config)
    selection_request = Phase6CAttackRequest(
        preprocessing_report=request.step3_request.preprocessing_report,
        client_partition_manifest=request.step3_request.client_partition_manifest,
        resolution_path=resolution_path,
        output_directory=request.step3_request.output_directory / "attack_contract",
        cache_directory=request.step3_request.cache_directory,
        cache_work_database=request.step3_request.cache_work_database,
        spec=spec,
        device=request.step3_request.device,
        cache_scan_batch_size=attack_config.cache_scan_batch_size,
        config=phase6c_config,
    )
    selection_id, selections = _selection_manifest(
        request=selection_request,
        cache=cache,
        resolution=resolution,
        realization=realization,
        attack_id=attack_id,
    )
    detector = DetectorConfig(
        corpus.input_dimension,
        corpus.num_classes,
        attack_config.hidden_dimensions,
        attack_config.dropout,
    )
    torch.manual_seed(attack_config.seed)
    initial_model = IntrusionMLP(detector)
    stable_input: dict[str, object] = {
        **request.config.stable_record(),
        "source_step3_comparison_input_id": step3_input_id,
        "source_step3_comparison_result_id": step3_result_id,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "client_partition_id": cache.client_partition_id,
        "cache_id": cache.cache_id,
        "attack_id": attack_id,
        "realization_id": realization.realization_id,
        "selection_manifest_id": selection_id,
        "schedule_id": schedule.schedule_id,
        "detector": detector.to_dict(),
        "initial_parameters_sha256": _tensor_sha256(
            flatten_parameters(initial_model, parameter_manifest(initial_model))
        ),
        "execution_profile": _resolve_execution(request.step3_request).to_dict(),
    }
    matrix_input_id = run_id(stable_input, length=64)
    request.output_directory.mkdir(parents=True, exist_ok=True)
    resolved = {"matrix_input_id": matrix_input_id, **stable_input}
    resolved_path = request.output_directory / "resolved_config.json"
    if resolved_path.exists():
        if _load_json(resolved_path, "Step 4 resolved configuration") != resolved:
            raise DataContractError("Existing Step 4 resolved configuration differs")
    else:
        _write_json(resolved_path, resolved)
        schedule.write(request.output_directory / "client_schedule.json")

    final_path = request.output_directory / "comparison_report.json"
    if final_path.exists():
        completed = _load_json(final_path, "completed Step 4 comparison")
        stable_completed = {
            key: value
            for key, value in completed.items()
            if key not in {"matrix_result_id", "status", "reporting", "completed_at_utc"}
        }
        if (
            completed.get("matrix_input_id") != matrix_input_id
            or completed.get("status") != "complete"
            or completed.get("matrix_result_id") != run_id(stable_completed, length=64)
        ):
            raise DataContractError("Completed Step 4 comparison identity is invalid")
        return completed

    baseline_reports = tuple(
        _run_baseline(
            request=request,
            method=method,
            matrix_input_id=matrix_input_id,
            step3_input_id=step3_input_id,
            corpus=corpus,
            cache=cache,
            schedule=schedule,
            detector=detector,
            resolution=resolution,
            realization=realization,
            attack_id=attack_id,
            selections=selections,
        )
        for method in _BASELINES
    )
    if any(report.get("status") != "complete" for report in baseline_reports):
        return {
            "status": "interrupted",
            "matrix_input_id": matrix_input_id,
            "source_step3_comparison_result_id": step3_result_id,
            "methods": list(baseline_reports),
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
    source_methods = step3_report.get("method_results")
    if not isinstance(source_methods, list):
        raise DataContractError("Step 3 method results are malformed")
    reports_by_method: dict[str, dict[str, object]] = {}
    for value in (*cast(list[object], source_methods), *baseline_reports):
        if not isinstance(value, dict) or not isinstance(value.get("method"), str):
            raise DataContractError("Step 4 source method report is malformed")
        reports_by_method[cast(str, value["method"])] = cast(dict[str, object], value)
    if set(reports_by_method) != set(_ALL_METHODS):
        raise DataContractError("Step 4 method population is incomplete")
    ordered = tuple(reports_by_method[method] for method in _ALL_METHODS)
    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_step4_reduced_baseline_matrix_v1",
        "matrix_input_id": matrix_input_id,
        "source_step3_comparison_result_id": step3_result_id,
        "method_results": [
            {
                "method": report["method"],
                "method_run_id": report["method_run_id"],
                "method_result_id": report["method_result_id"],
                "final_parameters_sha256": report["final_parameters_sha256"],
                "validation_metrics": report["validation_metrics"],
                "source_step3": report["method"] in {"fedavg", "semcred_fl"},
            }
            for report in ordered
        ],
        "gates": {
            "same_initialization_schedule_attack_and_local_optimizer": True,
            "standard_baselines_only": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    matrix_result_id = run_id(stable_result, length=64)
    records = _metric_records(
        ordered,
        seed=attack_config.seed,
        matrix_result_id=matrix_result_id,
    )
    metrics_path = request.output_directory / "run_metrics.jsonl"
    metric_bytes = "".join(canonical_json(asdict(record)) + "\n" for record in records).encode()
    if metrics_path.exists() and metrics_path.read_bytes() != metric_bytes:
        raise DataContractError("Existing Step 4 metrics differ")
    if not metrics_path.exists():
        metrics_path.write_bytes(metric_bytes)
    summary = write_summary_bundle(
        records,
        request.output_directory / "reporting",
        bootstrap_resamples=1_000,
        bootstrap_seed=attack_config.seed,
    )
    rendered = render_summary_artifacts(
        request.output_directory / "reporting" / "metric_summaries.json",
        request.output_directory / "reporting",
        metric_name="macro_f1",
    )
    report = {
        **stable_result,
        "matrix_result_id": matrix_result_id,
        "status": "complete",
        "reporting": {
            "summary_bundle_id": summary["summary_bundle_id"],
            "macro_f1_artifact_id": rendered["artifact_id"],
        },
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(final_path, report)
    return report
