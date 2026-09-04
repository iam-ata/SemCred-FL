"""Resumable benign-only update-norm calibration for Phase 6B.2."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from semcred_fl.attacks.artifacts import benign_norm_reference
from semcred_fl.data.client_cache import (
    ClientCacheBuildRequest,
    SelectedClientCache,
    build_selected_client_cache,
    iter_cached_client_batches,
    load_selected_client_cache,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    TrainingBatch,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import ParameterManifest
from semcred_fl.federation.sampling import ClientSchedule, generate_client_schedule
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation
from semcred_fl.federation.updates import ClientUpdate
from semcred_fl.identity import run_id, to_primitive
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


@dataclass(frozen=True, slots=True)
class Phase6BNormCalibrationConfig:
    """Frozen benign development-run controls used by the adaptive attack."""

    schema_version: int = 1
    phase: str = "6B.2"
    base_seed: int = 20260816
    profiles: tuple[str, ...] = (
        "iid",
        "dirichlet_0_1",
        "dirichlet_0_3",
        "dirichlet_1_0",
    )
    total_rounds: int = 3
    clients_per_round: int = 20
    aggregation: str = "fedavg"
    local_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    proximal_mu: float = 0.0
    maximum_class_weight: float = 10.0
    hidden_dimensions: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    reference_population: str = "benign_only_development_runs_same_dataset_model_and_partition"
    test_partition_accessed: bool = False
    anchor_records_accessed: bool = False
    performance_claims_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.phase != "6B.2" or self.base_seed < 0:
            raise ValueError("Phase 6B.2 calibration identity is invalid")
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Phase 6B.2 profiles must be nonempty and unique")
        if self.total_rounds < 1 or self.clients_per_round < 1:
            raise ValueError("Phase 6B.2 schedule controls are invalid")
        if self.aggregation != "fedavg":
            raise ValueError("Benign norm calibration is frozen to FedAvg")
        if self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("Phase 6B.2 local-training controls are invalid")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("Phase 6B.2 learning rate must be positive and finite")
        if self.proximal_mu < 0.0 or not math.isfinite(self.proximal_mu):
            raise ValueError("Phase 6B.2 proximal coefficient is invalid")
        if self.maximum_class_weight < 1.0 or not math.isfinite(self.maximum_class_weight):
            raise ValueError("Phase 6B.2 class-weight cap is invalid")
        if not self.hidden_dimensions or any(value < 1 for value in self.hidden_dimensions):
            raise ValueError("Phase 6B.2 detector dimensions are invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Phase 6B.2 dropout is invalid")
        if (self.lower_quantile, self.upper_quantile) != (0.25, 0.75):
            raise ValueError("The adaptive norm reference is frozen to the IQR")
        if self.reference_population != (
            "benign_only_development_runs_same_dataset_model_and_partition"
        ):
            raise ValueError("Phase 6B.2 reference population is invalid")
        if (
            self.test_partition_accessed
            or self.anchor_records_accessed
            or self.performance_claims_allowed
        ):
            raise ValueError("Phase 6B.2 crossed its evidence boundary")

    @property
    def contract_id(self) -> str:
        """Return the full deterministic identity of these controls."""

        return run_id(self, length=64)


def phase6b_norm_calibration_config() -> Phase6BNormCalibrationConfig:
    """Return the manuscript-aligned Phase 6B.2 calibration contract."""

    return Phase6BNormCalibrationConfig()


@dataclass(frozen=True, slots=True)
class Phase6BNormCalibrationRequest:
    """Artifact paths and runtime controls for one dataset/profile calibration."""

    preprocessing_report: Path
    client_partition_manifest: Path
    profile: str
    output_directory: Path
    cache_directory: Path
    cache_work_database: Path
    device: str = "auto"
    cache_scan_batch_size: int = 100_000
    max_rounds_this_call: int | None = None
    config: Phase6BNormCalibrationConfig = field(default_factory=phase6b_norm_calibration_config)

    def __post_init__(self) -> None:
        if self.profile not in self.config.profiles:
            raise ValueError("Requested profile is outside the Phase 6B.2 contract")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Phase 6B.2 device must be auto, cpu, or cuda")
        if self.cache_scan_batch_size < 1:
            raise ValueError("Client-cache scan batch size must be positive")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Maximum rounds per call must be positive")


@dataclass(frozen=True, slots=True)
class _CachedBatchSource:
    cache: SelectedClientCache
    client_index: int
    batch_size: int

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        for batch in iter_cached_client_batches(
            self.cache,
            self.client_index,
            batch_size=self.batch_size,
            shuffle_seed=seed,
        ):
            yield batch.training_batch()


class _CachedBenignProvider:
    def __init__(
        self,
        cache: SelectedClientCache,
        detector: DetectorConfig,
        config: Phase6BNormCalibrationConfig,
        device: str,
    ) -> None:
        self.cache = cache
        self.detector = detector
        self.batch_size = config.batch_size
        self.local_config = LocalTrainingConfig(
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            proximal_mu=config.proximal_mu,
            maximum_class_weight=config.maximum_class_weight,
            seed=config.base_seed,
            device=device,
        )

    @staticmethod
    def _index(client_id: str) -> int:
        prefix = "client-"
        suffix = client_id.removeprefix(prefix)
        if not client_id.startswith(prefix) or not suffix.isdigit():
            raise DataContractError("Phase 6B.2 client ID is malformed")
        return int(suffix)

    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        updates: list[ClientUpdate] = []
        for client_position, client_id in enumerate(selected_client_ids, start=1):
            client_index = self._index(client_id)
            print(
                f"phase6b_norm_training=round_{round_id + 1} "
                f"client_{client_position}/{len(selected_client_ids)} id={client_id}",
                flush=True,
            )
            update = train_local_client(
                detector=self.detector,
                manifest=manifest,
                global_parameters=global_parameters,
                round_id=round_id,
                client_id=client_id,
                declared_sample_count=self.cache.client_total(client_index),
                class_counts=self.cache.class_counts(client_index),
                batch_source=_CachedBatchSource(self.cache, client_index, self.batch_size),
                config=self.local_config,
                is_malicious=False,
            )
            print(
                f"phase6b_norm_update=round_{round_id + 1} id={client_id} "
                f"norm={update.update_norm:.12g}",
                flush=True,
            )
            updates.append(update)
        return updates


def _schedule(num_clients: int, config: Phase6BNormCalibrationConfig) -> ClientSchedule:
    if config.clients_per_round > num_clients:
        raise DataContractError("Phase 6B.2 requires more clients than Phase 3D provides")
    client_ids = tuple(f"client-{index:02d}" for index in range(num_clients))
    return generate_client_schedule(
        client_ids,
        rounds=config.total_rounds,
        clients_per_round=config.clients_per_round,
        seed=config.base_seed,
    )


def _load_round_norms(
    rounds_path: Path,
    schedule: ClientSchedule,
    federated_run_id: str,
) -> tuple[list[dict[str, object]], tuple[str, ...], tuple[float, ...]]:
    records: list[dict[str, object]] = []
    development_ids: list[str] = []
    norms: list[float] = []
    try:
        lines = rounds_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataContractError(f"Cannot read benign federated round log: {error}") from error
    if len(lines) != len(schedule.rounds):
        raise DataContractError("Benign federated round log is incomplete")
    for expected_round, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataContractError("Benign federated round log is malformed") from error
        if not isinstance(raw, dict):
            raise DataContractError("Benign federated round record is malformed")
        round_record = cast(dict[str, object], raw)
        selected = round_record.get("selected_client_ids")
        accepted = round_record.get("accepted_client_ids")
        rejected = round_record.get("rejected_updates")
        updates = round_record.get("client_updates")
        expected_clients = list(schedule.rounds[expected_round])
        if (
            round_record.get("round_id") != expected_round
            or selected != expected_clients
            or accepted != expected_clients
            or rejected != {}
            or round_record.get("round_failed") is not False
            or not isinstance(updates, list)
            or len(updates) != len(expected_clients)
        ):
            raise DataContractError("Benign federated round failed calibration gates")
        for expected_client, item in zip(expected_clients, updates, strict=True):
            if not isinstance(item, dict):
                raise DataContractError("Benign client-update record is malformed")
            update = cast(dict[str, object], item)
            norm = update.get("update_norm")
            if (
                update.get("client_id") != expected_client
                or not isinstance(norm, int | float)
                or not math.isfinite(float(norm))
                or float(norm) < 0.0
            ):
                raise DataContractError("Benign client-update norm is invalid")
            development_id = run_id(
                {
                    "phase": "6B.2-benign-update",
                    "federated_run_id": federated_run_id,
                    "round_id": expected_round,
                    "client_id": expected_client,
                    "update_norm": float(norm),
                },
                length=64,
            )
            records.append(
                {
                    "development_run_id": development_id,
                    "round_id": expected_round,
                    "client_id": expected_client,
                    "update_norm": float(norm),
                }
            )
            development_ids.append(development_id)
            norms.append(float(norm))
    if len(set(development_ids)) != len(development_ids):
        raise DataContractError("Benign development-update identities are not unique")
    return records, tuple(development_ids), tuple(norms)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _calibration_input(
    request: Phase6BNormCalibrationRequest,
    *,
    dataset: str,
    preprocessed_corpus_id: str,
    client_partition_id: str,
    schedule_id: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "6B.2",
        "dataset": dataset,
        "profile": request.profile,
        "preprocessed_corpus_id": preprocessed_corpus_id,
        "client_partition_id": client_partition_id,
        "schedule_id": schedule_id,
        "contract_id": request.config.contract_id,
        "device": request.device,
    }


def run_phase6b_norm_calibration(
    request: Phase6BNormCalibrationRequest,
) -> dict[str, object]:
    """Build the client cache, run benign FL, and freeze the prior norm IQR."""

    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(
        request.client_partition_manifest, corpus, request.profile
    )
    schedule = _schedule(profile.num_clients, request.config)
    calibration_input = _calibration_input(
        request,
        dataset=corpus.dataset,
        preprocessed_corpus_id=corpus.preprocessed_corpus_id,
        client_partition_id=profile.client_partition_id,
        schedule_id=schedule.schedule_id,
    )
    calibration_input_id = run_id(calibration_input, length=64)
    final_path = request.output_directory / "benign_norm_reference.json"
    if final_path.exists():
        existing = cast(dict[str, object], json.loads(final_path.read_text(encoding="utf-8")))
        if existing.get("calibration_input_id") != calibration_input_id:
            raise DataContractError("Completed norm reference belongs to other inputs")
        stable_existing = {
            key: value
            for key, value in existing.items()
            if key not in {"calibration_id", "created_at_utc"}
        }
        if existing.get("calibration_id") != run_id(stable_existing, length=64):
            raise DataContractError("Completed norm-reference identity is invalid")
        return existing

    selected_indices = tuple(
        sorted(
            {
                int(client_id.removeprefix("client-"))
                for selected_round in schedule.rounds
                for client_id in selected_round
            }
        )
    )
    cache_manifest = build_selected_client_cache(
        ClientCacheBuildRequest(
            preprocessing_report=request.preprocessing_report,
            client_partition_manifest=request.client_partition_manifest,
            profile=request.profile,
            selected_client_indices=selected_indices,
            schedule_id=schedule.schedule_id,
            output_directory=request.cache_directory,
            work_database=request.cache_work_database,
            batch_size=request.cache_scan_batch_size,
        )
    )
    cache = load_selected_client_cache(request.cache_directory / "client_cache.json")
    if (
        cache.dataset != corpus.dataset
        or cache.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or cache.client_partition_id != profile.client_partition_id
        or cache.profile != request.profile
        or cache.schedule_id != schedule.schedule_id
    ):
        raise DataContractError("Selected-client cache differs from calibration inputs")

    detector = DetectorConfig(
        corpus.input_dimension,
        corpus.num_classes,
        request.config.hidden_dimensions,
        request.config.dropout,
    )
    torch.manual_seed(request.config.base_seed)
    initial_model = IntrusionMLP(detector)
    provider = _CachedBenignProvider(cache, detector, request.config, request.device)
    federated_output = request.output_directory / "federation" / calibration_input_id
    federated_report = run_federated_simulation(
        initial_model=initial_model,
        schedule=schedule,
        update_provider=provider,
        config=FederatedRunConfig(
            output_directory=federated_output,
            total_rounds=request.config.total_rounds,
            aggregation=request.config.aggregation,
            seed=request.config.base_seed,
        ),
        max_rounds_this_call=request.max_rounds_this_call,
    )
    if federated_report.get("status") != "complete":
        return {
            "calibration_input_id": calibration_input_id,
            "federated_run_id": federated_report["run_id"],
            "status": "interrupted",
            "next_round": federated_report["next_round"],
            "cache_id": cache_manifest["cache_id"],
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
    federated_run_id = cast(str, federated_report["run_id"])
    run_directory = federated_output.resolve() / federated_run_id
    update_records, development_ids, norms = _load_round_norms(
        run_directory / "rounds.jsonl", schedule, federated_run_id
    )
    local_training = provider.local_config.to_dict()
    detector_config_id = run_id(detector.to_dict(), length=64)
    local_training_config_id = run_id(local_training, length=64)
    reference = benign_norm_reference(
        dataset=corpus.dataset,
        profile=request.profile,
        preprocessed_corpus_id=corpus.preprocessed_corpus_id,
        client_partition_id=profile.client_partition_id,
        detector_config_id=detector_config_id,
        local_training_config_id=local_training_config_id,
        development_run_ids=development_ids,
        update_norms=norms,
    )
    expected_updates = request.config.total_rounds * request.config.clients_per_round
    stable: dict[str, object] = {
        **calibration_input,
        "calibration_input_id": calibration_input_id,
        "cache_id": cache.cache_id,
        "federated_run_id": federated_run_id,
        "detector_config_id": detector_config_id,
        "detector": detector.to_dict(),
        "local_training_config_id": local_training_config_id,
        "local_training": local_training,
        "reference": {
            "reference_id": reference.reference_id,
            **cast(dict[str, object], to_primitive(reference)),
        },
        "update_norm_records": update_records,
        "gates": {
            "all_scheduled_updates_accepted": len(norms) == expected_updates,
            "all_development_clients_honest": True,
            "cache_bound_to_schedule": cache.schedule_id == schedule.schedule_id,
            "same_dataset_model_partition_and_training_contract": True,
            "reference_uses_prior_benign_development_updates_only": True,
            "current_round_benign_updates_accessed": False,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "anchor_records_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    if len(norms) != expected_updates:
        raise DataContractError("Norm calibration did not conserve scheduled updates")
    result: dict[str, object] = {
        "calibration_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(UTC).isoformat(),
        **stable,
    }
    _write_json(final_path, result)
    return result
