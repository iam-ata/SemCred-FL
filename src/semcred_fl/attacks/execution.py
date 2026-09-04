"""Resumable train-only Phase 6C attack execution on cached real clients."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from semcred_fl.attacks.artifacts import (
    BenignNormReference,
    Phase6BResolution,
    load_benign_norm_reference,
    load_phase6b_resolution,
)
from semcred_fl.attacks.execution_contracts import (
    AttackRealization,
    AttackRunSpec,
    Phase6CExecutionConfig,
    build_attack_realization,
    phase6c_execution_config,
)
from semcred_fl.attacks.transformations import (
    backdoor_poison_selected,
    clean_label_poison_selected,
    coalition_model_replacement,
    constrain_update_norm,
    reverse_update,
    scale_update,
    targeted_label_flip_selected,
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
    TrainingBatch,
    load_client_partition_profile,
    load_preprocessed_corpus,
)
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import ParameterManifest
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation
from semcred_fl.federation.updates import ClientUpdate
from semcred_fl.identity import run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


@dataclass(frozen=True, slots=True)
class Phase6CAttackRequest:
    """Artifact paths and runtime controls for one Phase 6C condition."""

    preprocessing_report: Path
    client_partition_manifest: Path
    resolution_path: Path
    output_directory: Path
    cache_directory: Path
    cache_work_database: Path
    spec: AttackRunSpec
    norm_reference_path: Path | None = None
    device: str = "auto"
    cache_scan_batch_size: int = 100_000
    max_rounds_this_call: int | None = None
    config: Phase6CExecutionConfig = field(default_factory=phase6c_execution_config)

    def __post_init__(self) -> None:
        self.spec.validate(self.config)
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Phase 6C device must be auto, cpu, or cuda")
        if self.cache_scan_batch_size < 1:
            raise ValueError("Phase 6C cache scan batch size must be positive")
        if self.max_rounds_this_call is not None and self.max_rounds_this_call < 1:
            raise ValueError("Phase 6C maximum rounds per call must be positive")
        if (self.spec.attack_name == "adaptive_norm") != (self.norm_reference_path is not None):
            raise ValueError("Only adaptive norm execution requires a norm reference")


@dataclass(frozen=True, slots=True)
class ClientPoisonSelection:
    """One client's exact population-level poison selection."""

    client_id: str
    eligible_count: int
    selected_record_ids: tuple[str, ...]
    selection_id: str


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _client_index(client_id: str) -> int:
    suffix = client_id.removeprefix("client-")
    if not client_id.startswith("client-") or not suffix.isdigit():
        raise DataContractError("Phase 6C client ID is malformed")
    return int(suffix)


def _full_profile_cache_scope(*, corpus_id: str, partition_id: str, profile: str) -> str:
    return run_id(
        {
            "schema_version": 1,
            "phase": "6C-full-profile-client-cache",
            "preprocessed_corpus_id": corpus_id,
            "client_partition_id": partition_id,
            "profile": profile,
        },
        length=64,
    )


def _rank_record_ids(
    record_ids: Sequence[str], *, seed: int, attack_id: str, client_id: str
) -> list[str]:
    def rank(record_id: str) -> tuple[bytes, str]:
        payload = f"{seed}\0{attack_id}\0{client_id}\0{record_id}".encode()
        return hashlib.sha256(payload).digest(), record_id

    return sorted(record_ids, key=rank)


def _client_selection(
    *,
    cache: SelectedClientCache,
    client_id: str,
    resolution: Phase6BResolution,
    spec: AttackRunSpec,
    attack_id: str,
    output_path: Path,
) -> ClientPoisonSelection:
    client_index = _client_index(client_id)
    eligible_count = cache.class_counts(client_index)[resolution.source_family_id]
    selected_count = math.floor(spec.poison_fraction * eligible_count)
    stable_scope = {
        "schema_version": 1,
        "phase": "6C-client-poison-selection",
        "cache_id": cache.cache_id,
        "attack_id": attack_id,
        "client_id": client_id,
        "source_family_id": resolution.source_family_id,
        "poison_fraction": spec.poison_fraction,
        "eligible_count": eligible_count,
        "selected_count": selected_count,
        "selection_rule": "lowest_sha256_rank_without_replacement_over_client_population",
    }
    if output_path.exists():
        completed = _load_json(output_path, "Phase 6C client poison selection")
        selected_value = completed.get("selected_record_ids")
        if not isinstance(selected_value, list) or not all(
            isinstance(value, str) for value in selected_value
        ):
            raise DataContractError("Phase 6C selected record IDs are malformed")
        stable = {**stable_scope, "selected_record_ids": selected_value}
        gates = completed.get("gates")
        if (
            len(selected_value) != selected_count
            or len(set(selected_value)) != len(selected_value)
            or completed.get("selection_id") != run_id(stable, length=64)
            or not isinstance(gates, dict)
            or gates.get("selection_population_is_complete_client_source_family") is not True
            or gates.get("selected_without_replacement") is not True
            or gates.get("selected_count_matches_floor_rule") is not True
            or gates.get("validation_test_and_anchors_excluded_by_cache") is not True
            or gates.get("test_partition_accessed") is not False
            or gates.get("performance_claims_allowed") is not False
        ):
            raise DataContractError("Phase 6C client selection identity is invalid")
        return ClientPoisonSelection(
            client_id,
            eligible_count,
            tuple(cast(list[str], selected_value)),
            cast(str, completed["selection_id"]),
        )

    selected: tuple[str, ...] = ()
    if spec.requires_data_poisoning and selected_count:
        eligible_ids: list[str] = []
        for batch in iter_cached_client_batches(
            cache,
            client_index,
            batch_size=100_000,
            shuffle_seed=spec.seed,
        ):
            eligible_ids.extend(
                record_id
                for record_id, target in zip(batch.record_ids, batch.targets.tolist(), strict=True)
                if int(target) == resolution.source_family_id
            )
        if len(eligible_ids) != eligible_count or len(set(eligible_ids)) != eligible_count:
            raise DataContractError("Phase 6C eligible source population differs from cache")
        selected = tuple(
            sorted(
                _rank_record_ids(
                    eligible_ids,
                    seed=spec.seed,
                    attack_id=attack_id,
                    client_id=client_id,
                )[:selected_count]
            )
        )
    stable = {**stable_scope, "selected_record_ids": list(selected)}
    selection_id = run_id(stable, length=64)
    _write_json(
        output_path,
        {
            "selection_id": selection_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            **stable,
            "gates": {
                "selection_population_is_complete_client_source_family": True,
                "selected_without_replacement": True,
                "selected_count_matches_floor_rule": True,
                "validation_test_and_anchors_excluded_by_cache": True,
                "test_partition_accessed": False,
                "performance_claims_allowed": False,
            },
        },
    )
    return ClientPoisonSelection(client_id, eligible_count, selected, selection_id)


def _selection_manifest(
    *,
    request: Phase6CAttackRequest,
    cache: SelectedClientCache,
    resolution: Phase6BResolution,
    realization: AttackRealization,
    attack_id: str,
) -> tuple[str, dict[str, ClientPoisonSelection]]:
    active_clients = (
        []
        if request.spec.attack_name == "clean"
        else sorted(
            set(realization.malicious_client_ids)
            & {client_id for selected in realization.schedule.rounds for client_id in selected}
        )
    )
    selection_directory = request.output_directory / "poison_selections"
    selections: dict[str, ClientPoisonSelection] = {}
    inventory: list[dict[str, object]] = []
    for position, client_id in enumerate(active_clients, start=1):
        path = selection_directory / f"{client_id}.json"
        selection = _client_selection(
            cache=cache,
            client_id=client_id,
            resolution=resolution,
            spec=request.spec,
            attack_id=attack_id,
            output_path=path,
        )
        selections[client_id] = selection
        inventory.append(
            {
                "client_id": client_id,
                "path": path.relative_to(request.output_directory).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
                "selection_id": selection.selection_id,
                "eligible_count": selection.eligible_count,
                "selected_count": len(selection.selected_record_ids),
            }
        )
        print(f"phase6c_selection={position}/{len(active_clients)} id={client_id}", flush=True)
    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "6C-poison-selection-manifest",
        "attack_id": attack_id,
        "realization_id": realization.realization_id,
        "cache_id": cache.cache_id,
        "resolution_id": resolution.resolution_id,
        "selected_malicious_client_ids": active_clients,
        "client_selections": inventory,
        "gates": {
            "selection_fixed_before_local_training": True,
            "client_specific_record_selection": True,
            "selected_counts_use_complete_client_populations": True,
            "test_partition_accessed": False,
            "anchor_records_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    manifest_id = run_id(stable, length=64)
    path = request.output_directory / "poison_selection_manifest.json"
    if path.exists():
        completed = _load_json(path, "Phase 6C poison selection manifest")
        completed_stable = {
            key: value
            for key, value in completed.items()
            if key not in {"selection_manifest_id", "created_at_utc"}
        }
        if completed.get("selection_manifest_id") != manifest_id or completed_stable != stable:
            raise DataContractError("Completed Phase 6C selection manifest differs")
    else:
        _write_json(
            path,
            {
                "selection_manifest_id": manifest_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                **stable,
            },
        )
    return manifest_id, selections


@dataclass(frozen=True, slots=True)
class _AttackedBatchSource:
    cache: SelectedClientCache
    client_index: int
    client_id: str
    batch_size: int
    spec: AttackRunSpec
    resolution: Phase6BResolution
    selected_record_ids: tuple[str, ...]
    attack_id: str
    malicious: bool

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        seen = 0
        for batch in iter_cached_client_batches(
            self.cache,
            self.client_index,
            batch_size=self.batch_size,
            shuffle_seed=seed,
        ):
            if not self.malicious or not self.spec.requires_data_poisoning:
                yield batch.training_batch()
                continue
            if self.spec.attack_name in {"label_flip", "adaptive_norm"}:
                poisoned = targeted_label_flip_selected(
                    features=batch.features,
                    targets=batch.targets,
                    record_ids=batch.record_ids,
                    selected_record_ids=self.selected_record_ids,
                    source_family_id=self.resolution.source_family_id,
                    target_family_id=self.resolution.target_family_id,
                    poison_fraction=self.spec.poison_fraction,
                    attack_id=self.attack_id,
                )
            elif self.spec.attack_name == "clean_label":
                if self.spec.strength is None:
                    raise RuntimeError("Clean-label execution lost its strength")
                poisoned = clean_label_poison_selected(
                    features=batch.features,
                    targets=batch.targets,
                    record_ids=batch.record_ids,
                    selected_record_ids=self.selected_record_ids,
                    resolution=self.resolution,
                    poison_fraction=self.spec.poison_fraction,
                    strength=self.spec.strength,
                    attack_id=self.attack_id,
                )
            elif self.spec.attack_name in {"backdoor", "model_replacement"}:
                poisoned = backdoor_poison_selected(
                    features=batch.features,
                    targets=batch.targets,
                    record_ids=batch.record_ids,
                    selected_record_ids=self.selected_record_ids,
                    resolution=self.resolution,
                    poison_fraction=self.spec.poison_fraction,
                    attack_id=self.attack_id,
                )
            else:
                raise RuntimeError("Unknown Phase 6C data-poisoning path")
            seen += len(poisoned.selected_indices)
            yield TrainingBatch(poisoned.features, poisoned.targets)
        if self.malicious and self.spec.requires_data_poisoning:
            if seen != len(self.selected_record_ids):
                raise DataContractError("Phase 6C streamed poison selection is incomplete")


def _effective_class_counts(
    cache: SelectedClientCache,
    client_index: int,
    spec: AttackRunSpec,
    resolution: Phase6BResolution,
    selected_count: int,
    malicious: bool,
) -> tuple[int, ...]:
    counts = list(cache.class_counts(client_index))
    if malicious and spec.attack_name in {
        "label_flip",
        "backdoor",
        "model_replacement",
        "adaptive_norm",
    }:
        counts[resolution.source_family_id] -= selected_count
        counts[resolution.target_family_id] += selected_count
    if any(value < 0 for value in counts):
        raise DataContractError("Phase 6C poisoned class counts are invalid")
    return tuple(counts)


class _AttackUpdateProvider:
    def __init__(
        self,
        *,
        request: Phase6CAttackRequest,
        cache: SelectedClientCache,
        detector: DetectorConfig,
        resolution: Phase6BResolution,
        norm_reference: BenignNormReference | None,
        realization: AttackRealization,
        attack_id: str,
        selection_manifest_id: str,
        selections: dict[str, ClientPoisonSelection],
        audit_directory: Path,
    ) -> None:
        self.request = request
        self.cache = cache
        self.detector = detector
        self.resolution = resolution
        self.norm_reference = norm_reference
        self.realization = realization
        self.attack_id = attack_id
        self.selection_manifest_id = selection_manifest_id
        self.selections = selections
        self.audit_directory = audit_directory
        self.malicious = set(realization.malicious_client_ids)
        config = request.config
        self.local_config = LocalTrainingConfig(
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            proximal_mu=config.proximal_mu,
            maximum_class_weight=config.maximum_class_weight,
            seed=request.spec.seed,
            device=request.device,
        )

    def _transform_delta(self, update: ClientUpdate, malicious: bool) -> Tensor:
        if not malicious:
            return update.delta
        name = self.request.spec.attack_name
        if name == "scaling":
            if self.request.spec.scale is None:
                raise RuntimeError("Scaling execution lost its factor")
            return scale_update(update.delta, self.request.spec.scale)
        if name == "sign_reversal":
            if self.request.spec.scale is None:
                raise RuntimeError("Sign-reversal execution lost its factor")
            return reverse_update(update.delta, self.request.spec.scale)
        if name == "adaptive_norm":
            if self.norm_reference is None:
                raise RuntimeError("Adaptive execution lost its norm reference")
            return constrain_update_norm(
                update.delta,
                self.norm_reference.lower_norm,
                self.norm_reference.upper_norm,
            )
        return update.delta

    def _round_audit(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        updates: Sequence[ClientUpdate],
    ) -> None:
        records: list[dict[str, object]] = []
        for update in updates:
            metadata = update.attack_metadata or {}
            records.append(
                {
                    "client_id": update.client_id,
                    "is_malicious": update.is_malicious,
                    "declared_sample_count": update.declared_sample_count,
                    "selected_poison_records": metadata.get("selected_poison_records", 0),
                    "selection_id": metadata.get("selection_id"),
                    "pre_manipulation_norm": metadata.get("pre_manipulation_norm"),
                    "submitted_norm": update.update_norm,
                    "submitted_delta_sha256": _tensor_sha256(update.delta),
                    "transformation": metadata.get("transformation"),
                }
            )
        designated_coalition = sorted(set(selected_client_ids) & self.malicious)
        active_malicious = [] if self.request.spec.attack_name == "clean" else designated_coalition
        stable: dict[str, object] = {
            "schema_version": 1,
            "phase": "6C-attack-round-audit",
            "round_id": round_id,
            "attack_id": self.attack_id,
            "realization_id": self.realization.realization_id,
            "selection_manifest_id": self.selection_manifest_id,
            "selected_client_ids": list(selected_client_ids),
            "designated_coalition_client_ids": designated_coalition,
            "active_malicious_client_ids": active_malicious,
            "client_updates": records,
            "gates": {
                "designated_coalition_count_matches_stratified_schedule": (
                    len(designated_coalition) == self.realization.active_malicious_per_round
                ),
                "active_malicious_count_matches_attack_condition": (
                    len(active_malicious)
                    == (
                        0
                        if self.request.spec.attack_name == "clean"
                        else self.realization.active_malicious_per_round
                    )
                ),
                "malicious_minority_preserved": 2 * len(active_malicious)
                < len(selected_client_ids),
                "declared_sample_counts_unchanged": True,
                "attack_metadata_excluded_from_defense_updates": True,
                "test_partition_accessed": False,
                "anchor_records_accessed": False,
                "performance_claims_allowed": False,
            },
        }
        if not all(
            value is True
            for key, value in cast(dict[str, bool], stable["gates"]).items()
            if key
            not in {
                "test_partition_accessed",
                "anchor_records_accessed",
                "performance_claims_allowed",
            }
        ):
            raise DataContractError("Phase 6C attack-round gates failed")
        record_id = run_id(stable, length=64)
        path = self.audit_directory / f"round-{round_id:04d}.json"
        if path.exists():
            completed = _load_json(path, "Phase 6C attack-round audit")
            completed_stable = {
                key: value
                for key, value in completed.items()
                if key not in {"attack_round_id", "created_at_utc"}
            }
            if completed.get("attack_round_id") != record_id or completed_stable != stable:
                raise DataContractError("Resumed Phase 6C attack round differs")
        else:
            _write_json(
                path,
                {
                    "attack_round_id": record_id,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    **stable,
                },
            )

    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        trained: list[ClientUpdate] = []
        for position, client_id in enumerate(selected_client_ids, start=1):
            client_index = _client_index(client_id)
            malicious = self.request.spec.attack_name != "clean" and client_id in self.malicious
            selection = self.selections.get(client_id)
            selected_ids = selection.selected_record_ids if selection is not None else ()
            class_counts = _effective_class_counts(
                self.cache,
                client_index,
                self.request.spec,
                self.resolution,
                len(selected_ids),
                malicious,
            )
            print(
                f"phase6c_training=round_{round_id + 1} "
                f"client_{position}/{len(selected_client_ids)} id={client_id} "
                f"malicious={str(malicious).lower()}",
                flush=True,
            )
            update = train_local_client(
                detector=self.detector,
                manifest=manifest,
                global_parameters=global_parameters,
                round_id=round_id,
                client_id=client_id,
                declared_sample_count=self.cache.client_total(client_index),
                class_counts=class_counts,
                batch_source=_AttackedBatchSource(
                    cache=self.cache,
                    client_index=client_index,
                    client_id=client_id,
                    batch_size=self.request.config.batch_size,
                    spec=self.request.spec,
                    resolution=self.resolution,
                    selected_record_ids=selected_ids,
                    attack_id=self.attack_id,
                    malicious=malicious,
                ),
                config=self.local_config,
                is_malicious=malicious,
            )
            transformed = self._transform_delta(update, malicious)
            metadata: dict[str, object] = {
                "attack_id": self.attack_id,
                "transformation": self.request.spec.attack_name if malicious else "none",
                "selected_poison_records": len(selected_ids),
                "selection_id": selection.selection_id if selection is not None else None,
                "pre_manipulation_norm": update.update_norm,
            }
            trained.append(
                replace(
                    update,
                    delta=transformed,
                    update_norm=float(torch.linalg.vector_norm(transformed).item()),
                    is_malicious=malicious,
                    attack_metadata=metadata,
                )
            )

        if self.request.spec.attack_name == "model_replacement":
            malicious_updates = [update for update in trained if update.is_malicious]
            coalition_size = len(malicious_updates)
            if coalition_size != self.realization.active_malicious_per_round:
                raise DataContractError("Model-replacement coalition size differs from schedule")
            total_weight = sum(update.declared_sample_count for update in malicious_updates)
            target_parameters = torch.zeros_like(global_parameters)
            for update in malicious_updates:
                weight = update.declared_sample_count / total_weight
                target_parameters.add_((global_parameters + update.delta) * weight)
            replacement_delta = coalition_model_replacement(
                target_parameters=target_parameters,
                global_parameters=global_parameters,
                participants_per_round=self.request.config.clients_per_round,
                coalition_size=coalition_size,
            )
            replaced: list[ClientUpdate] = []
            for update in trained:
                if not update.is_malicious:
                    replaced.append(update)
                    continue
                metadata = dict(update.attack_metadata or {})
                metadata["transformation"] = "model_replacement"
                metadata["coalition_size"] = coalition_size
                metadata["target_parameters_sha256"] = _tensor_sha256(target_parameters)
                replaced.append(
                    replace(
                        update,
                        delta=replacement_delta.clone(),
                        update_norm=float(torch.linalg.vector_norm(replacement_delta).item()),
                        attack_metadata=metadata,
                    )
                )
            trained = replaced
        self._round_audit(round_id, selected_client_ids, trained)
        return trained


def _finalize_attack_run(
    *,
    request: Phase6CAttackRequest,
    experiment_input: dict[str, object],
    experiment_input_id: str,
    federated_report: dict[str, object],
    audit_directory: Path,
) -> dict[str, object]:
    audit_files: list[dict[str, object]] = []
    for round_id in range(request.config.total_rounds):
        path = audit_directory / f"round-{round_id:04d}.json"
        if not path.is_file():
            raise DataContractError("Phase 6C attack audit is missing a completed round")
        audit_files.append(
            {
                "round_id": round_id,
                "path": path.relative_to(request.output_directory).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    stable: dict[str, object] = {
        **experiment_input,
        "experiment_input_id": experiment_input_id,
        "federated_run_id": federated_report["run_id"],
        "final_parameters_sha256": federated_report["final_parameters_sha256"],
        "attack_round_audits": audit_files,
        "gates": {
            "all_attack_rounds_audited": True,
            "all_rounds_checkpointed": True,
            "fixed_coalition_and_stratified_schedule": True,
            "malicious_minority_preserved_every_round": True,
            "complete_client_population_selection_used": True,
            "declared_sample_counts_unchanged": True,
            "attack_ground_truth_excluded_from_defense_updates": True,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "anchor_records_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    result: dict[str, object] = {
        "attack_run_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(UTC).isoformat(),
        **stable,
    }
    _write_json(request.output_directory / "phase6c_attack_run.json", result)
    return result


def run_phase6c_attack(request: Phase6CAttackRequest) -> dict[str, object]:
    """Build or resume one train-only Phase 6C attack-execution run."""

    config = request.config
    spec = request.spec
    corpus = load_preprocessed_corpus(request.preprocessing_report)
    profile = load_client_partition_profile(request.client_partition_manifest, corpus, spec.profile)
    if corpus.dataset != spec.dataset or profile.num_clients != config.total_clients:
        raise DataContractError("Phase 6C data artifacts differ from the attack condition")
    resolution = load_phase6b_resolution(request.resolution_path)
    if (
        resolution.dataset != corpus.dataset
        or resolution.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or resolution.client_partition_id != profile.client_partition_id
    ):
        raise DataContractError("Phase 6C resolution differs from the selected data")
    norm_reference: BenignNormReference | None = None
    if request.norm_reference_path is not None:
        norm_reference = load_benign_norm_reference(request.norm_reference_path)
        if (
            norm_reference.dataset != corpus.dataset
            or norm_reference.profile != profile.profile
            or norm_reference.preprocessed_corpus_id != corpus.preprocessed_corpus_id
            or norm_reference.client_partition_id != profile.client_partition_id
        ):
            raise DataContractError("Adaptive norm reference differs from this condition")

    realization = build_attack_realization(config, spec)
    cache_scope_id = _full_profile_cache_scope(
        corpus_id=corpus.preprocessed_corpus_id,
        partition_id=profile.client_partition_id,
        profile=profile.profile,
    )
    build_selected_client_cache(
        ClientCacheBuildRequest(
            preprocessing_report=request.preprocessing_report,
            client_partition_manifest=request.client_partition_manifest,
            profile=profile.profile,
            selected_client_indices=tuple(range(config.total_clients)),
            schedule_id=cache_scope_id,
            output_directory=request.cache_directory,
            work_database=request.cache_work_database,
            batch_size=request.cache_scan_batch_size,
        )
    )
    cache = load_selected_client_cache(request.cache_directory / "client_cache.json")
    if (
        cache.dataset != corpus.dataset
        or cache.profile != profile.profile
        or cache.preprocessed_corpus_id != corpus.preprocessed_corpus_id
        or cache.client_partition_id != profile.client_partition_id
        or cache.schedule_id != cache_scope_id
        or cache.selected_client_indices != tuple(range(config.total_clients))
    ):
        raise DataContractError("Phase 6C full-profile cache differs from the condition")
    attack_id = spec.attack_id(config)
    selection_manifest_id, selections = _selection_manifest(
        request=request,
        cache=cache,
        resolution=resolution,
        realization=realization,
        attack_id=attack_id,
    )
    detector = DetectorConfig(
        corpus.input_dimension,
        corpus.num_classes,
        config.hidden_dimensions,
        config.dropout,
    )
    norm_reference_id = norm_reference.reference_id if norm_reference is not None else None
    experiment_input: dict[str, object] = {
        **spec.stable_record(config),
        "attack_id": attack_id,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "client_partition_id": profile.client_partition_id,
        "resolution_id": resolution.resolution_id,
        "norm_reference_id": norm_reference_id,
        "cache_id": cache.cache_id,
        "realization_id": realization.realization_id,
        "schedule_id": realization.schedule.schedule_id,
        "selection_manifest_id": selection_manifest_id,
        "detector": detector.to_dict(),
        "device": request.device,
    }
    if norm_reference is not None and norm_reference.detector_config_id != run_id(
        detector.to_dict(), length=64
    ):
        raise DataContractError("Adaptive norm reference uses a different detector")
    experiment_input_id = run_id(experiment_input, length=64)
    final_path = request.output_directory / "phase6c_attack_run.json"
    if final_path.exists():
        completed = _load_json(final_path, "completed Phase 6C attack run")
        stable_completed = {
            key: value
            for key, value in completed.items()
            if key not in {"attack_run_id", "created_at_utc"}
        }
        if completed.get("experiment_input_id") != experiment_input_id or completed.get(
            "attack_run_id"
        ) != run_id(stable_completed, length=64):
            raise DataContractError("Completed Phase 6C run belongs to other inputs")
        return completed
    torch.manual_seed(spec.seed)
    initial_model = IntrusionMLP(detector)
    provider = _AttackUpdateProvider(
        request=request,
        cache=cache,
        detector=detector,
        resolution=resolution,
        norm_reference=norm_reference,
        realization=realization,
        attack_id=attack_id,
        selection_manifest_id=selection_manifest_id,
        selections=selections,
        audit_directory=request.output_directory / "attack_rounds",
    )
    federation_root = request.output_directory / "federation"
    federated_report = run_federated_simulation(
        initial_model=initial_model,
        schedule=realization.schedule,
        update_provider=provider,
        config=FederatedRunConfig(
            output_directory=federation_root,
            total_rounds=config.total_rounds,
            aggregation=spec.aggregation,
            seed=spec.seed,
            scientific_context_id=experiment_input_id,
        ),
        max_rounds_this_call=request.max_rounds_this_call,
    )
    if federated_report.get("status") != "complete":
        return {
            "status": "interrupted",
            "experiment_input_id": experiment_input_id,
            "attack_id": attack_id,
            "realization_id": realization.realization_id,
            "selection_manifest_id": selection_manifest_id,
            "federated_run_id": federated_report["run_id"],
            "next_round": federated_report["next_round"],
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        }
    return _finalize_attack_run(
        request=request,
        experiment_input=experiment_input,
        experiment_input_id=experiment_input_id,
        federated_report=federated_report,
        audit_directory=request.output_directory / "attack_rounds",
    )
