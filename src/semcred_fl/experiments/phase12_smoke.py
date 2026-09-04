"""Tiny synthetic Phase 12 execution and artifact-generation gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from semcred_fl.attacks.transformations import reverse_update
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.numerical_score import evaluate_numerical_updates
from semcred_fl.credibility.semantic_score import evaluate_semantic_updates
from semcred_fl.data.training_corpus import TrainingBatch
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
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import (
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.federation.server import process_semcred_round
from semcred_fl.federation.state import ClientState
from semcred_fl.federation.updates import ClientUpdate, validate_client_updates
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.modeling.metrics import ClassificationMetrics, classification_metrics
from semcred_fl.seeding import derive_seed
from semcred_fl.semantic.pipeline import SemanticPriorBundle
from semcred_fl.semantic.prior import FeatureGroupLayout


@dataclass(frozen=True, slots=True)
class Phase12SmokeConfig:
    """Frozen controls for the non-claim synthetic smoke experiment."""

    output_directory: Path
    seed: int = 20260816
    rounds: int = 3
    clients: int = 4
    malicious_client_id: str = "client-03"
    malicious_round: int = 1
    sign_reversal_scale: float = 5.0
    attribution_interval: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.rounds < 2 or self.clients != 4:
            raise ValueError("smoke seed, rounds, or fixed client population is invalid")
        expected_clients = {f"client-{index:02d}" for index in range(self.clients)}
        if self.malicious_client_id not in expected_clients:
            raise ValueError("malicious smoke client is outside the fixed population")
        if not 0 <= self.malicious_round < self.rounds:
            raise ValueError("malicious smoke round is outside the run")
        if not math.isfinite(self.sign_reversal_scale) or self.sign_reversal_scale <= 0.0:
            raise ValueError("sign-reversal scale must be finite and positive")
        if self.attribution_interval < 1 or self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("attribution interval or smoke device is invalid")

    def stable_record(self) -> dict[str, object]:
        """Return scientific controls without local output paths."""

        return {
            "schema": "semcred_phase12_synthetic_smoke_v1",
            "seed": self.seed,
            "rounds": self.rounds,
            "clients": self.clients,
            "malicious_client_id": self.malicious_client_id,
            "malicious_round": self.malicious_round,
            "attack": "sign_reversal",
            "sign_reversal_scale": self.sign_reversal_scale,
            "attribution_interval": self.attribution_interval,
            "device": self.device,
            "synthetic_only": True,
            "confirmatory": False,
            "performance_claims_allowed": False,
        }


class _Batches:
    def __init__(self, features: Tensor, targets: Tensor, batch_size: int) -> None:
        self._features = features
        self._targets = targets
        self._batch_size = batch_size

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self._targets), generator=generator)
        for start in range(0, len(order), self._batch_size):
            selected = order[start : start + self._batch_size]
            yield TrainingBatch(self._features[selected], self._targets[selected])


def _class_data(
    *,
    negative_count: int,
    positive_count: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    negative = torch.randn(negative_count, 2, generator=generator) * 0.18
    positive = torch.randn(positive_count, 2, generator=generator) * 0.18
    negative += torch.tensor((-1.0, -0.6))
    positive += torch.tensor((1.0, 0.6))
    features = torch.cat((negative, positive))
    targets = torch.cat(
        (
            torch.zeros(negative_count, dtype=torch.long),
            torch.ones(positive_count, dtype=torch.long),
        )
    )
    return features, targets


def _data(
    config: Phase12SmokeConfig,
) -> tuple[dict[str, tuple[Tensor, Tensor]], TrainingBatch, TrainingBatch, str]:
    class_splits = ((24, 8), (20, 12), (12, 20), (8, 24))
    clients = {
        f"client-{index:02d}": _class_data(
            negative_count=negative,
            positive_count=positive,
            seed=derive_seed(config.seed, f"phase12-smoke-client:{index}"),
        )
        for index, (negative, positive) in enumerate(class_splits)
    }
    anchor_features, anchor_targets = _class_data(
        negative_count=8,
        positive_count=8,
        seed=derive_seed(config.seed, "phase12-smoke-anchor"),
    )
    test_features, test_targets = _class_data(
        negative_count=32,
        positive_count=32,
        seed=derive_seed(config.seed, "phase12-smoke-test"),
    )
    tensor_hashes = {
        **{
            f"{client_id}:{name}": _tensor_sha256(value)
            for client_id, pair in clients.items()
            for name, value in zip(("features", "targets"), pair, strict=True)
        },
        "anchor:features": _tensor_sha256(anchor_features),
        "anchor:targets": _tensor_sha256(anchor_targets),
        "test:features": _tensor_sha256(test_features),
        "test:targets": _tensor_sha256(test_targets),
    }
    data_id = run_id(
        {
            "schema": "semcred_phase12_synthetic_data_v1",
            "seed": config.seed,
            "tensor_hashes": tensor_hashes,
        },
        length=64,
    )
    return (
        clients,
        TrainingBatch(anchor_features, anchor_targets),
        TrainingBatch(test_features, test_targets),
        data_id,
    )


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _semantic_contract(data_id: str) -> tuple[FeatureGroupLayout, SemanticPriorBundle]:
    layout = FeatureGroupLayout(
        dataset="synthetic_phase12_smoke",
        preprocessor_id=data_id,
        group_names=("direction", "magnitude"),
        coordinate_indices=((0,), (1,)),
        coordinate_names=("x0000", "x0001"),
    )
    prior = SemanticPriorBundle(
        bundle_id=run_id({"synthetic_prior": data_id}, length=64),
        dataset=layout.dataset,
        preprocessed_corpus_id=data_id,
        preprocessor_id=data_id,
        anchor_id=run_id({"synthetic_anchor": data_id}, length=64),
        feature_group_names=layout.group_names,
        class_names=("negative", "positive"),
        alpha_l=0.5,
        epsilon=1e-12,
        priors=((0.65, 0.35), (0.65, 0.35)),
    )
    return layout, prior


def _train_client(
    *,
    client_id: str,
    round_id: int,
    features: Tensor,
    targets: Tensor,
    detector: DetectorConfig,
    global_parameters: Tensor,
    config: Phase12SmokeConfig,
) -> ClientUpdate:
    manifest = parameter_manifest(IntrusionMLP(detector))
    counts = tuple(int(torch.sum(targets == class_id).item()) for class_id in range(2))
    return train_local_client(
        detector=detector,
        manifest=manifest,
        global_parameters=global_parameters,
        round_id=round_id,
        client_id=client_id,
        declared_sample_count=len(targets),
        class_counts=counts,
        batch_source=_Batches(features, targets, batch_size=8),
        config=LocalTrainingConfig(
            local_epochs=1,
            learning_rate=0.01,
            maximum_class_weight=10.0,
            seed=config.seed,
            device=config.device,
        ),
        is_malicious=False,
    )


def _apply_smoke_attack(
    update: ClientUpdate, config: Phase12SmokeConfig, round_id: int
) -> ClientUpdate:
    malicious = (
        update.client_id == config.malicious_client_id and round_id == config.malicious_round
    )
    if not malicious:
        return update
    delta = reverse_update(update.delta, config.sign_reversal_scale)
    return replace(
        update,
        delta=delta,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        is_malicious=True,
        attack_metadata={
            "attack": "sign_reversal",
            "scale": config.sign_reversal_scale,
        },
    )


def _evaluate_model(model: IntrusionMLP, batch: TrainingBatch) -> ClassificationMetrics:
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(batch.features), dim=1)
    return classification_metrics(
        batch.targets.tolist(), predictions.tolist(), model.config.num_classes
    )


def _write_immutable(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"immutable smoke artifact differs: {path.name}")
        return
    partial = path.with_suffix(f"{path.suffix}.partial")
    partial.write_bytes(value)
    partial.replace(path)


def _metric_records(
    *,
    smoke_run_id: str,
    result_id: str,
    seed: int,
    classification: ClassificationMetrics,
    ranking_auroc: float,
    ranking_auprc: float,
    false_rejection: float,
    false_acceptance: float,
    trust_mean_gap: float,
    trust_median_gap: float,
) -> tuple[RunMetricRecord, ...]:
    values = {
        "accuracy": classification.accuracy,
        "macro_precision": classification.macro_precision,
        "macro_recall": classification.macro_recall,
        "macro_f1": classification.macro_f1,
        "balanced_accuracy": classification.balanced_accuracy,
        "worst_class_recall": classification.worst_class_recall,
        "macro_false_positive_rate": classification.macro_false_positive_rate,
        "malicious_client_auroc": ranking_auroc,
        "malicious_client_auprc": ranking_auprc,
        "benign_false_rejection_rate": false_rejection,
        "malicious_false_acceptance_rate": false_acceptance,
        "trust_mean_gap": trust_mean_gap,
        "trust_median_gap": trust_median_gap,
    }
    return tuple(
        RunMetricRecord(
            run_id=smoke_run_id,
            experiment_family_id="phase12_synthetic_smoke",
            dataset="synthetic_phase12_smoke",
            method="semcred_fl",
            condition_id="sign_reversal_pipeline_fixture",
            seed=seed,
            metric_name=name,
            value=value,
            source_artifact_id=result_id,
            test_partition_accessed=True,
            performance_claims_allowed=False,
        )
        for name, value in sorted(values.items())
    )


def run_phase12_smoke(config: Phase12SmokeConfig) -> dict[str, object]:
    """Execute the deterministic synthetic pipeline and regenerate artifacts."""

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the smoke run but is unavailable")
    torch.manual_seed(config.seed)
    detector = DetectorConfig(2, 2, (8,), 0.0)
    model = IntrusionMLP(detector)
    manifest = parameter_manifest(model)
    initial_parameters = flatten_parameters(model, manifest)
    clients, scoring_anchor, synthetic_test, data_id = _data(config)
    layout, prior = _semantic_contract(data_id)
    stable_config = {
        **config.stable_record(),
        "data_id": data_id,
        "detector": detector.to_dict(),
        "initial_parameters_sha256": _tensor_sha256(initial_parameters),
        "semantic_layout_id": layout.layout_id,
        "semantic_prior_id": prior.bundle_id,
    }
    smoke_run_id = run_id(stable_config, length=64)
    run_directory = config.output_directory.resolve() / smoke_run_id
    report_path = run_directory / "smoke_report.json"
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            not isinstance(existing_report, dict)
            or existing_report.get("smoke_run_id") != smoke_run_id
        ):
            raise ValueError("existing smoke report is malformed or belongs to another run")
        return cast(dict[str, object], existing_report)

    states: dict[str, ClientState] = {}
    round_records: list[dict[str, object]] = []
    malicious_observations: list[bool] = []
    suspiciousness_observations: list[float] = []
    eligible_observations: list[bool] = []
    trust_observations: list[float] = []
    aggregation = AggregationConfig(
        burn_in_rounds=1,
        threshold_multiplier=1.0,
        minimum_retained_ratio=0.5,
        clipping_multiplier=2.5,
    )
    for round_id in range(config.rounds):
        broadcast = flatten_parameters(model, manifest).detach().cpu()
        simulation_updates = tuple(
            _apply_smoke_attack(
                _train_client(
                    client_id=client_id,
                    round_id=round_id,
                    features=clients[client_id][0],
                    targets=clients[client_id][1],
                    detector=detector,
                    global_parameters=broadcast,
                    config=config,
                ),
                config,
                round_id,
            )
            for client_id in sorted(clients)
        )
        validation = validate_client_updates(
            simulation_updates,
            expected_round=round_id,
            expected_values=manifest.total_values,
        )
        if validation.rejected or len(validation.accepted) != config.clients:
            raise RuntimeError("synthetic smoke update validation failed")
        defense_updates = validation.accepted
        semantic = evaluate_semantic_updates(
            model,
            defense_updates,
            manifest,
            scoring_anchor,
            layout,
            prior,
            attribution_enabled=round_id % config.attribution_interval == 0,
        )
        histories = {
            client_id: state.accepted_history
            for client_id, state in states.items()
            if state.accepted_history is not None
        }
        numerical = evaluate_numerical_updates(defense_updates, histories=histories)
        result = process_semcred_round(
            round_id=round_id,
            updates=defense_updates,
            semantic_scores={client_id: value.score for client_id, value in semantic.items()},
            numerical_scores={client_id: value.score for client_id, value in numerical.items()},
            client_states=states,
            config=aggregation,
        )
        apply_flat_delta(model, result.aggregate_delta, manifest)
        states = result.client_states
        malicious_by_client = {
            update.client_id: bool(update.is_malicious) for update in simulation_updates
        }
        eligible_set = set(result.eligible_client_ids)
        for client_id in sorted(result.participating_trust):
            malicious_observations.append(malicious_by_client[client_id])
            suspiciousness_observations.append(1.0 - result.participating_trust[client_id])
            eligible_observations.append(client_id in eligible_set)
            trust_observations.append(result.participating_trust[client_id])
        round_records.append(
            {
                "round_id": round_id,
                "burn_in": result.burn_in,
                "attribution_enabled": round_id % config.attribution_interval == 0,
                "malicious_client_ids": [
                    client_id
                    for client_id in sorted(malicious_by_client)
                    if malicious_by_client[client_id]
                ],
                "semantic_scores": {
                    client_id: semantic[client_id].score for client_id in sorted(semantic)
                },
                "numerical_scores": {
                    client_id: numerical[client_id].score for client_id in sorted(numerical)
                },
                "participating_trust": result.participating_trust,
                "eligible_client_ids": list(result.eligible_client_ids),
                "history_admitted_client_ids": list(result.history_admitted_client_ids),
                "aggregate_delta_sha256": _tensor_sha256(result.aggregate_delta),
                "global_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
            }
        )

    classification = _evaluate_model(model, synthetic_test)
    ranking = malicious_client_ranking(malicious_observations, suspiciousness_observations)
    eligibility = eligibility_metrics(malicious_observations, eligible_observations)
    separation = trust_separation(malicious_observations, trust_observations)
    required_values = (
        ranking.auroc,
        ranking.auprc,
        eligibility.benign_false_rejection.value,
        eligibility.malicious_false_acceptance.value,
        separation.mean_gap,
        separation.median_gap,
    )
    if any(value is None for value in required_values):
        raise RuntimeError("synthetic smoke assessment unexpectedly became undefined")
    defined_values = cast(tuple[float, float, float, float, float, float], required_values)
    stable_result: dict[str, object] = {
        "schema": "semcred_phase12_synthetic_smoke_result_v1",
        "smoke_run_id": smoke_run_id,
        "status": "complete",
        "stable_config": stable_config,
        "rounds": round_records,
        "final_parameters_sha256": _tensor_sha256(flatten_parameters(model, manifest)),
        "synthetic_test_metrics": classification.to_dict(),
        "client_assessment": {
            "malicious_client_auroc": defined_values[0],
            "malicious_client_auprc": defined_values[1],
            "benign_false_rejection": eligibility.benign_false_rejection.to_dict(),
            "malicious_false_acceptance": eligibility.malicious_false_acceptance.to_dict(),
            "trust_mean_gap": defined_values[4],
            "trust_median_gap": defined_values[5],
        },
        "gates": {
            "synthetic_only": True,
            "defense_inputs_exclude_attack_metadata": True,
            "scoring_anchor_used": True,
            "validation_partition_accessed": False,
            "synthetic_test_partition_accessed": True,
            "real_test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    result_id = run_id(stable_result, length=64)
    report: dict[str, object] = {**stable_result, "result_id": result_id}
    records = _metric_records(
        smoke_run_id=smoke_run_id,
        result_id=result_id,
        seed=config.seed,
        classification=classification,
        ranking_auroc=defined_values[0],
        ranking_auprc=defined_values[1],
        false_rejection=defined_values[2],
        false_acceptance=defined_values[3],
        trust_mean_gap=defined_values[4],
        trust_median_gap=defined_values[5],
    )
    metrics_bytes = "".join(canonical_json(asdict(record)) + "\n" for record in records).encode(
        "utf-8"
    )
    _write_immutable(run_directory / "run_metrics.jsonl", metrics_bytes)
    summary = write_summary_bundle(
        records,
        run_directory / "reporting",
        bootstrap_resamples=1_000,
        bootstrap_seed=config.seed,
    )
    rendered = render_summary_artifacts(
        run_directory / "reporting" / "metric_summaries.json",
        run_directory / "reporting",
        metric_name="macro_f1",
    )
    report["reporting"] = {
        "summary_bundle_id": summary["summary_bundle_id"],
        "macro_f1_artifact_id": rendered["artifact_id"],
        "output_directory": "reporting",
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_immutable(report_path, report_bytes)
    return report
