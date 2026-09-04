"""Anchor-loss and semantic-attribution credibility for one client population."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from semcred_fl.credibility.robust_stats import (
    one_sided_robust_z,
    renormalize_active_coefficients,
)
from semcred_fl.data.training_corpus import (
    PreprocessedCorpus,
    TrainingBatch,
    iter_partition_batches,
)
from semcred_fl.federation.parameters import ParameterManifest, apply_flat_delta
from semcred_fl.federation.updates import DefenseUpdate
from semcred_fl.semantic.attribution import GroupedAttribution, grouped_gradient_times_input
from semcred_fl.semantic.contracts import SemanticPriorError
from semcred_fl.semantic.pipeline import SemanticPriorBundle
from semcred_fl.semantic.prior import FeatureGroupLayout


@dataclass(frozen=True, slots=True)
class SemanticCoefficients:
    """Configured weights for anchor degradation and attribution discrepancy."""

    loss: float = 0.5
    attribution: float = 0.5

    def __post_init__(self) -> None:
        values = (self.loss, self.attribution)
        if any(not math.isfinite(value) or value < 0.0 for value in values) or sum(values) <= 0.0:
            raise ValueError("Semantic coefficients must be finite, nonnegative, and nonzero")

    def to_dict(self) -> dict[str, float]:
        """Return coefficients under stable indicator names."""

        return {"loss": self.loss, "attribution": self.attribution}


@dataclass(frozen=True, slots=True)
class SemanticDiscrepancy:
    """Class-averaged cosine discrepancy and its availability record."""

    value: float | None
    valid_class_ids: tuple[int, ...]
    unavailable_class_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SemanticIndicators:
    """Raw semantic evidence for one client update."""

    anchor_degradation: float
    attribution_discrepancy: float | None
    valid_attribution_classes: int
    missing_reasons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor_degradation):
            raise ValueError("Anchor degradation must be finite")
        if self.attribution_discrepancy is not None and (
            not math.isfinite(self.attribution_discrepancy) or self.attribution_discrepancy < 0.0
        ):
            raise ValueError("Attribution discrepancy must be finite and nonnegative")
        if self.valid_attribution_classes < 0:
            raise ValueError("Valid attribution class count cannot be negative")


@dataclass(frozen=True, slots=True)
class SemanticCredibility:
    """Missing-aware semantic credibility output for one client."""

    score: float
    indicators: SemanticIndicators
    robust_z: dict[str, float | None]
    active_coefficients: dict[str, float]
    missing_reasons: dict[str, str]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 < self.score <= 1.0:
            raise ValueError("Semantic credibility must be finite and in (0,1]")


def load_scoring_anchor_batch(
    corpus: PreprocessedCorpus,
    *,
    batch_size: int = 1024,
) -> TrainingBatch:
    """Load only the frozen scoring-anchor partition and verify class balance."""

    batches = tuple(iter_partition_batches(corpus, "anchor_scoring", batch_size=batch_size))
    if not batches:
        raise SemanticPriorError("The scoring-anchor partition is empty")
    features = torch.cat(tuple(batch.features for batch in batches), dim=0)
    targets = torch.cat(tuple(batch.targets for batch in batches), dim=0)
    if features.shape != (corpus.partition_counts["anchor_scoring"], corpus.input_dimension):
        raise SemanticPriorError("Scoring-anchor dimensions differ from Phase 3C")
    counts = torch.bincount(targets, minlength=corpus.num_classes)
    if counts.numel() != corpus.num_classes or torch.any(counts < 1).item():
        raise SemanticPriorError("Scoring anchors do not cover every primary class")
    if torch.any(counts != counts[0]).item():
        raise SemanticPriorError("Scoring anchors are not class-balanced")
    return TrainingBatch(features, targets)


def mean_anchor_cross_entropy(model: nn.Module, anchors: TrainingBatch) -> float:
    """Return mean negative log probability on the scoring anchors."""

    if anchors.records < 1 or anchors.features.ndim != 2 or anchors.targets.ndim != 1:
        raise ValueError("Anchor tensors are empty or malformed")
    if anchors.features.shape[0] != anchors.targets.shape[0]:
        raise ValueError("Anchor feature and target cardinalities differ")
    if not torch.isfinite(anchors.features).all().item():
        raise ValueError("Anchor features contain non-finite values")
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model(anchors.features)
            if logits.ndim != 2 or logits.shape[0] != anchors.records:
                raise ValueError("Detector logits differ from the anchor population")
            if torch.any((anchors.targets < 0) | (anchors.targets >= logits.shape[1])).item():
                raise ValueError("Anchor targets contain an invalid class ID")
            log_probabilities = torch.log_softmax(logits, dim=1)
            selected = log_probabilities.gather(1, anchors.targets[:, None]).squeeze(1)
            loss = float((-selected.mean()).detach().cpu().item())
    finally:
        model.train(was_training)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("Anchor loss is invalid")
    return loss


def reconstruct_client_model(
    global_model: nn.Module,
    update: DefenseUpdate,
    manifest: ParameterManifest,
) -> nn.Module:
    """Reconstruct ``theta + delta`` without mutating the global model."""

    reconstructed = copy.deepcopy(global_model)
    apply_flat_delta(reconstructed, update.delta, manifest)
    return reconstructed


def semantic_discrepancy(
    attribution: GroupedAttribution,
    prior: SemanticPriorBundle,
    *,
    epsilon: float = 1e-12,
    negligible_norm: float = 1e-12,
) -> SemanticDiscrepancy:
    """Compare each valid attribution row with its class-specific semantic prior."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(negligible_norm) or negligible_norm < 0.0:
        raise ValueError("negligible_norm must be finite and nonnegative")
    if attribution.group_names != prior.feature_group_names:
        raise SemanticPriorError("Attribution groups differ from the semantic prior")
    if len(attribution.values) != len(prior.priors):
        raise SemanticPriorError("Attribution classes differ from the semantic prior")
    discrepancies: list[float] = []
    valid: list[int] = []
    unavailable: list[int] = []
    for class_id, (attribution_row, prior_row) in enumerate(
        zip(attribution.values, prior.priors, strict=True)
    ):
        l1_norm = sum(attribution_row)
        normalized = tuple(value / (l1_norm + epsilon) for value in attribution_row)
        attribution_norm = math.sqrt(sum(value * value for value in normalized))
        prior_norm = math.sqrt(sum(value * value for value in prior_row))
        if attribution_norm <= negligible_norm or prior_norm <= negligible_norm:
            unavailable.append(class_id)
            continue
        dot_product = sum(left * right for left, right in zip(normalized, prior_row, strict=True))
        cosine = dot_product / (attribution_norm * prior_norm + epsilon)
        discrepancy = 1.0 - cosine
        if not math.isfinite(discrepancy) or discrepancy < -1e-9 or discrepancy > 2.0 + 1e-9:
            raise SemanticPriorError("Semantic cosine discrepancy is invalid")
        discrepancies.append(min(2.0, max(0.0, discrepancy)))
        valid.append(class_id)
    value = sum(discrepancies) / len(discrepancies) if discrepancies else None
    return SemanticDiscrepancy(value, tuple(valid), tuple(unavailable))


def semantic_credibility_population(
    indicators: Mapping[str, SemanticIndicators],
    *,
    coefficients: SemanticCoefficients | None = None,
    z_max: float = 20.0,
    epsilon: float = 1e-12,
) -> dict[str, SemanticCredibility]:
    """Transform population semantic indicators into missing-aware credibility scores."""

    if not indicators or any(not client_id for client_id in indicators):
        raise ValueError("Semantic scoring requires nonempty client IDs")
    client_ids = tuple(sorted(indicators))
    loss_values = tuple(indicators[client_id].anchor_degradation for client_id in client_ids)
    loss_z = dict(
        zip(
            client_ids,
            one_sided_robust_z(loss_values, z_max=z_max, epsilon=epsilon),
            strict=True,
        )
    )
    attribution_ids = tuple(
        client_id
        for client_id in client_ids
        if indicators[client_id].attribution_discrepancy is not None
    )
    attribution_z: dict[str, float] = {}
    if attribution_ids:
        attribution_values: list[float] = []
        for client_id in attribution_ids:
            value = indicators[client_id].attribution_discrepancy
            if value is None:
                raise RuntimeError("Available attribution unexpectedly became missing")
            attribution_values.append(value)
        attribution_z = dict(
            zip(
                attribution_ids,
                one_sided_robust_z(
                    tuple(attribution_values),
                    z_max=z_max,
                    epsilon=epsilon,
                ),
                strict=True,
            )
        )
    configured = (coefficients or SemanticCoefficients()).to_dict()
    results: dict[str, SemanticCredibility] = {}
    for client_id in client_ids:
        raw = indicators[client_id]
        attribution_available = raw.attribution_discrepancy is not None
        active = renormalize_active_coefficients(
            configured,
            {"loss": True, "attribution": attribution_available},
        )
        z_values: dict[str, float | None] = {
            "loss": loss_z[client_id],
            "attribution": attribution_z.get(client_id),
        }
        exponent = 0.0
        for name, coefficient in active.items():
            value = z_values[name]
            if value is None:
                raise RuntimeError("An active semantic indicator lacks a robust score")
            exponent += coefficient * value
        score = math.exp(-exponent)
        missing = dict(raw.missing_reasons)
        if not attribution_available:
            missing.setdefault("attribution", "attribution_unavailable_for_client_round")
        results[client_id] = SemanticCredibility(
            score=score,
            indicators=raw,
            robust_z=z_values,
            active_coefficients=active,
            missing_reasons=missing,
        )
    return results


def evaluate_semantic_updates(
    global_model: nn.Module,
    updates: tuple[DefenseUpdate, ...],
    manifest: ParameterManifest,
    anchors: TrainingBatch,
    layout: FeatureGroupLayout,
    prior: SemanticPriorBundle,
    *,
    attribution_enabled: bool,
    coefficients: SemanticCoefficients | None = None,
    z_max: float = 20.0,
    epsilon: float = 1e-12,
    negligible_norm: float = 1e-12,
) -> dict[str, SemanticCredibility]:
    """Reconstruct submitted models and score semantic credibility for one round."""

    if not updates or len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Semantic evaluation requires unique client updates")
    if layout.dataset != prior.dataset or layout.group_names != prior.feature_group_names:
        raise SemanticPriorError("Semantic layout differs from the frozen prior bundle")
    global_loss = mean_anchor_cross_entropy(global_model, anchors)
    indicators: dict[str, SemanticIndicators] = {}
    for update in sorted(updates, key=lambda item: item.client_id):
        client_model = reconstruct_client_model(global_model, update, manifest)
        client_loss = mean_anchor_cross_entropy(client_model, anchors)
        discrepancy_value: float | None = None
        valid_classes = 0
        missing: dict[str, str] = {}
        if attribution_enabled:
            attribution = grouped_gradient_times_input(
                client_model,
                anchors.features,
                anchors.targets,
                layout,
                num_classes=len(prior.class_names),
            )
            discrepancy = semantic_discrepancy(
                attribution,
                prior,
                epsilon=epsilon,
                negligible_norm=negligible_norm,
            )
            discrepancy_value = discrepancy.value
            valid_classes = len(discrepancy.valid_class_ids)
            if discrepancy.value is None:
                missing["attribution"] = "no_class_has_non_negligible_vectors"
        else:
            missing["attribution"] = "attribution_not_scheduled_this_round"
        indicators[update.client_id] = SemanticIndicators(
            anchor_degradation=client_loss - global_loss,
            attribution_discrepancy=discrepancy_value,
            valid_attribution_classes=valid_classes,
            missing_reasons=missing,
        )
    return semantic_credibility_population(
        indicators,
        coefficients=coefficients,
        z_max=z_max,
        epsilon=epsilon,
    )
