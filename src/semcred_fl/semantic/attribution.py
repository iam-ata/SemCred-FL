"""Class-specific grouped gradient-times-input attribution on scoring anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from semcred_fl.semantic.contracts import SemanticPriorError
from semcred_fl.semantic.prior import FeatureGroupLayout


@dataclass(frozen=True, slots=True)
class GroupedAttribution:
    """One class-by-semantic-group attribution matrix."""

    values: tuple[tuple[float, ...], ...]
    class_counts: tuple[int, ...]
    group_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.class_counts) or not self.values:
            raise SemanticPriorError("Attribution class dimensions are invalid")
        if any(count < 1 for count in self.class_counts):
            raise SemanticPriorError("Every class requires a scoring anchor")
        if any(
            len(row) != len(self.group_names)
            or any(not math.isfinite(value) or value < 0.0 for value in row)
            for row in self.values
        ):
            raise SemanticPriorError("Grouped attribution values are invalid")


def grouped_gradient_times_input(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    layout: FeatureGroupLayout,
    *,
    num_classes: int,
) -> GroupedAttribution:
    """Compute the manuscript RMS gradient-times-input attribution by class and group."""

    if features.ndim != 2 or features.shape[1] != len(layout.coordinate_names):
        raise ValueError("Scoring-anchor feature dimensions differ from the semantic layout")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("Scoring-anchor targets differ from the feature population")
    if num_classes < 2:
        raise ValueError("Attribution requires at least two classes")
    if not torch.isfinite(features).all().item():
        raise ValueError("Scoring-anchor features contain non-finite values")
    if targets.numel() == 0 or torch.any((targets < 0) | (targets >= num_classes)).item():
        raise ValueError("Scoring-anchor targets contain an invalid class ID")

    was_training = model.training
    rows: list[tuple[float, ...]] = []
    counts: list[int] = []
    try:
        model.eval()
        for class_id in range(num_classes):
            class_features = features[targets == class_id]
            count = int(class_features.shape[0])
            if count < 1:
                raise SemanticPriorError("Every class requires at least one scoring anchor")
            inputs = class_features.detach().clone().requires_grad_(True)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape != (count, num_classes):
                raise ValueError("Detector logits differ from the declared class system")
            log_probabilities = torch.log_softmax(logits, dim=1)
            gradients = torch.autograd.grad(
                log_probabilities[:, class_id].sum(),
                inputs,
                create_graph=False,
                retain_graph=False,
            )[0]
            contribution = inputs * gradients
            group_values: list[float] = []
            for indices in layout.coordinate_indices:
                group = contribution[:, list(indices)]
                per_record = torch.sqrt(torch.mean(torch.square(group), dim=1))
                value = float(torch.mean(per_record).detach().cpu().item())
                if not math.isfinite(value) or value < 0.0:
                    raise SemanticPriorError("Grouped attribution produced an invalid value")
                group_values.append(value)
            rows.append(tuple(group_values))
            counts.append(count)
    finally:
        model.train(was_training)
    return GroupedAttribution(tuple(rows), tuple(counts), layout.group_names)
