"""Class-weighted and proximal objectives matching the manuscript equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True, slots=True)
class ClassWeightPolicy:
    """Capped inverse-frequency class weighting for one training population."""

    maximum_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.maximum_weight < 1.0:
            raise ValueError("maximum_weight must be at least one")


def class_weights_from_counts(
    counts: Sequence[int], policy: ClassWeightPolicy | None = None
) -> tuple[float, ...]:
    """Return capped inverse-frequency weights and zero for absent classes.

    For an observed class ``c``, the uncapped value is ``N / (C+ * n_c)``, where
    ``C+`` is the number of classes present in this training population. This makes
    the record-weighted mean equal to one before capping.
    """

    resolved_policy = policy or ClassWeightPolicy()
    if len(counts) < 2 or any(count < 0 for count in counts):
        raise ValueError("counts must contain at least two nonnegative class counts")
    total = sum(counts)
    observed = sum(count > 0 for count in counts)
    if total < 1 or observed < 1:
        raise ValueError("At least one class must be observed")
    return tuple(
        0.0 if count == 0 else min(resolved_policy.maximum_weight, total / (observed * count))
        for count in counts
    )


def class_weighted_cross_entropy(logits: Tensor, targets: Tensor, class_weights: Tensor) -> Tensor:
    """Compute ``-(1/n) sum_i w[y_i] log p(y_i|x_i)`` exactly."""

    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be two-dimensional and targets one-dimensional")
    if logits.shape[0] != targets.shape[0] or logits.shape[0] < 1:
        raise ValueError("logits and targets require the same positive batch size")
    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError("class_weights must match the logit class dimension")
    losses = functional.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    return losses.mean()


def proximal_penalty(
    model: nn.Module, reference_parameters: Sequence[Tensor], proximal_mu: float
) -> Tensor:
    """Compute ``mu/2 * ||theta-theta_reference||_2^2``."""

    if proximal_mu < 0.0:
        raise ValueError("proximal_mu cannot be negative")
    parameters = tuple(model.parameters())
    if len(parameters) != len(reference_parameters):
        raise ValueError("Reference parameter structure differs from the model")
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    total = torch.zeros((), dtype=parameters[0].dtype, device=parameters[0].device)
    for parameter, reference in zip(parameters, reference_parameters, strict=True):
        if parameter.shape != reference.shape:
            raise ValueError("Reference parameter shape differs from the model")
        total = total + torch.sum((parameter - reference.to(parameter.device)) ** 2)
    return total * (proximal_mu / 2.0)


def local_objective(
    model: nn.Module,
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor,
    *,
    proximal_mu: float = 0.0,
    reference_parameters: Sequence[Tensor] | None = None,
) -> Tensor:
    """Compute class-weighted cross-entropy plus an optional FedProx penalty."""

    data_loss = class_weighted_cross_entropy(logits, targets, class_weights)
    if proximal_mu == 0.0:
        return data_loss
    if reference_parameters is None:
        raise ValueError("Nonzero proximal_mu requires reference_parameters")
    return data_loss + proximal_penalty(model, reference_parameters, proximal_mu)
