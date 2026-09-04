"""Detector, objective, metric, and checkpoint components."""

from semcred_fl.modeling.checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.modeling.losses import (
    ClassWeightPolicy,
    class_weighted_cross_entropy,
    class_weights_from_counts,
    local_objective,
    proximal_penalty,
)
from semcred_fl.modeling.metrics import (
    ClassificationMetrics,
    ConfusionMatrixAccumulator,
    classification_metrics,
)

__all__ = [
    "ClassWeightPolicy",
    "ClassificationMetrics",
    "ConfusionMatrixAccumulator",
    "DetectorConfig",
    "IntrusionMLP",
    "LoadedCheckpoint",
    "class_weighted_cross_entropy",
    "class_weights_from_counts",
    "classification_metrics",
    "load_checkpoint",
    "local_objective",
    "proximal_penalty",
    "save_checkpoint",
]
