"""Deterministic primary-task classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Clean-data metrics and exact confusion counts."""

    records: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    balanced_accuracy: float
    worst_class_recall: float
    macro_false_positive_rate: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    per_class_false_positive_rate: tuple[float, ...]
    support: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metric values."""

        return {
            "records": self.records,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "worst_class_recall": self.worst_class_recall,
            "macro_false_positive_rate": self.macro_false_positive_rate,
            "per_class_precision": list(self.per_class_precision),
            "per_class_recall": list(self.per_class_recall),
            "per_class_f1": list(self.per_class_f1),
            "per_class_false_positive_rate": list(self.per_class_false_positive_rate),
            "support": list(self.support),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def _metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> ClassificationMetrics:
    num_classes = len(matrix)
    if num_classes < 2 or any(len(row) != num_classes for row in matrix):
        raise ValueError("confusion matrix must be square with at least two classes")
    total = sum(sum(row) for row in matrix)
    if total < 1 or any(value < 0 for row in matrix for value in row):
        raise ValueError("confusion matrix must contain nonnegative counts and records")

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    false_positive_rate: list[float] = []
    support: list[int] = []
    for class_id in range(num_classes):
        true_positive = matrix[class_id][class_id]
        class_support = sum(matrix[class_id])
        predicted = sum(row[class_id] for row in matrix)
        false_positive = predicted - true_positive
        true_negative = total - class_support - false_positive
        class_precision = _safe_ratio(true_positive, predicted)
        class_recall = _safe_ratio(true_positive, class_support)
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(_safe_ratio(2 * class_precision * class_recall, class_precision + class_recall))
        false_positive_rate.append(_safe_ratio(false_positive, false_positive + true_negative))
        support.append(class_support)

    observed_recall = [value for value, count in zip(recall, support, strict=True) if count > 0]
    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    return ClassificationMetrics(
        records=total,
        accuracy=sum(matrix[index][index] for index in range(num_classes)) / total,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=sum(f1) / num_classes,
        balanced_accuracy=macro_recall,
        worst_class_recall=min(observed_recall),
        macro_false_positive_rate=sum(false_positive_rate) / num_classes,
        per_class_precision=tuple(precision),
        per_class_recall=tuple(recall),
        per_class_f1=tuple(f1),
        per_class_false_positive_rate=tuple(false_positive_rate),
        support=tuple(support),
        confusion_matrix=tuple(tuple(row) for row in matrix),
    )


class ConfusionMatrixAccumulator:
    """Accumulate exact classification counts without retaining predictions."""

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self._num_classes = num_classes
        self._matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    def update(self, targets: Sequence[int], predictions: Sequence[int]) -> None:
        """Add one prediction batch."""

        if len(targets) != len(predictions):
            raise ValueError("targets and predictions require the same length")
        for target, prediction in zip(targets, predictions, strict=True):
            if not 0 <= target < self._num_classes or not 0 <= prediction < self._num_classes:
                raise ValueError("targets and predictions must be valid class IDs")
            self._matrix[target][prediction] += 1

    def compute(self) -> ClassificationMetrics:
        """Return metrics for all accumulated records."""

        return _metrics_from_matrix(self._matrix)


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> ClassificationMetrics:
    """Compute multiclass metrics over the declared class coordinate system."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if not targets or len(targets) != len(predictions):
        raise ValueError("targets and predictions require the same positive length")
    if any(value < 0 or value >= num_classes for value in (*targets, *predictions)):
        raise ValueError("targets and predictions must be valid class IDs")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target][prediction] += 1

    return _metrics_from_matrix(matrix)
