"""Phase 4 detector, objective, metric, and checkpoint gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from sklearn import metrics as sklearn_metrics

from semcred_fl.modeling import (
    ClassWeightPolicy,
    ConfusionMatrixAccumulator,
    DetectorConfig,
    IntrusionMLP,
    class_weighted_cross_entropy,
    class_weights_from_counts,
    classification_metrics,
    load_checkpoint,
    local_objective,
    proximal_penalty,
    save_checkpoint,
)


def test_detector_returns_logits_and_probabilities() -> None:
    model = IntrusionMLP(DetectorConfig(4, 3, hidden_dimensions=(8, 4), dropout=0.0))
    features = torch.ones((5, 4))
    logits = model(features)
    probabilities = model.probabilities(features)
    assert logits.shape == (5, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5))
    with pytest.raises(ValueError, match="dimension 4"):
        model(torch.ones((5, 3)))


def test_detector_config_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="input_dimension"):
        DetectorConfig(0, 2)
    with pytest.raises(ValueError, match="at least two"):
        DetectorConfig(4, 1)
    with pytest.raises(ValueError, match="hidden_dimensions"):
        DetectorConfig(4, 2, hidden_dimensions=())
    with pytest.raises(ValueError, match="dropout"):
        DetectorConfig(4, 2, dropout=1.0)
    valid: dict[str, object] = {
        "input_dimension": 4,
        "num_classes": 2,
        "hidden_dimensions": [8, 4],
        "dropout": 0.2,
    }
    assert DetectorConfig.from_dict(valid) == DetectorConfig(4, 2, (8, 4), 0.2)
    for key, value in (
        ("input_dimension", "4"),
        ("hidden_dimensions", [8, "4"]),
        ("dropout", "0.2"),
    ):
        invalid = {**valid, key: value}
        with pytest.raises(ValueError, match="checkpoint"):
            DetectorConfig.from_dict(invalid)


def test_class_weights_and_objective_match_declared_formula() -> None:
    weights = class_weights_from_counts((90, 10, 0), ClassWeightPolicy(4.0))
    assert weights == pytest.approx((100 / 180, 4.0, 0.0))
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    weight_tensor = torch.tensor([0.5, 2.0])
    loss = class_weighted_cross_entropy(logits, targets, weight_tensor)
    log_probabilities = torch.log_softmax(logits, dim=1)
    manual = (-0.5 * log_probabilities[0, 0] - 2.0 * log_probabilities[1, 1]) / 2
    assert torch.allclose(loss, manual)


def test_weight_and_loss_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ClassWeightPolicy(0.5)
    with pytest.raises(ValueError, match="at least two"):
        class_weights_from_counts((1,))
    with pytest.raises(ValueError, match="nonnegative"):
        class_weights_from_counts((1, -1))
    with pytest.raises(ValueError, match="observed"):
        class_weights_from_counts((0, 0))
    logits = torch.ones((2, 2))
    targets = torch.tensor([0, 1])
    weights = torch.ones(2)
    with pytest.raises(ValueError, match="two-dimensional"):
        class_weighted_cross_entropy(logits[0], targets, weights)
    with pytest.raises(ValueError, match="same positive"):
        class_weighted_cross_entropy(logits, targets[:1], weights)
    with pytest.raises(ValueError, match="class dimension"):
        class_weighted_cross_entropy(logits, targets, torch.ones(3))


def test_proximal_objective_is_zero_at_reference_and_positive_after_change() -> None:
    model = IntrusionMLP(DetectorConfig(2, 2, hidden_dimensions=(3,), dropout=0.0))
    reference = tuple(parameter.detach().clone() for parameter in model.parameters())
    assert proximal_penalty(model, reference, 0.5).item() == pytest.approx(0.0)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    penalty = proximal_penalty(model, reference, 0.5)
    assert penalty.item() > 0.0
    logits = model(torch.ones((2, 2)))
    objective = local_objective(
        model,
        logits,
        torch.tensor([0, 1]),
        torch.ones(2),
        proximal_mu=0.5,
        reference_parameters=reference,
    )
    assert objective.item() > penalty.item()


def test_proximal_validation_rejects_structure_errors() -> None:
    model = IntrusionMLP(DetectorConfig(2, 2, hidden_dimensions=(3,), dropout=0.0))
    reference = tuple(parameter.detach().clone() for parameter in model.parameters())
    with pytest.raises(ValueError, match="cannot be negative"):
        proximal_penalty(model, reference, -0.1)
    with pytest.raises(ValueError, match="structure"):
        proximal_penalty(model, reference[:-1], 0.1)
    wrong = (torch.zeros(1), *reference[1:])
    with pytest.raises(ValueError, match="shape"):
        proximal_penalty(model, wrong, 0.1)
    empty = torch.nn.Identity()
    with pytest.raises(ValueError, match="no trainable"):
        proximal_penalty(empty, (), 0.1)
    logits = model(torch.ones((2, 2)))
    with pytest.raises(ValueError, match="requires reference"):
        local_objective(
            model,
            logits,
            torch.tensor([0, 1]),
            torch.ones(2),
            proximal_mu=0.1,
        )


def test_loss_decreases_on_tiny_learnable_fixture() -> None:
    torch.manual_seed(7)
    negative = torch.randn((64, 2)) * 0.15 - 1.0
    positive = torch.randn((64, 2)) * 0.15 + 1.0
    features = torch.cat((negative, positive))
    targets = torch.cat((torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)))
    model = IntrusionMLP(DetectorConfig(2, 2, hidden_dimensions=(8,), dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    weights = torch.ones(2)
    initial = class_weighted_cross_entropy(model(features), targets, weights).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = class_weighted_cross_entropy(model(features), targets, weights)
        loss.backward()
        optimizer.step()
    final = class_weighted_cross_entropy(model(features), targets, weights).item()
    assert final < initial * 0.1


def test_metrics_match_known_sklearn_fixture() -> None:
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 0, 2]
    result = classification_metrics(targets, predictions, 3)
    assert result.accuracy == pytest.approx(sklearn_metrics.accuracy_score(targets, predictions))
    assert result.macro_precision == pytest.approx(
        sklearn_metrics.precision_score(targets, predictions, average="macro", zero_division=0)
    )
    assert result.macro_recall == pytest.approx(
        sklearn_metrics.recall_score(targets, predictions, average="macro", zero_division=0)
    )
    assert result.macro_f1 == pytest.approx(
        sklearn_metrics.f1_score(targets, predictions, average="macro", zero_division=0)
    )
    assert result.balanced_accuracy == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(targets, predictions)
    )
    assert result.confusion_matrix == tuple(
        tuple(int(value) for value in row)
        for row in sklearn_metrics.confusion_matrix(targets, predictions, labels=[0, 1, 2])
    )
    accumulator = ConfusionMatrixAccumulator(3)
    accumulator.update(targets[:3], predictions[:3])
    accumulator.update(targets[3:], predictions[3:])
    assert accumulator.compute() == result


def test_metric_validation_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        classification_metrics([0], [0], 1)
    with pytest.raises(ValueError, match="same positive"):
        classification_metrics([], [], 2)
    with pytest.raises(ValueError, match="valid class"):
        classification_metrics([0, 2], [0, 1], 2)
    with pytest.raises(ValueError, match="at least two"):
        ConfusionMatrixAccumulator(1)
    accumulator = ConfusionMatrixAccumulator(2)
    with pytest.raises(ValueError, match="same length"):
        accumulator.update([0], [0, 1])
    with pytest.raises(ValueError, match="valid class"):
        accumulator.update([2], [0])
    with pytest.raises(ValueError, match="records"):
        accumulator.compute()


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = IntrusionMLP(DetectorConfig(3, 2, hidden_dimensions=(5,), dropout=0.0))
    model.eval()
    features = torch.randn((4, 3))
    expected = model(features).detach()
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, model, epoch=2, metadata={"purpose": "test"})
    loaded = load_checkpoint(path, expected_sha256=digest)
    actual = loaded.model(features).detach()
    assert torch.equal(expected, actual)
    assert loaded.epoch == 2
    assert loaded.metadata == {"purpose": "test"}
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_checkpoint_validation_rejects_malformed_payloads(tmp_path: Path) -> None:
    model = IntrusionMLP(DetectorConfig(2, 2, hidden_dimensions=(3,), dropout=0.0))
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "negative.pt", model, epoch=-1)

    valid: dict[str, Any] = {
        "checkpoint_version": 1,
        "detector_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "epoch": 0,
        "metadata": {},
    }
    cases = (
        ([], "not a dictionary"),
        ({**valid, "checkpoint_version": 99}, "Unsupported"),
        ({**valid, "model_state": "bad"}, "configuration or state"),
        ({**valid, "epoch": -1}, "epoch or metadata"),
        ({**valid, "optimizer_state": "bad"}, "optimizer state"),
    )
    for index, (payload, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            load_checkpoint(path)
