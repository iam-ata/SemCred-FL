"""Tests for deterministic local client optimization."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from semcred_fl.baselines import fedavg
from semcred_fl.data.training_corpus import TrainingBatch
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client
from semcred_fl.federation.parameters import (
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


class _Batches:
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, batch_size: int = 8) -> None:
        self.features = features
        self.targets = targets
        self.batch_size = batch_size

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        order = torch.randperm(len(self.targets), generator=torch.Generator().manual_seed(seed))
        for start in range(0, len(order), self.batch_size):
            selected = order[start : start + self.batch_size]
            yield TrainingBatch(self.features[selected], self.targets[selected])


def _fixture() -> tuple[IntrusionMLP, _Batches]:
    torch.manual_seed(10)
    negative = torch.randn(20, 2) * 0.1 - 1.0
    positive = torch.randn(20, 2) * 0.1 + 1.0
    features = torch.cat((negative, positive))
    targets = torch.cat((torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)))
    model = IntrusionMLP(DetectorConfig(2, 2, (8,), 0.0))
    return model, _Batches(features, targets)


def test_local_training_is_deterministic_and_one_client_fedavg_matches() -> None:
    model, batches = _fixture()
    manifest = parameter_manifest(model)
    global_parameters = flatten_parameters(model, manifest).detach().cpu()
    config = LocalTrainingConfig(
        local_epochs=2,
        learning_rate=0.02,
        seed=17,
        device="cpu",
    )
    kwargs = {
        "detector": model.config,
        "manifest": manifest,
        "global_parameters": global_parameters,
        "round_id": 0,
        "client_id": "client-00",
        "declared_sample_count": 40,
        "class_counts": (20, 20),
        "batch_source": batches,
        "config": config,
    }

    first = train_local_client(**kwargs)  # type: ignore[arg-type]
    second = train_local_client(**kwargs)  # type: ignore[arg-type]

    torch.testing.assert_close(first.delta, second.delta, rtol=0.0, atol=0.0)
    assert first.mean_local_loss == second.mean_local_loss
    assert first.finite and first.update_norm > 0.0
    aggregate = fedavg((first.to_defense_update(),))
    torch.testing.assert_close(aggregate.delta, first.delta, rtol=0.0, atol=0.0)
    updated = IntrusionMLP(model.config)
    updated.load_state_dict(model.state_dict())
    apply_flat_delta(updated, aggregate.delta, manifest)
    torch.testing.assert_close(
        flatten_parameters(updated, manifest), global_parameters + first.delta, rtol=0.0, atol=0.0
    )


def test_local_training_records_missing_classes() -> None:
    torch.manual_seed(5)
    features = torch.randn(12, 2)
    targets = torch.zeros(12, dtype=torch.long)
    model = IntrusionMLP(DetectorConfig(2, 2, (4,), 0.0))
    manifest = parameter_manifest(model)
    update = train_local_client(
        detector=model.config,
        manifest=manifest,
        global_parameters=flatten_parameters(model, manifest),
        round_id=2,
        client_id="client-01",
        declared_sample_count=12,
        class_counts=(12, 0),
        batch_source=_Batches(features, targets),
        config=LocalTrainingConfig(device="cpu"),
    )

    assert update.missing_class_ids == (1,)


def test_local_training_rejects_incomplete_batch_source() -> None:
    model, batches = _fixture()
    manifest = parameter_manifest(model)
    batches.targets = batches.targets[:-1]
    batches.features = batches.features[:-1]
    with pytest.raises(ValueError, match="declared population"):
        train_local_client(
            detector=model.config,
            manifest=manifest,
            global_parameters=flatten_parameters(model, manifest),
            round_id=0,
            client_id="client-00",
            declared_sample_count=40,
            class_counts=(20, 20),
            batch_source=batches,
            config=LocalTrainingConfig(device="cpu"),
        )


def test_local_training_fast_controls_are_explicit_and_cpu_fused_fails() -> None:
    config = LocalTrainingConfig(
        device="cpu",
        optimizer_backend="standard",
        prevalidated_batches=True,
    )
    assert config.to_dict()["prevalidated_batches"] is True
    with pytest.raises(ValueError, match="optimizer_backend"):
        LocalTrainingConfig(optimizer_backend="unknown")

    model, batches = _fixture()
    manifest = parameter_manifest(model)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_local_client(
            detector=model.config,
            manifest=manifest,
            global_parameters=flatten_parameters(model, manifest),
            round_id=0,
            client_id="client-00",
            declared_sample_count=40,
            class_counts=(20, 20),
            batch_source=batches,
            config=LocalTrainingConfig(device="cpu", optimizer_backend="fused"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_cuda_local_training_is_deterministic() -> None:
    model, batches = _fixture()
    manifest = parameter_manifest(model)
    kwargs = {
        "detector": model.config,
        "manifest": manifest,
        "global_parameters": flatten_parameters(model, manifest).detach().cpu(),
        "round_id": 0,
        "client_id": "client-00",
        "declared_sample_count": 40,
        "class_counts": (20, 20),
        "batch_source": batches,
        "config": LocalTrainingConfig(
            device="cuda",
            optimizer_backend="fused",
            prevalidated_batches=True,
            seed=23,
        ),
    }
    first = train_local_client(**kwargs)  # type: ignore[arg-type]
    second = train_local_client(**kwargs)  # type: ignore[arg-type]
    torch.testing.assert_close(first.delta, second.delta, rtol=0.0, atol=0.0)
    assert first.mean_local_loss == second.mean_local_loss
