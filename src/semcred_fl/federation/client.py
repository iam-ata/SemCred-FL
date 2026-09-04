"""Deterministic local client optimization for the basic FL engine."""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from semcred_fl.data.training_corpus import TrainingBatch
from semcred_fl.federation.parameters import (
    ParameterManifest,
    assign_flat_parameters,
    flatten_parameters,
)
from semcred_fl.federation.updates import ClientUpdate
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.modeling.losses import ClassWeightPolicy, class_weights_from_counts, local_objective
from semcred_fl.seeding import derive_seed


class LocalBatchSource(Protocol):
    """Re-iterable client batch source with an explicit epoch seed."""

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        """Yield every local training record once for the selected epoch."""


@dataclass(frozen=True, slots=True)
class LocalTrainingConfig:
    """Frozen local-optimizer controls shared by compared methods."""

    local_epochs: int = 1
    learning_rate: float = 1e-3
    proximal_mu: float = 0.0
    maximum_class_weight: float = 10.0
    seed: int = 20260816
    device: str = "auto"
    optimizer_backend: str = "standard"
    prevalidated_batches: bool = False

    def __post_init__(self) -> None:
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.proximal_mu < 0.0 or not math.isfinite(self.proximal_mu):
            raise ValueError("proximal_mu must be nonnegative and finite")
        if self.maximum_class_weight < 1.0 or not math.isfinite(self.maximum_class_weight):
            raise ValueError("maximum_class_weight must be finite and at least one")
        if self.seed < 0 or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("seed or device is invalid")
        if self.optimizer_backend not in {"standard", "fused"}:
            raise ValueError("optimizer_backend must be standard or fused")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible local-training record."""

        return {
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "proximal_mu": self.proximal_mu,
            "maximum_class_weight": self.maximum_class_weight,
            "seed": self.seed,
            "device": self.device,
            "optimizer": "adam",
            "optimizer_backend": self.optimizer_backend,
            "prevalidated_batches": self.prevalidated_batches,
        }


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_batch(batch: TrainingBatch, detector: DetectorConfig) -> None:
    if batch.records < 1 or batch.features.shape != (batch.records, detector.input_dimension):
        raise ValueError("Local batch has the wrong feature shape")
    if batch.targets.ndim != 1:
        raise ValueError("Local batch targets must be one-dimensional")
    if not torch.isfinite(batch.features).all().item():
        raise ValueError("Local batch contains non-finite features")
    if (
        torch.any(batch.targets < 0).item()
        or torch.any(batch.targets >= detector.num_classes).item()
    ):
        raise ValueError("Local batch contains an out-of-range class ID")


def _validate_prevalidated_batch_shape(batch: TrainingBatch, detector: DetectorConfig) -> None:
    if batch.records < 1 or batch.features.shape != (
        batch.records,
        detector.input_dimension,
    ):
        raise ValueError("Local batch has the wrong feature shape")
    if batch.targets.ndim != 1:
        raise ValueError("Local batch targets must be one-dimensional")


def train_local_client(
    *,
    detector: DetectorConfig,
    manifest: ParameterManifest,
    global_parameters: Tensor,
    round_id: int,
    client_id: str,
    declared_sample_count: int,
    class_counts: Sequence[int],
    batch_source: LocalBatchSource,
    config: LocalTrainingConfig,
    is_malicious: bool | None = None,
    attack_metadata: dict[str, object] | None = None,
) -> ClientUpdate:
    """Optimize one client from the broadcast global state and return its delta."""

    if round_id < 0 or not client_id:
        raise ValueError("round_id and client_id are invalid")
    if len(class_counts) != detector.num_classes or any(count < 0 for count in class_counts):
        raise ValueError("class_counts do not match the detector class system")
    if declared_sample_count < 1 or sum(class_counts) != declared_sample_count:
        raise ValueError("declared_sample_count differs from exact local class counts")
    if global_parameters.ndim != 1 or global_parameters.numel() != manifest.total_values:
        raise ValueError("Broadcast global parameter vector has the wrong shape")
    if not torch.isfinite(global_parameters).all().item():
        raise ValueError("Broadcast global parameter vector is non-finite")

    local_seed = derive_seed(config.seed, f"phase5-local:round={round_id}:client={client_id}")
    _set_reproducibility(local_seed)
    device = _device(config.device)
    model = IntrusionMLP(detector).to(device)
    assign_flat_parameters(model, global_parameters, manifest)
    reference_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    if config.optimizer_backend == "fused" and device.type != "cuda":
        raise RuntimeError("Fused Adam requires CUDA local training")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        fused=config.optimizer_backend == "fused",
    )
    weight_values = class_weights_from_counts(
        class_counts, ClassWeightPolicy(config.maximum_class_weight)
    )
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    total_loss = 0.0
    total_records = 0
    durations: list[float] = []

    for epoch in range(config.local_epochs):
        epoch_seed = derive_seed(local_seed, f"epoch:{epoch}")
        _set_reproducibility(epoch_seed)
        epoch_started = time.perf_counter()
        epoch_records = 0
        epoch_losses: list[Tensor] = []
        epoch_batch_records: list[int] = []
        model.train()
        for batch in batch_source(epoch, epoch_seed):
            if config.prevalidated_batches:
                _validate_prevalidated_batch_shape(batch, detector)
            else:
                _validate_batch(batch, detector)
            features = batch.features.to(device)
            targets = batch.targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = local_objective(
                model,
                logits,
                targets,
                class_weights,
                proximal_mu=config.proximal_mu,
                reference_parameters=reference_parameters,
            )
            torch.autograd.backward(loss)
            optimizer.step()
            epoch_losses.append(loss.detach())
            epoch_batch_records.append(batch.records)
            epoch_records += batch.records
        if epoch_records != declared_sample_count:
            raise ValueError("Local batch source did not emit the declared population exactly once")
        loss_values = torch.stack(epoch_losses)
        if not torch.isfinite(loss_values).all().item():
            raise RuntimeError("Non-finite local training loss")
        if not torch.isfinite(flatten_parameters(model, manifest)).all().item():
            raise RuntimeError("Non-finite local model parameter")
        for loss_value, records in zip(
            loss_values.detach().cpu().tolist(), epoch_batch_records, strict=True
        ):
            total_loss += float(loss_value) * records
            total_records += records
        durations.append(time.perf_counter() - epoch_started)

    local_parameters = flatten_parameters(model, manifest).detach().cpu()
    delta = local_parameters - global_parameters.detach().cpu().to(local_parameters.dtype)
    update_norm = float(torch.linalg.vector_norm(delta).item())
    finite = bool(torch.isfinite(delta).all().item() and math.isfinite(update_norm))
    return ClientUpdate(
        round_id=round_id,
        client_id=client_id,
        delta=delta,
        declared_sample_count=declared_sample_count,
        mean_local_loss=total_loss / total_records,
        epoch_durations_seconds=tuple(durations),
        update_norm=update_norm,
        finite=finite,
        missing_class_ids=tuple(index for index, count in enumerate(class_counts) if count == 0),
        is_malicious=is_malicious,
        attack_metadata=attack_metadata,
    )
