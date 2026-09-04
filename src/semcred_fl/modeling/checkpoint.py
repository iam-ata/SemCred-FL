"""Checksum-bound detector checkpoints with safe tensor-only loading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from semcred_fl.identity import canonical_json
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP

_CHECKPOINT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated model and recorded training state."""

    model: IntrusionMLP
    epoch: int
    optimizer_state: dict[str, Any] | None
    metadata: dict[str, object]
    sha256: str


def save_checkpoint(
    path: Path,
    model: IntrusionMLP,
    *,
    epoch: int,
    optimizer: nn.Module | torch.optim.Optimizer | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Atomically save a checkpoint and return its SHA-256 digest."""

    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    stable_metadata = dict(metadata or {})
    canonical_json(stable_metadata)
    optimizer_state = None if optimizer is None else optimizer.state_dict()
    payload: dict[str, object] = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "detector_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "metadata": stable_metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)
    return _sha256(path)


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    """Load a validated checkpoint using PyTorch's weights-only unpickler."""

    digest = _sha256(path)
    if expected_sha256 is not None and digest.casefold() != expected_sha256.casefold():
        raise ValueError("Checkpoint SHA-256 differs from the expected digest")
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint payload is not a dictionary")
    payload = cast(dict[str, Any], raw)
    if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    config_value = payload.get("detector_config")
    state_value = payload.get("model_state")
    epoch = payload.get("epoch")
    metadata = payload.get("metadata")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(config_value, dict) or not isinstance(state_value, dict):
        raise ValueError("Checkpoint model configuration or state is invalid")
    if not isinstance(epoch, int) or epoch < 0 or not isinstance(metadata, dict):
        raise ValueError("Checkpoint epoch or metadata is invalid")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint optimizer state is invalid")
    config = DetectorConfig.from_dict(cast(dict[str, object], config_value))
    model = IntrusionMLP(config).to(device)
    model.load_state_dict(cast(dict[str, Any], state_value), strict=True)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        epoch=epoch,
        optimizer_state=cast(dict[str, Any] | None, optimizer_state),
        metadata=cast(dict[str, object], metadata),
        sha256=digest,
    )
