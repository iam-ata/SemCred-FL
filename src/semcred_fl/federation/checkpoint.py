"""Checksum-bound checkpoints for exact federated-round resumption."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from semcred_fl.federation.parameters import ParameterManifest, parameter_manifest
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP

_CHECKPOINT_VERSION = 1


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedFederatedCheckpoint:
    """Validated model and exact round-resume cursor."""

    model: IntrusionMLP
    manifest: ParameterManifest
    next_round: int
    run_id: str
    schedule_id: str
    last_record_sha256: str | None
    sha256: str


def save_federated_checkpoint(
    path: Path,
    model: IntrusionMLP,
    manifest: ParameterManifest,
    *,
    next_round: int,
    run_id: str,
    schedule_id: str,
    last_record_sha256: str | None,
) -> str:
    """Atomically persist a federated model and deterministic resume state."""

    if next_round < 0 or not run_id or not schedule_id:
        raise ValueError("Federated checkpoint metadata is invalid")
    payload: dict[str, object] = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "detector_config": model.config.to_dict(),
        "parameter_manifest": manifest.to_dict(),
        "model_state": model.state_dict(),
        "next_round": next_round,
        "run_id": run_id,
        "schedule_id": schedule_id,
        "last_record_sha256": last_record_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)
    return file_sha256(path)


def load_federated_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> LoadedFederatedCheckpoint:
    """Load a tensor-only federated checkpoint and validate its layout."""

    digest = file_sha256(path)
    if expected_sha256 is not None and digest.casefold() != expected_sha256.casefold():
        raise ValueError("Federated checkpoint SHA-256 differs from the expected digest")
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("Federated checkpoint payload is not a dictionary")
    payload = cast(dict[str, Any], raw)
    if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise ValueError("Unsupported federated checkpoint version")
    config_value = payload.get("detector_config")
    state_value = payload.get("model_state")
    manifest_value = payload.get("parameter_manifest")
    next_round = payload.get("next_round")
    run_identity = payload.get("run_id")
    schedule_id = payload.get("schedule_id")
    last_hash = payload.get("last_record_sha256")
    if not isinstance(config_value, dict) or not isinstance(state_value, dict):
        raise ValueError("Federated checkpoint model payload is invalid")
    if not isinstance(manifest_value, dict):
        raise ValueError("Federated checkpoint parameter manifest is missing")
    if not isinstance(next_round, int) or next_round < 0:
        raise ValueError("Federated checkpoint next-round cursor is invalid")
    if not isinstance(run_identity, str) or not isinstance(schedule_id, str):
        raise ValueError("Federated checkpoint identities are invalid")
    if last_hash is not None and not isinstance(last_hash, str):
        raise ValueError("Federated checkpoint round-record hash is invalid")
    config = DetectorConfig.from_dict(cast(dict[str, object], config_value))
    model = IntrusionMLP(config).to(device)
    model.load_state_dict(cast(dict[str, Any], state_value), strict=True)
    model.eval()
    manifest = parameter_manifest(model)
    if manifest.to_dict() != cast(dict[str, object], manifest_value):
        raise ValueError("Federated checkpoint parameter manifest differs from the model")
    return LoadedFederatedCheckpoint(
        model=model,
        manifest=manifest,
        next_round=next_round,
        run_id=run_identity,
        schedule_id=schedule_id,
        last_record_sha256=last_hash,
        sha256=digest,
    )
