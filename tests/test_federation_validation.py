"""Fail-closed validation coverage for the Phase 5 engine."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from semcred_fl.baselines import coordinate_median, fedavg
from semcred_fl.data.training_corpus import TrainingBatch
from semcred_fl.federation.checkpoint import (
    load_federated_checkpoint,
    save_federated_checkpoint,
)
from semcred_fl.federation.client import (
    LocalTrainingConfig,
    _device,
    _validate_batch,
    train_local_client,
)
from semcred_fl.federation.parameters import (
    ParameterManifest,
    ParameterSpec,
    apply_flat_delta,
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.federation.sampling import ClientSchedule, generate_client_schedule
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


def _model() -> IntrusionMLP:
    torch.manual_seed(9)
    return IntrusionMLP(DetectorConfig(2, 2, (3,), 0.0))


def _defense(client_id: str, delta: torch.Tensor, count: int = 1) -> DefenseUpdate:
    return DefenseUpdate(
        round_id=0,
        client_id=client_id,
        delta=delta,
        declared_sample_count=count,
        mean_local_loss=1.0,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        finite=True,
        missing_class_ids=(),
    )


def test_parameter_manifest_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        ParameterManifest((), 0)


@pytest.mark.parametrize(
    "parameters,total,message",
    [
        ((ParameterSpec("a", (2,), 2),), 3, "cardinality"),
        ((ParameterSpec("a", (1,), 1), ParameterSpec("a", (1,), 1)), 2, "unique"),
        ((ParameterSpec("", (1,), 1),), 1, "malformed"),
        ((ParameterSpec("a", (2,), 1),), 1, "malformed"),
    ],
)
def test_parameter_manifest_validation(
    parameters: tuple[ParameterSpec, ...], total: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ParameterManifest(parameters, total)


def test_apply_delta_rejects_nonfinite() -> None:
    model = _model()
    manifest = parameter_manifest(model)
    with pytest.raises(ValueError, match="non-finite"):
        apply_flat_delta(model, torch.full((manifest.total_values,), torch.nan), manifest)


@pytest.mark.parametrize(
    "updates,message",
    [
        ((_defense("a", torch.ones(2)), _defense("a", torch.ones(2))), "unique"),
        ((_defense("a", torch.ones(2)), _defense("b", torch.ones(3))), "shapes"),
        ((_defense("a", torch.ones(2), 0),), "positive"),
        ((_defense("a", torch.tensor([float("nan")])),), "non-finite"),
    ],
)
def test_fedavg_rejects_invalid_inputs(updates: tuple[DefenseUpdate, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fedavg(updates)


@pytest.mark.parametrize(
    "updates,message",
    [
        ((_defense("a", torch.ones(2)), _defense("a", torch.ones(2))), "unique"),
        ((_defense("a", torch.ones(2)), _defense("b", torch.ones(3))), "shapes"),
        ((_defense("a", torch.tensor([float("nan")])),), "non-finite"),
    ],
)
def test_coordinate_median_rejects_invalid_inputs(
    updates: tuple[DefenseUpdate, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        coordinate_median(updates)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_epochs": 0},
        {"learning_rate": 0.0},
        {"proximal_mu": -1.0},
        {"maximum_class_weight": 0.5},
        {"seed": -1},
        {"device": "tpu"},
    ],
)
def test_local_training_config_rejects_invalid_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LocalTrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_cuda_request_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _device("cuda")
    assert _device("auto").type == "cpu"


@pytest.mark.parametrize(
    "batch,message",
    [
        (TrainingBatch(torch.ones(2, 3), torch.zeros(2, dtype=torch.long)), "feature shape"),
        (TrainingBatch(torch.ones(2, 2), torch.zeros((2, 1), dtype=torch.long)), "one-dimensional"),
        (
            TrainingBatch(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, dtype=torch.long)),
            "non-finite",
        ),
        (TrainingBatch(torch.ones(1, 2), torch.tensor([2])), "out-of-range"),
    ],
)
def test_local_batch_validation(batch: TrainingBatch, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_batch(batch, DetectorConfig(2, 2, (3,), 0.0))


class _OneBatch:
    def __call__(self, epoch: int, seed: int) -> object:
        del epoch, seed
        return iter((TrainingBatch(torch.zeros(2, 2), torch.tensor([0, 1])),))


@pytest.mark.parametrize("case", ["identity", "counts", "declared", "shape", "nonfinite"])
def test_local_training_rejects_invalid_inputs(case: str) -> None:
    model = _model()
    manifest = parameter_manifest(model)
    vector = flatten_parameters(model, manifest)
    kwargs: dict[str, object] = {
        "detector": model.config,
        "manifest": manifest,
        "global_parameters": vector,
        "round_id": 0,
        "client_id": "client",
        "declared_sample_count": 2,
        "class_counts": (1, 1),
        "batch_source": _OneBatch(),
        "config": LocalTrainingConfig(device="cpu"),
    }
    if case == "identity":
        kwargs["client_id"] = ""
    elif case == "counts":
        kwargs["class_counts"] = (1,)
    elif case == "declared":
        kwargs["declared_sample_count"] = 3
    elif case == "shape":
        kwargs["global_parameters"] = torch.zeros(2)
    else:
        kwargs["global_parameters"] = torch.full((manifest.total_values,), torch.nan)
    with pytest.raises(ValueError):
        train_local_client(**kwargs)  # type: ignore[arg-type]


def _client_update(**changes: object) -> ClientUpdate:
    values: dict[str, object] = {
        "round_id": 0,
        "client_id": "client",
        "delta": torch.tensor([1.0, 2.0]),
        "declared_sample_count": 2,
        "mean_local_loss": 1.0,
        "epoch_durations_seconds": (),
        "update_norm": float(torch.sqrt(torch.tensor(5.0)).item()),
        "finite": True,
    }
    values.update(changes)
    return ClientUpdate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "update,reason",
    [
        (_client_update(client_id=""), "empty_client_id"),
        (_client_update(delta=torch.ones(3)), "wrong_update_shape"),
        (_client_update(declared_sample_count=0), "invalid_declared_sample_count"),
        (_client_update(mean_local_loss=-1.0), "invalid_local_loss"),
        (_client_update(update_norm=-1.0), "invalid_update_norm"),
        (_client_update(update_norm=99.0), "update_norm_mismatch"),
    ],
)
def test_update_validation_rejection_variants(update: ClientUpdate, reason: str) -> None:
    result = validate_client_updates((update,), expected_round=0, expected_values=2)
    assert list(result.rejected.values()) == [reason]


def test_update_validation_rejects_invalid_expected_contract() -> None:
    with pytest.raises(ValueError, match="Expected"):
        validate_client_updates((), expected_round=-1, expected_values=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_rounds": 0},
        {"seed": -1},
        {"aggregation": "unknown"},
    ],
)
def test_federated_run_config_rejects_invalid_controls(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    values: dict[str, object] = {"output_directory": tmp_path, "total_rounds": 1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        FederatedRunConfig(**values)  # type: ignore[arg-type]


def test_schedule_validation_and_generation_fail_closed() -> None:
    valid = generate_client_schedule(("a", "b"), rounds=1, clients_per_round=1, seed=2)
    with pytest.raises(ValueError, match="identity"):
        replace(valid, schedule_id="0" * 64)
    with pytest.raises(ValueError, match="nonempty"):
        generate_client_schedule((), rounds=1, clients_per_round=1, seed=2)
    with pytest.raises(ValueError, match="controls"):
        generate_client_schedule(("a",), rounds=0, clients_per_round=1, seed=2)
    stable = valid._stable_record()
    assert stable["schema_version"] == 1
    with pytest.raises(ValueError, match="unique"):
        ClientSchedule(("a", "a"), (("a",),), 1, 2, valid.schedule_id)
    with pytest.raises(ValueError, match="at least one"):
        ClientSchedule(("a",), (), 1, 2, valid.schedule_id)


def _checkpoint_payload(model: IntrusionMLP) -> dict[str, object]:
    return {
        "checkpoint_version": 1,
        "detector_config": model.config.to_dict(),
        "parameter_manifest": parameter_manifest(model).to_dict(),
        "model_state": model.state_dict(),
        "next_round": 1,
        "run_id": "run",
        "schedule_id": "schedule",
        "last_record_sha256": "hash",
    }


def test_federated_checkpoint_round_trip_and_checksum(tmp_path: Path) -> None:
    model = _model()
    manifest = parameter_manifest(model)
    path = tmp_path / "round.pt"
    digest = save_federated_checkpoint(
        path,
        model,
        manifest,
        next_round=1,
        run_id="run",
        schedule_id="schedule",
        last_record_sha256="hash",
    )
    loaded = load_federated_checkpoint(path, expected_sha256=digest)
    assert loaded.next_round == 1 and loaded.last_record_sha256 == "hash"
    with pytest.raises(ValueError, match="SHA-256"):
        load_federated_checkpoint(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="metadata"):
        save_federated_checkpoint(
            tmp_path / "bad.pt",
            model,
            manifest,
            next_round=-1,
            run_id="run",
            schedule_id="schedule",
            last_record_sha256=None,
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("checkpoint_version", 2, "Unsupported"),
        ("detector_config", None, "model payload"),
        ("parameter_manifest", None, "manifest"),
        ("next_round", -1, "next-round"),
        ("run_id", 3, "identities"),
        ("last_record_sha256", 4, "record hash"),
        ("parameter_manifest", {"parameters": [], "total_values": 0}, "differs"),
    ],
)
def test_federated_checkpoint_rejects_malformed_payload(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _checkpoint_payload(_model())
    payload[field] = value
    path = tmp_path / f"{field}-{message}.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_federated_checkpoint(path)


def test_federated_checkpoint_rejects_non_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "list.pt"
    torch.save([1, 2], path)
    with pytest.raises(ValueError, match="dictionary"):
        load_federated_checkpoint(path)


class _NoUpdates:
    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: torch.Tensor,
        manifest: ParameterManifest,
    ) -> tuple[ClientUpdate, ...]:
        del round_id, selected_client_ids, global_parameters, manifest
        return ()


def test_server_rejects_schedule_or_call_mismatch(tmp_path: Path) -> None:
    schedule = generate_client_schedule(("a",), rounds=1, clients_per_round=1, seed=2)
    with pytest.raises(ValueError, match="round count"):
        run_federated_simulation(
            initial_model=_model(),
            schedule=schedule,
            update_provider=_NoUpdates(),
            config=FederatedRunConfig(tmp_path, total_rounds=2, seed=2),
        )
    with pytest.raises(ValueError, match="root seed"):
        run_federated_simulation(
            initial_model=_model(),
            schedule=schedule,
            update_provider=_NoUpdates(),
            config=FederatedRunConfig(tmp_path, total_rounds=1, seed=3),
        )
    with pytest.raises(ValueError, match="positive"):
        run_federated_simulation(
            initial_model=_model(),
            schedule=schedule,
            update_provider=_NoUpdates(),
            config=FederatedRunConfig(tmp_path, total_rounds=1, seed=2),
            max_rounds_this_call=0,
        )


def test_server_cached_final_and_log_corruption_detection(tmp_path: Path) -> None:
    schedule = generate_client_schedule(("a",), rounds=1, clients_per_round=1, seed=5)
    config = FederatedRunConfig(tmp_path, total_rounds=1, seed=5)
    first = run_federated_simulation(
        initial_model=_model(), schedule=schedule, update_provider=_NoUpdates(), config=config
    )
    cached = run_federated_simulation(
        initial_model=_model(), schedule=schedule, update_provider=_NoUpdates(), config=config
    )
    assert cached == first
    run_directory = tmp_path / str(first["run_id"])
    (run_directory / "federated_run.json").unlink()
    rounds_path = run_directory / "rounds.jsonl"
    record = cast(dict[str, object], json.loads(rounds_path.read_text(encoding="utf-8")))
    record["record_sha256"] = "0" * 64
    rounds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        run_federated_simulation(
            initial_model=_model(), schedule=schedule, update_provider=_NoUpdates(), config=config
        )
