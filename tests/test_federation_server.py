"""Integration gates for deterministic multi-round execution and exact resume."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from semcred_fl.federation.parameters import ParameterManifest
from semcred_fl.federation.sampling import generate_client_schedule
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation
from semcred_fl.federation.updates import ClientUpdate
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.seeding import derive_seed


class _Provider:
    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: torch.Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        assert global_parameters.numel() == manifest.total_values
        updates: list[ClientUpdate] = []
        for client_id in selected_client_ids:
            generator = torch.Generator().manual_seed(
                derive_seed(91, f"fixture:{round_id}:{client_id}")
            )
            delta = torch.randn(manifest.total_values, generator=generator) * 0.001
            updates.append(
                ClientUpdate(
                    round_id=round_id,
                    client_id=client_id,
                    delta=delta,
                    declared_sample_count=int(client_id[-1]) + 1,
                    mean_local_loss=1.0 / (round_id + 1),
                    epoch_durations_seconds=(0.01,),
                    update_norm=float(torch.linalg.vector_norm(delta).item()),
                    finite=True,
                    is_malicious=True,
                    attack_metadata={"hidden": "simulation-only"},
                )
            )
        return updates


def _model() -> IntrusionMLP:
    torch.manual_seed(123)
    return IntrusionMLP(DetectorConfig(2, 2, (4,), 0.0))


def _run(output: Path, *, max_rounds: int | None = None) -> dict[str, object]:
    schedule = generate_client_schedule(
        ("client-0", "client-1", "client-2", "client-3"),
        rounds=4,
        clients_per_round=2,
        seed=77,
    )
    return run_federated_simulation(
        initial_model=_model(),
        schedule=schedule,
        update_provider=_Provider(),
        config=FederatedRunConfig(output, total_rounds=4, aggregation="fedavg", seed=77),
        max_rounds_this_call=max_rounds,
    )


def _round_log(output: Path, run_id: object) -> bytes:
    return (output / str(run_id) / "rounds.jsonl").read_bytes()


def test_multi_round_run_is_deterministic_and_hides_attack_metadata(tmp_path: Path) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = _run(first_output)
    second = _run(second_output)

    assert first["status"] == second["status"] == "complete"
    assert first["final_parameters_sha256"] == second["final_parameters_sha256"]
    assert first["last_record_sha256"] == second["last_record_sha256"]
    assert _round_log(first_output, first["run_id"]) == _round_log(second_output, second["run_id"])
    assert b"attack_metadata" not in _round_log(first_output, first["run_id"])
    assert first["gates"]["test_partition_accessed"] is False


def test_resume_matches_uninterrupted_run_exactly(tmp_path: Path) -> None:
    uninterrupted_output = tmp_path / "uninterrupted"
    resumed_output = tmp_path / "resumed"
    uninterrupted = _run(uninterrupted_output)
    interrupted = _run(resumed_output, max_rounds=2)

    assert interrupted["status"] == "interrupted"
    assert interrupted["next_round"] == 2
    resumed = _run(resumed_output)
    assert resumed["final_parameters_sha256"] == uninterrupted["final_parameters_sha256"]
    assert resumed["last_record_sha256"] == uninterrupted["last_record_sha256"]
    assert _round_log(resumed_output, resumed["run_id"]) == _round_log(
        uninterrupted_output, uninterrupted["run_id"]
    )
    pointer = json.loads(
        (resumed_output / str(resumed["run_id"]) / "latest_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["next_round"] == 4


class _InvalidProvider:
    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: torch.Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        del selected_client_ids, global_parameters
        delta = torch.full((manifest.total_values,), torch.nan)
        return (
            ClientUpdate(
                round_id=round_id,
                client_id="client-0",
                delta=delta,
                declared_sample_count=1,
                mean_local_loss=1.0,
                epoch_durations_seconds=(),
                update_norm=float("nan"),
                finite=False,
            ),
        )


def test_all_invalid_updates_keep_global_model_unchanged(tmp_path: Path) -> None:
    schedule = generate_client_schedule(("client-0",), rounds=1, clients_per_round=1, seed=3)
    report = run_federated_simulation(
        initial_model=_model(),
        schedule=schedule,
        update_provider=_InvalidProvider(),
        config=FederatedRunConfig(tmp_path, total_rounds=1, seed=3),
    )
    record = json.loads(
        (tmp_path / str(report["run_id"]) / "rounds.jsonl").read_text(encoding="utf-8")
    )
    assert record["round_failed"] is True
    assert record["rejected_updates"] == {"client-0": "nonfinite_update"}
    assert report["initial_parameters_sha256"] == report["final_parameters_sha256"]


def test_schedule_is_reproducible_and_round_unique() -> None:
    first = generate_client_schedule(("c", "a", "b"), rounds=5, clients_per_round=2, seed=8)
    second = generate_client_schedule(("b", "c", "a"), rounds=5, clients_per_round=2, seed=8)

    assert first == second
    assert all(len(set(selected)) == 2 for selected in first.rounds)
