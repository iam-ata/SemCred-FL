"""Tests for the sequential resumable Phase 6C matrix runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import semcred_fl.attacks.matrix as matrix_module
from semcred_fl.attacks.execution import Phase6CAttackRequest
from semcred_fl.attacks.matrix import (
    Phase6CMatrixRequest,
    condition_relative_directory,
    inspect_phase6c_matrix,
    phase6c_matrix_specs,
    run_phase6c_matrix,
)


def _write_checkpoint(output: Path, next_round: int) -> None:
    path = output / "federation" / ("f" * 64) / "latest_checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next_round": next_round}), encoding="utf-8")


def test_all_types_matrix_contains_control_and_every_family() -> None:
    specs = phase6c_matrix_specs(
        dataset="edge_iiotset",
        profile="iid",
        malicious_ratio=0.1,
        seed=20260816,
        matrix="all_types",
    )
    assert len(specs) == 9
    assert {spec.attack_name for spec in specs} == {
        "clean",
        "label_flip",
        "clean_label",
        "backdoor",
        "scaling",
        "sign_reversal",
        "model_replacement",
        "adaptive_norm",
    }
    assert specs[0].attack_name == "clean"
    assert specs[1].attack_name == "label_flip"
    assert specs[1].poison_fraction == 0.0
    assert len({spec.attack_id(matrix_module.phase6c_execution_config()) for spec in specs}) == 9
    assert condition_relative_directory(specs[0]) == Path(
        "edge_iiotset/iid/clean/mr_0_1/seed_20260816"
    )


def test_full_grid_contains_every_frozen_parameter_combination() -> None:
    specs = phase6c_matrix_specs(
        dataset="edge_iiotset",
        profile="iid",
        malicious_ratio=0.1,
        seed=7,
        matrix="full_grid",
    )
    assert len(specs) == 25
    assert sum(spec.attack_name == "clean_label" for spec in specs) == 6
    assert sum(spec.attack_name == "label_flip" for spec in specs) == 4
    assert sum(spec.attack_name == "adaptive_norm" for spec in specs) == 3


def test_matrix_skips_existing_target_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "phase6c"
    clean = phase6c_matrix_specs(
        dataset="edge_iiotset",
        profile="iid",
        malicious_ratio=0.1,
        seed=17,
        matrix="all_types",
    )[0]
    _write_checkpoint(output_root / condition_relative_directory(clean), 1)
    calls: list[str] = []

    def fake_run(request: Phase6CAttackRequest) -> dict[str, object]:
        calls.append(request.spec.attack_id(request.config))
        _write_checkpoint(request.output_directory, 1)
        return {"federated_run_id": "f" * 64, "status": "interrupted"}

    monkeypatch.setattr(matrix_module, "run_phase6c_attack", fake_run)
    request = Phase6CMatrixRequest(
        preprocessing_report=tmp_path / "preprocessing.json",
        client_partition_manifest=tmp_path / "clients.json",
        resolution_path=tmp_path / "resolution.json",
        output_root=output_root,
        cache_root=tmp_path / "cache",
        norm_reference_root=tmp_path / "norms",
        dataset="edge_iiotset",
        profiles=("iid",),
        malicious_ratios=(0.1,),
        seeds=(17,),
        target_round=1,
    )
    plan = inspect_phase6c_matrix(request)
    assert plan["conditions_total"] == 9
    assert plan["conditions_at_target"] == 1
    assert plan["conditions_pending"] == 8
    first = run_phase6c_matrix(request)
    assert first["status"] == "target_reached"
    assert first["conditions_total"] == 9
    assert first["conditions_at_target"] == 9
    assert len(calls) == 8
    first_conditions = cast(list[dict[str, object]], first["conditions"])
    assert first_conditions[0]["rounds_at_target"] == 1

    second = run_phase6c_matrix(request)
    assert second == first
    assert len(calls) == 8
