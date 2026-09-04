"""Deterministic and leakage-safe gates for the Phase 12 synthetic smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semcred_fl.evaluation.reporting import read_metric_jsonl
from semcred_fl.experiments.phase12_smoke import (
    Phase12SmokeConfig,
    _write_immutable,
    run_phase12_smoke,
)


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("phase12-smoke")
    report = run_phase12_smoke(Phase12SmokeConfig(output_directory=root))
    return root, report


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"rounds": 1},
        {"clients": 3},
        {"malicious_client_id": "client-99"},
        {"malicious_round": 3},
        {"sign_reversal_scale": 0.0},
        {"sign_reversal_scale": float("nan")},
        {"attribution_interval": 0},
        {"device": "tpu"},
    ],
)
def test_smoke_config_rejects_invalid_controls(tmp_path: Path, change: dict[str, object]) -> None:
    values: dict[str, object] = {"output_directory": tmp_path}
    values.update(change)
    with pytest.raises(ValueError):
        Phase12SmokeConfig(**values)  # type: ignore[arg-type]


def test_smoke_exercises_all_required_pipeline_boundaries(
    completed_smoke: tuple[Path, dict[str, object]],
) -> None:
    _, report = completed_smoke
    assert report["schema"] == "semcred_phase12_synthetic_smoke_result_v1"
    assert report["status"] == "complete"
    rounds = report["rounds"]
    gates = report["gates"]
    assert isinstance(rounds, list) and len(rounds) == 3
    assert isinstance(gates, dict)
    assert [round_record["burn_in"] for round_record in rounds] == [True, False, False]
    assert [round_record["attribution_enabled"] for round_record in rounds] == [
        True,
        False,
        True,
    ]
    assert [round_record["malicious_client_ids"] for round_record in rounds] == [
        [],
        ["client-03"],
        [],
    ]
    assert gates == {
        "synthetic_only": True,
        "defense_inputs_exclude_attack_metadata": True,
        "scoring_anchor_used": True,
        "validation_partition_accessed": False,
        "synthetic_test_partition_accessed": True,
        "real_test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    for round_record in rounds:
        assert "attack_metadata" not in round_record
        assert "is_malicious" not in round_record
        assert len(round_record["participating_trust"]) == 4


def test_smoke_writes_traceable_nonclaim_metrics_and_phase11_outputs(
    completed_smoke: tuple[Path, dict[str, object]],
) -> None:
    root, report = completed_smoke
    smoke_run_id = report["smoke_run_id"]
    result_id = report["result_id"]
    assert isinstance(smoke_run_id, str) and isinstance(result_id, str)
    run_directory = root / smoke_run_id
    records = read_metric_jsonl(run_directory / "run_metrics.jsonl")
    assert len(records) == 13
    assert {record.run_id for record in records} == {smoke_run_id}
    assert {record.source_artifact_id for record in records} == {result_id}
    assert all(record.test_partition_accessed for record in records)
    assert not any(record.performance_claims_allowed for record in records)
    reporting = run_directory / "reporting"
    expected = {
        "metric_summaries.json",
        "metric_provenance.json",
        "macro_f1_table.csv",
        "macro_f1_table.tex",
        "macro_f1_plot.svg",
        "macro_f1_artifact_manifest.json",
    }
    assert expected <= {path.name for path in reporting.iterdir()}


def test_completed_smoke_is_idempotent_and_report_is_immutable(
    completed_smoke: tuple[Path, dict[str, object]],
) -> None:
    root, first = completed_smoke
    report_path = root / str(first["smoke_run_id"]) / "smoke_report.json"
    before = report_path.read_bytes()
    second = run_phase12_smoke(Phase12SmokeConfig(output_directory=root))
    assert second == first
    assert report_path.read_bytes() == before


def test_existing_malformed_smoke_report_fails_closed(
    tmp_path: Path,
    completed_smoke: tuple[Path, dict[str, object]],
) -> None:
    _, report = completed_smoke
    run_directory = tmp_path / str(report["smoke_run_id"])
    run_directory.mkdir(parents=True)
    (run_directory / "smoke_report.json").write_text(
        json.dumps({"smoke_run_id": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed"):
        run_phase12_smoke(Phase12SmokeConfig(output_directory=tmp_path))


def test_immutable_writer_accepts_identical_content_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    _write_immutable(path, b"frozen")
    _write_immutable(path, b"frozen")
    with pytest.raises(ValueError, match="differs"):
        _write_immutable(path, b"changed")
