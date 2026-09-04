"""Fast gates for the Phase 12 Step 4 reduced baseline matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from conftest import Phase4CorpusFixture
from test_phase12_step3 import _upgrade_fixture

from semcred_fl.data.contracts import DataContractError
from semcred_fl.experiments.phase12_attack import Phase12AttackConfig, Phase12AttackRequest
from semcred_fl.experiments.phase12_baselines import (
    Phase12BaselineConfig,
    Phase12BaselineRequest,
    _aggregate,
    _load_json,
    _metric_records,
    run_phase12_reduced_baselines,
)
from semcred_fl.federation.updates import DefenseUpdate


def _step3_request(fixture: Phase4CorpusFixture, tmp_path: Path) -> Phase12AttackRequest:
    prior_path, resolution_path = _upgrade_fixture(fixture)
    config = Phase12AttackConfig(
        seed=17,
        malicious_ratio=0.2,
        poison_fraction=0.5,
        total_clients=5,
        clients_per_round=5,
        total_rounds=2,
        local_epochs=1,
        batch_size=8,
        hidden_dimensions=(8,),
        dropout=0.0,
        burn_in_rounds=1,
        validation_batch_size=16,
        cache_scan_batch_size=17,
        prefer_fused_adam=False,
    )
    cache = tmp_path / "cache"
    return Phase12AttackRequest(
        preprocessing_report=fixture.preprocessing_report,
        client_partition_manifest=fixture.client_partition_manifest,
        resolution_path=resolution_path,
        semantic_prior_path=prior_path,
        output_directory=tmp_path / "step3",
        cache_directory=cache,
        cache_work_database=cache / "cache.sqlite3",
        device="cpu",
        config=config,
    )


def _baseline_config(attack: Phase12AttackConfig) -> Phase12BaselineConfig:
    return Phase12BaselineConfig(
        attack=attack,
        trim_count=1,
        krum_byzantine_count=1,
        krum_selected_count=4,
    )


def _request(
    step3: Phase12AttackRequest,
    tmp_path: Path,
    *,
    max_rounds: int | None,
) -> Phase12BaselineRequest:
    return Phase12BaselineRequest(
        step3_request=step3,
        output_directory=tmp_path / "step4",
        max_rounds_this_call=max_rounds,
        config=_baseline_config(step3.config),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"trim_count": 3},
        {"krum_byzantine_count": 2},
        {"krum_selected_count": 5},
    ],
)
def test_step4_config_rejects_inadmissible_controls(change: dict[str, object]) -> None:
    attack = Phase12AttackConfig(
        total_clients=5,
        clients_per_round=5,
        malicious_ratio=0.2,
        total_rounds=2,
    )
    with pytest.raises(ValueError):
        Phase12BaselineConfig(attack=attack, **change)  # type: ignore[arg-type]


def test_step4_config_rejects_excessive_krum_selection() -> None:
    attack = Phase12AttackConfig(
        total_clients=5,
        clients_per_round=5,
        malicious_ratio=0.2,
        total_rounds=2,
    )
    with pytest.raises(ValueError, match="selected count"):
        Phase12BaselineConfig(
            attack=attack,
            trim_count=1,
            krum_byzantine_count=1,
            krum_selected_count=5,
        )


def test_step4_json_and_metric_contracts_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(DataContractError, match="Cannot read"):
        _load_json(missing, "fixture")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    with pytest.raises(DataContractError, match="JSON object"):
        _load_json(sequence, "fixture")
    with pytest.raises(DataContractError, match="method metrics"):
        _metric_records(({},), seed=1, matrix_result_id="result")
    malformed = {
        "method": "fedavg",
        "method_run_id": "run",
        "validation_metrics": {"accuracy": True},
    }
    with pytest.raises(DataContractError, match="validation metric"):
        _metric_records((malformed,), seed=1, matrix_result_id="result")


def test_step4_request_rejects_mismatched_or_unsafe_paths(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    step3 = _step3_request(phase4_corpus, tmp_path)
    config = _baseline_config(step3.config)
    with pytest.raises(ValueError, match="exact Step 3"):
        Phase12BaselineRequest(
            step3,
            tmp_path / "step4",
            config=Phase12BaselineConfig(),
        )
    bounded_step3 = Phase12AttackRequest(
        preprocessing_report=step3.preprocessing_report,
        client_partition_manifest=step3.client_partition_manifest,
        resolution_path=step3.resolution_path,
        semantic_prior_path=step3.semantic_prior_path,
        output_directory=step3.output_directory,
        cache_directory=step3.cache_directory,
        cache_work_database=step3.cache_work_database,
        device=step3.device,
        max_rounds_this_call=1,
        config=step3.config,
    )
    with pytest.raises(ValueError, match="unrestricted"):
        Phase12BaselineRequest(bounded_step3, tmp_path / "step4", config=config)
    with pytest.raises(ValueError, match="overwrite"):
        Phase12BaselineRequest(step3, step3.output_directory, config=config)
    with pytest.raises(ValueError, match="maximum rounds"):
        Phase12BaselineRequest(step3, tmp_path / "step4", 0, config)


def _update(client_id: str, values: list[float]) -> DefenseUpdate:
    return DefenseUpdate(
        round_id=0,
        client_id=client_id,
        delta=torch.tensor(values),
        declared_sample_count=1,
        mean_local_loss=0.1,
        update_norm=float(torch.linalg.vector_norm(torch.tensor(values)).item()),
        finite=True,
        missing_class_ids=(),
    )


def test_step4_dispatches_all_standard_aggregators() -> None:
    updates = tuple(
        _update(f"client-{index:02d}", [float(index), float(index + 1)]) for index in range(5)
    )
    attack = Phase12AttackConfig(
        total_clients=5,
        clients_per_round=5,
        malicious_ratio=0.2,
        total_rounds=2,
    )
    config = _baseline_config(attack)
    assert _aggregate("coordinate_median", updates, config).tolist() == [2.0, 3.0]
    assert _aggregate("trimmed_mean", updates, config).tolist() == [2.0, 3.0]
    assert _aggregate("multi_krum", updates, config).shape == (2,)
    with pytest.raises(ValueError, match="Unknown"):
        _aggregate("fedavg", updates, config)


def test_step4_resumes_reuses_step3_and_emits_five_method_matrix(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    step3 = _step3_request(phase4_corpus, tmp_path)
    interrupted = run_phase12_reduced_baselines(_request(step3, tmp_path, max_rounds=1))
    assert interrupted["status"] == "interrupted"
    methods = cast(list[dict[str, object]], interrupted["methods"])
    assert {item["next_round"] for item in methods} == {1}

    completed = run_phase12_reduced_baselines(_request(step3, tmp_path, max_rounds=None))
    assert completed["status"] == "complete"
    results = cast(list[dict[str, object]], completed["method_results"])
    assert [item["method"] for item in results] == [
        "fedavg",
        "coordinate_median",
        "trimmed_mean",
        "multi_krum",
        "semcred_fl",
    ]
    assert sum(bool(item["source_step3"]) for item in results) == 2
    assert all(
        cast(dict[str, object], item["validation_metrics"])["records"] == 40 for item in results
    )
    lines = (tmp_path / "step4" / "run_metrics.jsonl").read_text(encoding="utf-8")
    assert len(lines.splitlines()) == 35
    runtime = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "step4" / "methods").glob("*/runtime/round-*.json")
    ]
    assert sum(item["updates_reused"] for item in runtime) >= 15
    round_record = json.loads(
        (tmp_path / "step4/methods/coordinate_median/rounds/round-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert "malicious" not in json.dumps(
        {key: value for key, value in round_record.items() if key != "post_decision_attack_audit"}
    )
    assert (tmp_path / "step4/reporting/macro_f1_plot.svg").is_file()

    repeated = run_phase12_reduced_baselines(_request(step3, tmp_path, max_rounds=None))
    assert repeated == completed

    (tmp_path / "step4/comparison_report.json").unlink()
    rebuilt = run_phase12_reduced_baselines(_request(step3, tmp_path, max_rounds=None))
    assert rebuilt["matrix_result_id"] == completed["matrix_result_id"]


def test_step4_requires_completed_step3(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step3 = _step3_request(phase4_corpus, tmp_path)
    monkeypatch.setattr(
        "semcred_fl.experiments.phase12_baselines.run_phase12_attack_comparison",
        lambda request: {"status": "interrupted"},
    )
    with pytest.raises(DataContractError, match="completed Step 3"):
        run_phase12_reduced_baselines(_request(step3, tmp_path, max_rounds=None))
