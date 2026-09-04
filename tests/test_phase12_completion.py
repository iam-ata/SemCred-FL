"""Final audit and reporting gates for measured Phase 12 clean campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import Phase4CorpusFixture
from test_phase12_step3 import _upgrade_fixture
from test_phase12_step5 import _config

from semcred_fl.data.contracts import DataContractError
from semcred_fl.experiments.phase12_clean_matrix import (
    Phase12CleanMatrixRequest,
    run_phase12_clean_matrix,
)
from semcred_fl.experiments.phase12_completion import (
    audit_phase12_clean_campaign,
    finalize_phase12_clean_campaign,
)


def _campaign(
    fixture: Phase4CorpusFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    prior_path = fixture.preprocessing_report.parent / "semantic_prior_bundle.json"
    if not prior_path.exists():
        prior_path, _ = _upgrade_fixture(fixture)
    monkeypatch.setattr(
        "semcred_fl.experiments.phase12_clean_matrix._artifact_paths",
        lambda project_root, dataset: (
            fixture.preprocessing_report,
            fixture.client_partition_manifest,
            prior_path,
        ),
    )
    config = _config()
    output = tmp_path / "measured"
    report = run_phase12_clean_matrix(
        Phase12CleanMatrixRequest(
            project_root=tmp_path,
            output_directory=output,
            cache_root=tmp_path / "cache",
            datasets=("edge_iiotset",),
            profiles=("dirichlet_0_3",),
            seeds=(17,),
            device="cpu",
            config_overrides={
                "total_clients": config.total_clients,
                "clients_per_round": config.clients_per_round,
                "total_rounds": config.total_rounds,
                "local_epochs": config.local_epochs,
                "batch_size": config.batch_size,
                "hidden_dimensions": config.hidden_dimensions,
                "dropout": config.dropout,
                "burn_in_rounds": config.burn_in_rounds,
                "validation_batch_size": config.validation_batch_size,
                "cache_scan_batch_size": config.cache_scan_batch_size,
                "trim_count": config.trim_count,
                "krum_byzantine_count": config.krum_byzantine_count,
                "krum_selected_count": config.krum_selected_count,
                "prefer_fused_adam": False,
            },
        )
    )
    return output / str(report["matrix_input_id"])


def test_completion_audits_and_finalizes_development_registry(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(phase4_corpus, tmp_path, monkeypatch)
    audit, records = audit_phase12_clean_campaign(
        campaign,
        require_confirmatory_registry=False,
    )
    assert audit["status"] == "passed"
    assert len(records) == 35

    result = finalize_phase12_clean_campaign(
        campaign,
        require_confirmatory_registry=False,
        bootstrap_resamples=40,
        bootstrap_seed=5,
    )
    output = campaign / "final_reporting"
    assert result["status"] == "clean_matrix_reporting_complete"
    assert (output / "campaign_audit.json").is_file()
    assert (output / "paired_statistics.json").is_file()
    assert (output / "macro_f1_table.csv").is_file()
    assert (output / "macro_f1_table.tex").is_file()
    assert (output / "macro_f1_plot.svg").is_file()
    assert (
        json.loads((output / "final_manifest.json").read_text())["gates"]["manuscript_modified"]
        is False
    )
    assert (
        finalize_phase12_clean_campaign(
            campaign,
            require_confirmatory_registry=False,
            bootstrap_resamples=40,
            bootstrap_seed=5,
        )
        == result
    )


def test_completion_rejects_reduced_registry_by_default(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(phase4_corpus, tmp_path, monkeypatch)
    with pytest.raises(DataContractError, match="confirmatory registry"):
        audit_phase12_clean_campaign(campaign)


def test_completion_rejects_claim_enabled_method_report(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(phase4_corpus, tmp_path, monkeypatch)
    method = next(campaign.glob("conditions/*/*/seed_*/methods/*/method_report.json"))
    value = json.loads(method.read_text(encoding="utf-8"))
    value["gates"]["performance_claims_allowed"] = True
    method.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataContractError):
        audit_phase12_clean_campaign(campaign, require_confirmatory_registry=False)
