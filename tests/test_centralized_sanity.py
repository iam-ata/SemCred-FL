"""End-to-end bounded Phase 4 sanity baseline test."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import Phase4CorpusFixture

from semcred_fl.modeling import load_checkpoint
from semcred_fl.training import CentralizedSanityConfig, run_centralized_sanity


def test_centralized_sanity_tracks_artifacts_without_test_access(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    config = CentralizedSanityConfig(
        preprocessing_report=phase4_corpus.preprocessing_report,
        client_partition_manifest=phase4_corpus.client_partition_manifest,
        output_directory=tmp_path / "runs",
        seed=41,
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        train_record_limit=None,
        validation_record_limit=None,
        device="cpu",
        progress_every_batches=2,
    )
    report = run_centralized_sanity(config)
    assert report["test_partition_accessed"] is False
    assert report["gates"]["performance_claims_allowed"] is False
    assert len(report["history"]) == 2
    first_loss = report["history"][0]["train_loss"]
    final_loss = report["history"][-1]["train_loss"]
    assert final_loss < first_loss
    run_directory = config.output_directory / report["sanity_id"]
    checkpoint = report["best_checkpoint"]
    loaded = load_checkpoint(
        run_directory / checkpoint["path"], expected_sha256=checkpoint["sha256"]
    )
    with torch.no_grad():
        prediction = loaded.model(torch.zeros((1, 4)))
    assert prediction.shape == (1, 2)
    assert (run_directory / "centralized_sanity.json").is_file()
    assert (run_directory / "epochs.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert run_centralized_sanity(config)["sanity_id"] == report["sanity_id"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"maximum_class_weight": 0.5},
        {"train_record_limit": 1},
        {"validation_record_limit": 1},
        {"device": "tpu"},
        {"progress_every_batches": 0},
    ],
)
def test_centralized_config_rejects_invalid_controls(
    phase4_corpus: Phase4CorpusFixture,
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CentralizedSanityConfig(
            preprocessing_report=phase4_corpus.preprocessing_report,
            client_partition_manifest=phase4_corpus.client_partition_manifest,
            output_directory=tmp_path,
            **kwargs,  # type: ignore[arg-type]
        )
