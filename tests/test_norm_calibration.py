"""Integration tests for resumable Phase 6B.2 benign norm calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import yaml
from conftest import Phase4CorpusFixture

from semcred_fl.attacks.norm_calibration import (
    Phase6BNormCalibrationRequest,
    phase6b_norm_calibration_config,
    run_phase6b_norm_calibration,
)
from semcred_fl.identity import to_primitive


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upgrade_fixture_identities(fixture: Phase4CorpusFixture) -> None:
    corpus_id = "a" * 64
    preprocessor_id = "b" * 64
    split_id = "c" * 64
    anchor_id = "d" * 64
    partition_id = "e" * 64
    report = cast(
        dict[str, object], json.loads(fixture.preprocessing_report.read_text(encoding="utf-8"))
    )
    preprocessor_record = cast(dict[str, object], report["preprocessor"])
    preprocessor_path = fixture.preprocessing_report.parent / cast(str, preprocessor_record["path"])
    preprocessor = cast(
        dict[str, object], json.loads(preprocessor_path.read_text(encoding="utf-8"))
    )
    preprocessor.update(
        {
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    _write_json(preprocessor_path, preprocessor)
    report.update(
        {
            "preprocessed_corpus_id": corpus_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    preprocessor_record.update(
        {"preprocessor_id": preprocessor_id, "sha256": _sha256(preprocessor_path)}
    )
    _write_json(fixture.preprocessing_report, report)
    manifest = cast(
        dict[str, object],
        json.loads(fixture.client_partition_manifest.read_text(encoding="utf-8")),
    )
    manifest.update(
        {
            "client_partition_id": partition_id,
            "preprocessed_corpus_id": corpus_id,
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    _write_json(fixture.client_partition_manifest, manifest)


def test_norm_calibration_yaml_matches_typed_contract() -> None:
    config_path = Path("configs/attacks/phase6b_norm_calibration.yaml")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert loaded == to_primitive(phase6b_norm_calibration_config())


def test_norm_calibration_resumes_and_freezes_exact_iqr(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    _upgrade_fixture_identities(phase4_corpus)
    config = replace(
        phase6b_norm_calibration_config(),
        profiles=("iid",),
        total_rounds=2,
        clients_per_round=2,
        local_epochs=1,
        batch_size=8,
        hidden_dimensions=(8,),
        dropout=0.0,
    )
    output = tmp_path / "norms"
    cache = tmp_path / "cache"
    first = run_phase6b_norm_calibration(
        Phase6BNormCalibrationRequest(
            preprocessing_report=phase4_corpus.preprocessing_report,
            client_partition_manifest=phase4_corpus.client_partition_manifest,
            profile="iid",
            output_directory=output,
            cache_directory=cache,
            cache_work_database=cache / "cache.sqlite3",
            device="cpu",
            cache_scan_batch_size=17,
            max_rounds_this_call=1,
            config=config,
        )
    )
    assert first["status"] == "interrupted"
    assert first["next_round"] == 1

    request = Phase6BNormCalibrationRequest(
        preprocessing_report=phase4_corpus.preprocessing_report,
        client_partition_manifest=phase4_corpus.client_partition_manifest,
        profile="iid",
        output_directory=output,
        cache_directory=cache,
        cache_work_database=cache / "cache.sqlite3",
        device="cpu",
        cache_scan_batch_size=17,
        config=config,
    )
    completed = run_phase6b_norm_calibration(request)
    repeated = run_phase6b_norm_calibration(request)
    assert completed == repeated
    reference = cast(dict[str, object], completed["reference"])
    assert reference["update_count"] == 4
    assert isinstance(reference["lower_norm"], float)
    assert isinstance(reference["upper_norm"], float)
    assert cast(float, reference["lower_norm"]) <= cast(float, reference["upper_norm"])
    records = cast(list[dict[str, object]], completed["update_norm_records"])
    assert len(records) == 4
    gates = cast(dict[str, object], completed["gates"])
    assert gates["all_scheduled_updates_accepted"] is True
    assert gates["all_development_clients_honest"] is True
    assert gates["test_partition_accessed"] is False
    assert gates["anchor_records_accessed"] is False
    assert gates["performance_claims_allowed"] is False
