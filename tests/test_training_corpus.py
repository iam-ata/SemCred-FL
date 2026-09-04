"""Streaming and identity tests for Phase 4 corpus loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import Phase4CorpusFixture

import semcred_fl.data.training_corpus as training_corpus_module
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.training_corpus import (
    iter_client_batches,
    iter_partition_batches,
    load_client_partition_profile,
    load_preprocessed_corpus,
    load_training_class_counts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_preprocessor(fixture: Phase4CorpusFixture, update: dict[str, object]) -> None:
    report = json.loads(fixture.preprocessing_report.read_text(encoding="utf-8"))
    preprocessor_path = fixture.preprocessing_report.parent / report["preprocessor"]["path"]
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    preprocessor.update(update)
    preprocessor_path.write_text(json.dumps(preprocessor), encoding="utf-8")
    report["preprocessor"]["sha256"] = _sha256(preprocessor_path)
    fixture.preprocessing_report.write_text(json.dumps(report), encoding="utf-8")


def test_corpus_loader_binds_phase3c_and_phase3d(phase4_corpus: Phase4CorpusFixture) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    assert corpus.input_dimension == 4
    assert corpus.num_classes == 2
    assert corpus.partition_counts["test"] == 20
    assert load_training_class_counts(phase4_corpus.client_partition_manifest, corpus) == (
        40,
        40,
    )


def test_partition_stream_respects_boundary_limit_and_seed(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    batches = list(
        iter_partition_batches(
            corpus,
            "train",
            batch_size=17,
            record_limit=31,
            shuffle_seed=91,
        )
    )
    repeated = list(
        iter_partition_batches(
            corpus,
            "train",
            batch_size=17,
            record_limit=31,
            shuffle_seed=91,
        )
    )
    assert sum(batch.records for batch in batches) == 31
    assert all(batch.features.shape[1] == 4 for batch in batches)
    assert all(
        batch.targets.min().item() >= 0 and batch.targets.max().item() <= 1 for batch in batches
    )
    assert len(batches) == len(repeated)
    assert all(
        first.features.equal(second.features) and first.targets.equal(second.targets)
        for first, second in zip(batches, repeated, strict=True)
    )


def test_phase3d_client_stream_matches_exact_profile_counts(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    profile = load_client_partition_profile(phase4_corpus.client_partition_manifest, corpus, "iid")
    batches = list(iter_client_batches(corpus, profile, 0, batch_size=13, shuffle_seed=20260816))
    repeated = list(iter_client_batches(corpus, profile, 0, batch_size=13, shuffle_seed=20260816))

    assert profile.client_partition_id == "clients-test"
    assert profile.class_counts(0) == (20, 20)
    assert sum(batch.records for batch in batches) == 40
    assert sum(int((batch.targets == 0).sum()) for batch in batches) == 20
    assert all(
        left.features.equal(right.features) and left.targets.equal(right.targets)
        for left, right in zip(batches, repeated, strict=True)
    )


def test_phase3d_client_profile_rejects_unknown_profile_or_bad_checksum(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    with pytest.raises(DataContractError, match="profile"):
        load_client_partition_profile(phase4_corpus.client_partition_manifest, corpus, "missing")
    manifest = json.loads(phase4_corpus.client_partition_manifest.read_text(encoding="utf-8"))
    manifest["assignment_files"][0]["sha256"] = "0" * 64
    phase4_corpus.client_partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="checksum"):
        load_client_partition_profile(phase4_corpus.client_partition_manifest, corpus, "iid")


def test_loader_fails_closed_on_identity_or_gate_change(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    manifest = json.loads(phase4_corpus.client_partition_manifest.read_text(encoding="utf-8"))
    manifest["preprocessor_id"] = "different"
    phase4_corpus.client_partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    with pytest.raises(DataContractError, match="does not bind"):
        load_training_class_counts(phase4_corpus.client_partition_manifest, corpus)


def test_stream_rejects_unknown_partition(phase4_corpus: Phase4CorpusFixture) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    with pytest.raises(ValueError, match="Unknown partition"):
        list(iter_partition_batches(corpus, "holdout", batch_size=8))
    with pytest.raises(ValueError, match="positive"):
        list(iter_partition_batches(corpus, "train", batch_size=0))
    with pytest.raises(ValueError, match="positive"):
        list(iter_partition_batches(corpus, "train", batch_size=8, record_limit=0))


def test_corpus_loader_rejects_failed_gate_and_bad_preprocessor_hash(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    report = json.loads(phase4_corpus.preprocessing_report.read_text(encoding="utf-8"))
    report["gates"]["no_nonfinite_model_input"] = False
    phase4_corpus.preprocessing_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataContractError, match="gates"):
        load_preprocessed_corpus(phase4_corpus.preprocessing_report)


def test_corpus_loader_rejects_bad_family_map_and_partition_counts(
    phase4_corpus: Phase4CorpusFixture,
) -> None:
    _rewrite_preprocessor(phase4_corpus, {"family_label_to_id": {"benign": 1, "attack": 2}})
    with pytest.raises(DataContractError, match="contiguous"):
        load_preprocessed_corpus(phase4_corpus.preprocessing_report)


def test_phase3d_count_and_gate_validation(phase4_corpus: Phase4CorpusFixture) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    manifest = json.loads(phase4_corpus.client_partition_manifest.read_text(encoding="utf-8"))
    manifest["gates"]["minimum_client_size_satisfied"] = False
    phase4_corpus.client_partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="gates"):
        load_training_class_counts(phase4_corpus.client_partition_manifest, corpus)


def test_phase3d_family_count_conservation(phase4_corpus: Phase4CorpusFixture) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    manifest = json.loads(phase4_corpus.client_partition_manifest.read_text(encoding="utf-8"))
    manifest["family_training_counts"]["attack"] = 39
    phase4_corpus.client_partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="conserve"):
        load_training_class_counts(phase4_corpus.client_partition_manifest, corpus)


@pytest.mark.parametrize(
    "case",
    [
        "identity",
        "preprocessor_reference",
        "preprocessor_path",
        "preprocessor_hash",
        "partition_counts",
        "partition_count_value",
        "inventory_missing",
        "inventory_entry_type",
        "inventory_entry_value",
        "inventory_file",
        "record_total",
    ],
)
def test_phase3c_report_validation_variants(phase4_corpus: Phase4CorpusFixture, case: str) -> None:
    report = json.loads(phase4_corpus.preprocessing_report.read_text(encoding="utf-8"))
    if case == "identity":
        report["dataset"] = None
    elif case == "preprocessor_reference":
        report["preprocessor"] = None
    elif case == "preprocessor_path":
        report["preprocessor"]["path"] = 7
    elif case == "preprocessor_hash":
        report["preprocessor"]["sha256"] = "0" * 64
    elif case == "partition_counts":
        report["partition_counts"] = None
    elif case == "partition_count_value":
        report["partition_counts"]["train"] = -1
    elif case == "inventory_missing":
        report["transformed_files"] = []
    elif case == "inventory_entry_type":
        report["transformed_files"] = ["bad"]
    elif case == "inventory_entry_value":
        report["transformed_files"][0]["row_count"] = -1
    elif case == "inventory_file":
        report["transformed_files"][0]["path"] = "shards/missing.parquet"
    elif case == "record_total":
        report["records"] = 999
    phase4_corpus.preprocessing_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataContractError):
        load_preprocessed_corpus(phase4_corpus.preprocessing_report)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"model_input_dimension": 0}, "identity or dimension"),
        ({"dataset": "different"}, "dataset differs"),
        ({"family_label_to_id": None}, "family map"),
        ({"family_label_to_id": {"benign": 0, "attack": "1"}}, "malformed"),
    ],
)
def test_preprocessor_validation_variants(
    phase4_corpus: Phase4CorpusFixture,
    update: dict[str, object],
    message: str,
) -> None:
    _rewrite_preprocessor(phase4_corpus, update)
    with pytest.raises(DataContractError, match=message):
        load_preprocessed_corpus(phase4_corpus.preprocessing_report)


@pytest.mark.parametrize("case", ["missing", "malformed"])
def test_phase3d_family_count_validation(phase4_corpus: Phase4CorpusFixture, case: str) -> None:
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    manifest = json.loads(phase4_corpus.client_partition_manifest.read_text(encoding="utf-8"))
    if case == "missing":
        manifest["family_training_counts"] = None
    else:
        manifest["family_training_counts"]["attack"] = "40"
    phase4_corpus.client_partition_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="counts"):
        load_training_class_counts(phase4_corpus.client_partition_manifest, corpus)


def test_json_dependency_and_stream_cardinality_failures(
    phase4_corpus: Phase4CorpusFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(DataContractError, match="Cannot read"):
        load_preprocessed_corpus(tmp_path / "missing.json")
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)

    def missing_dependency(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(training_corpus_module.importlib, "import_module", missing_dependency)
    with pytest.raises(DataContractError, match="ML dependencies"):
        list(iter_partition_batches(corpus, "train", batch_size=8))


def test_stream_cardinality_must_match_manifest(phase4_corpus: Phase4CorpusFixture) -> None:
    report = json.loads(phase4_corpus.preprocessing_report.read_text(encoding="utf-8"))
    report["partition_counts"]["train"] = 79
    phase4_corpus.preprocessing_report.write_text(json.dumps(report), encoding="utf-8")
    corpus = load_preprocessed_corpus(phase4_corpus.preprocessing_report)
    with pytest.raises(DataContractError, match="declares 79"):
        list(iter_partition_batches(corpus, "train", batch_size=16))
