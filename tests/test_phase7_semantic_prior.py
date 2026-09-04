"""Equation-level and artifact-bound tests for Phase 7 semantic priors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from semcred_fl.data.training_corpus import load_preprocessed_corpus
from semcred_fl.semantic import (
    OPENAI_MODEL_SNAPSHOT,
    LLMInvocationMetadata,
    OpenAIProviderError,
    PublicClass,
    PublicFeatureGroup,
    PublicSemanticMetadata,
    PublicSource,
    SemanticPriorError,
    build_semantic_prior_bundle,
    compute_anchor_data_prior,
    fuse_priors,
    invoke_openai_semantic_prior,
    load_semantic_prior_bundle,
    normalize_prior,
    parse_llm_response,
    prepare_semantic_prompt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class SemanticFixture:
    preprocessing_report: Path
    anchor_manifest: Path
    metadata: PublicSemanticMetadata
    invocation: LLMInvocationMetadata


@pytest.fixture
def semantic_fixture(tmp_path: Path) -> SemanticFixture:
    """Create exact calibration/scoring anchors with a two-group transformed layout."""

    artifact = tmp_path / "preprocessing"
    shards = artifact / "shards"
    shards.mkdir(parents=True)
    calibration = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 3.0, 1.0],
            [4.0, 4.0, 1.0, 5.0],
            [6.0, 4.0, 3.0, 5.0],
        ],
        dtype=np.float32,
    )
    scoring = np.full((4, 4), 10_000.0, dtype=np.float32)
    extra = np.zeros((3, 4), dtype=np.float32)
    features = np.concatenate((calibration, scoring, extra), axis=0)
    record_ids = [f"anchor-{index}" for index in range(8)] + [
        "train-0",
        "validation-0",
        "test-0",
    ]
    partitions = (
        ["anchor_calibration"] * 4 + ["anchor_scoring"] * 4 + ["train", "validation", "test"]
    )
    family_ids = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.int16)
    table = pa.table(
        {
            "record_id": pa.array(record_ids, type=pa.string()),
            "partition": pa.array(partitions, type=pa.string()),
            "family_id": pa.array(family_ids, type=pa.int16()),
            **{
                f"x{index:04d}": pa.array(features[:, index], type=pa.float32())
                for index in range(4)
            },
        }
    )
    shard = shards / "0000-preprocessed.parquet"
    pq.write_table(table, shard)
    preprocessor_id = "1" * 64
    anchor_id = "2" * 64
    corpus_id = "3" * 64
    preprocessor = {
        "schema_version": 1,
        "dataset": "edge_iiotset",
        "split_id": "split-test",
        "anchor_id": anchor_id,
        "preprocessor_id": preprocessor_id,
        "model_input_dimension": 4,
        "family_label_to_id": {"benign": 0, "attack": 1},
        "coordinates": [
            {
                "coordinate": f"x{index:04d}",
                "semantic_group": "group_a" if index < 2 else "group_b",
            }
            for index in range(4)
        ],
    }
    preprocessor_path = artifact / "preprocessor.json"
    _write_json(preprocessor_path, preprocessor)
    report = {
        "schema_version": 1,
        "dataset": "edge_iiotset",
        "preprocessed_corpus_id": corpus_id,
        "split_id": "split-test",
        "anchor_id": anchor_id,
        "records": 11,
        "partition_counts": {
            "train": 1,
            "validation": 1,
            "test": 1,
            "anchor_calibration": 4,
            "anchor_scoring": 4,
        },
        "preprocessor": {
            "path": "preprocessor.json",
            "preprocessor_id": preprocessor_id,
            "sha256": _sha256(preprocessor_path),
        },
        "transformed_files": [
            {
                "path": "shards/0000-preprocessed.parquet",
                "row_count": 11,
                "byte_size": shard.stat().st_size,
                "sha256": _sha256(shard),
            }
        ],
        "gates": {
            "fit_cardinality_matches_federated_training_pool": True,
            "fit_excludes_validation_test_and_anchors": True,
            "no_nonfinite_model_input": True,
            "partition_cardinality_matches_phase3b": True,
            "preprocessing_fitted": True,
            "stable_feature_order_and_dimension": True,
            "transformed_cardinality_matches_canonical": True,
        },
    }
    report_path = artifact / "edge_iiotset_phase3c_preprocessed.json"
    _write_json(report_path, report)

    split = tmp_path / "splits"
    split.mkdir()
    anchor_file = split / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "record_id": pa.array(record_ids[:8], type=pa.string()),
                "family_label": pa.array(
                    ["benign", "benign", "attack", "attack"] * 2,
                    type=pa.string(),
                ),
                "subset": pa.array(
                    ["calibration"] * 4 + ["scoring"] * 4,
                    type=pa.string(),
                ),
            }
        ),
        anchor_file,
    )
    manifest = {
        "schema_version": 1,
        "dataset": "edge_iiotset",
        "anchor_id": anchor_id,
        "canonical_corpus_id": "4" * 64,
        "file": {
            "path": "anchors.parquet",
            "byte_size": anchor_file.stat().st_size,
            "sha256": _sha256(anchor_file),
        },
        "gates": {
            "all_anchors_selected_from_train": True,
            "calibration_scoring_disjoint": True,
            "exact_class_balance": True,
        },
    }
    manifest_path = split / "anchor_manifest.json"
    _write_json(manifest_path, manifest)

    source = PublicSource("dataset-paper", "Dataset paper", "https://example.test/paper")
    metadata = PublicSemanticMetadata(
        dataset="edge_iiotset",
        reviewed_at_utc="2026-08-31T00:00:00+00:00",
        reviewer="fixture-reviewer",
        sources=(source,),
        feature_groups=(
            PublicFeatureGroup(
                "group_a", "First public group.", "unitless", "documented", (source.source_id,)
            ),
            PublicFeatureGroup(
                "group_b", "Second public group.", "unitless", "documented", (source.source_id,)
            ),
        ),
        classes=(
            PublicClass("benign", "Normal traffic.", (source.source_id,)),
            PublicClass("attack", "Attack traffic.", (source.source_id,)),
        ),
    )
    invocation = LLMInvocationMetadata(
        provider="fixture-provider",
        model="fixture-model",
        accessed_at_utc="2026-08-31T00:00:00+00:00",
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=1000,
        seed=17,
        seed_behavior="fixture deterministic",
    )
    return SemanticFixture(report_path, manifest_path, metadata, invocation)


def _valid_response(metadata: PublicSemanticMetadata) -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "dataset": metadata.dataset,
            "classes": {
                class_name: {
                    group_name: {"score": (class_index + group_index) % 4, "reason": "public"}
                    for group_index, group_name in enumerate(metadata.feature_group_names)
                }
                for class_index, class_name in enumerate(metadata.class_names)
            },
        }
    )


@dataclass(slots=True)
class FakeOpenAITransport:
    """Capture one provider request and return a controlled response envelope."""

    envelope: dict[str, object]
    calls: int = 0
    payload: dict[str, object] | None = None
    saw_nonempty_key: bool = False

    async def post_json(
        self,
        *,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls += 1
        self.payload = payload
        self.saw_nonempty_key = bool(api_key)
        assert timeout_seconds > 0.0
        return self.envelope


def _openai_invocation() -> LLMInvocationMetadata:
    return LLMInvocationMetadata(
        provider="openai",
        model=OPENAI_MODEL_SNAPSHOT,
        accessed_at_utc="2026-08-31T00:00:00+00:00",
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=32_000,
        seed=None,
        seed_behavior="No seed parameter; reuse the saved immutable response.",
    )


def _openai_envelope(response_text: str) -> dict[str, object]:
    return {
        "id": "resp_fixture_001",
        "created_at": 1_788_134_400,
        "status": "completed",
        "error": None,
        "model": OPENAI_MODEL_SNAPSHOT,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": response_text}],
            }
        ],
        "usage": {"input_tokens": 101, "output_tokens": 202, "total_tokens": 303},
    }


def test_strict_llm_response_rejects_malformed_values(
    semantic_fixture: SemanticFixture,
) -> None:
    metadata = semantic_fixture.metadata
    valid = _valid_response(metadata)
    response = parse_llm_response(valid, metadata)
    assert response.class_names == ("benign", "attack")
    assert response.feature_group_names == ("group_a", "group_b")

    duplicate = valid.replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0", "schema_version": "1.0",',
    )
    with pytest.raises(SemanticPriorError, match="Duplicate JSON key"):
        parse_llm_response(duplicate, metadata)

    invalid_bool = valid.replace('"score": 0', '"score": true', 1)
    with pytest.raises(SemanticPriorError, match="integers"):
        parse_llm_response(invalid_bool, metadata)

    missing_group = json.loads(valid)
    del missing_group["classes"]["benign"]["group_b"]
    with pytest.raises(SemanticPriorError, match="keys differ"):
        parse_llm_response(json.dumps(missing_group), metadata)

    extra_prose = json.loads(valid)
    extra_prose["comment"] = "repair me"
    with pytest.raises(SemanticPriorError, match="keys differ"):
        parse_llm_response(json.dumps(extra_prose), metadata)


def test_anchor_prior_matches_hand_calculated_rms_and_excludes_scoring(
    semantic_fixture: SemanticFixture,
) -> None:
    corpus = load_preprocessed_corpus(semantic_fixture.preprocessing_report)
    prior = compute_anchor_data_prior(corpus, semantic_fixture.anchor_manifest, epsilon=1e-12)
    expected_a = math.sqrt(3.6)
    expected_b = math.sqrt(2.0)
    assert prior.class_counts == (2, 2)
    assert prior.feature_group_names == ("group_a", "group_b")
    for row in prior.scores:
        assert row == pytest.approx((expected_a, expected_b), rel=1e-12, abs=1e-12)
    assert prior.validation_partition_accessed is False
    assert prior.test_partition_accessed is False
    assert prior.scoring_anchor_accessed is False


def test_prior_normalization_and_fusion_endpoints_are_exact() -> None:
    llm = normalize_prior((0.0, 1.0, 3.0), epsilon=1e-12)
    data = normalize_prior((2.0, 2.0, 0.0), epsilon=1e-12)
    assert fuse_priors(llm, data, alpha_l=0.0) is data
    assert fuse_priors(llm, data, alpha_l=1.0) is llm
    midpoint = fuse_priors(llm, data, alpha_l=0.5)
    assert midpoint == pytest.approx(
        tuple((left + right) / 2.0 for left, right in zip(llm, data, strict=True))
    )


def test_prompt_and_bundle_are_immutable_and_load_without_llm(
    semantic_fixture: SemanticFixture, tmp_path: Path
) -> None:
    prompt_directory = tmp_path / "prompt"
    manifest = prepare_semantic_prompt(
        preprocessing_report=semantic_fixture.preprocessing_report,
        metadata=semantic_fixture.metadata,
        invocation=semantic_fixture.invocation,
        output_directory=prompt_directory,
    )
    assert manifest["llm_called"] is False
    prompt = (prompt_directory / "prompt.txt").read_text(encoding="utf-8")
    assert "record_id" not in prompt
    assert "client_id" not in prompt
    assert "submitted_delta" not in prompt

    response_path = tmp_path / "provider-response.json"
    response_path.write_text(_valid_response(semantic_fixture.metadata), encoding="utf-8")
    output = tmp_path / "bundle"
    report = build_semantic_prior_bundle(
        preprocessing_report=semantic_fixture.preprocessing_report,
        anchor_manifest=semantic_fixture.anchor_manifest,
        prompt_directory=prompt_directory,
        raw_response=response_path,
        output_directory=output,
        alpha_l=0.5,
    )
    bundle_path = output / "semantic_prior_bundle.json"
    loaded = load_semantic_prior_bundle(bundle_path, expected_bundle_id=report["bundle_id"])
    assert loaded.bundle_id == report["bundle_id"]
    assert loaded.class_names == ("benign", "attack")
    assert loaded.feature_group_names == ("group_a", "group_b")
    assert report["gates"]["online_llm_call_required"] is False
    assert report["gates"]["test_partition_accessed"] is False

    with pytest.raises(SemanticPriorError, match="expected run input"):
        load_semantic_prior_bundle(bundle_path, expected_bundle_id="0" * 64)

    repeated = build_semantic_prior_bundle(
        preprocessing_report=semantic_fixture.preprocessing_report,
        anchor_manifest=semantic_fixture.anchor_manifest,
        prompt_directory=prompt_directory,
        raw_response=response_path,
        output_directory=output,
        alpha_l=0.5,
    )
    assert repeated["bundle_id"] == report["bundle_id"]

    tampered = json.loads(bundle_path.read_text(encoding="utf-8"))
    tampered["alpha_l"] = 0.75
    _write_json(bundle_path, tampered)
    with pytest.raises(SemanticPriorError, match="identity"):
        load_semantic_prior_bundle(bundle_path)


def test_openai_adapter_calls_once_without_storing_or_enabling_tools(
    semantic_fixture: SemanticFixture, tmp_path: Path
) -> None:
    prompt_directory = tmp_path / "prompt"
    prepare_semantic_prompt(
        preprocessing_report=semantic_fixture.preprocessing_report,
        metadata=semantic_fixture.metadata,
        invocation=_openai_invocation(),
        output_directory=prompt_directory,
    )
    output_directory = tmp_path / "provider"
    transport = FakeOpenAITransport(_openai_envelope(_valid_response(semantic_fixture.metadata)))
    secret = "test-secret-that-must-not-be-persisted"
    report = asyncio.run(
        invoke_openai_semantic_prior(
            prompt_directory=prompt_directory,
            output_directory=output_directory,
            environment={"OPENAI_API_KEY": secret},
            transport=transport,
        )
    )

    assert transport.calls == 1
    assert transport.saw_nonempty_key is True
    assert transport.payload is not None
    assert transport.payload["model"] == OPENAI_MODEL_SNAPSHOT
    assert transport.payload["store"] is False
    assert "tools" not in transport.payload
    assert transport.payload["reasoning"] == {"effort": "none"}
    text_format = transport.payload["text"]
    assert isinstance(text_format, dict)
    assert text_format["format"]["strict"] is True
    assert report["api_called"] is True
    assert report["reused"] is False
    assert report["usage"] == {"input_tokens": 101, "output_tokens": 202, "total_tokens": 303}
    raw_response = (output_directory / "raw_response.json").read_text(encoding="utf-8")
    parse_llm_response(raw_response, semantic_fixture.metadata)
    persisted = raw_response + (output_directory / "provider_receipt.json").read_text(
        encoding="utf-8"
    )
    assert secret not in persisted
    assert "record_id" not in persisted
    assert "client_id" not in persisted

    second_transport = FakeOpenAITransport({})
    repeated = asyncio.run(
        invoke_openai_semantic_prior(
            prompt_directory=prompt_directory,
            output_directory=output_directory,
            environment={},
            transport=second_transport,
        )
    )
    assert second_transport.calls == 0
    assert repeated["api_called"] is False
    assert repeated["reused"] is True
    assert repeated["receipt_id"] == report["receipt_id"]


def test_openai_adapter_rejects_refusal_without_writing_artifacts(
    semantic_fixture: SemanticFixture, tmp_path: Path
) -> None:
    prompt_directory = tmp_path / "prompt"
    prepare_semantic_prompt(
        preprocessing_report=semantic_fixture.preprocessing_report,
        metadata=semantic_fixture.metadata,
        invocation=_openai_invocation(),
        output_directory=prompt_directory,
    )
    envelope = _openai_envelope(_valid_response(semantic_fixture.metadata))
    envelope["output"] = [
        {
            "type": "message",
            "content": [{"type": "refusal", "refusal": "Cannot comply."}],
        }
    ]
    output_directory = tmp_path / "provider"
    with pytest.raises(OpenAIProviderError, match="refused"):
        asyncio.run(
            invoke_openai_semantic_prior(
                prompt_directory=prompt_directory,
                output_directory=output_directory,
                environment={"OPENAI_API_KEY": "test-secret"},
                transport=FakeOpenAITransport(envelope),
            )
        )
    assert not (output_directory / "raw_response.json").exists()
    assert not (output_directory / "provider_receipt.json").exists()


def test_openai_adapter_rejects_unpinned_model_before_call(
    semantic_fixture: SemanticFixture, tmp_path: Path
) -> None:
    prompt_directory = tmp_path / "prompt"
    invalid = LLMInvocationMetadata(
        provider="openai",
        model="gpt-5.4-mini",
        accessed_at_utc="2026-08-31T00:00:00+00:00",
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=32_000,
        seed=None,
        seed_behavior="Alias is intentionally invalid for the confirmatory adapter.",
    )
    prepare_semantic_prompt(
        preprocessing_report=semantic_fixture.preprocessing_report,
        metadata=semantic_fixture.metadata,
        invocation=invalid,
        output_directory=prompt_directory,
    )
    transport = FakeOpenAITransport({})
    with pytest.raises(OpenAIProviderError, match="pinned model"):
        asyncio.run(
            invoke_openai_semantic_prior(
                prompt_directory=prompt_directory,
                output_directory=tmp_path / "provider",
                environment={"OPENAI_API_KEY": "test-secret"},
                transport=transport,
            )
        )
    assert transport.calls == 0
