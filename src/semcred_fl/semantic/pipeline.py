"""Provider-neutral prompt preparation and immutable Phase 7 prior bundles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from semcred_fl.data.training_corpus import load_preprocessed_corpus
from semcred_fl.identity import canonical_json, run_id
from semcred_fl.semantic.contracts import (
    LLMInvocationMetadata,
    LLMRelevanceResponse,
    PublicSemanticMetadata,
    SemanticPriorError,
    invocation_metadata_from_dict,
    parse_llm_response,
    public_metadata_from_dict,
    strict_json_object,
)
from semcred_fl.semantic.prior import (
    AnchorDataPrior,
    compute_anchor_data_prior,
    fuse_priors,
    load_feature_group_layout,
    normalize_prior,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise SemanticPriorError(f"Immutable Phase 7 artifact differs: {path.name}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(value)
    partial.replace(path)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _public_snapshot(metadata: PublicSemanticMetadata) -> dict[str, object]:
    return cast(dict[str, object], asdict(metadata))


def _invocation_snapshot(invocation: LLMInvocationMetadata) -> dict[str, object]:
    return cast(dict[str, object], asdict(invocation))


def _response_snapshot(response: LLMRelevanceResponse) -> dict[str, object]:
    classes: dict[str, object] = {}
    for class_index, class_name in enumerate(response.class_names):
        classes[class_name] = {
            group_name: {
                "score": response.scores[class_index][group_index],
                "reason": response.reasons[class_index][group_index],
            }
            for group_index, group_name in enumerate(response.feature_group_names)
        }
    return {
        "schema_version": response.schema_version,
        "dataset": response.dataset,
        "classes": classes,
    }


def _prompt(metadata: PublicSemanticMetadata) -> str:
    public_payload = {
        "dataset": metadata.dataset,
        "sources": [asdict(item) for item in metadata.sources],
        "feature_groups": [asdict(item) for item in metadata.feature_groups],
        "classes": [asdict(item) for item in metadata.classes],
    }
    output_contract = {
        "schema_version": "1.0",
        "dataset": metadata.dataset,
        "classes": {
            class_name: {
                group_name: {"score": "integer 0, 1, 2, or 3", "reason": "short rationale"}
                for group_name in metadata.feature_group_names
            }
            for class_name in metadata.class_names
        },
    }
    return "\n".join(
        (
            "Assign cybersecurity relevance between each declared traffic class and feature group.",
            "Use only the public metadata below. Do not request private records or client data.",
            "model parameters, gradients, updates, trust values, or online decisions.",
            "Scores are integers: 0=irrelevant, 1=weak, 2=moderate, 3=strong.",
            "Return one JSON object matching the output contract. Do not add prose or keys.",
            "PUBLIC_METADATA=",
            canonical_json(public_payload),
            "OUTPUT_CONTRACT=",
            canonical_json(output_contract),
            "",
        )
    )


def validate_public_boundary(text: str) -> None:
    """Reject prompt text containing known private or online-state field names."""

    lowered = text.casefold()
    forbidden = (
        "record_id",
        "client_id",
        "client-00",
        "submitted_delta",
        "model_state",
        "gradient_tensor",
        "trust_value",
        "attack_metadata",
        "is_malicious",
        "anchor_record",
    )
    found = [token for token in forbidden if token in lowered]
    if found:
        raise SemanticPriorError(f"Prompt crossed the public-only boundary: {found}")


def prepare_semantic_prompt(
    *,
    preprocessing_report: Path,
    metadata: PublicSemanticMetadata,
    invocation: LLMInvocationMetadata,
    output_directory: Path,
) -> dict[str, object]:
    """Freeze a public-only prompt and invocation manifest without calling an LLM."""

    corpus = load_preprocessed_corpus(preprocessing_report)
    layout = load_feature_group_layout(corpus)
    ordered_classes = tuple(
        label for label, _ in sorted(corpus.family_label_to_id.items(), key=lambda item: item[1])
    )
    if metadata.dataset != corpus.dataset:
        raise SemanticPriorError("Public semantic metadata names another dataset")
    if metadata.feature_group_names != layout.group_names:
        raise SemanticPriorError("Reviewed feature groups differ from the Phase 3C layout")
    if metadata.class_names != ordered_classes:
        raise SemanticPriorError("Reviewed classes differ from the model class-ID order")
    prompt = _prompt(metadata)
    validate_public_boundary(prompt)
    prompt_bytes = prompt.encode("utf-8")
    metadata_value = _public_snapshot(metadata)
    invocation_value = _invocation_snapshot(invocation)
    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "7-semantic-prompt",
        "dataset": corpus.dataset,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "feature_group_layout_id": layout.layout_id,
        "public_metadata_id": metadata.metadata_id,
        "invocation_id": invocation.invocation_id,
        "prompt_sha256": _sha256_bytes(prompt_bytes),
        "public_only": True,
        "llm_called": False,
        "private_records_accessed": False,
        "client_updates_accessed": False,
        "validation_partition_accessed": False,
        "test_partition_accessed": False,
    }
    manifest = {"prompt_id": run_id(stable, length=64), **stable}
    _write_immutable_bytes(output_directory / "public_metadata.json", _json_bytes(metadata_value))
    _write_immutable_bytes(
        output_directory / "llm_invocation_metadata.json", _json_bytes(invocation_value)
    )
    _write_immutable_bytes(output_directory / "prompt.txt", prompt_bytes)
    _write_immutable_bytes(output_directory / "prompt_manifest.json", _json_bytes(manifest))
    return manifest


def _load_prompt_inputs(
    directory: Path,
) -> tuple[dict[str, object], PublicSemanticMetadata, LLMInvocationMetadata]:
    try:
        manifest_text = (directory / "prompt_manifest.json").read_text(encoding="utf-8")
        metadata_text = (directory / "public_metadata.json").read_text(encoding="utf-8")
        invocation_text = (directory / "llm_invocation_metadata.json").read_text(encoding="utf-8")
        prompt_bytes = (directory / "prompt.txt").read_bytes()
    except OSError as error:
        raise SemanticPriorError(f"Cannot read frozen Phase 7 prompt inputs: {error}") from error
    manifest = strict_json_object(manifest_text, "Phase 7 prompt manifest")
    metadata = public_metadata_from_dict(
        strict_json_object(metadata_text, "public semantic metadata")
    )
    invocation = invocation_metadata_from_dict(
        strict_json_object(invocation_text, "LLM invocation metadata")
    )
    stable = {key: value for key, value in manifest.items() if key != "prompt_id"}
    if (
        manifest.get("prompt_id") != run_id(stable, length=64)
        or manifest.get("public_metadata_id") != metadata.metadata_id
        or manifest.get("invocation_id") != invocation.invocation_id
        or manifest.get("prompt_sha256") != _sha256_bytes(prompt_bytes)
    ):
        raise SemanticPriorError("Frozen Phase 7 prompt identities are inconsistent")
    validate_public_boundary(prompt_bytes.decode("utf-8"))
    return manifest, metadata, invocation


@dataclass(frozen=True, slots=True)
class SemanticPriorBundle:
    """Validated no-LLM runtime view of one immutable Phase 7 bundle."""

    bundle_id: str
    dataset: str
    preprocessed_corpus_id: str
    preprocessor_id: str
    anchor_id: str
    feature_group_names: tuple[str, ...]
    class_names: tuple[str, ...]
    alpha_l: float
    epsilon: float
    priors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        identities = (
            self.bundle_id,
            self.preprocessed_corpus_id,
            self.preprocessor_id,
            self.anchor_id,
        )
        if any(len(value) != 64 for value in identities):
            raise SemanticPriorError("Semantic-prior bundle identities are invalid")
        if not 0.0 <= self.alpha_l <= 1.0 or self.epsilon <= 0.0:
            raise SemanticPriorError("Semantic-prior bundle coefficients are invalid")
        if len(self.priors) != len(self.class_names):
            raise SemanticPriorError("Semantic-prior bundle class dimensions differ")
        for row in self.priors:
            if len(row) != len(self.feature_group_names) or any(
                not math.isfinite(value) or value < 0.0 for value in row
            ):
                raise SemanticPriorError("Semantic-prior bundle values are invalid")


def _anchor_snapshot(prior: AnchorDataPrior) -> dict[str, object]:
    value = asdict(prior)
    return {"anchor_prior_id": prior.anchor_prior_id, **value}


def build_semantic_prior_bundle(
    *,
    preprocessing_report: Path,
    anchor_manifest: Path,
    prompt_directory: Path,
    raw_response: Path,
    output_directory: Path,
    alpha_l: float = 0.5,
    epsilon: float = 1e-12,
) -> dict[str, object]:
    """Validate one saved response and fuse it with calibration-anchor evidence."""

    if not math.isfinite(alpha_l) or not 0.0 <= alpha_l <= 1.0:
        raise ValueError("alpha_l must be finite and in [0,1]")
    manifest, metadata, invocation = _load_prompt_inputs(prompt_directory)
    try:
        raw_bytes = raw_response.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SemanticPriorError(f"Cannot read raw LLM response: {error}") from error
    response = parse_llm_response(raw_text, metadata)
    corpus = load_preprocessed_corpus(preprocessing_report)
    if (
        manifest.get("preprocessed_corpus_id") != corpus.preprocessed_corpus_id
        or manifest.get("preprocessor_id") != corpus.preprocessor_id
    ):
        raise SemanticPriorError("Prompt manifest differs from the selected Phase 3C corpus")
    anchor_prior = compute_anchor_data_prior(corpus, anchor_manifest, epsilon=epsilon)
    if (
        response.class_names != anchor_prior.class_names
        or response.feature_group_names != anchor_prior.feature_group_names
    ):
        raise SemanticPriorError("LLM and anchor priors use different coordinates")
    normalized_llm: list[tuple[float, ...]] = []
    normalized_data: list[tuple[float, ...]] = []
    fused: list[tuple[float, ...]] = []
    for llm_row, data_row in zip(response.scores, anchor_prior.scores, strict=True):
        llm_normalized = normalize_prior(tuple(float(value) for value in llm_row), epsilon=epsilon)
        data_normalized = normalize_prior(data_row, epsilon=epsilon)
        normalized_llm.append(llm_normalized)
        normalized_data.append(data_normalized)
        fused.append(fuse_priors(llm_normalized, data_normalized, alpha_l=alpha_l))
    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "7-semantic-prior-bundle",
        "dataset": corpus.dataset,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "anchor_id": corpus.anchor_id,
        "prompt_id": manifest["prompt_id"],
        "public_metadata_id": metadata.metadata_id,
        "invocation_id": invocation.invocation_id,
        "response_id": response.response_id,
        "raw_response_sha256": _sha256_bytes(raw_bytes),
        "anchor_prior_id": anchor_prior.anchor_prior_id,
        "alpha_l": alpha_l,
        "epsilon": epsilon,
        "class_names": list(response.class_names),
        "feature_group_names": list(response.feature_group_names),
        "llm_scores": [list(row) for row in response.scores],
        "anchor_scores": [list(row) for row in anchor_prior.scores],
        "normalized_llm_priors": [list(row) for row in normalized_llm],
        "normalized_anchor_priors": [list(row) for row in normalized_data],
        "fused_priors": [list(row) for row in fused],
        "gates": {
            "response_schema_valid": True,
            "class_and_group_coverage_exact": True,
            "anchor_calibration_only": True,
            "prompt_public_only": True,
            "client_records_sent_to_llm": False,
            "client_updates_sent_to_llm": False,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "scoring_anchor_accessed": False,
            "online_llm_call_required": False,
            "performance_claims_allowed": False,
        },
    }
    bundle = {"bundle_id": run_id(stable, length=64), **stable}
    validation = {
        "validation_id": run_id(
            {
                "prompt_id": manifest["prompt_id"],
                "response_id": response.response_id,
                "raw_response_sha256": _sha256_bytes(raw_bytes),
            },
            length=64,
        ),
        "response_id": response.response_id,
        "raw_response_sha256": _sha256_bytes(raw_bytes),
        "schema_valid": True,
        "repaired": False,
        "unknown_keys_accepted": False,
    }
    _write_immutable_bytes(output_directory / "raw_response.txt", raw_bytes)
    _write_immutable_bytes(
        output_directory / "validated_response.json", _json_bytes(_response_snapshot(response))
    )
    _write_immutable_bytes(output_directory / "validation.json", _json_bytes(validation))
    _write_immutable_bytes(
        output_directory / "anchor_prior.json", _json_bytes(_anchor_snapshot(anchor_prior))
    )
    _write_immutable_bytes(output_directory / "semantic_prior_bundle.json", _json_bytes(bundle))
    return bundle


def load_semantic_prior_bundle(
    path: Path, *, expected_bundle_id: str | None = None
) -> SemanticPriorBundle:
    """Load and identity-check a saved bundle without making an LLM call."""

    try:
        value = strict_json_object(path.read_text(encoding="utf-8"), "semantic-prior bundle")
    except OSError as error:
        raise SemanticPriorError(f"Cannot read semantic-prior bundle: {error}") from error
    stable = {key: item for key, item in value.items() if key != "bundle_id"}
    calculated_id = run_id(stable, length=64)
    if value.get("bundle_id") != calculated_id:
        raise SemanticPriorError("Semantic-prior bundle identity is invalid")
    if expected_bundle_id is not None and calculated_id != expected_bundle_id:
        raise SemanticPriorError("Semantic-prior bundle differs from the expected run input")
    gates = value.get("gates")
    if not isinstance(gates, dict) or not all(
        (
            gates.get("response_schema_valid") is True,
            gates.get("class_and_group_coverage_exact") is True,
            gates.get("anchor_calibration_only") is True,
            gates.get("prompt_public_only") is True,
            gates.get("client_records_sent_to_llm") is False,
            gates.get("client_updates_sent_to_llm") is False,
            gates.get("validation_partition_accessed") is False,
            gates.get("test_partition_accessed") is False,
            gates.get("scoring_anchor_accessed") is False,
            gates.get("online_llm_call_required") is False,
            gates.get("performance_claims_allowed") is False,
        )
    ):
        raise SemanticPriorError("Semantic-prior bundle gates are incomplete")
    class_names = value.get("class_names")
    group_names = value.get("feature_group_names")
    priors = value.get("fused_priors")
    if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
        raise SemanticPriorError("Semantic-prior bundle class names are invalid")
    if not isinstance(group_names, list) or not all(isinstance(item, str) for item in group_names):
        raise SemanticPriorError("Semantic-prior bundle feature groups are invalid")
    if not isinstance(priors, list) or not all(isinstance(row, list) for row in priors):
        raise SemanticPriorError("Semantic-prior bundle values are invalid")
    typed_priors: list[tuple[float, ...]] = []
    for row in cast(list[list[object]], priors):
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in row):
            raise SemanticPriorError("Semantic-prior bundle contains a nonnumeric value")
        numeric_row = cast(list[int | float], row)
        typed_priors.append(tuple(float(item) for item in numeric_row))
    alpha_l = value.get("alpha_l")
    epsilon = value.get("epsilon")
    if isinstance(alpha_l, bool) or not isinstance(alpha_l, int | float):
        raise SemanticPriorError("Semantic-prior bundle alpha_l is invalid")
    if isinstance(epsilon, bool) or not isinstance(epsilon, int | float):
        raise SemanticPriorError("Semantic-prior bundle epsilon is invalid")
    required_strings = (
        "bundle_id",
        "dataset",
        "preprocessed_corpus_id",
        "preprocessor_id",
        "anchor_id",
    )
    if any(not isinstance(value.get(name), str) for name in required_strings):
        raise SemanticPriorError("Semantic-prior bundle identity fields are invalid")
    return SemanticPriorBundle(
        bundle_id=cast(str, value["bundle_id"]),
        dataset=cast(str, value["dataset"]),
        preprocessed_corpus_id=cast(str, value["preprocessed_corpus_id"]),
        preprocessor_id=cast(str, value["preprocessor_id"]),
        anchor_id=cast(str, value["anchor_id"]),
        feature_group_names=tuple(cast(list[str], group_names)),
        class_names=tuple(cast(list[str], class_names)),
        alpha_l=float(alpha_l),
        epsilon=float(epsilon),
        priors=tuple(typed_priors),
    )
