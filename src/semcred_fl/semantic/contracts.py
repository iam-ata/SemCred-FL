"""Strict public-metadata and LLM-response contracts for Phase 7."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from semcred_fl.identity import run_id


class SemanticPriorError(ValueError):
    """Raised when a Phase 7 input or artifact violates its frozen contract."""


def _exact_keys(value: dict[str, object], expected: set[str], description: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SemanticPriorError(f"{description} keys differ; missing={missing}, unknown={unknown}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticPriorError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_object(text: str, description: str) -> dict[str, object]:
    """Parse one JSON object while rejecting duplicate keys and non-object roots."""

    try:
        value = json.loads(text, object_pairs_hook=_strict_pairs)
    except (json.JSONDecodeError, TypeError) as error:
        raise SemanticPriorError(f"Cannot parse {description}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticPriorError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticPriorError(f"{description} must be a nonempty string")
    cleaned = value.strip()
    if len(cleaned) > 20_000:
        raise SemanticPriorError(f"{description} exceeds the Phase 7 text limit")
    return cleaned


def _utc_timestamp(value: object, description: str) -> str:
    text = _nonempty_string(value, description)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SemanticPriorError(f"{description} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SemanticPriorError(f"{description} must include the UTC offset")
    return text


def _string_tuple(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SemanticPriorError(f"{description} must be a nonempty string list")
    items = tuple(_nonempty_string(item, description) for item in value)
    if len(items) != len(set(items)):
        raise SemanticPriorError(f"{description} contains duplicates")
    return items


@dataclass(frozen=True, slots=True)
class PublicSource:
    """One public source used to define features or traffic classes."""

    source_id: str
    citation: str
    url: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.citation or not self.url:
            raise SemanticPriorError("Public source fields must be nonempty")
        if not self.url.startswith(("https://", "http://")):
            raise SemanticPriorError("Public source URL must use HTTP or HTTPS")


@dataclass(frozen=True, slots=True)
class PublicFeatureGroup:
    """Public definition of one post-preprocessing semantic feature group."""

    name: str
    definition: str
    unit: str
    admissible_range: str
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all((self.name, self.definition, self.unit, self.admissible_range)):
            raise SemanticPriorError("Public feature-group fields must be nonempty")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise SemanticPriorError("Public feature groups require unique source IDs")


@dataclass(frozen=True, slots=True)
class PublicClass:
    """Public definition of one primary traffic or attack family."""

    name: str
    description: str
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise SemanticPriorError("Public class fields must be nonempty")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise SemanticPriorError("Public classes require unique source IDs")


@dataclass(frozen=True, slots=True)
class PublicSemanticMetadata:
    """Human-reviewed public-only input to the offline LLM prompt."""

    dataset: str
    reviewed_at_utc: str
    reviewer: str
    sources: tuple[PublicSource, ...]
    feature_groups: tuple[PublicFeatureGroup, ...]
    classes: tuple[PublicClass, ...]
    schema_version: str = "1.0"
    review_status: str = "approved"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.review_status != "approved":
            raise SemanticPriorError("Public semantic metadata must be approved schema 1.0")
        if not self.dataset or not self.reviewed_at_utc or not self.reviewer:
            raise SemanticPriorError("Public semantic metadata identity is incomplete")
        source_ids = tuple(item.source_id for item in self.sources)
        group_names = tuple(item.name for item in self.feature_groups)
        class_names = tuple(item.name for item in self.classes)
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise SemanticPriorError("Public source IDs must be nonempty and unique")
        if not group_names or len(group_names) != len(set(group_names)):
            raise SemanticPriorError("Public feature-group names must be nonempty and unique")
        if len(class_names) < 2 or len(class_names) != len(set(class_names)):
            raise SemanticPriorError("Public class names must be unique and contain two classes")
        known_sources = set(source_ids)
        referenced = {
            source_id for item in self.feature_groups for source_id in item.source_ids
        } | {source_id for item in self.classes for source_id in item.source_ids}
        if not referenced <= known_sources:
            raise SemanticPriorError("Public metadata references an unknown source ID")

    @property
    def metadata_id(self) -> str:
        """Return the deterministic full identity of this reviewed metadata."""

        return run_id(self, length=64)

    @property
    def feature_group_names(self) -> tuple[str, ...]:
        """Return semantic feature groups in their reviewed order."""

        return tuple(item.name for item in self.feature_groups)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return primary classes in model-ID order."""

        return tuple(item.name for item in self.classes)


def _public_source(value: object) -> PublicSource:
    if not isinstance(value, dict):
        raise SemanticPriorError("Public source entry must be an object")
    typed = cast(dict[str, object], value)
    _exact_keys(typed, {"source_id", "citation", "url"}, "public source")
    return PublicSource(
        _nonempty_string(typed["source_id"], "source_id"),
        _nonempty_string(typed["citation"], "citation"),
        _nonempty_string(typed["url"], "url"),
    )


def _public_feature(value: object) -> PublicFeatureGroup:
    if not isinstance(value, dict):
        raise SemanticPriorError("Public feature-group entry must be an object")
    typed = cast(dict[str, object], value)
    _exact_keys(
        typed,
        {"name", "definition", "unit", "admissible_range", "source_ids"},
        "public feature group",
    )
    return PublicFeatureGroup(
        _nonempty_string(typed["name"], "feature name"),
        _nonempty_string(typed["definition"], "feature definition"),
        _nonempty_string(typed["unit"], "feature unit"),
        _nonempty_string(typed["admissible_range"], "feature admissible range"),
        _string_tuple(typed["source_ids"], "feature source_ids"),
    )


def _public_class(value: object) -> PublicClass:
    if not isinstance(value, dict):
        raise SemanticPriorError("Public class entry must be an object")
    typed = cast(dict[str, object], value)
    _exact_keys(typed, {"name", "description", "source_ids"}, "public class")
    return PublicClass(
        _nonempty_string(typed["name"], "class name"),
        _nonempty_string(typed["description"], "class description"),
        _string_tuple(typed["source_ids"], "class source_ids"),
    )


def public_metadata_from_dict(value: dict[str, object]) -> PublicSemanticMetadata:
    """Validate a reviewed public-metadata object without silently repairing it."""

    _exact_keys(
        value,
        {
            "schema_version",
            "dataset",
            "review_status",
            "reviewed_at_utc",
            "reviewer",
            "sources",
            "feature_groups",
            "classes",
        },
        "public semantic metadata",
    )
    sources_value = value["sources"]
    features_value = value["feature_groups"]
    classes_value = value["classes"]
    if not isinstance(sources_value, list):
        raise SemanticPriorError("Public sources must be a list")
    if not isinstance(features_value, list):
        raise SemanticPriorError("Public feature groups must be a list")
    if not isinstance(classes_value, list):
        raise SemanticPriorError("Public classes must be a list")
    return PublicSemanticMetadata(
        schema_version=_nonempty_string(value["schema_version"], "schema_version"),
        dataset=_nonempty_string(value["dataset"], "dataset"),
        review_status=_nonempty_string(value["review_status"], "review_status"),
        reviewed_at_utc=_utc_timestamp(value["reviewed_at_utc"], "reviewed_at_utc"),
        reviewer=_nonempty_string(value["reviewer"], "reviewer"),
        sources=tuple(_public_source(item) for item in sources_value),
        feature_groups=tuple(_public_feature(item) for item in features_value),
        classes=tuple(_public_class(item) for item in classes_value),
    )


def load_public_metadata(path: Path) -> PublicSemanticMetadata:
    """Load strictly reviewed public semantic metadata from JSON."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SemanticPriorError(f"Cannot read public semantic metadata: {error}") from error
    return public_metadata_from_dict(strict_json_object(text, "public semantic metadata"))


@dataclass(frozen=True, slots=True)
class LLMInvocationMetadata:
    """Recorded provider configuration for one offline semantic-prior request."""

    provider: str
    model: str
    accessed_at_utc: str
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int | None
    seed_behavior: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise SemanticPriorError("LLM invocation metadata must use schema 1.0")
        identity_fields = (self.provider, self.model, self.accessed_at_utc, self.seed_behavior)
        if not all(identity_fields):
            raise SemanticPriorError("LLM invocation metadata identity is incomplete")
        if not 0.0 <= self.temperature <= 2.0 or not 0.0 < self.top_p <= 1.0:
            raise SemanticPriorError("LLM decoding parameters are invalid")
        if self.max_output_tokens < 1 or (self.seed is not None and self.seed < 0):
            raise SemanticPriorError("LLM output-token or seed setting is invalid")

    @property
    def invocation_id(self) -> str:
        """Return the deterministic full identity of this invocation declaration."""

        return run_id(self, length=64)


def invocation_metadata_from_dict(value: dict[str, object]) -> LLMInvocationMetadata:
    """Validate provider metadata without accepting credentials or unknown fields."""

    _exact_keys(
        value,
        {
            "schema_version",
            "provider",
            "model",
            "accessed_at_utc",
            "temperature",
            "top_p",
            "max_output_tokens",
            "seed",
            "seed_behavior",
        },
        "LLM invocation metadata",
    )
    temperature = value["temperature"]
    top_p = value["top_p"]
    max_tokens = value["max_output_tokens"]
    seed = value["seed"]
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise SemanticPriorError("LLM temperature must be numeric")
    if isinstance(top_p, bool) or not isinstance(top_p, int | float):
        raise SemanticPriorError("LLM top_p must be numeric")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise SemanticPriorError("LLM max_output_tokens must be an integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SemanticPriorError("LLM seed must be an integer or null")
    return LLMInvocationMetadata(
        schema_version=_nonempty_string(value["schema_version"], "schema_version"),
        provider=_nonempty_string(value["provider"], "provider"),
        model=_nonempty_string(value["model"], "model"),
        accessed_at_utc=_utc_timestamp(value["accessed_at_utc"], "accessed_at_utc"),
        temperature=float(temperature),
        top_p=float(top_p),
        max_output_tokens=max_tokens,
        seed=seed,
        seed_behavior=_nonempty_string(value["seed_behavior"], "seed_behavior"),
    )


def load_invocation_metadata(path: Path) -> LLMInvocationMetadata:
    """Load a strict provider declaration that contains no API credential."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SemanticPriorError(f"Cannot read LLM invocation metadata: {error}") from error
    return invocation_metadata_from_dict(strict_json_object(text, "LLM invocation metadata"))


@dataclass(frozen=True, slots=True)
class LLMRelevanceResponse:
    """Strictly validated class-by-feature relevance scores and rationales."""

    dataset: str
    class_names: tuple[str, ...]
    feature_group_names: tuple[str, ...]
    scores: tuple[tuple[int, ...], ...]
    reasons: tuple[tuple[str, ...], ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or not self.dataset:
            raise SemanticPriorError("LLM relevance response identity is invalid")
        if len(self.scores) != len(self.class_names) or len(self.reasons) != len(self.class_names):
            raise SemanticPriorError("LLM relevance response class dimensions differ")
        for score_row, reason_row in zip(self.scores, self.reasons, strict=True):
            if len(score_row) != len(self.feature_group_names) or len(reason_row) != len(
                self.feature_group_names
            ):
                raise SemanticPriorError("LLM relevance response feature dimensions differ")
            if any(type(score) is not int or score not in {0, 1, 2, 3} for score in score_row):
                raise SemanticPriorError("LLM relevance scores must be integers in {0,1,2,3}")
            if any(not reason.strip() for reason in reason_row):
                raise SemanticPriorError("Every LLM relevance score requires a rationale")

    @property
    def response_id(self) -> str:
        """Return the deterministic identity of the validated semantic response."""

        return run_id(self, length=64)


def parse_llm_response(text: str, metadata: PublicSemanticMetadata) -> LLMRelevanceResponse:
    """Validate exact class/group coverage and reject every malformed response."""

    value = strict_json_object(text, "LLM relevance response")
    _exact_keys(value, {"schema_version", "dataset", "classes"}, "LLM relevance response")
    if value["schema_version"] != "1.0" or value["dataset"] != metadata.dataset:
        raise SemanticPriorError("LLM relevance response identity differs from the prompt")
    classes_value = value["classes"]
    if not isinstance(classes_value, dict):
        raise SemanticPriorError("LLM response classes must be an object")
    classes = cast(dict[str, object], classes_value)
    _exact_keys(classes, set(metadata.class_names), "LLM response classes")
    score_rows: list[tuple[int, ...]] = []
    reason_rows: list[tuple[str, ...]] = []
    for class_name in metadata.class_names:
        groups_value = classes[class_name]
        if not isinstance(groups_value, dict):
            raise SemanticPriorError("Each LLM response class must contain a group object")
        groups = cast(dict[str, object], groups_value)
        _exact_keys(groups, set(metadata.feature_group_names), f"LLM class {class_name}")
        scores: list[int] = []
        reasons: list[str] = []
        for group_name in metadata.feature_group_names:
            item_value = groups[group_name]
            if not isinstance(item_value, dict):
                raise SemanticPriorError("Each LLM class/group value must be an object")
            item = cast(dict[str, object], item_value)
            _exact_keys(item, {"score", "reason"}, "LLM class/group score")
            score = item["score"]
            if type(score) is not int or score not in {0, 1, 2, 3}:
                raise SemanticPriorError("LLM relevance scores must be integers in {0,1,2,3}")
            scores.append(score)
            reason = _nonempty_string(item["reason"], "LLM relevance rationale")
            if len(reason) > 1_000:
                raise SemanticPriorError("LLM relevance rationale is not short")
            reasons.append(reason)
        score_rows.append(tuple(scores))
        reason_rows.append(tuple(reasons))
    return LLMRelevanceResponse(
        dataset=metadata.dataset,
        class_names=metadata.class_names,
        feature_group_names=metadata.feature_group_names,
        scores=tuple(score_rows),
        reasons=tuple(reason_rows),
    )
