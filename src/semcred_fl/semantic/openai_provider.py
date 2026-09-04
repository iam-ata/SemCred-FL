"""One-shot OpenAI Responses API adapter for the offline Phase 7 prior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from semcred_fl.identity import canonical_json, run_id
from semcred_fl.semantic.contracts import (
    LLMInvocationMetadata,
    PublicSemanticMetadata,
    SemanticPriorError,
    parse_llm_response,
    strict_json_object,
)
from semcred_fl.semantic.pipeline import _load_prompt_inputs

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL_SNAPSHOT = "gpt-5.4-mini-2026-03-17"
OPENAI_API_KEY_ENVIRONMENT = "OPENAI_API_KEY"
_MAX_PROVIDER_RESPONSE_BYTES = 16 * 1024 * 1024


class OpenAIProviderError(SemanticPriorError):
    """Raised when the one-shot OpenAI request or its receipt is invalid."""


class OpenAITransport(Protocol):
    """Minimal asynchronous transport used by the OpenAI provider adapter."""

    async def post_json(
        self,
        *,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        """POST one JSON request and return one decoded JSON object."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenAIProviderError(f"{description} must be a nonempty string")
    return value.strip()


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIProviderError(f"{description} must be a nonnegative integer")
    return value


def _safe_api_error(body: bytes, status: int) -> OpenAIProviderError:
    error_type = "unknown"
    error_code = "unknown"
    message = "The provider returned an error response."
    try:
        parsed = strict_json_object(body.decode("utf-8"), "OpenAI error response")
        error_value = parsed.get("error")
        if isinstance(error_value, dict):
            typed = cast(dict[str, object], error_value)
            if isinstance(typed.get("type"), str):
                error_type = cast(str, typed["type"])
            if isinstance(typed.get("code"), str):
                error_code = cast(str, typed["code"])
            if isinstance(typed.get("message"), str):
                message = cast(str, typed["message"])
    except (UnicodeDecodeError, SemanticPriorError):
        pass
    return OpenAIProviderError(
        f"OpenAI request failed: http_status={status}, type={error_type}, "
        f"code={error_code}, message={message}"
    )


@dataclass(frozen=True, slots=True)
class UrllibOpenAITransport:
    """Dependency-free HTTPS transport for a single Responses API request."""

    endpoint: str = OPENAI_RESPONSES_URL

    async def post_json(
        self,
        *,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        """POST JSON without logging the bearer credential."""

        return await asyncio.to_thread(
            self._post_json_sync,
            api_key=api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

    def _post_json_sync(
        self,
        *,
        api_key: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        request = urllib.request.Request(
            self.endpoint,
            data=canonical_json(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "semcred-fl-phase7/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = response.status
                body = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            body = error.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
            raise _safe_api_error(body, error.code) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OpenAIProviderError(
                f"OpenAI request could not complete: {type(error).__name__}"
            ) from error
        if len(body) > _MAX_PROVIDER_RESPONSE_BYTES:
            raise OpenAIProviderError("OpenAI response exceeded the configured size limit")
        if status != 200:
            raise _safe_api_error(body, status)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise OpenAIProviderError("OpenAI response was not UTF-8") from error
        return strict_json_object(text, "OpenAI response envelope")


def _score_item_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }


def relevance_json_schema(metadata: PublicSemanticMetadata) -> dict[str, object]:
    """Return the strict provider schema for the reviewed class/group layout."""

    class_properties: dict[str, object] = {}
    for class_name in metadata.class_names:
        class_properties[class_name] = {
            "type": "object",
            "properties": {
                group_name: _score_item_schema() for group_name in metadata.feature_group_names
            },
            "required": list(metadata.feature_group_names),
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": "1.0"},
            "dataset": {"type": "string", "const": metadata.dataset},
            "classes": {
                "type": "object",
                "properties": class_properties,
                "required": list(metadata.class_names),
                "additionalProperties": False,
            },
        },
        "required": ["schema_version", "dataset", "classes"],
        "additionalProperties": False,
    }


def _validate_openai_invocation(invocation: LLMInvocationMetadata) -> None:
    if invocation.provider.casefold() != "openai":
        raise OpenAIProviderError("The frozen invocation provider must be openai")
    if invocation.model != OPENAI_MODEL_SNAPSHOT:
        raise OpenAIProviderError(
            f"The OpenAI adapter requires the pinned model {OPENAI_MODEL_SNAPSHOT}"
        )
    if invocation.temperature != 0.0 or invocation.top_p != 1.0:
        raise OpenAIProviderError("The OpenAI adapter requires temperature=0 and top_p=1")
    if invocation.seed is not None:
        raise OpenAIProviderError("The Responses API invocation must declare seed=null")


def build_openai_request(
    *,
    prompt: str,
    metadata: PublicSemanticMetadata,
    invocation: LLMInvocationMetadata,
) -> dict[str, object]:
    """Build the one-shot, no-tools, non-stored structured-output request."""

    _validate_openai_invocation(invocation)
    return {
        "model": invocation.model,
        "input": prompt,
        "store": False,
        "max_output_tokens": invocation.max_output_tokens,
        "reasoning": {"effort": "none"},
        "temperature": invocation.temperature,
        "top_p": invocation.top_p,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "semcred_fl_relevance_v1",
                "strict": True,
                "schema": relevance_json_schema(metadata),
            },
        },
    }


def _extract_output_text(envelope: dict[str, object]) -> str:
    status = envelope.get("status")
    if status != "completed" or envelope.get("error") is not None:
        raise OpenAIProviderError(f"OpenAI response did not complete successfully: status={status}")
    output_value = envelope.get("output")
    if not isinstance(output_value, list):
        raise OpenAIProviderError("OpenAI response output must be a list")
    texts: list[str] = []
    for output_item in output_value:
        if not isinstance(output_item, dict):
            raise OpenAIProviderError("OpenAI output item must be an object")
        typed_item = cast(dict[str, object], output_item)
        content_value = typed_item.get("content")
        if content_value is None:
            continue
        if not isinstance(content_value, list):
            raise OpenAIProviderError("OpenAI message content must be a list")
        for content_item in content_value:
            if not isinstance(content_item, dict):
                raise OpenAIProviderError("OpenAI content item must be an object")
            typed_content = cast(dict[str, object], content_item)
            content_type = typed_content.get("type")
            if content_type == "refusal":
                raise OpenAIProviderError("OpenAI refused the Phase 7 structured-output request")
            if content_type == "output_text":
                text_value = typed_content.get("text")
                if not isinstance(text_value, str) or not text_value.strip():
                    raise OpenAIProviderError("OpenAI output text must be a nonempty string")
                texts.append(text_value)
    if len(texts) != 1:
        raise OpenAIProviderError("OpenAI response must contain exactly one output_text item")
    return texts[0]


def _usage(envelope: dict[str, object]) -> dict[str, int]:
    usage_value = envelope.get("usage")
    if not isinstance(usage_value, dict):
        raise OpenAIProviderError("OpenAI response usage must be an object")
    typed = cast(dict[str, object], usage_value)
    return {
        "input_tokens": _integer(typed.get("input_tokens"), "OpenAI input_tokens"),
        "output_tokens": _integer(typed.get("output_tokens"), "OpenAI output_tokens"),
        "total_tokens": _integer(typed.get("total_tokens"), "OpenAI total_tokens"),
    }


def _write_artifact_pair(
    *, response_path: Path, response_bytes: bytes, receipt_path: Path, receipt_bytes: bytes
) -> None:
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_partial = response_path.with_suffix(response_path.suffix + ".partial")
    receipt_partial = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    try:
        response_partial.write_bytes(response_bytes)
        receipt_partial.write_bytes(receipt_bytes)
        response_partial.replace(response_path)
        receipt_partial.replace(receipt_path)
    finally:
        response_partial.unlink(missing_ok=True)
        receipt_partial.unlink(missing_ok=True)


def _load_existing(
    *,
    response_path: Path,
    receipt_path: Path,
    metadata: PublicSemanticMetadata,
    invocation: LLMInvocationMetadata,
    prompt_id: str,
) -> dict[str, object] | None:
    if not response_path.exists() and not receipt_path.exists():
        return None
    if not response_path.exists() or not receipt_path.exists():
        raise OpenAIProviderError("Phase 7 provider artifacts are incomplete; do not call again")
    try:
        response_bytes = response_path.read_bytes()
        receipt = strict_json_object(
            receipt_path.read_text(encoding="utf-8"), "OpenAI provider receipt"
        )
    except OSError as error:
        raise OpenAIProviderError(f"Cannot read existing OpenAI artifacts: {error}") from error
    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenAIProviderError("Saved OpenAI response is not UTF-8") from error
    parsed = parse_llm_response(response_text, metadata)
    stable = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        receipt.get("receipt_id") != run_id(stable, length=64)
        or receipt.get("prompt_id") != prompt_id
        or receipt.get("provider") != "openai"
        or receipt.get("requested_model") != invocation.model
        or receipt.get("raw_response_sha256") != _sha256_bytes(response_bytes)
        or receipt.get("validated_response_id") != parsed.response_id
        or receipt.get("store_requested") is not False
    ):
        raise OpenAIProviderError("Existing OpenAI provider artifacts failed identity validation")
    return receipt


async def invoke_openai_semantic_prior(
    *,
    prompt_directory: Path,
    output_directory: Path,
    timeout_seconds: float = 180.0,
    environment: Mapping[str, str] | None = None,
    transport: OpenAITransport | None = None,
) -> dict[str, object]:
    """Call OpenAI once for a frozen public-only prompt and save immutable artifacts."""

    if timeout_seconds <= 0.0:
        raise OpenAIProviderError("OpenAI timeout must be positive")
    if prompt_directory.resolve() == output_directory.resolve():
        raise OpenAIProviderError("Provider responses must be separate from frozen prompt inputs")
    manifest, metadata, invocation = _load_prompt_inputs(prompt_directory)
    _validate_openai_invocation(invocation)
    prompt_id = _nonempty_string(manifest.get("prompt_id"), "Phase 7 prompt_id")
    prompt_path = prompt_directory / "prompt.txt"
    try:
        prompt_bytes = prompt_path.read_bytes()
        prompt = prompt_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OpenAIProviderError(f"Cannot read frozen OpenAI prompt: {error}") from error
    if manifest.get("prompt_sha256") != _sha256_bytes(prompt_bytes):
        raise OpenAIProviderError("Frozen OpenAI prompt hash changed before invocation")

    response_path = output_directory / "raw_response.json"
    receipt_path = output_directory / "provider_receipt.json"
    existing = _load_existing(
        response_path=response_path,
        receipt_path=receipt_path,
        metadata=metadata,
        invocation=invocation,
        prompt_id=prompt_id,
    )
    if existing is not None:
        return {**existing, "api_called": False, "reused": True}

    selected_environment = os.environ if environment is None else environment
    api_key = selected_environment.get(OPENAI_API_KEY_ENVIRONMENT, "").strip()
    if not api_key:
        raise OpenAIProviderError(
            f"{OPENAI_API_KEY_ENVIRONMENT} is unavailable; credentials must not be stored in files"
        )
    payload = build_openai_request(prompt=prompt, metadata=metadata, invocation=invocation)
    selected_transport = UrllibOpenAITransport() if transport is None else transport
    started_at = datetime.now(UTC)
    started_counter = time.perf_counter()
    envelope = await selected_transport.post_json(
        api_key=api_key,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    duration_ms = round((time.perf_counter() - started_counter) * 1000.0, 3)
    completed_at = datetime.now(UTC)
    output_text = _extract_output_text(envelope)
    parsed_response = parse_llm_response(output_text, metadata)
    resolved_model = _nonempty_string(envelope.get("model"), "OpenAI resolved model")
    if resolved_model != invocation.model:
        raise OpenAIProviderError("OpenAI resolved a model different from the frozen snapshot")
    provider_response_id = _nonempty_string(envelope.get("id"), "OpenAI provider response_id")
    provider_created_at = _integer(envelope.get("created_at"), "OpenAI created_at")
    usage = _usage(envelope)
    response_bytes = output_text.encode("utf-8")
    stable: dict[str, object] = {
        "schema_version": 1,
        "phase": "7-openai-provider-response",
        "dataset": metadata.dataset,
        "prompt_id": prompt_id,
        "invocation_id": invocation.invocation_id,
        "provider": "openai",
        "requested_model": invocation.model,
        "resolved_model": resolved_model,
        "provider_response_id": provider_response_id,
        "provider_created_at_unix": provider_created_at,
        "request_started_at_utc": started_at.isoformat(),
        "request_completed_at_utc": completed_at.isoformat(),
        "duration_ms": duration_ms,
        "request_sha256": _sha256_bytes(canonical_json(payload).encode("utf-8")),
        "provider_envelope_sha256": _sha256_bytes(canonical_json(envelope).encode("utf-8")),
        "raw_response_sha256": _sha256_bytes(response_bytes),
        "validated_response_id": parsed_response.response_id,
        "usage": usage,
        "reasoning_effort": "none",
        "store_requested": False,
        "tools_enabled": False,
        "automatic_retry_enabled": False,
        "private_records_accessed": False,
        "client_updates_accessed": False,
        "validation_partition_accessed": False,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    receipt = {"receipt_id": run_id(stable, length=64), **stable}
    _write_artifact_pair(
        response_path=response_path,
        response_bytes=response_bytes,
        receipt_path=receipt_path,
        receipt_bytes=_json_bytes(receipt),
    )
    return {**receipt, "api_called": True, "reused": False}
