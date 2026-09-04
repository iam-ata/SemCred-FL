"""Typed efficiency and communication accounting records."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OnlineTiming:
    """Disjoint online wall-clock components measured after warm-up."""

    client_training_seconds: float
    anchor_inference_seconds: float
    attribution_seconds: float
    numerical_scoring_seconds: float
    fusion_trust_eligibility_clipping_seconds: float
    aggregation_seconds: float

    def __post_init__(self) -> None:
        values = tuple(self.to_dict().values())
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("online timing components must be finite and nonnegative")

    @property
    def total_seconds(self) -> float:
        """Return the sum of the separately measured components."""

        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible timing components."""

        return {
            "client_training_seconds": self.client_training_seconds,
            "anchor_inference_seconds": self.anchor_inference_seconds,
            "attribution_seconds": self.attribution_seconds,
            "numerical_scoring_seconds": self.numerical_scoring_seconds,
            "fusion_trust_eligibility_clipping_seconds": (
                self.fusion_trust_eligibility_clipping_seconds
            ),
            "aggregation_seconds": self.aggregation_seconds,
        }


@dataclass(frozen=True, slots=True)
class CommunicationRecord:
    """Per-round model and auxiliary payload accounting."""

    broadcast_model_bytes: int
    submitted_update_bytes: int
    declared_sample_count_bytes: int
    auxiliary_payload_bytes: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.to_dict().values()):
            raise ValueError("communication byte counts must be nonnegative")

    @property
    def total_bytes(self) -> int:
        """Return total transmitted bytes for the recorded scope."""

        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible byte counts."""

        return {
            "broadcast_model_bytes": self.broadcast_model_bytes,
            "submitted_update_bytes": self.submitted_update_bytes,
            "declared_sample_count_bytes": self.declared_sample_count_bytes,
            "auxiliary_payload_bytes": self.auxiliary_payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class OfflineLLMCost:
    """One-time semantic-prior request cost, separate from online training."""

    model_id: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    billed_cost_usd: float | None
    validation_passed: bool

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("LLM model identifier is required")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0.0:
            raise ValueError("LLM latency must be finite and nonnegative")
        if self.billed_cost_usd is not None and (
            not math.isfinite(self.billed_cost_usd) or self.billed_cost_usd < 0.0
        ):
            raise ValueError("billed LLM cost must be finite and nonnegative when known")
