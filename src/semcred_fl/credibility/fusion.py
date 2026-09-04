"""Disagreement-aware fusion of semantic and numerical credibility views."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedCredibility:
    """One fused score with an explicit active-view record."""

    semantic_score: float | None
    numerical_score: float | None
    score: float
    active_views: tuple[str, ...]
    disagreement_penalty_applied: bool
    missing_reasons: dict[str, str]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Fused credibility must be finite and in [0,1]")


def _validate_view(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} credibility must be finite and in [0,1]")


def fuse_credibility(
    semantic_score: float | None,
    numerical_score: float | None,
    *,
    semantic_coefficient: float = 0.5,
    disagreement_penalty: float = 1.0,
    epsilon: float = 1e-12,
) -> FusedCredibility:
    """Fuse both complete views or use the sole available view without substitution."""

    _validate_view("Semantic", semantic_score)
    _validate_view("Numerical", numerical_score)
    if not math.isfinite(semantic_coefficient) or not 0.0 <= semantic_coefficient <= 1.0:
        raise ValueError("semantic_coefficient must be finite and in [0,1]")
    if not math.isfinite(disagreement_penalty) or disagreement_penalty < 0.0:
        raise ValueError("disagreement_penalty must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    missing: dict[str, str] = {}
    if semantic_score is None and numerical_score is None:
        raise ValueError("At least one complete credibility view is required")
    if semantic_score is None:
        if numerical_score is None:
            raise RuntimeError("Credibility availability changed unexpectedly")
        missing["semantic"] = "complete_semantic_view_unavailable"
        return FusedCredibility(
            semantic_score=None,
            numerical_score=numerical_score,
            score=numerical_score,
            active_views=("numerical",),
            disagreement_penalty_applied=False,
            missing_reasons=missing,
        )
    if numerical_score is None:
        missing["numerical"] = "complete_numerical_view_unavailable"
        return FusedCredibility(
            semantic_score=semantic_score,
            numerical_score=None,
            score=semantic_score,
            active_views=("semantic",),
            disagreement_penalty_applied=False,
            missing_reasons=missing,
        )
    fused = (
        (semantic_score + epsilon) ** semantic_coefficient
        * (numerical_score + epsilon) ** (1.0 - semantic_coefficient)
        * math.exp(-disagreement_penalty * abs(semantic_score - numerical_score))
    )
    return FusedCredibility(
        semantic_score=semantic_score,
        numerical_score=numerical_score,
        score=min(1.0, max(0.0, fused)),
        active_views=("semantic", "numerical"),
        disagreement_penalty_applied=True,
        missing_reasons=missing,
    )
