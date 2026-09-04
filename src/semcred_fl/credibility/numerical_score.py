"""Round-wise numerical credibility from update geometry and accepted history."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from semcred_fl.credibility.robust_stats import (
    median,
    one_sided_robust_z,
    renormalize_active_coefficients,
)
from semcred_fl.federation.updates import DefenseUpdate


@dataclass(frozen=True, slots=True)
class NumericalCoefficients:
    """Configured weights for the four numerical discrepancy indicators."""

    center: float = 0.25
    norm: float = 0.25
    direction: float = 0.25
    history: float = 0.25

    def __post_init__(self) -> None:
        values = (self.center, self.norm, self.direction, self.history)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Numerical coefficients must be finite and nonnegative")
        if sum(values) <= 0.0:
            raise ValueError("At least one numerical coefficient must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return coefficients under stable indicator names."""

        return {
            "center": self.center,
            "norm": self.norm,
            "direction": self.direction,
            "history": self.history,
        }


@dataclass(frozen=True, slots=True)
class NumericalIndicators:
    """Raw numerical discrepancies for one submitted update."""

    center: float
    norm: float
    direction: float | None
    history: float | None
    missing_reasons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (self.center, self.norm)
        optional = (self.direction, self.history)
        if any(not math.isfinite(value) or value < 0.0 for value in required):
            raise ValueError("Required numerical indicators must be finite and nonnegative")
        if any(
            value is not None and (not math.isfinite(value) or value < 0.0) for value in optional
        ):
            raise ValueError("Available numerical indicators must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class NumericalCredibility:
    """Missing-aware numerical credibility output for one client."""

    score: float
    indicators: NumericalIndicators
    robust_z: dict[str, float | None]
    active_coefficients: dict[str, float]
    missing_reasons: dict[str, str]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 < self.score <= 1.0:
            raise ValueError("Numerical credibility must be finite and in (0,1]")


def coordinate_median(vectors: tuple[Tensor, ...]) -> Tensor:
    """Return the ordinary midpoint median independently for every coordinate."""

    if not vectors:
        raise ValueError("Coordinate median requires at least one vector")
    shape = vectors[0].shape
    if len(shape) != 1 or vectors[0].numel() < 1:
        raise ValueError("Coordinate median requires nonempty flat vectors")
    if any(vector.shape != shape for vector in vectors):
        raise ValueError("Coordinate median vectors must have identical shapes")
    if any(not torch.isfinite(vector).all().item() for vector in vectors):
        raise ValueError("Coordinate median vectors must be finite")
    ordered = torch.sort(torch.stack(vectors, dim=0), dim=0).values
    midpoint = len(vectors) // 2
    if len(vectors) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def cosine_discrepancy(
    left: Tensor,
    right: Tensor,
    *,
    epsilon: float = 1e-12,
    negligible_norm: float = 1e-12,
) -> float | None:
    """Return half cosine distance, or unavailable for negligible vectors."""

    if left.ndim != 1 or left.shape != right.shape or left.numel() < 1:
        raise ValueError("Cosine discrepancy requires equal nonempty flat vectors")
    if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
        raise ValueError("Cosine discrepancy vectors must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(negligible_norm) or negligible_norm < 0.0:
        raise ValueError("negligible_norm must be finite and nonnegative")
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm <= negligible_norm or right_norm <= negligible_norm:
        return None
    dot_product = float(torch.dot(left, right).item())
    cosine = dot_product / (left_norm * right_norm + epsilon)
    discrepancy = 0.5 * (1.0 - cosine)
    if not math.isfinite(discrepancy) or discrepancy < -1e-7 or discrepancy > 1.0 + 1e-7:
        raise ValueError("Cosine discrepancy is outside its numerical range")
    return min(1.0, max(0.0, discrepancy))


def numerical_indicators(
    updates: tuple[DefenseUpdate, ...],
    *,
    histories: Mapping[str, Tensor] | None = None,
    epsilon: float = 1e-12,
    negligible_norm: float = 1e-12,
    minimum_history_clients: int = 3,
) -> dict[str, NumericalIndicators]:
    """Compute the manuscript's numerical indicators for one round population."""

    if not updates or len({update.client_id for update in updates}) != len(updates):
        raise ValueError("Numerical evaluation requires unique client updates")
    if any(not update.client_id for update in updates):
        raise ValueError("Numerical evaluation requires nonempty client IDs")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(negligible_norm) or negligible_norm < 0.0:
        raise ValueError("negligible_norm must be finite and nonnegative")
    if minimum_history_clients < 3:
        raise ValueError("History requires at least three valid participating clients")

    ordered = tuple(sorted(updates, key=lambda item: item.client_id))
    deltas = tuple(update.delta for update in ordered)
    center = coordinate_median(deltas)
    distances = tuple(float(torch.linalg.vector_norm(delta - center).item()) for delta in deltas)
    distance_reference = median(distances)
    norms = tuple(float(torch.linalg.vector_norm(delta).item()) for delta in deltas)
    norm_reference = median(norms)

    history_values: dict[str, float] = {}
    history_missing: dict[str, str] = {}
    available_histories = histories or {}
    for update in ordered:
        accepted_history = available_histories.get(update.client_id)
        if accepted_history is None:
            history_missing[update.client_id] = "accepted_history_missing"
            continue
        if accepted_history.shape != update.delta.shape or accepted_history.ndim != 1:
            raise ValueError("Accepted-history shape differs from its client update")
        discrepancy = cosine_discrepancy(
            update.delta,
            accepted_history,
            epsilon=epsilon,
            negligible_norm=negligible_norm,
        )
        if discrepancy is None:
            history_missing[update.client_id] = "accepted_history_negligible"
        else:
            history_values[update.client_id] = discrepancy

    history_globally_enabled = len(history_values) >= minimum_history_clients
    results: dict[str, NumericalIndicators] = {}
    for index, update in enumerate(ordered):
        direction = cosine_discrepancy(
            update.delta,
            center,
            epsilon=epsilon,
            negligible_norm=negligible_norm,
        )
        missing: dict[str, str] = {}
        if direction is None:
            missing["direction"] = "update_or_round_center_negligible"
        historical_discrepancy: float | None
        if history_globally_enabled:
            historical_discrepancy = history_values.get(update.client_id)
            if historical_discrepancy is None:
                missing["history"] = history_missing[update.client_id]
        else:
            historical_discrepancy = None
            missing["history"] = "fewer_than_three_valid_participating_histories"
        results[update.client_id] = NumericalIndicators(
            center=distances[index] / (distance_reference + epsilon),
            norm=abs(math.log((norms[index] + epsilon) / (norm_reference + epsilon))),
            direction=direction,
            history=historical_discrepancy,
            missing_reasons=missing,
        )
    return results


def numerical_credibility_population(
    indicators: Mapping[str, NumericalIndicators],
    *,
    coefficients: NumericalCoefficients | None = None,
    z_max: float = 20.0,
    epsilon: float = 1e-12,
) -> dict[str, NumericalCredibility]:
    """Transform numerical discrepancies into missing-aware credibility scores."""

    if not indicators or any(not client_id for client_id in indicators):
        raise ValueError("Numerical scoring requires nonempty client IDs")
    client_ids = tuple(sorted(indicators))
    indicator_names = ("center", "norm", "direction", "history")
    population_z: dict[str, dict[str, float]] = {}
    for name in indicator_names:
        available_ids = tuple(
            client_id
            for client_id in client_ids
            if getattr(indicators[client_id], name) is not None
        )
        if not available_ids:
            population_z[name] = {}
            continue
        values = tuple(float(getattr(indicators[client_id], name)) for client_id in available_ids)
        population_z[name] = dict(
            zip(
                available_ids,
                one_sided_robust_z(values, z_max=z_max, epsilon=epsilon),
                strict=True,
            )
        )

    configured = (coefficients or NumericalCoefficients()).to_dict()
    results: dict[str, NumericalCredibility] = {}
    for client_id in client_ids:
        raw = indicators[client_id]
        availability = {name: getattr(raw, name) is not None for name in indicator_names}
        active = renormalize_active_coefficients(configured, availability)
        z_values: dict[str, float | None] = {
            name: population_z[name].get(client_id) for name in indicator_names
        }
        exponent = 0.0
        for name, coefficient in active.items():
            value = z_values[name]
            if value is None:
                raise RuntimeError("An active numerical indicator lacks a robust score")
            exponent += coefficient * value
        results[client_id] = NumericalCredibility(
            score=math.exp(-exponent),
            indicators=raw,
            robust_z=z_values,
            active_coefficients=active,
            missing_reasons=dict(raw.missing_reasons),
        )
    return results


def evaluate_numerical_updates(
    updates: tuple[DefenseUpdate, ...],
    *,
    histories: Mapping[str, Tensor] | None = None,
    coefficients: NumericalCoefficients | None = None,
    z_max: float = 20.0,
    epsilon: float = 1e-12,
    negligible_norm: float = 1e-12,
    minimum_history_clients: int = 3,
) -> dict[str, NumericalCredibility]:
    """Compute indicators and numerical credibility for a submitted population."""

    indicators = numerical_indicators(
        updates,
        histories=histories,
        epsilon=epsilon,
        negligible_norm=negligible_norm,
        minimum_history_clients=minimum_history_clients,
    )
    return numerical_credibility_population(
        indicators,
        coefficients=coefficients,
        z_max=z_max,
        epsilon=epsilon,
    )
