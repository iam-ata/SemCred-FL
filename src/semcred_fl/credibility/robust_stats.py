"""Robust population statistics and missing-aware coefficient handling."""

from __future__ import annotations

import math
from collections.abc import Mapping


def median(values: tuple[float, ...]) -> float:
    """Return the midpoint median of a nonempty finite population."""

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Median requires a nonempty finite population")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def median_absolute_deviation(values: tuple[float, ...]) -> float:
    """Return the unscaled median absolute deviation."""

    center = median(values)
    return median(tuple(abs(value - center) for value in values))


def one_sided_robust_z(
    values: tuple[float, ...],
    *,
    z_max: float = 20.0,
    epsilon: float = 1e-12,
) -> tuple[float, ...]:
    """Apply the manuscript's capped, one-sided median/MAD anomaly transform."""

    if not math.isfinite(z_max) or z_max <= 0.0:
        raise ValueError("z_max must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    center = median(values)
    denominator = 1.4826 * median_absolute_deviation(values) + epsilon
    return tuple(min(z_max, max(0.0, (value - center) / denominator)) for value in values)


def renormalize_active_coefficients(
    coefficients: Mapping[str, float],
    availability: Mapping[str, bool],
) -> dict[str, float]:
    """Remove unavailable indicators and normalize the remaining nonzero weights."""

    if not coefficients or set(coefficients) != set(availability):
        raise ValueError("Coefficient and availability keys must be nonempty and identical")
    if any(not math.isfinite(value) or value < 0.0 for value in coefficients.values()):
        raise ValueError("Credibility coefficients must be finite and nonnegative")
    active = {
        name: coefficients[name]
        for name, available in availability.items()
        if available and coefficients[name] > 0.0
    }
    total = sum(active.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("No positive credibility coefficient remains active")
    return {name: value / total for name, value in active.items()}
