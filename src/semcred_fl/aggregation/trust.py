"""Asymmetric persistent-trust update."""

from __future__ import annotations

import math


def asymmetric_trust(
    previous_trust: float,
    credibility: float,
    *,
    decrease_rate: float = 0.6,
    recovery_rate: float = 0.1,
) -> float:
    """Apply faster decrease than recovery to one participating client."""

    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in (previous_trust, credibility)
    ):
        raise ValueError("Trust and credibility must be finite and in [0,1]")
    if not 0.0 < recovery_rate < decrease_rate <= 1.0:
        raise ValueError("Trust recovery must be slower than trust decrease")
    rate = decrease_rate if credibility < previous_trust else recovery_rate
    updated = (1.0 - rate) * previous_trust + rate * credibility
    return min(1.0, max(0.0, updated))
