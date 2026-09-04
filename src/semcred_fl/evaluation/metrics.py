"""Leakage-isolated Phase 11 evaluation metrics.

Defense code must finish before these functions receive simulated malicious
identity, attack success labels, or final-test predictions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateMetric:
    """A rate together with its auditable count evidence."""

    numerator: int
    denominator: int
    value: float | None
    undefined_reason: str | None = None

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator < 0 or self.numerator > self.denominator:
            raise ValueError("rate counts require 0 <= numerator <= denominator")
        if self.denominator == 0:
            if self.value is not None or not self.undefined_reason:
                raise ValueError("an empty denominator requires an undefined reason")
        elif (
            self.value is None
            or not math.isfinite(self.value)
            or not math.isclose(self.value, self.numerator / self.denominator)
            or self.undefined_reason is not None
        ):
            raise ValueError("a defined rate must equal numerator / denominator")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""

        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "undefined_reason": self.undefined_reason,
        }


def rate(numerator: int, denominator: int, *, empty_reason: str) -> RateMetric:
    """Construct a rate without replacing an empty denominator by zero."""

    if denominator == 0:
        return RateMetric(numerator, denominator, None, empty_reason)
    return RateMetric(numerator, denominator, numerator / denominator)


@dataclass(frozen=True, slots=True)
class RobustnessMetrics:
    """Utility degradation and attack-specific rate metrics."""

    clean_macro_f1: float
    attacked_macro_f1: float
    absolute_degradation: float
    relative_degradation: float
    attack_success_rate: RateMetric
    backdoor_success_rate: RateMetric


def robustness_metrics(
    clean_macro_f1: float,
    attacked_macro_f1: float,
    *,
    attack_successes: int,
    attack_eligible_records: int,
    backdoor_successes: int,
    backdoor_triggered_source_records: int,
    epsilon: float = 1e-12,
) -> RobustnessMetrics:
    """Compute manuscript equations (absolute and relative degradation)."""

    if (
        not math.isfinite(clean_macro_f1)
        or not math.isfinite(attacked_macro_f1)
        or not 0.0 <= clean_macro_f1 <= 1.0
        or not 0.0 <= attacked_macro_f1 <= 1.0
    ):
        raise ValueError("macro-F1 values must be finite and in [0,1]")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    absolute = clean_macro_f1 - attacked_macro_f1
    return RobustnessMetrics(
        clean_macro_f1=clean_macro_f1,
        attacked_macro_f1=attacked_macro_f1,
        absolute_degradation=absolute,
        relative_degradation=absolute / max(clean_macro_f1, epsilon),
        attack_success_rate=rate(
            attack_successes,
            attack_eligible_records,
            empty_reason="no attack-eligible records",
        ),
        backdoor_success_rate=rate(
            backdoor_successes,
            backdoor_triggered_source_records,
            empty_reason="no triggered source-class records",
        ),
    )


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Malicious-client ranking quality with explicit undefined states."""

    auroc: float | None
    auprc: float | None
    malicious_clients: int
    benign_clients: int
    undefined_reason: str | None = None


def malicious_client_ranking(
    malicious: Sequence[bool], suspiciousness: Sequence[float]
) -> RankingMetrics:
    """Compute tie-aware empirical AUROC and average precision.

    AUPRC uses threshold groups, so tied suspiciousness values cannot gain an
    advantage from arbitrary input order.
    """

    if not malicious or len(malicious) != len(suspiciousness):
        raise ValueError("ranking inputs require the same positive length")
    if any(not math.isfinite(score) for score in suspiciousness):
        raise ValueError("suspiciousness scores must be finite")
    positives = sum(malicious)
    negatives = len(malicious) - positives
    if positives == 0 or negatives == 0:
        return RankingMetrics(
            auroc=None,
            auprc=None,
            malicious_clients=positives,
            benign_clients=negatives,
            undefined_reason="both malicious and benign clients are required",
        )

    favorable = 0.0
    positive_scores = [
        score for label, score in zip(malicious, suspiciousness, strict=True) if label
    ]
    negative_scores = [
        score for label, score in zip(malicious, suspiciousness, strict=True) if not label
    ]
    for positive in positive_scores:
        for negative in negative_scores:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    auroc = favorable / (positives * negatives)

    groups: dict[float, tuple[int, int]] = {}
    for label, score in zip(malicious, suspiciousness, strict=True):
        group_positive, group_total = groups.get(score, (0, 0))
        groups[score] = (group_positive + int(label), group_total + 1)
    cumulative_positive = 0
    cumulative_total = 0
    average_precision = 0.0
    for score in sorted(groups, reverse=True):
        group_positive, group_total = groups[score]
        cumulative_positive += group_positive
        cumulative_total += group_total
        average_precision += (group_positive / positives) * (cumulative_positive / cumulative_total)
    return RankingMetrics(auroc, average_precision, positives, negatives)


@dataclass(frozen=True, slots=True)
class EligibilityMetrics:
    """Benign false rejection and malicious false acceptance."""

    benign_false_rejection: RateMetric
    malicious_false_acceptance: RateMetric


def eligibility_metrics(malicious: Sequence[bool], eligible: Sequence[bool]) -> EligibilityMetrics:
    """Score eligibility only after the defense decision is complete."""

    if not malicious or len(malicious) != len(eligible):
        raise ValueError("eligibility inputs require the same positive length")
    benign_total = sum(not label for label in malicious)
    malicious_total = sum(malicious)
    return EligibilityMetrics(
        benign_false_rejection=rate(
            sum(
                not label and not accepted
                for label, accepted in zip(malicious, eligible, strict=True)
            ),
            benign_total,
            empty_reason="no participating benign clients",
        ),
        malicious_false_acceptance=rate(
            sum(label and accepted for label, accepted in zip(malicious, eligible, strict=True)),
            malicious_total,
            empty_reason="no participating malicious clients",
        ),
    )


@dataclass(frozen=True, slots=True)
class TrustSeparation:
    """Signed benign-minus-malicious trust gaps."""

    mean_gap: float | None
    median_gap: float | None
    undefined_reason: str | None = None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def trust_separation(malicious: Sequence[bool], trust: Sequence[float]) -> TrustSeparation:
    """Compute raw trust gaps; positive values favor benign clients."""

    if not malicious or len(malicious) != len(trust):
        raise ValueError("trust inputs require the same positive length")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in trust):
        raise ValueError("trust values must be finite and in [0,1]")
    benign = [value for label, value in zip(malicious, trust, strict=True) if not label]
    adversarial = [value for label, value in zip(malicious, trust, strict=True) if label]
    if not benign or not adversarial:
        return TrustSeparation(None, None, "both malicious and benign clients are required")
    return TrustSeparation(
        mean_gap=sum(benign) / len(benign) - sum(adversarial) / len(adversarial),
        median_gap=_median(benign) - _median(adversarial),
    )


@dataclass(frozen=True, slots=True)
class DelayMetric:
    """Participation-event delay with right-censoring evidence."""

    participating_events: int
    delay: int | None
    censored: bool
    undefined_reason: str | None = None


def threshold_crossing_delay(
    trust: Sequence[float], threshold: Sequence[float], *, crossing: str
) -> DelayMetric:
    """Count participating events through the first threshold crossing.

    Use ``crossing='below'`` for attack detection and ``crossing='at_or_above'``
    for recovery after the recovery observation window has been selected.
    """

    if len(trust) != len(threshold):
        raise ValueError("trust and threshold sequences require the same length")
    if crossing not in {"below", "at_or_above"}:
        raise ValueError("crossing must be below or at_or_above")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (*trust, *threshold)):
        raise ValueError("trust and thresholds must be finite and in [0,1]")
    if not trust:
        return DelayMetric(0, None, True, "no participating events in observation window")
    for index, (value, boundary) in enumerate(zip(trust, threshold, strict=True), start=1):
        crossed = value < boundary if crossing == "below" else value >= boundary
        if crossed:
            return DelayMetric(index, index, False)
    return DelayMetric(len(trust), None, True, "threshold was not crossed before observation ended")


def trust_volatility(trust_by_client: Sequence[Sequence[float]]) -> tuple[float | None, int]:
    """Return RMS trust change over benign participation transitions."""

    squared_changes: list[float] = []
    for trajectory in trust_by_client:
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in trajectory):
            raise ValueError("trust trajectories must contain finite values in [0,1]")
        squared_changes.extend(
            (current - previous) ** 2
            for previous, current in zip(trajectory[:-1], trajectory[1:], strict=True)
        )
    if not squared_changes:
        return None, 0
    return math.sqrt(sum(squared_changes) / len(squared_changes)), len(squared_changes)
