"""Deterministic descriptive and paired statistical analysis."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    """Seed-level summary with a deterministic percentile-bootstrap interval."""

    count: int
    mean: float
    sample_standard_deviation: float
    median: float
    confidence_level: float
    bootstrap_lower: float
    bootstrap_upper: float
    bootstrap_resamples: int
    bootstrap_seed: int


def describe(
    values: Sequence[float],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> DescriptiveStatistics:
    """Summarize independent seeds and bootstrap their mean."""

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("descriptive statistics require finite observations")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0,1)")
    count = len(values)
    mean = sum(values) / count
    standard_deviation = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1)) if count > 1 else 0.0
    )
    generator = random.Random(bootstrap_seed)
    bootstrap_means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return DescriptiveStatistics(
        count=count,
        mean=mean,
        sample_standard_deviation=standard_deviation,
        median=_median(values),
        confidence_level=confidence_level,
        bootstrap_lower=_percentile(bootstrap_means, alpha),
        bootstrap_upper=_percentile(bootstrap_means, 1.0 - alpha),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """Exact paired Wilcoxon result using the Wilcox zero convention."""

    pairs_total: int
    nonzero_pairs: int
    zero_differences: int
    tied_absolute_ranks: bool
    positive_rank_sum: float
    negative_rank_sum: float
    statistic: float
    exact_two_sided_p_value: float
    rank_biserial_correlation: float
    zero_convention: str = "wilcox_discard_exact_zero"


@dataclass(frozen=True, slots=True)
class PairedComparisonInput:
    """One preregistered contrast in a declared Holm family."""

    comparison_id: str
    comparison_family_id: str
    left_by_pair: Mapping[str, float]
    right_by_pair: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.comparison_id or not self.comparison_family_id:
            raise ValueError("comparison and family identifiers are required")


@dataclass(frozen=True, slots=True)
class CorrectedComparison:
    """Exact paired result with its family-wise Holm-adjusted p-value."""

    comparison_id: str
    comparison_family_id: str
    result: WilcoxonResult
    holm_adjusted_p_value: float


def _midranks(values: Sequence[float]) -> tuple[list[float], bool]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    tied = False
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        tied = tied or end - start > 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks, tied


def paired_wilcoxon_exact(
    left_by_pair: Mapping[str, float],
    right_by_pair: Mapping[str, float],
) -> WilcoxonResult:
    """Run an exact two-sided sign-permutation Wilcoxon signed-rank test.

    Pair identifiers must match exactly. Absolute-difference ties use midranks;
    exact zero differences are discarded and logged.
    """

    if not left_by_pair or set(left_by_pair) != set(right_by_pair):
        raise ValueError("paired Wilcoxon inputs require identical nonempty pair IDs")
    pair_ids = sorted(left_by_pair)
    differences = [left_by_pair[pair_id] - right_by_pair[pair_id] for pair_id in pair_ids]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("paired differences must be finite")
    nonzero = [value for value in differences if value != 0.0]
    zero_count = len(differences) - len(nonzero)
    if not nonzero:
        return WilcoxonResult(
            pairs_total=len(differences),
            nonzero_pairs=0,
            zero_differences=zero_count,
            tied_absolute_ranks=False,
            positive_rank_sum=0.0,
            negative_rank_sum=0.0,
            statistic=0.0,
            exact_two_sided_p_value=1.0,
            rank_biserial_correlation=0.0,
        )

    ranks, tied = _midranks([abs(value) for value in nonzero])
    positive = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0.0)
    negative = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value < 0.0)
    scaled_ranks = [round(rank * 2.0) for rank in ranks]
    observed_scaled = round(positive * 2.0)
    counts = [0] * (sum(scaled_ranks) + 1)
    counts[0] = 1
    reachable = 0
    for rank in scaled_ranks:
        for subtotal in range(reachable, -1, -1):
            if counts[subtotal]:
                counts[subtotal + rank] += counts[subtotal]
        reachable += rank
    permutations = 2 ** len(scaled_ranks)
    lower = sum(counts[: observed_scaled + 1]) / permutations
    upper = sum(counts[observed_scaled:]) / permutations
    p_value = min(1.0, 2.0 * min(lower, upper))
    total_rank = positive + negative
    return WilcoxonResult(
        pairs_total=len(differences),
        nonzero_pairs=len(nonzero),
        zero_differences=zero_count,
        tied_absolute_ranks=tied,
        positive_rank_sum=positive,
        negative_rank_sum=negative,
        statistic=min(positive, negative),
        exact_two_sided_p_value=p_value,
        rank_biserial_correlation=(positive - negative) / total_rank,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise error correction."""

    if not p_values:
        raise ValueError("Holm correction requires at least one comparison")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must be finite and in [0,1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (comparison_id, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[comparison_id] = running
    return {comparison_id: adjusted[comparison_id] for comparison_id in sorted(adjusted)}


def analyze_comparison_family(
    comparisons: Sequence[PairedComparisonInput],
) -> tuple[CorrectedComparison, ...]:
    """Run a declared paired-comparison family and apply Holm correction once."""

    if not comparisons:
        raise ValueError("comparison analysis requires a declared family")
    family_ids = {comparison.comparison_family_id for comparison in comparisons}
    comparison_ids = [comparison.comparison_id for comparison in comparisons]
    if len(family_ids) != 1:
        raise ValueError("one analysis call may contain only one correction family")
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("comparison identifiers must be unique within a family")
    results = {
        comparison.comparison_id: paired_wilcoxon_exact(
            comparison.left_by_pair, comparison.right_by_pair
        )
        for comparison in comparisons
    }
    adjusted = holm_adjust(
        {comparison_id: result.exact_two_sided_p_value for comparison_id, result in results.items()}
    )
    family_id = next(iter(family_ids))
    return tuple(
        CorrectedComparison(
            comparison_id=comparison_id,
            comparison_family_id=family_id,
            result=results[comparison_id],
            holm_adjusted_p_value=adjusted[comparison_id],
        )
        for comparison_id in sorted(results)
    )
