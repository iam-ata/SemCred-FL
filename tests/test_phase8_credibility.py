"""Equation fixtures and scientific safeguards for Phase 8 credibility."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from semcred_fl.credibility.numerical_score import (
    coordinate_median,
    cosine_discrepancy,
    evaluate_numerical_updates,
    numerical_indicators,
)
from semcred_fl.credibility.robust_stats import (
    median,
    median_absolute_deviation,
    one_sided_robust_z,
    renormalize_active_coefficients,
)
from semcred_fl.credibility.semantic_score import (
    SemanticIndicators,
    mean_anchor_cross_entropy,
    reconstruct_client_model,
    semantic_credibility_population,
    semantic_discrepancy,
)
from semcred_fl.data.training_corpus import TrainingBatch
from semcred_fl.federation.parameters import flatten_parameters, parameter_manifest
from semcred_fl.federation.updates import DefenseUpdate
from semcred_fl.semantic.attribution import GroupedAttribution, grouped_gradient_times_input
from semcred_fl.semantic.pipeline import SemanticPriorBundle
from semcred_fl.semantic.prior import FeatureGroupLayout


def _update(client_id: str, values: tuple[float, ...]) -> DefenseUpdate:
    delta = torch.tensor(values, dtype=torch.float32)
    return DefenseUpdate(
        round_id=4,
        client_id=client_id,
        delta=delta,
        declared_sample_count=10,
        mean_local_loss=0.5,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        finite=True,
        missing_class_ids=(),
    )


def _layout() -> FeatureGroupLayout:
    return FeatureGroupLayout(
        dataset="synthetic",
        preprocessor_id="3" * 64,
        group_names=("g0", "g1"),
        coordinate_indices=((0,), (1,)),
        coordinate_names=("x0000", "x0001"),
    )


def _prior() -> SemanticPriorBundle:
    return SemanticPriorBundle(
        bundle_id="1" * 64,
        dataset="synthetic",
        preprocessed_corpus_id="2" * 64,
        preprocessor_id="3" * 64,
        anchor_id="4" * 64,
        feature_group_names=("g0", "g1"),
        class_names=("c0", "c1"),
        alpha_l=0.5,
        epsilon=1e-12,
        priors=((1.0, 0.0), (0.0, 1.0)),
    )


def test_median_mad_and_one_sided_robust_z_match_equations() -> None:
    assert median((-2.0, 4.0, 10.0, 20.0)) == 7.0
    assert median_absolute_deviation((1.0, 2.0, 3.0)) == 1.0
    assert one_sided_robust_z((5.0, 5.0, 5.0)) == (0.0, 0.0, 0.0)
    scores = one_sided_robust_z((0.0, 1.0, 2.0, 100.0), z_max=7.0)
    assert scores[0] == 0.0
    assert scores[-1] == 7.0
    assert list(scores) == sorted(scores)


def test_missing_indicator_coefficients_are_renormalized() -> None:
    semantic = renormalize_active_coefficients(
        {"loss": 0.5, "attribution": 0.5},
        {"loss": True, "attribution": False},
    )
    numerical = renormalize_active_coefficients(
        {"center": 0.25, "norm": 0.25, "direction": 0.25, "history": 0.25},
        {"center": True, "norm": True, "direction": True, "history": False},
    )
    assert semantic == {"loss": 1.0}
    assert numerical == pytest.approx(
        {"center": 1.0 / 3.0, "norm": 1.0 / 3.0, "direction": 1.0 / 3.0}
    )


def test_grouped_gradient_times_input_is_finite_and_restores_mode() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(((2.0, 0.0), (0.0, 2.0))))
    model.train()
    features = torch.tensor(((1.0, 0.1), (0.8, 0.2), (0.1, 1.0), (0.2, 0.8)))
    targets = torch.tensor((0, 0, 1, 1))
    attribution = grouped_gradient_times_input(
        model,
        features,
        targets,
        _layout(),
        num_classes=2,
    )
    assert model.training
    assert attribution.class_counts == (2, 2)
    assert len(attribution.values) == 2
    assert all(value >= 0.0 and math.isfinite(value) for row in attribution.values for value in row)


def test_semantic_discrepancy_rewards_prior_alignment_and_skips_zero_rows() -> None:
    aligned = GroupedAttribution(((2.0, 0.0), (0.0, 4.0)), (2, 2), ("g0", "g1"))
    reversed_rows = GroupedAttribution(
        ((0.0, 2.0), (4.0, 0.0)),
        (2, 2),
        ("g0", "g1"),
    )
    unavailable = GroupedAttribution(((0.0, 0.0), (0.0, 4.0)), (2, 2), ("g0", "g1"))
    aligned_value = semantic_discrepancy(aligned, _prior()).value
    reversed_value = semantic_discrepancy(reversed_rows, _prior()).value
    assert aligned_value is not None and reversed_value is not None
    assert aligned_value < reversed_value
    skipped = semantic_discrepancy(unavailable, _prior())
    assert skipped.valid_class_ids == (1,)
    assert skipped.unavailable_class_ids == (0,)


def test_anchor_loss_and_reconstruction_do_not_mutate_global_model() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    anchors = TrainingBatch(torch.eye(2), torch.tensor((0, 1)))
    assert mean_anchor_cross_entropy(model, anchors) == pytest.approx(math.log(2.0))
    manifest = parameter_manifest(model)
    before = flatten_parameters(model, manifest)
    reconstructed = reconstruct_client_model(
        model,
        _update("client-0", tuple(1.0 for _ in range(manifest.total_values))),
        manifest,
    )
    assert torch.equal(flatten_parameters(model, manifest), before)
    assert not torch.equal(flatten_parameters(reconstructed, manifest), before)


def test_adverse_semantic_population_indicators_lower_credibility() -> None:
    indicators = {
        "good": SemanticIndicators(0.0, 0.0, 2),
        "near": SemanticIndicators(0.1, 0.1, 2),
        "middle": SemanticIndicators(0.2, 0.2, 2),
        "adverse": SemanticIndicators(10.0, 1.0, 2),
        "loss-only": SemanticIndicators(
            0.1,
            None,
            0,
            {"attribution": "attribution_not_scheduled_this_round"},
        ),
    }
    scored = semantic_credibility_population(indicators)
    assert scored["adverse"].score < scored["middle"].score <= scored["good"].score
    assert scored["loss-only"].active_coefficients == {"loss": 1.0}
    assert scored["loss-only"].robust_z["attribution"] is None
    assert all(0.0 < result.score <= 1.0 for result in scored.values())


def test_coordinate_median_uses_midpoint_and_cosine_handles_negligible_vectors() -> None:
    center = coordinate_median(
        (
            torch.tensor((0.0, 10.0)),
            torch.tensor((2.0, 8.0)),
            torch.tensor((4.0, 6.0)),
            torch.tensor((6.0, 4.0)),
        )
    )
    assert torch.equal(center, torch.tensor((3.0, 7.0)))
    assert cosine_discrepancy(torch.zeros(2), torch.ones(2)) is None
    discrepancy = cosine_discrepancy(
        torch.tensor((1.0, 0.0)),
        torch.tensor((-1.0, 0.0)),
    )
    assert discrepancy == pytest.approx(1.0)


def test_numerical_history_gate_and_adverse_update_monotonicity() -> None:
    updates = (
        _update("a", (1.0, 0.0)),
        _update("b", (1.1, 0.0)),
        _update("c", (0.9, 0.0)),
        _update("adverse", (-10.0, 0.0)),
    )
    only_two_histories = {
        "a": torch.tensor((1.0, 0.0)),
        "b": torch.tensor((1.0, 0.0)),
    }
    gated = evaluate_numerical_updates(updates, histories=only_two_histories)
    assert all(result.indicators.history is None for result in gated.values())
    assert all("history" not in result.active_coefficients for result in gated.values())
    assert gated["adverse"].score < gated["a"].score

    valid_histories = {
        "a": torch.tensor((1.0, 0.0)),
        "b": torch.tensor((1.0, 0.0)),
        "c": torch.tensor((1.0, 0.0)),
        "adverse": torch.tensor((1.0, 0.0)),
    }
    scored = evaluate_numerical_updates(updates, histories=valid_histories)
    assert scored["adverse"].indicators.history == pytest.approx(1.0)
    assert scored["adverse"].score < scored["a"].score
    assert all(0.0 < result.score <= 1.0 for result in scored.values())


def test_negligible_round_center_disables_direction_per_client() -> None:
    updates = (
        _update("a", (-1.0, 0.0)),
        _update("b", (0.0, 0.0)),
        _update("c", (1.0, 0.0)),
    )
    indicators = numerical_indicators(updates)
    assert all(value.direction is None for value in indicators.values())
    scored = evaluate_numerical_updates(updates)
    assert all("direction" not in value.active_coefficients for value in scored.values())
    assert all(value.missing_reasons["direction"] for value in scored.values())
