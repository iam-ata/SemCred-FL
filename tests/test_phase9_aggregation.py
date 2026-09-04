"""Analytical and state-transition tests for Phase 9 SemCred aggregation."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from semcred_fl.aggregation.burn_in import aggregate_burn_in
from semcred_fl.aggregation.clipping import ClippingResult, clip_eligible_updates, clip_update
from semcred_fl.aggregation.eligibility import (
    EligibilityResult,
    adaptive_trust_threshold,
    minimum_retained_count,
    select_burn_in_history_clients,
    select_eligible_clients,
)
from semcred_fl.aggregation.semcred import (
    AggregationRecord,
    aggregate_semcred_updates,
    weighted_clipped_sum,
)
from semcred_fl.aggregation.trust import asymmetric_trust
from semcred_fl.aggregation.weighting import WeightingResult, trust_aware_weights
from semcred_fl.config import AggregationConfig
from semcred_fl.credibility.fusion import fuse_credibility
from semcred_fl.federation.server import process_semcred_round
from semcred_fl.federation.state import (
    ClientState,
    admit_clipped_histories,
    update_participating_trust,
)
from semcred_fl.federation.updates import DefenseUpdate


def _update(
    client_id: str,
    values: tuple[float, ...],
    *,
    sample_count: int = 10,
    round_id: int = 0,
) -> DefenseUpdate:
    delta = torch.tensor(values, dtype=torch.float64)
    return DefenseUpdate(
        round_id=round_id,
        client_id=client_id,
        delta=delta,
        declared_sample_count=sample_count,
        mean_local_loss=0.5,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        finite=True,
        missing_class_ids=(),
    )


def test_fusion_is_symmetric_at_half_and_penalizes_disagreement() -> None:
    left = fuse_credibility(0.8, 0.2, semantic_coefficient=0.5)
    right = fuse_credibility(0.2, 0.8, semantic_coefficient=0.5)
    agreement = fuse_credibility(0.4, 0.4, semantic_coefficient=0.5)
    assert left.score == pytest.approx(right.score)
    assert left.score < agreement.score
    assert left.disagreement_penalty_applied
    numerical_only = fuse_credibility(None, 0.7)
    assert numerical_only.score == 0.7
    assert numerical_only.active_views == ("numerical",)
    assert not numerical_only.disagreement_penalty_applied


def test_trust_falls_faster_than_it_recovers_for_equal_evidence_distance() -> None:
    fallen = asymmetric_trust(0.5, 0.0, decrease_rate=0.6, recovery_rate=0.1)
    recovered = asymmetric_trust(0.5, 1.0, decrease_rate=0.6, recovery_rate=0.1)
    assert 0.5 - fallen > recovered - 0.5
    assert fallen == pytest.approx(0.2)
    assert recovered == pytest.approx(0.55)


def test_inactive_trust_and_rejected_history_remain_unchanged() -> None:
    inactive = ClientState(0.8, torch.tensor((1.0, 0.0)), 0, 0, (0,))
    states = {"inactive": inactive}
    updated = update_participating_trust(states, {"active": 0.6}, round_id=1)
    assert updated["inactive"] is inactive
    before_rejection = updated["active"]
    after_rejection = admit_clipped_histories(updated, {}, round_id=1)
    assert after_rejection["active"] is before_rejection


def test_history_initializes_then_uses_event_driven_ema() -> None:
    state0 = update_participating_trust({}, {"a": 0.9}, round_id=0)
    state1 = admit_clipped_histories(
        state0,
        {"a": torch.tensor((1.0, 0.0))},
        round_id=0,
        history_decay=0.8,
    )
    assert torch.equal(state1["a"].accepted_history, torch.tensor((1.0, 0.0)))
    state2 = update_participating_trust(state1, {"a": 0.9}, round_id=1)
    state3 = admit_clipped_histories(
        state2,
        {"a": torch.tensor((0.0, 1.0))},
        round_id=1,
        history_decay=0.8,
    )
    expected = torch.tensor((0.8, 0.2))
    assert torch.allclose(state3["a"].accepted_history, expected)
    assert state3["a"].accepted_rounds == (0, 1)


def test_minimum_retention_is_nonempty_and_breaks_ties_by_client_id() -> None:
    eligibility = select_eligible_clients(
        {"d": 0.05, "c": 0.05, "b": 0.05, "a": 0.05},
        minimum_trust=0.1,
        retained_ratio=0.5,
    )
    assert eligibility.threshold_client_ids == ()
    assert eligibility.eligible_client_ids == ("a", "b")
    assert eligibility.retention_added_client_ids == ("a", "b")


def test_clipping_respects_bound_and_preserves_positive_scale_direction() -> None:
    updates = {
        update.client_id: update
        for update in (
            _update("a", (1.0, 0.0)),
            _update("b", (2.0, 0.0)),
            _update("c", (10.0, 0.0)),
        )
    }
    result = clip_eligible_updates(updates, ("a", "b", "c"), clipping_multiplier=2.5)
    assert result.bound == pytest.approx(5.0)
    assert result.clipped_norms["c"] <= result.bound + 1e-9
    assert torch.dot(updates["c"].delta, result.clipped_updates["c"]).item() > 0.0
    zero_bound = clip_eligible_updates(
        {"zero": _update("zero", (0.0, 0.0))},
        ("zero",),
    )
    assert torch.equal(zero_bound.clipped_updates["zero"], torch.zeros(2, dtype=torch.float64))


def test_sample_cap_and_weights_are_finite_normalized_and_count_independent_at_xi_zero() -> None:
    updates = {
        update.client_id: update
        for update in (
            _update("a", (1.0,), sample_count=10),
            _update("b", (1.0,), sample_count=20),
            _update("c", (1.0,), sample_count=100),
        )
    }
    trusts = {"a": 0.5, "b": 0.5, "c": 0.5}
    capped = trust_aware_weights(
        updates,
        trusts,
        ("a", "b", "c"),
        sample_count_exponent=0.5,
        sample_cap_median_multiplier=2.0,
    )
    assert capped.sample_count_cap == pytest.approx(40.0)
    assert capped.capped_sample_counts["c"] == 40.0
    assert math.isclose(sum(capped.weights.values()), 1.0)
    confirmatory = trust_aware_weights(updates, trusts, ("a", "b", "c"))
    assert confirmatory.sample_count_cap is None
    assert confirmatory.weights == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_post_burn_in_weighted_sum_matches_analytical_fixture() -> None:
    updates = (_update("a", (1.0,)), _update("b", (3.0,)))
    controls = AggregationConfig(
        threshold_multiplier=0.0,
        minimum_retained_ratio=1.0,
        clipping_multiplier=10.0,
    )
    result = aggregate_semcred_updates(updates, {"a": 0.5, "b": 0.5}, config=controls)
    assert result.weights == pytest.approx({"a": 0.5, "b": 0.5})
    assert result.aggregate_delta.item() == pytest.approx(2.0)


def test_burn_in_uses_coordinate_median_and_controls_history_admission() -> None:
    updates = (
        _update("a", (1.0, 0.0)),
        _update("b", (2.0, 0.0)),
        _update("c", (3.0, 0.0)),
        _update("d", (100.0, 0.0)),
    )
    result = aggregate_burn_in(updates, {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    assert torch.equal(result.aggregate_delta, torch.tensor((2.5, 0.0), dtype=torch.float64))
    assert result.admitted_client_ids == ("a", "b")
    assert set(result.history_clipping.clipped_updates) == {"a", "b"}


def test_server_round_integration_is_deterministic_and_admits_only_eligible_history() -> None:
    controls = AggregationConfig(
        burn_in_rounds=1,
        threshold_multiplier=0.0,
        minimum_retained_ratio=0.5,
    )
    inactive = ClientState(0.7, torch.tensor((0.5, 0.0), dtype=torch.float64), 0, 0, (0,))
    round0_updates = tuple(
        _update(client_id, values, round_id=0)
        for client_id, values in {
            "a": (1.0, 0.0),
            "b": (2.0, 0.0),
            "c": (3.0, 0.0),
            "d": (100.0, 0.0),
        }.items()
    )
    semantic0 = {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}
    numerical0 = dict(semantic0)
    first = process_semcred_round(
        round_id=0,
        updates=round0_updates,
        semantic_scores=semantic0,
        numerical_scores=numerical0,
        client_states={"inactive": inactive},
        config=controls,
    )
    assert first.burn_in
    assert first.client_states["inactive"] is inactive
    assert first.history_admitted_client_ids == ("a", "b")
    assert first.client_states["c"].accepted_history is None

    round1_updates = tuple(
        _update(update.client_id, tuple(float(value) for value in update.delta), round_id=1)
        for update in round0_updates
    )
    semantic1 = {"a": 0.9, "b": 0.8, "c": 0.0, "d": 0.0}
    numerical1 = dict(semantic1)
    second = process_semcred_round(
        round_id=1,
        updates=round1_updates,
        semantic_scores=semantic1,
        numerical_scores=numerical1,
        client_states=first.client_states,
        config=controls,
    )
    repeated = process_semcred_round(
        round_id=1,
        updates=round1_updates,
        semantic_scores=semantic1,
        numerical_scores=numerical1,
        client_states=first.client_states,
        config=controls,
    )
    assert not second.burn_in
    assert second.eligible_client_ids == ("a", "b")
    assert second.history_admitted_client_ids == ("a", "b")
    assert second.client_states["c"].accepted_history is None
    assert second.client_states["d"].accepted_history is None
    assert second.client_states["inactive"] is inactive
    assert torch.equal(second.aggregate_delta, repeated.aggregate_delta)
    assert second.weights == repeated.weights


def test_phase9_result_contracts_fail_closed() -> None:
    vector = torch.tensor((1.0,))
    with pytest.raises(ValueError, match="Clipping bound"):
        ClippingResult(-1.0, {"a": vector}, {"a": 1.0}, {"a": 1.0})
    with pytest.raises(ValueError, match="inventories"):
        ClippingResult(1.0, {}, {}, {})
    with pytest.raises(ValueError, match="exceeds"):
        ClippingResult(1.0, {"a": vector}, {"a": 2.0}, {"a": 2.0})
    with pytest.raises(ValueError, match="threshold"):
        EligibilityResult(float("nan"), 1, (), ("a",), ())
    with pytest.raises(ValueError, match="minimum retention"):
        EligibilityResult(0.1, 2, (), ("a",), ())
    with pytest.raises(ValueError, match="sample-count cap"):
        WeightingResult(0.0, {"a": 1.0}, {"a": 1.0})
    with pytest.raises(ValueError, match="inventories"):
        WeightingResult(None, {}, {})
    with pytest.raises(ValueError, match="nonnegative"):
        WeightingResult(None, {"a": 1.0}, {"a": -1.0})
    with pytest.raises(ValueError, match="sum to one"):
        WeightingResult(None, {"a": 1.0}, {"a": 0.5})


def test_aggregation_record_contract_rejects_inconsistent_outputs() -> None:
    vector = torch.tensor((1.0,))
    valid = {
        "threshold": 0.1,
        "minimum_count": 1,
        "eligible_client_ids": ("a",),
        "retention_added_client_ids": (),
        "clipping_bound": 1.0,
        "sample_count_cap": None,
        "weights": {"a": 1.0},
        "clipped_updates": {"a": vector},
        "aggregate_delta": vector,
    }
    with pytest.raises(ValueError, match="threshold"):
        AggregationRecord(**{**valid, "threshold": -1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="minimum retention"):
        AggregationRecord(**{**valid, "eligible_client_ids": ()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weights differ"):
        AggregationRecord(**{**valid, "weights": {"b": 1.0}})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clipped updates differ"):
        AggregationRecord(**{**valid, "clipped_updates": {"b": vector}})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite flat"):
        AggregationRecord(  # type: ignore[arg-type]
            **{**valid, "aggregate_delta": torch.tensor((float("nan"),))}
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: minimum_retained_count(0), "nonempty"),
        (lambda: minimum_retained_count(2, 0.0), "retained_ratio"),
        (lambda: adaptive_trust_threshold({}), "participating"),
        (lambda: adaptive_trust_threshold({"a": 2.0}), "trust values"),
        (lambda: adaptive_trust_threshold({"a": 0.5}, minimum_trust=-1.0), "minimum_trust"),
        (
            lambda: adaptive_trust_threshold({"a": 0.5}, threshold_multiplier=-1.0),
            "threshold_multiplier",
        ),
        (lambda: select_burn_in_history_clients({}), "participating"),
        (lambda: select_burn_in_history_clients({"a": float("nan")}), "credibility"),
        (lambda: asymmetric_trust(2.0, 0.5), "Trust and credibility"),
        (lambda: asymmetric_trust(0.5, 0.5, recovery_rate=0.7), "recovery"),
    ],
)
def test_phase9_scalar_validation(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_burn_in_full_retention_and_input_validation() -> None:
    scores = {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}
    retained = select_burn_in_history_clients(scores, retained_ratio=1.0)
    assert retained.eligible_client_ids == ("a", "b", "c", "d")
    assert retained.retention_added_client_ids == ("c", "d")
    with pytest.raises(ValueError, match="unique"):
        aggregate_burn_in((), {})
    with pytest.raises(ValueError, match="population differs"):
        aggregate_burn_in((_update("a", (1.0,)),), {"b": 0.5})


def test_clipping_and_weighting_input_validation() -> None:
    valid = _update("a", (1.0,))
    updates = {"a": valid}
    with pytest.raises(ValueError, match="finite nonempty"):
        clip_update(torch.tensor(((1.0,),)), 1.0)
    with pytest.raises(ValueError, match="bound"):
        clip_update(torch.tensor((1.0,)), -1.0)
    with pytest.raises(ValueError, match="epsilon"):
        clip_update(torch.tensor((1.0,)), 1.0, epsilon=0.0)
    with pytest.raises(ValueError, match="unique eligible"):
        clip_eligible_updates(updates, ())
    with pytest.raises(ValueError, match="no accepted"):
        clip_eligible_updates(updates, ("b",))
    with pytest.raises(ValueError, match="clipping_multiplier"):
        clip_eligible_updates(updates, ("a",), clipping_multiplier=0.0)
    nan_update = _update("nan", (float("nan"),))
    with pytest.raises(ValueError, match="non-finite"):
        clip_eligible_updates({"nan": nan_update}, ("nan",))

    with pytest.raises(ValueError, match="unique eligible"):
        trust_aware_weights(updates, {"a": 0.5}, ())
    with pytest.raises(ValueError, match="lacks"):
        trust_aware_weights(updates, {"a": 0.5}, ("b",))
    for kwargs, message in (
        ({"sample_count_exponent": 2.0}, "sample_count_exponent"),
        ({"sample_cap_median_multiplier": 0.0}, "sample_cap"),
        ({"trust_exponent": 0.0}, "trust_exponent"),
        ({"epsilon": 0.0}, "epsilon"),
    ):
        with pytest.raises(ValueError, match=message):
            trust_aware_weights(updates, {"a": 0.5}, ("a",), **kwargs)
    with pytest.raises(ValueError, match="sample counts"):
        trust_aware_weights({"a": _update("a", (1.0,), sample_count=0)}, {"a": 0.5}, ("a",))
    with pytest.raises(ValueError, match="trust values"):
        trust_aware_weights(updates, {"a": float("nan")}, ("a",))


def test_semcred_aggregation_rejects_malformed_populations() -> None:
    with pytest.raises(ValueError, match="unique"):
        aggregate_semcred_updates((), {})
    with pytest.raises(ValueError, match="flat-vector shape"):
        aggregate_semcred_updates(
            (_update("a", (1.0, 2.0)), _update("b", (1.0,))),
            {"a": 0.5, "b": 0.5},
        )
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_semcred_updates((_update("a", (float("nan"),)),), {"a": 0.5})
    with pytest.raises(ValueError, match="population differs"):
        aggregate_semcred_updates((_update("a", (1.0,)),), {"b": 0.5})
    clipping = ClippingResult(1.0, {"a": torch.tensor((1.0,))}, {"a": 1.0}, {"a": 1.0})
    weighting = WeightingResult(None, {"b": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="inventories differ"):
        weighted_clipped_sum(clipping, weighting)


def test_fusion_and_state_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="At least one"):
        fuse_credibility(None, None)
    with pytest.raises(ValueError, match="Semantic credibility"):
        fuse_credibility(2.0, 0.5)
    with pytest.raises(ValueError, match="semantic_coefficient"):
        fuse_credibility(0.5, 0.5, semantic_coefficient=2.0)
    with pytest.raises(ValueError, match="disagreement_penalty"):
        fuse_credibility(0.5, 0.5, disagreement_penalty=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        fuse_credibility(0.5, 0.5, epsilon=0.0)
    semantic_only = fuse_credibility(0.6, None)
    assert semantic_only.score == 0.6
    with pytest.raises(ValueError, match="valid round"):
        update_participating_trust({}, {}, round_id=0)
    with pytest.raises(ValueError, match="recovery"):
        update_participating_trust({}, {"a": 0.5}, round_id=0, recovery_rate=0.7)
    with pytest.raises(ValueError, match="finite"):
        update_participating_trust({}, {"a": float("nan")}, round_id=0)
