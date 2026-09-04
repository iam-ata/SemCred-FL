"""Analytical and source-parity checks for the Phase 10 baseline cores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from semcred_fl.baselines import (
    agat_trust_aggregate,
    aggregate_public_labels,
    apply_local_verifier_mask,
    centered_kernel_alignment,
    fed_lsae_aggregate,
    fed_lsae_partition,
    multi_krum,
    rbf_cka,
    rbf_kernel,
    trimmed_mean,
)
from semcred_fl.federation.parameters import ParameterManifest
from semcred_fl.federation.sampling import generate_client_schedule
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


def _update(client_id: str, values: list[float], count: int = 1) -> DefenseUpdate:
    delta = torch.tensor(values, dtype=torch.float64)
    return DefenseUpdate(
        round_id=0,
        client_id=client_id,
        delta=delta,
        declared_sample_count=count,
        mean_local_loss=1.0,
        update_norm=float(torch.linalg.vector_norm(delta).item()),
        finite=True,
        missing_class_ids=(),
    )


def test_trimmed_mean_matches_coordinatewise_analytical_fixture() -> None:
    updates = (
        _update("a", [0.0, 0.0]),
        _update("b", [1.0, 10.0]),
        _update("c", [2.0, 20.0]),
        _update("d", [3.0, 30.0]),
        _update("e", [100.0, 40.0]),
    )

    result = trimmed_mean(updates, trim_count=1)

    torch.testing.assert_close(result.delta, torch.tensor([2.0, 20.0], dtype=torch.float64))
    assert result.method == "trimmed_mean" and result.weights == {}


@pytest.mark.parametrize("trim_count", [-1, 2, True])
def test_trimmed_mean_rejects_invalid_trim_counts(trim_count: int) -> None:
    updates = (_update("a", [0.0]), _update("b", [1.0]), _update("c", [2.0]))
    with pytest.raises(ValueError):
        trimmed_mean(updates, trim_count=trim_count)


@pytest.mark.parametrize(
    "updates",
    [
        (),
        (_update("a", [0.0]), _update("a", [1.0])),
        (_update("a", [0.0]), _update("b", [1.0, 2.0])),
        (_update("a", [float("nan")]),),
    ],
)
def test_trimmed_mean_rejects_malformed_update_populations(
    updates: tuple[DefenseUpdate, ...],
) -> None:
    with pytest.raises(ValueError):
        trimmed_mean(updates, trim_count=0)


def test_multi_krum_excludes_distant_update_and_uses_equal_selected_weights() -> None:
    updates = tuple(
        _update(client_id, [value])
        for client_id, value in (
            ("a", 0.0),
            ("b", 0.1),
            ("c", 0.2),
            ("d", 0.3),
            ("outlier", 10.0),
        )
    )

    result = multi_krum(updates, assumed_byzantine=1)

    torch.testing.assert_close(result.delta, torch.tensor([0.15], dtype=torch.float64))
    assert result.weights == {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}


def test_multi_krum_enforces_admissibility_and_selection_contract() -> None:
    four = tuple(_update(str(index), [float(index)]) for index in range(4))
    five = tuple(_update(str(index), [float(index)]) for index in range(5))
    with pytest.raises(ValueError, match=r"n >= 2f \+ 3"):
        multi_krum(four, assumed_byzantine=1)
    with pytest.raises(ValueError, match="selected count"):
        multi_krum(five, assumed_byzantine=1, selected_count=5)


@pytest.mark.parametrize(
    "updates,assumed_byzantine",
    [
        ((), 0),
        ((_update("a", [0.0]), _update("a", [1.0]), _update("c", [2.0])), 0),
        ((_update("a", [0.0]), _update("b", [1.0, 2.0]), _update("c", [2.0])), 0),
        ((_update("a", [float("nan")]), _update("b", [1.0]), _update("c", [2.0])), 0),
        ((_update("a", [0.0]), _update("b", [1.0]), _update("c", [2.0])), -1),
    ],
)
def test_multi_krum_rejects_malformed_contracts(
    updates: tuple[DefenseUpdate, ...], assumed_byzantine: int
) -> None:
    with pytest.raises(ValueError):
        multi_krum(updates, assumed_byzantine=assumed_byzantine)


def test_multi_krum_rejects_finite_inputs_whose_squared_distances_overflow() -> None:
    updates = (
        _update("a", [-1e200]),
        _update("b", [0.0]),
        _update("c", [1e200]),
    )
    with pytest.raises(ValueError, match="candidate score"):
        multi_krum(updates, assumed_byzantine=0)


def test_agat_official_archive_weighted_sum_core_matches_fixture() -> None:
    result = agat_trust_aggregate(
        (_update("a", [1.0, 3.0]), _update("b", [3.0, 1.0])),
        {"a": 0.25, "b": 0.75},
    )

    torch.testing.assert_close(result.delta, torch.tensor([2.5, 1.5], dtype=torch.float64))
    assert result.method == "agat_fl_trust_core"


def test_agat_core_requires_exact_population_and_normalized_trust() -> None:
    updates = (_update("a", [1.0]), _update("b", [2.0]))
    with pytest.raises(ValueError, match="populations"):
        agat_trust_aggregate(updates, {"a": 1.0})
    with pytest.raises(ValueError, match="sum to one"):
        agat_trust_aggregate(updates, {"a": 0.2, "b": 0.2})


@pytest.mark.parametrize(
    "updates,trust,tolerance",
    [
        ((), {}, 1e-6),
        ((_update("a", [1.0]), _update("a", [2.0])), {"a": 1.0}, 1e-6),
        (
            (_update("a", [1.0]), _update("b", [2.0, 3.0])),
            {"a": 0.5, "b": 0.5},
            1e-6,
        ),
        (
            (_update("a", [float("nan")]), _update("b", [2.0])),
            {"a": 0.5, "b": 0.5},
            1e-6,
        ),
        ((_update("a", [1.0]),), {"a": 1.0}, 0.0),
        ((_update("a", [1.0]), _update("b", [2.0])), {"a": -1.0, "b": 2.0}, 1e-6),
    ],
)
def test_agat_core_rejects_malformed_inputs(
    updates: tuple[DefenseUpdate, ...], trust: dict[str, float], tolerance: float
) -> None:
    with pytest.raises(ValueError):
        agat_trust_aggregate(updates, trust, tolerance=tolerance)


def test_rbf_cka_is_one_for_identical_representations() -> None:
    representation = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=torch.float64
    )
    assert rbf_cka(representation, representation.clone(), bandwidth=1.0) == pytest.approx(1.0)


def test_rbf_and_cka_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        rbf_kernel(torch.tensor([1.0]), bandwidth=1.0)
    with pytest.raises(ValueError, match="finite"):
        rbf_kernel(torch.tensor([[0.0], [float("nan")]]), bandwidth=1.0)
    with pytest.raises(ValueError, match="bandwidth"):
        rbf_kernel(torch.tensor([[0.0], [1.0]]), bandwidth=0.0)
    with pytest.raises(ValueError, match="equal square"):
        centered_kernel_alignment(torch.eye(2), torch.eye(3))
    with pytest.raises(ValueError, match="finite"):
        centered_kernel_alignment(torch.eye(2), torch.full((2, 2), float("nan")))
    with pytest.raises(ValueError, match="epsilon"):
        centered_kernel_alignment(torch.eye(2), torch.eye(2), epsilon=0.0)
    with pytest.raises(ValueError, match="degenerate"):
        centered_kernel_alignment(torch.ones((3, 3)), torch.ones((3, 3)))
    with pytest.raises(ValueError, match="sample population"):
        rbf_cka(torch.ones((2, 1)), torch.ones((3, 1)), bandwidth=1.0)


def test_fed_lsae_filters_smaller_cluster_and_applies_fedavg() -> None:
    updates = (
        _update("a", [1.0]),
        _update("b", [2.0]),
        _update("c", [3.0]),
        _update("suspect", [100.0]),
    )
    scores = {"a": 0.90, "b": 0.92, "c": 0.88, "suspect": 0.10}

    benign, suspected = fed_lsae_partition(scores)
    result = fed_lsae_aggregate(updates, scores)

    assert benign == ("a", "b", "c") and suspected == ("suspect",)
    torch.testing.assert_close(result.delta, torch.tensor([2.0], dtype=torch.float64))
    assert set(result.weights) == {"a", "b", "c"}


def test_fed_lsae_fails_when_paper_cluster_label_rule_is_ambiguous() -> None:
    with pytest.raises(ValueError, match="equal-size"):
        fed_lsae_partition({"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})


@pytest.mark.parametrize(
    "scores,message",
    [
        ({"a": 0.1, "b": 0.9}, "at least three"),
        ({"": 0.1, "b": 0.8, "c": 0.9}, "nonempty"),
        ({"a": -0.1, "b": 0.8, "c": 0.9}, r"in \[0,1\]"),
        ({"a": 0.5, "b": 0.5, "c": 0.5}, "distinct"),
        ({"a": 0.0, "b": 0.5, "c": 1.0}, "optimum is ambiguous"),
    ],
)
def test_fed_lsae_partition_rejects_unresolved_inputs(
    scores: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fed_lsae_partition(scores)


def test_fed_lsae_aggregate_requires_matching_score_population() -> None:
    with pytest.raises(ValueError, match="populations"):
        fed_lsae_aggregate((_update("a", [1.0]),), {"b": 0.9})


def test_fedkd_local_verifier_mask_matches_equation_15() -> None:
    masked = apply_local_verifier_mask(
        torch.tensor([1, 3, 2]),
        torch.tensor([0, 1, 0]),
    )
    torch.testing.assert_close(masked, torch.tensor([1, -1, 2]))


@pytest.mark.parametrize(
    "detector,verifier",
    [
        (torch.tensor([[1]]), torch.tensor([0])),
        (torch.tensor([True]), torch.tensor([False])),
        (torch.tensor([1.0]), torch.tensor([0.0])),
        (torch.tensor([1]), torch.tensor([2])),
    ],
)
def test_fedkd_local_verifier_rejects_invalid_label_contracts(
    detector: torch.Tensor, verifier: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        apply_local_verifier_mask(detector, verifier)


def test_fedkd_server_core_matches_paper_hard_voting_example() -> None:
    result = aggregate_public_labels(
        {
            "collaborator-1": torch.tensor([1, 3, 2]),
            "collaborator-2": torch.tensor([2, 2, 1]),
            "collaborator-3": torch.tensor([1, 3, 2]),
        },
        class_count=4,
        threshold_multiplier=0.5,
    )

    torch.testing.assert_close(result.global_labels, torch.tensor([1, 3, 2]))
    assert result.eligible_client_ids == ("collaborator-1", "collaborator-3")
    assert result.harmonic_means["collaborator-1"] == 0.0


def test_fedkd_fails_closed_for_unspecified_vote_tie_or_all_abstentions() -> None:
    with pytest.raises(ValueError, match="tie break"):
        aggregate_public_labels(
            {"a": torch.tensor([0]), "b": torch.tensor([1])},
            class_count=2,
            threshold_multiplier=2.0,
        )
    with pytest.raises(ValueError, match="no eligible vote"):
        aggregate_public_labels(
            {"a": torch.tensor([-1]), "b": torch.tensor([-1]), "c": torch.tensor([0])},
            class_count=2,
            threshold_multiplier=0.5,
        )


@pytest.mark.parametrize(
    "labels,class_count,multiplier",
    [
        ({"a": torch.tensor([0])}, 2, 0.5),
        ({"a": torch.tensor([0]), "b": torch.tensor([0])}, 1, 0.5),
        ({"a": torch.tensor([0]), "b": torch.tensor([0])}, 2, 0.0),
        ({"": torch.tensor([0]), "b": torch.tensor([0])}, 2, 0.5),
        (
            {"a": torch.tensor([], dtype=torch.int64), "b": torch.tensor([], dtype=torch.int64)},
            2,
            0.5,
        ),
        ({"a": torch.tensor([0]), "b": torch.tensor([0, 1])}, 2, 0.5),
        ({"a": torch.tensor([0]), "b": torch.tensor([2])}, 2, 0.5),
        ({"a": torch.tensor([0]), "b": torch.tensor([0])}, 2, 0.5),
    ],
)
def test_fedkd_server_rejects_invalid_or_empty_protocol_states(
    labels: dict[str, torch.Tensor], class_count: int, multiplier: float
) -> None:
    with pytest.raises(ValueError):
        aggregate_public_labels(
            labels,
            class_count=class_count,
            threshold_multiplier=multiplier,
        )


def test_phase10_standard_controls_enter_stable_run_identity(tmp_path: Path) -> None:
    trimmed = FederatedRunConfig(
        tmp_path,
        total_rounds=1,
        aggregation="trimmed_mean",
        trim_count=1,
    ).stable_record()
    krum = FederatedRunConfig(
        tmp_path,
        total_rounds=1,
        aggregation="multi_krum",
        krum_byzantine_count=1,
        krum_selected_count=3,
    ).stable_record()

    assert trimmed["phase"] == "phase_10_standard_baselines" and trimmed["trim_count"] == 1
    assert krum["krum_byzantine_count"] == 1 and krum["krum_selected_count"] == 3


def test_phase10_run_controls_reject_cross_method_leakage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only for trimmed"):
        FederatedRunConfig(tmp_path, total_rounds=1, trim_count=1)
    with pytest.raises(ValueError, match="only for Multi-Krum"):
        FederatedRunConfig(tmp_path, total_rounds=1, krum_byzantine_count=1)


class _SharedScheduleProvider:
    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: torch.Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        del global_parameters
        updates: list[ClientUpdate] = []
        for client_id in selected_client_ids:
            index = int(client_id[-1])
            value = 10.0 if index == 4 else index / 10.0
            delta = torch.full((manifest.total_values,), value)
            updates.append(
                ClientUpdate(
                    round_id=round_id,
                    client_id=client_id,
                    delta=delta,
                    declared_sample_count=1,
                    mean_local_loss=1.0,
                    epoch_durations_seconds=(),
                    update_norm=float(torch.linalg.vector_norm(delta).item()),
                    finite=True,
                )
            )
        return updates


def _tiny_model() -> IntrusionMLP:
    torch.manual_seed(10)
    return IntrusionMLP(DetectorConfig(2, 2, (3,), 0.0))


def test_standard_baselines_use_the_exact_same_frozen_client_schedule(tmp_path: Path) -> None:
    schedule = generate_client_schedule(
        tuple(f"client-{index}" for index in range(5)),
        rounds=1,
        clients_per_round=5,
        seed=19,
    )
    trimmed = run_federated_simulation(
        initial_model=_tiny_model(),
        schedule=schedule,
        update_provider=_SharedScheduleProvider(),
        config=FederatedRunConfig(
            tmp_path / "trimmed",
            total_rounds=1,
            aggregation="trimmed_mean",
            seed=19,
            trim_count=1,
        ),
    )
    krum = run_federated_simulation(
        initial_model=_tiny_model(),
        schedule=schedule,
        update_provider=_SharedScheduleProvider(),
        config=FederatedRunConfig(
            tmp_path / "krum",
            total_rounds=1,
            aggregation="multi_krum",
            seed=19,
            krum_byzantine_count=1,
        ),
    )

    assert trimmed["status"] == krum["status"] == "complete"
    assert trimmed["schedule_id"] == krum["schedule_id"] == schedule.schedule_id
