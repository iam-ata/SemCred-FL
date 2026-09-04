"""Analytical tests for safe update validation and Phase 5 aggregators."""

from __future__ import annotations

import math

import pytest
import torch

from semcred_fl.baselines import coordinate_median, fedavg
from semcred_fl.federation.updates import ClientUpdate, DefenseUpdate, validate_client_updates


def _defense(client_id: str, delta: list[float], count: int = 1) -> DefenseUpdate:
    tensor = torch.tensor(delta)
    return DefenseUpdate(
        round_id=0,
        client_id=client_id,
        delta=tensor,
        declared_sample_count=count,
        mean_local_loss=1.0,
        update_norm=float(torch.linalg.vector_norm(tensor).item()),
        finite=True,
        missing_class_ids=(),
    )


def _client(
    client_id: str,
    delta: list[float],
    *,
    round_id: int = 0,
    count: int = 1,
    finite: bool = True,
) -> ClientUpdate:
    tensor = torch.tensor(delta)
    return ClientUpdate(
        round_id=round_id,
        client_id=client_id,
        delta=tensor,
        declared_sample_count=count,
        mean_local_loss=1.0,
        epoch_durations_seconds=(0.1,),
        update_norm=float(torch.linalg.vector_norm(tensor).item()),
        finite=finite,
        is_malicious=True,
        attack_metadata={"private_simulation_label": "must_not_reach_defense"},
    )


def test_weighted_fedavg_matches_analytical_fixture() -> None:
    result = fedavg((_defense("a", [1.0, 3.0]), _defense("b", [3.0, 1.0], 3)))

    torch.testing.assert_close(result.delta, torch.tensor([2.5, 1.5]))
    assert result.weights == {"a": 0.25, "b": 0.75}


def test_coordinate_median_uses_midpoint_for_even_population() -> None:
    odd = coordinate_median(
        (_defense("a", [0.0, 10.0]), _defense("b", [2.0, 5.0]), _defense("c", [9.0, 1.0]))
    )
    even = coordinate_median(
        (
            _defense("a", [0.0, 10.0]),
            _defense("b", [2.0, 5.0]),
            _defense("c", [9.0, 1.0]),
            _defense("d", [4.0, 3.0]),
        )
    )

    torch.testing.assert_close(odd.delta, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(even.delta, torch.tensor([3.0, 4.0]))


def test_validation_rejects_bad_updates_and_strips_attack_metadata() -> None:
    valid = _client("valid", [1.0, 2.0])
    wrong_round = _client("round", [1.0, 2.0], round_id=2)
    nonfinite = _client("nan", [math.nan, 0.0], finite=False)
    duplicated = _client("duplicate", [1.0, 2.0])
    validation = validate_client_updates(
        (valid, wrong_round, nonfinite, duplicated, duplicated),
        expected_round=0,
        expected_values=2,
    )

    assert [item.client_id for item in validation.accepted] == ["valid"]
    assert validation.rejected == {
        "duplicate": "duplicate_client_id",
        "nan": "nonfinite_update",
        "round": "wrong_round",
    }
    assert not hasattr(validation.accepted[0], "is_malicious")
    assert not hasattr(validation.accepted[0], "attack_metadata")


@pytest.mark.parametrize("aggregator", [fedavg, coordinate_median])
def test_aggregators_reject_empty_input(aggregator: object) -> None:
    with pytest.raises(ValueError):
        aggregator(())  # type: ignore[operator]
