"""Tests for frozen Phase 6C conditions and stratified realizations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from semcred_fl.attacks.execution_contracts import (
    AttackRunSpec,
    build_attack_realization,
    phase6c_execution_config,
)
from semcred_fl.identity import to_primitive


def test_phase6c_yaml_matches_typed_contract() -> None:
    loaded = yaml.safe_load(Path("configs/attacks/phase6c.yaml").read_text(encoding="utf-8"))
    assert loaded == to_primitive(phase6c_execution_config())


def test_realization_has_fixed_coalition_and_exact_minority_each_round() -> None:
    config = phase6c_execution_config()
    spec = AttackRunSpec(
        dataset="edge_iiotset",
        profile="dirichlet_0_1",
        attack_name="label_flip",
        malicious_ratio=0.2,
        poison_fraction=0.3,
        seed=20260816,
    )
    first = build_attack_realization(config, spec)
    second = build_attack_realization(config, spec)
    assert first == second
    assert len(first.malicious_client_ids) == 10
    assert first.active_malicious_per_round == 4
    malicious = set(first.malicious_client_ids)
    for selected in first.schedule.rounds:
        assert len(set(selected) & malicious) == 4
        assert 2 * len(set(selected) & malicious) < len(selected)


def test_clean_and_attack_share_realization_but_not_attack_identity() -> None:
    config = phase6c_execution_config()
    clean = AttackRunSpec("edge_iiotset", "iid", "clean", 0.1, 7)
    attacked = AttackRunSpec("edge_iiotset", "iid", "label_flip", 0.1, 7, poison_fraction=0.0)
    clean_realization = build_attack_realization(config, clean)
    attacked_realization = build_attack_realization(config, attacked)
    assert clean_realization == attacked_realization
    assert clean.attack_id(config) != attacked.attack_id(config)


def test_attack_specific_parameters_fail_closed() -> None:
    config = phase6c_execution_config()
    with pytest.raises(ValueError):
        AttackRunSpec(
            "edge_iiotset",
            "iid",
            "scaling",
            0.1,
            1,
            scale=3.0,
        ).validate(config)
    with pytest.raises(ValueError):
        AttackRunSpec(
            "edge_iiotset",
            "iid",
            "clean_label",
            0.1,
            1,
            poison_fraction=0.3,
        ).validate(config)
