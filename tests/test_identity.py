"""Tests for canonical experiment identities and seed derivation."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from semcred_fl.config import DatasetName, ExperimentConfig
from semcred_fl.identity import canonical_json, run_id, to_primitive
from semcred_fl.seeding import derive_seed, seed_standard_library


def test_canonical_identity_is_stable_and_sensitive() -> None:
    first = ExperimentConfig()
    second = ExperimentConfig(dataset=DatasetName.CIC_IOT_2023)
    assert canonical_json(first) == canonical_json(first)
    assert run_id(first) == run_id(first)
    assert run_id(first) != run_id(second)
    assert len(run_id(first)) == 16


def test_supported_primitive_conversions() -> None:
    converted = to_primitive(
        {
            "path": Path("a") / "b",
            "tuple": (DatasetName.EDGE_IIOTSET, 2),
            "list": [True, None],
        }
    )
    assert converted == {
        "path": "a/b",
        "tuple": ["edge_iiotset", 2],
        "list": [True, None],
    }


def test_canonical_conversion_rejects_unsupported_values() -> None:
    with pytest.raises(TypeError, match="string keys"):
        to_primitive({1: "value"})
    with pytest.raises(TypeError, match="Unsupported"):
        to_primitive({1, 2})
    with pytest.raises(ValueError, match="between 8 and 64"):
        run_id(ExperimentConfig(), length=7)


def test_seed_derivation_is_stable_and_namespaced() -> None:
    assert derive_seed(42, "partition") == derive_seed(42, "partition")
    assert derive_seed(42, "partition") != derive_seed(42, "attack")
    with pytest.raises(ValueError, match="negative"):
        derive_seed(-1, "partition")
    with pytest.raises(ValueError, match="empty"):
        derive_seed(1, "  ")


def test_standard_library_seed_replays_sequence() -> None:
    derived = seed_standard_library(17, "unit-test")
    first = random.random()
    assert seed_standard_library(17, "unit-test") == derived
    assert random.random() == first
