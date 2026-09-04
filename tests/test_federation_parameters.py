"""Tests for deterministic flattened parameter manifests."""

from __future__ import annotations

import pytest
import torch

from semcred_fl.federation.parameters import (
    apply_flat_delta,
    assign_flat_parameters,
    flatten_parameters,
    parameter_manifest,
    unflatten_parameters,
)
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP


def _model() -> IntrusionMLP:
    torch.manual_seed(7)
    return IntrusionMLP(DetectorConfig(3, 2, (4,), 0.0))


def test_parameter_flatten_round_trip_preserves_predictions() -> None:
    source = _model()
    manifest = parameter_manifest(source)
    flattened = flatten_parameters(source, manifest)
    reconstructed = unflatten_parameters(flattened, manifest)

    assert tuple(reconstructed) == tuple(item.name for item in manifest.parameters)
    assert sum(value.numel() for value in reconstructed.values()) == manifest.total_values
    target = IntrusionMLP(source.config)
    assign_flat_parameters(target, flattened, manifest)
    features = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.0, -1.0]])
    torch.testing.assert_close(source(features), target(features), rtol=0.0, atol=0.0)


def test_apply_flat_delta_is_exact() -> None:
    model = _model()
    manifest = parameter_manifest(model)
    before = flatten_parameters(model, manifest)
    delta = torch.linspace(-0.01, 0.01, manifest.total_values)

    apply_flat_delta(model, delta, manifest)

    torch.testing.assert_close(flatten_parameters(model, manifest), before + delta)


def test_parameter_utilities_reject_shape_or_layout_mismatch() -> None:
    model = _model()
    manifest = parameter_manifest(model)
    with pytest.raises(ValueError, match="wrong shape"):
        unflatten_parameters(torch.zeros(manifest.total_values + 1), manifest)
    with pytest.raises(ValueError, match="wrong shape"):
        apply_flat_delta(model, torch.zeros(2), manifest)
    different = IntrusionMLP(DetectorConfig(3, 2, (5,), 0.0))
    with pytest.raises(ValueError, match="layout"):
        flatten_parameters(different, manifest)
