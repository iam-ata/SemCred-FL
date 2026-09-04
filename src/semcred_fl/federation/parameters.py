"""Deterministic flattening and reconstruction of trainable model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Name and shape of one trainable parameter in canonical order."""

    name: str
    shape: tuple[int, ...]
    number_of_values: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible parameter specification."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "number_of_values": self.number_of_values,
        }


@dataclass(frozen=True, slots=True)
class ParameterManifest:
    """Stable flattened-vector layout for one model architecture."""

    parameters: tuple[ParameterSpec, ...]
    total_values: int

    def __post_init__(self) -> None:
        if not self.parameters or self.total_values < 1:
            raise ValueError("Parameter manifest cannot be empty")
        if sum(item.number_of_values for item in self.parameters) != self.total_values:
            raise ValueError("Parameter manifest cardinality is inconsistent")
        if len({item.name for item in self.parameters}) != len(self.parameters):
            raise ValueError("Parameter manifest names must be unique")
        for item in self.parameters:
            if not item.name or item.number_of_values != prod(item.shape):
                raise ValueError("Parameter specification is malformed")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible manifest."""

        return {
            "parameters": [item.to_dict() for item in self.parameters],
            "total_values": self.total_values,
        }


def parameter_manifest(model: nn.Module) -> ParameterManifest:
    """Construct the canonical manifest from ``named_parameters`` order."""

    specifications = tuple(
        ParameterSpec(name, tuple(parameter.shape), parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return ParameterManifest(specifications, sum(item.number_of_values for item in specifications))


def _validated_named_parameters(
    model: nn.Module, manifest: ParameterManifest
) -> tuple[nn.Parameter, ...]:
    selected = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if len(selected) != len(manifest.parameters):
        raise ValueError("Model trainable-parameter count differs from the manifest")
    result: list[nn.Parameter] = []
    for (name, parameter), specification in zip(selected, manifest.parameters, strict=True):
        if name != specification.name or tuple(parameter.shape) != specification.shape:
            raise ValueError("Model trainable-parameter layout differs from the manifest")
        result.append(parameter)
    return tuple(result)


def flatten_parameters(model: nn.Module, manifest: ParameterManifest) -> Tensor:
    """Return one detached vector in manifest order."""

    parameters = _validated_named_parameters(model, manifest)
    return torch.cat(tuple(parameter.detach().reshape(-1) for parameter in parameters)).clone()


def unflatten_parameters(vector: Tensor, manifest: ParameterManifest) -> dict[str, Tensor]:
    """Reconstruct named tensors from one canonical vector."""

    if vector.ndim != 1 or vector.numel() != manifest.total_values:
        raise ValueError("Flattened vector has the wrong shape")
    result: dict[str, Tensor] = {}
    offset = 0
    for specification in manifest.parameters:
        end = offset + specification.number_of_values
        result[specification.name] = vector[offset:end].reshape(specification.shape).clone()
        offset = end
    return result


@torch.no_grad()
def assign_flat_parameters(model: nn.Module, vector: Tensor, manifest: ParameterManifest) -> None:
    """Replace trainable model parameters from one manifest-bound vector."""

    values = unflatten_parameters(vector, manifest)
    parameters = _validated_named_parameters(model, manifest)
    for parameter, specification in zip(parameters, manifest.parameters, strict=True):
        parameter.copy_(values[specification.name].to(parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def apply_flat_delta(model: nn.Module, delta: Tensor, manifest: ParameterManifest) -> None:
    """Apply one validated flattened parameter update to a model."""

    current = flatten_parameters(model, manifest)
    if delta.ndim != 1 or delta.numel() != manifest.total_values:
        raise ValueError("Flattened update has the wrong shape")
    if not torch.isfinite(delta).all().item():
        raise ValueError("Flattened update contains non-finite values")
    assign_flat_parameters(model, current + delta.to(current.device, dtype=current.dtype), manifest)
