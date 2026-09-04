"""Tabular intrusion-detection backbone frozen by the SemCred-FL protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Shape and regularization controls for the default detector."""

    input_dimension: int
    num_classes: int
    hidden_dimensions: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_dimension < 1:
            raise ValueError("input_dimension must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not self.hidden_dimensions or any(value < 1 for value in self.hidden_dimensions):
            raise ValueError("hidden_dimensions must contain positive values")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "input_dimension": self.input_dimension,
            "num_classes": self.num_classes,
            "hidden_dimensions": list(self.hidden_dimensions),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DetectorConfig:
        """Construct a validated configuration from checkpoint metadata."""

        input_dimension = value.get("input_dimension")
        num_classes = value.get("num_classes")
        hidden = value.get("hidden_dimensions")
        dropout = value.get("dropout")
        if not isinstance(input_dimension, int) or not isinstance(num_classes, int):
            raise ValueError("Detector checkpoint dimensions are invalid")
        if not isinstance(hidden, list) or not all(isinstance(item, int) for item in hidden):
            raise ValueError("Detector checkpoint hidden dimensions are invalid")
        if not isinstance(dropout, int | float):
            raise ValueError("Detector checkpoint dropout is invalid")
        return cls(input_dimension, num_classes, tuple(hidden), float(dropout))


class IntrusionMLP(nn.Module):
    """MLP returning logits for the primary attack-family task."""

    config: DetectorConfig
    network: nn.Sequential

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        previous = config.input_dimension
        for width in config.hidden_dimensions:
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            previous = width
        layers.append(nn.Linear(previous, config.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Return unnormalized class logits."""

        if features.ndim != 2 or features.shape[1] != self.config.input_dimension:
            raise ValueError(
                "Expected a two-dimensional feature tensor with "
                f"dimension {self.config.input_dimension}"
            )
        return cast(Tensor, self.network(features))

    @torch.no_grad()
    def probabilities(self, features: Tensor) -> Tensor:
        """Return class probabilities for evaluation or inference."""

        return torch.softmax(self(features), dim=1)
