"""Schema and bounded-inspection validation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from semcred_fl.data.contracts import (
    DataContractError,
    DatasetSchema,
    FileLoadResult,
    InspectionSummary,
)


class DatasetAdapter(Protocol):
    """Minimal interface required by bounded source inspection."""

    @property
    def root(self) -> Path:
        """Return the dataset source root."""

    @property
    def schema(self) -> DatasetSchema:
        """Return the adapter's frozen schema."""

    def discover_files(self) -> tuple[Path, ...]:
        """Return source files in deterministic order."""

    def load_file(self, source_file: Path, *, max_rows: int) -> FileLoadResult:
        """Read and validate at most ``max_rows`` source records."""


def validate_header(actual: Sequence[str], expected: Sequence[str], source_file: str) -> None:
    """Require an exact source header, including column order."""

    if tuple(actual) != tuple(expected):
        raise DataContractError(f"Unexpected header or column order in {source_file}")


def require_positive_limit(value: int, name: str) -> None:
    """Require a positive bounded-read limit."""

    if value < 1:
        raise ValueError(f"{name} must be positive")


def is_finite_number(value: str) -> bool:
    """Return whether a string is a finite floating-point value."""

    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def inspect_adapter(
    adapter: DatasetAdapter, *, max_files: int, max_rows_per_file: int
) -> InspectionSummary:
    """Inspect a deterministic, bounded prefix of an adapter's source files."""

    require_positive_limit(max_files, "max_files")
    require_positive_limit(max_rows_per_file, "max_rows_per_file")
    discovered = adapter.discover_files()
    selected = discovered[:max_files]
    results = [
        adapter.load_file(source_file, max_rows=max_rows_per_file) for source_file in selected
    ]
    records = [record for result in results for record in result.records]
    return InspectionSummary(
        dataset=adapter.schema.dataset,
        schema_id=adapter.schema.schema_id,
        discovered_files=len(discovered),
        sampled_files=len(selected),
        rows_seen=sum(result.report.rows_seen for result in results),
        rows_emitted=sum(result.report.rows_emitted for result in results),
        invalid_rows=sum(result.report.invalid_rows for result in results),
        fine_labels_observed=tuple(sorted({record.fine_label for record in records})),
        family_labels_observed=tuple(sorted({record.family_label for record in records})),
    )
