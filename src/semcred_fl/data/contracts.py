"""Typed contracts shared by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass

from semcred_fl.config import DatasetName
from semcred_fl.identity import run_id


class DataContractError(ValueError):
    """Raised when source data violates a frozen ingestion contract."""


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Immutable raw and model-facing schema for one dataset adapter."""

    dataset: DatasetName
    source_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    model_feature_columns: tuple[str, ...]
    grouping_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    unresolved_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        named_sets = {
            "source": self.source_columns,
            "label": self.label_columns,
            "model feature": self.model_feature_columns,
            "grouping": self.grouping_columns,
            "categorical": self.categorical_columns,
            "continuous": self.continuous_columns,
            "unresolved": self.unresolved_columns,
        }
        for name, columns in named_sets.items():
            if len(columns) != len(set(columns)):
                raise DataContractError(f"{name} columns must be unique")

        source = set(self.source_columns)
        model = set(self.model_feature_columns)
        if not model <= source:
            raise DataContractError("Model feature columns must exist in the source schema")
        if set(self.label_columns) & model:
            raise DataContractError("Label columns must not enter model features")

        typed = (
            set(self.categorical_columns)
            | set(self.continuous_columns)
            | set(self.unresolved_columns)
        )
        if typed != model:
            missing = sorted(model - typed)
            extra = sorted(typed - model)
            raise DataContractError(
                f"Feature-kind partition mismatch; missing={missing}, extra={extra}"
            )
        if (
            set(self.categorical_columns) & set(self.continuous_columns)
            or set(self.categorical_columns) & set(self.unresolved_columns)
            or set(self.continuous_columns) & set(self.unresolved_columns)
        ):
            raise DataContractError("Feature-kind partitions must be disjoint")

    @property
    def schema_id(self) -> str:
        """Return a deterministic identifier for this exact schema."""

        return run_id(self)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One validated raw record with labels and grouping evidence separated."""

    dataset: DatasetName
    record_id: str
    source_file: str
    source_row: int
    fine_label: str
    family_label: str
    features: tuple[tuple[str, str], ...]
    grouping: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.source_row < 1:
            raise DataContractError("source_row must be one-based and positive")
        if len(self.record_id) != 64 or not set(self.record_id) <= set("0123456789abcdef"):
            raise DataContractError("record_id must be a full SHA-256 hexadecimal digest")
        for name, values in (("feature", self.features), ("grouping", self.grouping)):
            keys = [key for key, _ in values]
            if len(keys) != len(set(keys)):
                raise DataContractError(f"{name} keys must be unique")

    @property
    def feature_map(self) -> dict[str, str]:
        """Return model-facing feature values keyed by their stable names."""

        return dict(self.features)

    @property
    def grouping_map(self) -> dict[str, str]:
        """Return split-only grouping evidence keyed by name."""

        return dict(self.grouping)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Bounded file-read accounting without silent invalid-row removal."""

    source_file: str
    rows_seen: int
    rows_emitted: int
    invalid_width: int = 0
    invalid_label: int = 0
    non_finite_numeric: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.rows_seen,
            self.rows_emitted,
            self.invalid_width,
            self.invalid_label,
            self.non_finite_numeric,
        )
        if any(count < 0 for count in counts):
            raise DataContractError("Ingestion counts must be non-negative")
        if self.rows_emitted + self.invalid_rows > self.rows_seen:
            raise DataContractError("Ingestion report accounts for more rows than were seen")

    @property
    def invalid_rows(self) -> int:
        """Return the total number of rejected rows in this report."""

        return self.invalid_width + self.invalid_label + self.non_finite_numeric


@dataclass(frozen=True, slots=True)
class FileLoadResult:
    """Validated records and accounting from one bounded source-file read."""

    records: tuple[DatasetRecord, ...]
    report: IngestionReport

    def __post_init__(self) -> None:
        if len(self.records) != self.report.rows_emitted:
            raise DataContractError("rows_emitted must match the number of returned records")


@dataclass(frozen=True, slots=True)
class InspectionSummary:
    """Read-only, bounded inspection result for a dataset source tree."""

    dataset: DatasetName
    schema_id: str
    discovered_files: int
    sampled_files: int
    rows_seen: int
    rows_emitted: int
    invalid_rows: int
    fine_labels_observed: tuple[str, ...]
    family_labels_observed: tuple[str, ...]
