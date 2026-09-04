"""Frozen post-audit preparation policies and scalar normalization rules."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from semcred_fl.config import DatasetName
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.edge_iiotset import EDGE_MODEL_FEATURE_COLUMNS
from semcred_fl.identity import run_id

_HEX_RE: Final = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
ABSENT_CATEGORY: Final = "__ABSENT__"
UNKNOWN_CATEGORY: Final = "__UNKNOWN__"


class FeatureValueError(DataContractError):
    """Raised when a raw feature cannot satisfy the frozen preparation policy."""


class ConflictAction(StrEnum):
    """Permitted handling of feature-identical records with conflicting labels."""

    EXCLUDE_COMPONENT = "exclude_entire_component"
    KEEP_FOR_PRIMARY_ONLY = "keep_for_primary_exclude_from_fine_analysis"


class DuplicateAction(StrEnum):
    """Permitted handling of label-consistent exact feature duplicates."""

    CANONICAL_REPRESENTATIVE = "keep_one_canonical_representative"


@dataclass(frozen=True, slots=True)
class ConflictEvidence:
    """Verified Phase 2B/2C counts supporting one dataset's ambiguity policy."""

    phase_2b_audit_id: str
    phase_2c_analysis_id: str
    valid_rows: int
    raw_unique_fingerprints: int
    primary_conflict_fingerprints: int
    primary_conflict_occurrences: int
    fine_only_conflict_fingerprints: int
    fine_only_conflict_occurrences: int

    def __post_init__(self) -> None:
        for name in ("phase_2b_audit_id", "phase_2c_analysis_id"):
            value = getattr(self, name)
            if len(value) != 64 or not set(value) <= set("0123456789abcdef"):
                raise DataContractError(f"{name} must be a full lowercase SHA-256 identifier")
        counts = (
            self.valid_rows,
            self.raw_unique_fingerprints,
            self.primary_conflict_fingerprints,
            self.primary_conflict_occurrences,
            self.fine_only_conflict_fingerprints,
            self.fine_only_conflict_occurrences,
        )
        if any(count < 0 for count in counts):
            raise DataContractError("Conflict evidence counts must be non-negative")
        if self.raw_unique_fingerprints > self.valid_rows:
            raise DataContractError("Unique fingerprints cannot exceed valid rows")
        if self.primary_conflict_fingerprints + self.fine_only_conflict_fingerprints > (
            self.raw_unique_fingerprints
        ):
            raise DataContractError("Conflict fingerprints cannot exceed all fingerprints")

    @property
    def raw_primary_consistent_unique_fingerprints(self) -> int:
        """Return the raw-string lower bound before numeric canonicalization."""

        return self.raw_unique_fingerprints - self.primary_conflict_fingerprints


@dataclass(frozen=True, slots=True)
class CategoryRule:
    """Closed raw domain for one categorical feature before train-only encoding."""

    column: str
    zero_is_absent: bool = False
    allowed_text_tokens: tuple[str, ...] = ()
    allowed_numeric_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.allowed_text_tokens) == bool(self.allowed_numeric_codes):
            raise DataContractError(
                "Each category rule must define exactly one text or numeric-code domain"
            )
        if len(self.allowed_text_tokens) != len(set(self.allowed_text_tokens)):
            raise DataContractError("Allowed categorical text tokens must be unique")
        if len(self.allowed_numeric_codes) != len(set(self.allowed_numeric_codes)):
            raise DataContractError("Allowed categorical numeric codes must be unique")


@dataclass(frozen=True, slots=True)
class FeaturePreparationPolicy:
    """Complete typed partition and raw normalization rules for one source schema."""

    source_feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    category_rules: tuple[CategoryRule, ...]

    def __post_init__(self) -> None:
        partitions = (
            set(self.categorical_columns),
            set(self.continuous_columns),
            set(self.dropped_columns),
        )
        if any(partitions[left] & partitions[right] for left in range(3) for right in range(left)):
            raise DataContractError("Prepared feature partitions must be disjoint")
        if set().union(*partitions) != set(self.source_feature_columns):
            raise DataContractError("Prepared feature partitions must cover the raw model schema")
        rule_columns = tuple(rule.column for rule in self.category_rules)
        if len(rule_columns) != len(set(rule_columns)):
            raise DataContractError("Categorical rules must have unique columns")
        if set(rule_columns) != set(self.categorical_columns):
            raise DataContractError(
                "Every categorical feature must have exactly one raw-domain rule"
            )

    @property
    def prepared_feature_columns(self) -> tuple[str, ...]:
        """Return stable source order after audit-frozen feature removal."""

        dropped = set(self.dropped_columns)
        return tuple(column for column in self.source_feature_columns if column not in dropped)

    @property
    def policy_id(self) -> str:
        """Return the deterministic identifier for this exact feature policy."""

        return run_id(self, length=64)


@dataclass(frozen=True, slots=True)
class DatasetPreparationPolicy:
    """Frozen cleaning decisions required before split construction."""

    dataset: DatasetName
    evidence: ConflictEvidence
    features: FeaturePreparationPolicy
    primary_conflict_action: ConflictAction = ConflictAction.EXCLUDE_COMPONENT
    fine_only_conflict_action: ConflictAction = ConflictAction.KEEP_FOR_PRIMARY_ONLY
    duplicate_action: DuplicateAction = DuplicateAction.CANONICAL_REPRESENTATIVE
    representative_order: tuple[str, ...] = ("source_file", "source_row")
    fingerprint_stage: str = "after_validation_and_canonicalization"

    def __post_init__(self) -> None:
        if self.representative_order != ("source_file", "source_row"):
            raise DataContractError(
                "Canonical duplicate selection must use source_file, source_row"
            )
        if self.fingerprint_stage != "after_validation_and_canonicalization":
            raise DataContractError("Final fingerprints must use canonical prepared feature values")

    @property
    def policy_id(self) -> str:
        """Return the deterministic identifier for the complete dataset policy."""

        return run_id(self, length=64)


def _parse_numeric(value: str, column: str) -> float:
    stripped = value.strip()
    try:
        parsed = float(int(stripped, 0)) if _HEX_RE.fullmatch(stripped) else float(stripped)
    except ValueError as exc:
        raise FeatureValueError(f"Non-numeric value for {column}: {stripped!r}") from exc
    if not math.isfinite(parsed):
        raise FeatureValueError(f"Non-finite value for {column}: {stripped!r}")
    return parsed


def canonical_numeric(value: str, column: str) -> str:
    """Return one stable decimal representation for finite decimal or hexadecimal input."""

    parsed = _parse_numeric(value, column)
    if parsed == 0.0:
        return "0"
    return format(parsed, ".17g")


def _normalize_category(value: str, rule: CategoryRule) -> str:
    stripped = value.strip()
    if rule.allowed_numeric_codes:
        parsed = _parse_numeric(stripped, rule.column)
        if not parsed.is_integer():
            raise FeatureValueError(f"Non-integer categorical code for {rule.column}: {stripped!r}")
        code = int(parsed)
        if code not in rule.allowed_numeric_codes:
            raise FeatureValueError(
                f"Out-of-domain categorical code for {rule.column}: {stripped!r}"
            )
        return str(code)

    if rule.zero_is_absent:
        try:
            if _parse_numeric(stripped, rule.column) == 0.0:
                return ABSENT_CATEGORY
        except FeatureValueError:
            pass
    if stripped not in rule.allowed_text_tokens:
        raise FeatureValueError(f"Out-of-domain categorical token for {rule.column}: {stripped!r}")
    return stripped


def normalize_feature_values(
    values: Mapping[str, str], policy: FeaturePreparationPolicy
) -> tuple[tuple[str, str], ...]:
    """Validate and canonicalize one raw model-feature mapping in stable source order."""

    missing = sorted(set(policy.source_feature_columns) - set(values))
    if missing:
        raise FeatureValueError(f"Missing raw model features: {missing}")
    rules = {rule.column: rule for rule in policy.category_rules}
    continuous = set(policy.continuous_columns)
    dropped = set(policy.dropped_columns)
    normalized: list[tuple[str, str]] = []
    for column in policy.source_feature_columns:
        if column in dropped:
            continue
        value = values[column]
        if column in continuous:
            normalized.append((column, canonical_numeric(value, column)))
        else:
            normalized.append((column, _normalize_category(value, rules[column])))
    return tuple(normalized)


_EDGE_CATEGORICAL = (
    "http.request.method",
    "http.request.version",
    "mqtt.conack.flags",
    "mqtt.protoname",
)
_EDGE_DROPPED = ("icmp.unused",)
_EDGE_CONTINUOUS = tuple(
    column
    for column in EDGE_MODEL_FEATURE_COLUMNS
    if column not in set(_EDGE_CATEGORICAL) | set(_EDGE_DROPPED)
)

EDGE_FEATURE_POLICY = FeaturePreparationPolicy(
    source_feature_columns=EDGE_MODEL_FEATURE_COLUMNS,
    categorical_columns=_EDGE_CATEGORICAL,
    continuous_columns=_EDGE_CONTINUOUS,
    dropped_columns=_EDGE_DROPPED,
    category_rules=(
        CategoryRule(
            "http.request.method",
            zero_is_absent=True,
            allowed_text_tokens=("GET", "OPTIONS", "POST", "PROPFIND", "PUT", "SEARCH", "TRACE"),
        ),
        CategoryRule(
            "http.request.version",
            zero_is_absent=True,
            allowed_text_tokens=("HTTP/1.0", "HTTP/1.1"),
        ),
        CategoryRule("mqtt.conack.flags", allowed_numeric_codes=(0, 1)),
        CategoryRule("mqtt.protoname", zero_is_absent=True, allowed_text_tokens=("MQTT",)),
    ),
)

CIC_FEATURE_POLICY = FeaturePreparationPolicy(
    source_feature_columns=CIC_SOURCE_COLUMNS,
    categorical_columns=(),
    continuous_columns=CIC_SOURCE_COLUMNS,
    dropped_columns=(),
    category_rules=(),
)

EDGE_PREPARATION_POLICY = DatasetPreparationPolicy(
    dataset=DatasetName.EDGE_IIOTSET,
    evidence=ConflictEvidence(
        phase_2b_audit_id="534caa215cbd6ca4fd668e803ad4b1519b6ce89de6bc9045215c8984f6b09973",
        phase_2c_analysis_id="c023ee61bf7d21f465e14b5b30f255ae957e244599c5ca0726a41040af1fe766",
        valid_rows=20_939_622,
        raw_unique_fingerprints=15_235_695,
        primary_conflict_fingerprints=36_289,
        primary_conflict_occurrences=135_857,
        fine_only_conflict_fingerprints=71,
        fine_only_conflict_occurrences=150,
    ),
    features=EDGE_FEATURE_POLICY,
)

CIC_PREPARATION_POLICY = DatasetPreparationPolicy(
    dataset=DatasetName.CIC_IOT_2023,
    evidence=ConflictEvidence(
        phase_2b_audit_id="eb41b084eeb07a425e28985b989f9e4ba2f9966f5827ea33e57ebe4443514e24",
        phase_2c_analysis_id="4ca1ad75e446bb98d7a1836d07d7331564040897527a6445523855b0ad824211",
        valid_rows=46_775_660,
        raw_unique_fingerprints=21_055_707,
        primary_conflict_fingerprints=461_347,
        primary_conflict_occurrences=13_807_409,
        fine_only_conflict_fingerprints=81_553,
        fine_only_conflict_occurrences=513_304,
    ),
    features=CIC_FEATURE_POLICY,
)
