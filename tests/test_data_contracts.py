"""Tests for typed dataset schema contracts."""

from __future__ import annotations

import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.ciciot2023 import CIC_SCHEMA, CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import (
    DataContractError,
    DatasetRecord,
    DatasetSchema,
    FileLoadResult,
    IngestionReport,
)
from semcred_fl.data.edge_iiotset import (
    EDGE_LABEL_COLUMNS,
    EDGE_MODEL_FEATURE_COLUMNS,
    EDGE_PROHIBITED_MODEL_COLUMNS,
    EDGE_SCHEMA,
)


def test_cic_schema_exposes_all_official_features_as_continuous() -> None:
    assert len(CIC_SCHEMA.source_columns) == 39
    assert CIC_SCHEMA.model_feature_columns == CIC_SOURCE_COLUMNS
    assert CIC_SCHEMA.continuous_columns == CIC_SOURCE_COLUMNS
    assert CIC_SCHEMA.label_columns == ()
    assert len(CIC_SCHEMA.schema_id) == 16


def test_edge_schema_separates_labels_and_prohibited_fields() -> None:
    assert len(EDGE_SCHEMA.source_columns) == 63
    assert EDGE_SCHEMA.model_feature_columns == EDGE_MODEL_FEATURE_COLUMNS
    assert set(EDGE_LABEL_COLUMNS).isdisjoint(EDGE_MODEL_FEATURE_COLUMNS)
    assert set(EDGE_PROHIBITED_MODEL_COLUMNS).isdisjoint(EDGE_MODEL_FEATURE_COLUMNS)
    assert set(EDGE_SCHEMA.categorical_columns) | set(EDGE_SCHEMA.unresolved_columns) == set(
        EDGE_MODEL_FEATURE_COLUMNS
    )


def test_schema_rejects_incomplete_feature_kind_partition() -> None:
    with pytest.raises(DataContractError, match="partition mismatch"):
        DatasetSchema(
            dataset=DatasetName.CIC_IOT_2023,
            source_columns=("x",),
            label_columns=(),
            model_feature_columns=("x",),
            grouping_columns=("source_file",),
            categorical_columns=(),
            continuous_columns=(),
        )


def test_record_rejects_duplicate_feature_keys() -> None:
    with pytest.raises(DataContractError, match="feature keys"):
        DatasetRecord(
            dataset=DatasetName.CIC_IOT_2023,
            record_id="a" * 64,
            source_file="capture.csv",
            source_row=1,
            fine_label="BenignTraffic",
            family_label="benign",
            features=(("x", "1"), ("x", "2")),
            grouping=(("source_file", "capture.csv"),),
        )


def test_ingestion_report_and_result_enforce_accounting() -> None:
    with pytest.raises(DataContractError, match="more rows"):
        IngestionReport(source_file="a.csv", rows_seen=1, rows_emitted=1, invalid_width=1)
    report = IngestionReport(source_file="a.csv", rows_seen=1, rows_emitted=1)
    with pytest.raises(DataContractError, match="rows_emitted"):
        FileLoadResult(records=(), report=report)
