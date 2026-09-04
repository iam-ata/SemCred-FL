"""Tests for stable source-row identities."""

from __future__ import annotations

from pathlib import Path

import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.record_ids import normalized_source_file, record_id


def test_record_id_is_stable_namespaced_and_row_sensitive() -> None:
    first = record_id(DatasetName.CIC_IOT_2023, "Benign/capture.csv", 1)
    assert first == record_id(DatasetName.CIC_IOT_2023, "Benign\\capture.csv", 1)
    assert first != record_id(DatasetName.CIC_IOT_2023, "Benign/capture.csv", 2)
    assert first != record_id(DatasetName.EDGE_IIOTSET, "Benign/capture.csv", 1)
    assert len(first) == 64


def test_record_id_rejects_unsafe_source_identity() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        record_id(DatasetName.CIC_IOT_2023, "../capture.csv", 1)
    with pytest.raises(ValueError, match="positive"):
        record_id(DatasetName.CIC_IOT_2023, "capture.csv", 0)
    with pytest.raises(ValueError, match="empty"):
        record_id(DatasetName.CIC_IOT_2023, "", 1)


def test_normalized_source_file_requires_containment(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    source = root / "label" / "capture.csv"
    source.parent.mkdir()
    source.touch()
    assert normalized_source_file(root, source) == "label/capture.csv"
    with pytest.raises(ValueError, match="outside"):
        normalized_source_file(root, tmp_path / "other.csv")
