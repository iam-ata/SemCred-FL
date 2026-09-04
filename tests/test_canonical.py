"""Tests for Phase 2D canonical-corpus construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
import pytest

from semcred_fl.config import DatasetName
from semcred_fl.data.canonical import CanonicalBuildRequest, build_canonical_corpus
from semcred_fl.data.ciciot2023 import CIC_SOURCE_COLUMNS
from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.edge_iiotset import EDGE_SOURCE_COLUMNS

_HASH = "a" * 64


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _request(dataset: DatasetName, root: Path, output: Path) -> CanonicalBuildRequest:
    return CanonicalBuildRequest(
        dataset=dataset,
        root=root,
        archive_sha256=_HASH,
        output_directory=output,
        work_database=output / f"{dataset.value}_canonical.sqlite3",
        row_batch_size=2,
        parquet_batch_size=2,
    )


def _read_shards(report: dict[str, object], output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard in cast(list[dict[str, object]], report["shards"]):
        rows.extend(pq.read_table(output / cast(str, shard["file"])).to_pylist())
    return rows


def test_cic_build_excludes_primary_conflicts_and_resumes(tmp_path: Path) -> None:
    root = tmp_path / "cic"
    one = ["1"] * len(CIC_SOURCE_COLUMNS)
    two = ["2"] * len(CIC_SOURCE_COLUMNS)
    three = ["3"] * len(CIC_SOURCE_COLUMNS)
    four = ["4"] * len(CIC_SOURCE_COLUMNS)
    invalid = ["5"] * len(CIC_SOURCE_COLUMNS)
    invalid[0] = "nan"
    _write_csv(
        root / "DDoS-HTTP_Flood" / "a.csv",
        CIC_SOURCE_COLUMNS,
        [one, two, three, ["3.0"] * len(CIC_SOURCE_COLUMNS), four, invalid],
    )
    _write_csv(
        root / "DDoS-HTTP_Flood" / "b.csv",
        CIC_SOURCE_COLUMNS,
        [["3.00"] * len(CIC_SOURCE_COLUMNS)],
    )
    _write_csv(
        root / "DDoS-SYN_Flood" / "c.csv",
        CIC_SOURCE_COLUMNS,
        [["2.0"] * len(CIC_SOURCE_COLUMNS)],
    )
    _write_csv(
        root / "DoS-HTTP_Flood" / "d.csv",
        CIC_SOURCE_COLUMNS,
        [["1.0"] * len(CIC_SOURCE_COLUMNS)],
    )
    output = tmp_path / "canonical"
    request = _request(DatasetName.CIC_IOT_2023, root, output)

    first = build_canonical_corpus(request, progress_every=100)
    resumed = build_canonical_corpus(request, progress_every=100)

    assert first["corpus_id"] == resumed["corpus_id"]
    assert first["coverage"] == resumed["coverage"]
    coverage = cast(dict[str, object], first["coverage"])
    assert coverage["rows_seen"] == 9
    assert coverage["valid_rows"] == 8
    assert coverage["invalid_feature_rows"] == 1
    assert coverage["emitted_rows"] == 3
    classification = cast(dict[str, int], first["classification"])
    assert classification["canonical_fingerprints"] == 4
    assert classification["primary_conflict_fingerprints"] == 1
    assert classification["primary_conflict_occurrences"] == 2
    assert classification["fine_ambiguous_fingerprints"] == 1
    assert classification["eligible_duplicate_occurrences_removed"] == 3

    rows = _read_shards(first, output)
    assert len(rows) == 3
    by_value = {row[CIC_SOURCE_COLUMNS[0]]: row for row in rows}
    assert 1.0 not in by_value
    assert by_value[2.0]["fine_label_ambiguous"] is True
    assert by_value[2.0]["fine_label"] == "DDoS-HTTP_Flood"
    assert by_value[3.0]["duplicate_count"] == 3
    assert by_value[3.0]["source_file_count"] == 2
    assert by_value[3.0]["source_file"] == "DDoS-HTTP_Flood/a.csv"
    assert by_value[3.0]["source_row"] == 3
    assert len({row["canonical_fingerprint"] for row in rows}) == 3
    persisted = json.loads((output / "cic_iot_2023_canonical.json").read_text(encoding="utf-8"))
    assert persisted["corpus_id"] == first["corpus_id"]


def _edge_row(*, fine: str = "MITM", binary: str = "1") -> list[str]:
    row = ["0"] * len(EDGE_SOURCE_COLUMNS)
    index = {column: position for position, column in enumerate(EDGE_SOURCE_COLUMNS)}
    row[index["frame.time"]] = "2022-01-01 12:00:00"
    row[index["ip.src_host"]] = "10.0.0.2"
    row[index["ip.dst_host"]] = "10.0.0.1"
    row[index["Attack_type"]] = fine
    row[index["Attack_label"]] = binary
    return row


def test_edge_build_drops_frozen_column_and_rejects_bad_numeric(tmp_path: Path) -> None:
    root = tmp_path / "edge"
    first = _edge_row()
    duplicate = _edge_row()
    invalid = _edge_row()
    index = {column: position for position, column in enumerate(EDGE_SOURCE_COLUMNS)}
    first[index["icmp.unused"]] = "discarded-a"
    duplicate[index["icmp.unused"]] = "discarded-b"
    invalid[index["arp.opcode"]] = "request"
    _write_csv(
        root / "Attack traffic" / "MITM_attack.csv",
        EDGE_SOURCE_COLUMNS,
        [first, duplicate, invalid],
    )
    output = tmp_path / "canonical"

    report = build_canonical_corpus(
        _request(DatasetName.EDGE_IIOTSET, root, output), progress_every=100
    )

    coverage = cast(dict[str, object], report["coverage"])
    assert coverage["rows_seen"] == 3
    assert coverage["valid_rows"] == 2
    assert coverage["invalid_feature_rows"] == 1
    assert coverage["emitted_rows"] == 1
    rows = _read_shards(report, output)
    assert len(rows) == 1
    assert rows[0]["duplicate_count"] == 2
    assert rows[0]["endpoint_pair"] == "10.0.0.1|10.0.0.2"
    assert "icmp.unused" not in rows[0]
    assert rows[0]["http.request.method"] == "__ABSENT__"


def test_build_rejects_invalid_controls_and_changed_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full SHA-256"):
        CanonicalBuildRequest(
            DatasetName.CIC_IOT_2023, tmp_path, "bad", tmp_path, tmp_path / "work.db"
        )
    with pytest.raises(ValueError, match="positive"):
        CanonicalBuildRequest(
            DatasetName.CIC_IOT_2023,
            tmp_path,
            _HASH,
            tmp_path,
            tmp_path / "work.db",
            row_batch_size=0,
        )

    root = tmp_path / "cic"
    source = root / "Benign_Final" / "a.csv"
    _write_csv(source, CIC_SOURCE_COLUMNS, [["1"] * len(CIC_SOURCE_COLUMNS)])
    request = _request(DatasetName.CIC_IOT_2023, root, tmp_path / "canonical")
    with pytest.raises(ValueError, match="positive"):
        build_canonical_corpus(request, progress_every=0)
    build_canonical_corpus(request, progress_every=100)
    _write_csv(
        source,
        CIC_SOURCE_COLUMNS,
        [["1"] * len(CIC_SOURCE_COLUMNS), ["2"] * len(CIC_SOURCE_COLUMNS)],
    )
    with pytest.raises(DataContractError, match="different build identity"):
        build_canonical_corpus(request, progress_every=100)
