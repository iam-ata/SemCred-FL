from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from semcred_fl.portability.colab_bundle import (
    BUNDLE_MANIFEST_NAME,
    build_colab_bundle,
    discover_bundle_files,
    verify_colab_bundle,
)


def _minimal_project(root: Path) -> None:
    for name in (
        "README.md",
        "pyproject.toml",
        "environment.yml",
        "environment-win-64.lock.yml",
        "requirements-colab.txt",
    ):
        (root / name).write_text(name, encoding="utf-8")
    for directory in ("src", "scripts", "configs", "tests", "docs", "notebooks"):
        path = root / directory
        path.mkdir(parents=True)
        (path / "kept.txt").write_text(directory, encoding="utf-8")

    required = (
        "artifacts/preprocessing/phase_3c/edge_iiotset/edge_iiotset_phase3c_preprocessed.json",
        "artifacts/preprocessing/phase_3c/cic_iot_2023/cic_iot_2023_phase3c_preprocessed.json",
        "artifacts/client_partitions/phase_3d/edge_iiotset/edge_iiotset_phase3d_clients.json",
        "artifacts/client_partitions/phase_3d/cic_iot_2023/cic_iot_2023_phase3d_clients.json",
        "artifacts/attacks/phase_6b/edge_iiotset/edge_iiotset_phase6b_resolution.json",
        "artifacts/attacks/phase_6b/cic_iot_2023/cic_iot_2023_phase6b_resolution.json",
        "artifacts/attacks/phase_6b/norm_references/phase6b_norm_reference_index.json",
        *(
            f"artifacts/attacks/phase_6b/norm_references/{dataset}/{profile}/benign_norm_reference.json"
            for dataset in ("edge_iiotset", "cic_iot_2023")
            for profile in ("iid", "dirichlet_0_1", "dirichlet_0_3", "dirichlet_1_0")
        ),
    )
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (root / "artifacts/preprocessing/phase_3c/edge_iiotset/shards").mkdir(parents=True)
    (root / "artifacts/preprocessing/phase_3c/edge_iiotset/shards/0000.parquet").write_bytes(
        b"parquet-fixture"
    )
    (root / "artifacts/preprocessing/phase_3c/edge_iiotset/work.sqlite3").write_bytes(b"excluded")


def test_discovery_excludes_resume_databases(tmp_path: Path) -> None:
    _minimal_project(tmp_path)
    relative_paths = {str(item.relative) for item in discover_bundle_files(tmp_path)}
    assert "artifacts/preprocessing/phase_3c/edge_iiotset/shards/0000.parquet" in relative_paths
    assert (
        "artifacts/attacks/phase_6b/norm_references/edge_iiotset/iid/benign_norm_reference.json"
        in relative_paths
    )
    assert "artifacts/preprocessing/phase_3c/edge_iiotset/work.sqlite3" not in relative_paths


def test_build_and_verify_round_trip(tmp_path: Path) -> None:
    _minimal_project(tmp_path)
    output = tmp_path / "output" / "bundle.zip"
    built = build_colab_bundle(tmp_path, output, progress_every=100)
    verified = verify_colab_bundle(output)
    assert verified["bundle_id"] == built["bundle_id"]
    assert output.with_suffix(".zip.sha256").is_file()


def test_verifier_rejects_unmanifested_file(tmp_path: Path) -> None:
    _minimal_project(tmp_path)
    output = tmp_path / "bundle.zip"
    build_colab_bundle(tmp_path, output, progress_every=100)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            target.writestr(info.filename, source.read(info.filename))
        target.writestr("unexpected.txt", b"not manifested")
    with pytest.raises(ValueError, match="unmanifested"):
        verify_colab_bundle(tampered)


def test_manifest_is_valid_json(tmp_path: Path) -> None:
    _minimal_project(tmp_path)
    output = tmp_path / "bundle.zip"
    build_colab_bundle(tmp_path, output, progress_every=100)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read(BUNDLE_MANIFEST_NAME))
    assert manifest["raw_datasets_included"] is False
    assert manifest["resume_databases_included"] is False
