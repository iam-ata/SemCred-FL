"""Build and verify the portable SemCred-FL Colab fail-safe bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "COLAB_BUNDLE_MANIFEST.json"
DEFAULT_OUTPUT = Path("artifacts/colab_bundle/semcred_fl_colab_bundle.zip")

ROOT_FILES = (
    Path("README.md"),
    Path("pyproject.toml"),
    Path("environment.yml"),
    Path("environment-win-64.lock.yml"),
    Path("requirements-colab.txt"),
)
SOURCE_DIRECTORIES = (
    Path("src"),
    Path("scripts"),
    Path("configs"),
    Path("tests"),
    Path("docs"),
    Path("notebooks"),
)
ARTIFACT_DIRECTORIES = (
    Path("artifacts/preprocessing/phase_3c"),
    Path("artifacts/client_partitions/phase_3d"),
    Path("artifacts/attacks/phase_6b"),
)
REQUIRED_ARTIFACTS = (
    Path("artifacts/preprocessing/phase_3c/edge_iiotset/edge_iiotset_phase3c_preprocessed.json"),
    Path("artifacts/preprocessing/phase_3c/cic_iot_2023/cic_iot_2023_phase3c_preprocessed.json"),
    Path("artifacts/client_partitions/phase_3d/edge_iiotset/edge_iiotset_phase3d_clients.json"),
    Path("artifacts/client_partitions/phase_3d/cic_iot_2023/cic_iot_2023_phase3d_clients.json"),
    Path("artifacts/attacks/phase_6b/edge_iiotset/edge_iiotset_phase6b_resolution.json"),
    Path("artifacts/attacks/phase_6b/cic_iot_2023/cic_iot_2023_phase6b_resolution.json"),
    Path("artifacts/attacks/phase_6b/norm_references/phase6b_norm_reference_index.json"),
    *(
        Path("artifacts/attacks/phase_6b/norm_references")
        / dataset
        / profile
        / "benign_norm_reference.json"
        for dataset in ("edge_iiotset", "cic_iot_2023")
        for profile in ("iid", "dirichlet_0_1", "dirichlet_0_3", "dirichlet_1_0")
    ),
)
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".tmp",
}
STORED_SUFFIXES = {".parquet", ".pt", ".pth", ".zip", ".gz"}
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One source file and its project-relative archive path."""

    source: Path
    relative: PurePosixPath
    size: int


def _is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts):
        return True
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _iter_files(root: Path, relative_paths: Iterable[Path]) -> Iterable[Path]:
    for relative in relative_paths:
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(f"Required Colab bundle input is missing: {source}")
        if source.is_file():
            yield source
            continue
        for candidate in sorted(source.rglob("*")):
            if candidate.is_file() and not _is_excluded(candidate.relative_to(root)):
                yield candidate


def discover_bundle_files(project_root: Path) -> tuple[BundleFile, ...]:
    """Return the deterministic, validated bundle inventory."""
    root = project_root.resolve()
    for artifact_relative in REQUIRED_ARTIFACTS:
        required = root / artifact_relative
        if not required.is_file():
            raise FileNotFoundError(f"Required prepared artifact is missing: {required}")

    selected = ROOT_FILES + SOURCE_DIRECTORIES + ARTIFACT_DIRECTORIES
    unique: dict[PurePosixPath, BundleFile] = {}
    for source in _iter_files(root, selected):
        archive_relative = PurePosixPath(source.relative_to(root).as_posix())
        if archive_relative == PurePosixPath(BUNDLE_MANIFEST_NAME):
            raise ValueError(f"Reserved bundle path encountered: {archive_relative}")
        unique[archive_relative] = BundleFile(
            source=source,
            relative=archive_relative,
            size=source.stat().st_size,
        )
    return tuple(unique[path] for path in sorted(unique, key=str))


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _bundle_identity(files: list[dict[str, object]]) -> str:
    identity_input = [
        {"path": item["path"], "size": item["size"], "sha256": item["sha256"]} for item in files
    ]
    return hashlib.sha256(_canonical_json(identity_input)).hexdigest()


def build_colab_bundle(
    project_root: Path,
    output_path: Path,
    *,
    progress_every: int = 25,
) -> dict[str, object]:
    """Create an atomic ZIP64 bundle and return its checksum manifest."""
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    root = project_root.resolve()
    output = output_path.resolve()
    inventory = discover_bundle_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.unlink(missing_ok=True)

    manifest_files: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for index, item in enumerate(inventory, start=1):
                digest = _sha256_file(item.source)
                compression = (
                    zipfile.ZIP_STORED
                    if item.source.suffix.lower() in STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(item.source, arcname=str(item.relative), compress_type=compression)
                manifest_files.append(
                    {"path": str(item.relative), "size": item.size, "sha256": digest}
                )
                if index % progress_every == 0 or index == len(inventory):
                    print(f"colab_bundle_files={index}/{len(inventory)}", flush=True)

            manifest: dict[str, object] = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "bundle_id": _bundle_identity(manifest_files),
                "file_count": len(manifest_files),
                "total_uncompressed_bytes": sum(item.size for item in inventory),
                "files": manifest_files,
                "raw_datasets_included": False,
                "resume_databases_included": False,
            }
            archive.writestr(BUNDLE_MANIFEST_NAME, _canonical_json(manifest))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    bundle_sha256 = _sha256_file(output)
    output.with_suffix(f"{output.suffix}.sha256").write_text(
        f"{bundle_sha256}  {output.name}\n", encoding="utf-8"
    )
    manifest["bundle_path"] = str(output)
    manifest["bundle_sha256"] = bundle_sha256
    return manifest


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe bundle path: {name}")
    return path


def verify_colab_bundle(bundle_path: Path) -> dict[str, object]:
    """Stream every bundled file and fail if inventory, size, or hash differs."""
    bundle = bundle_path.resolve()
    with zipfile.ZipFile(bundle, "r", allowZip64=True) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Colab bundle contains duplicate archive paths")
        for archive_name in names:
            _safe_archive_path(archive_name)
        if BUNDLE_MANIFEST_NAME not in names:
            raise ValueError("Colab bundle manifest is missing")

        manifest = cast(dict[str, object], json.loads(archive.read(BUNDLE_MANIFEST_NAME)))
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported Colab bundle schema version")
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("Colab bundle manifest files field is invalid")

        expected_names = {BUNDLE_MANIFEST_NAME}
        verified_files: list[dict[str, object]] = []
        for raw_item in raw_files:
            if not isinstance(raw_item, dict):
                raise ValueError("Colab bundle manifest file entry is invalid")
            item = cast(dict[str, object], raw_item)
            raw_name = item.get("path")
            size = item.get("size")
            expected_digest = item.get("sha256")
            if (
                not isinstance(raw_name, str)
                or not isinstance(size, int)
                or not isinstance(expected_digest, str)
            ):
                raise ValueError("Colab bundle manifest file metadata is invalid")
            name = raw_name
            _safe_archive_path(name)
            expected_names.add(name)
            info = archive.getinfo(name)
            if info.file_size != size:
                raise ValueError(f"Colab bundle size mismatch: {name}")
            with archive.open(name, "r") as stream:
                actual_digest = _sha256_stream(stream)
            if actual_digest != expected_digest:
                raise ValueError(f"Colab bundle checksum mismatch: {name}")
            verified_files.append({"path": name, "size": size, "sha256": actual_digest})

        if set(names) != expected_names:
            raise ValueError("Colab bundle contains unmanifested files")
        if manifest.get("bundle_id") != _bundle_identity(verified_files):
            raise ValueError("Colab bundle identity mismatch")
        if manifest.get("file_count") != len(verified_files):
            raise ValueError("Colab bundle file count mismatch")
        if manifest.get("total_uncompressed_bytes") != sum(
            cast(int, item["size"]) for item in verified_files
        ):
            raise ValueError("Colab bundle byte count mismatch")
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path)
    return parser


def main() -> int:
    """Run the Colab bundle CLI."""
    args = _parser().parse_args()
    project_root = cast(Path, args.project_root).resolve()
    verify_path = cast(Path | None, args.verify)
    if verify_path is not None:
        manifest = verify_colab_bundle(verify_path)
        print(f"bundle_id={manifest['bundle_id']}")
        print(f"file_count={manifest['file_count']}")
        return 0

    if cast(bool, args.dry_run):
        inventory = discover_bundle_files(project_root)
        print(f"file_count={len(inventory)}")
        print(f"total_uncompressed_bytes={sum(item.size for item in inventory)}")
        return 0

    output_arg = cast(Path | None, args.output)
    output = output_arg if output_arg is not None else project_root / DEFAULT_OUTPUT
    manifest = build_colab_bundle(
        project_root,
        output,
        progress_every=cast(int, args.progress_every),
    )
    print(f"bundle_id={manifest['bundle_id']}")
    print(f"bundle_sha256={manifest['bundle_sha256']}")
    print(f"bundle_path={manifest['bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
