"""Stable record identity derived before filtering or partitioning."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from semcred_fl.config import DatasetName
from semcred_fl.identity import canonical_json


def normalized_source_file(dataset_root: Path, source_file: Path) -> str:
    """Return a stable POSIX path relative to the declared dataset root.

    Args:
        dataset_root: Root directory owned by one adapter.
        source_file: Source file that must be located beneath the root.

    Returns:
        A non-empty relative path with POSIX separators.

    Raises:
        ValueError: If the source file is outside the dataset root.
    """

    root = dataset_root.resolve()
    resolved = source_file.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source file is outside dataset root: {source_file}") from exc
    normalized = relative.as_posix()
    if not normalized or normalized == ".":
        raise ValueError("Source file identity must not be empty")
    return normalized


def record_id(dataset: DatasetName, source_file: str, source_row: int) -> str:
    """Return the full deterministic SHA-256 identity for one source row."""

    if source_row < 1:
        raise ValueError("source_row must be one-based and positive")
    normalized = source_file.replace("\\", "/")
    if not normalized or normalized == ".":
        raise ValueError("source_file must not be empty")
    source_path = PurePosixPath(normalized)
    if source_path.is_absolute() or ".." in source_path.parts or ":" in source_path.parts[0]:
        raise ValueError("source_file must be a safe relative path")

    identity = {
        "dataset": dataset.value,
        "source_file": source_path.as_posix(),
        "source_row": source_row,
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
