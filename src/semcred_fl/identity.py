"""Canonical serialization and deterministic run identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def to_primitive(value: object) -> object:
    """Convert supported typed configuration values to JSON primitives.

    Args:
        value: A dataclass, enum, path, tuple, list, dictionary, or JSON primitive.

    Returns:
        A recursively normalized object suitable for canonical JSON serialization.

    Raises:
        TypeError: If the value has no supported deterministic representation.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple | list):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Canonical dictionaries require string keys")
        return {key: to_primitive(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize a supported value with stable key order and separators."""

    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def run_id(value: object, *, length: int = 16) -> str:
    """Return a deterministic hexadecimal identifier for a configuration."""

    if not 8 <= length <= 64:
        raise ValueError("Run identifier length must be between 8 and 64")
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]
