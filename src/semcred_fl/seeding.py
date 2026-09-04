"""Deterministic namespace-specific seed derivation."""

from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable 32-bit seed without coupling independent subsystems.

    Args:
        base_seed: Non-negative experiment seed.
        namespace: Stable subsystem name, such as ``partition`` or ``attack``.

    Returns:
        A deterministic unsigned 32-bit integer.
    """

    if base_seed < 0:
        raise ValueError("base_seed cannot be negative")
    if not namespace.strip():
        raise ValueError("namespace cannot be empty")
    payload = f"semcred-fl-v1\0{base_seed}\0{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_standard_library(base_seed: int, namespace: str = "python") -> int:
    """Seed Python's pseudorandom generator and return the derived seed."""

    seed = derive_seed(base_seed, namespace)
    random.seed(seed)
    return seed
