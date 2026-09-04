"""Audit the complete Phase 6B.2 norm-reference matrix and write its closure index."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.norm_reference_audit import (  # noqa: E402
    NormReferenceAuditRequest,
    audit_phase6b_norm_references,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_phase6b_norm_references(
        NormReferenceAuditRequest(
            root_directory=args.root_directory,
            output_path=args.output,
        )
    )
    print(f"closure_id={report['closure_id']}")
    print(f"reference_count={report['expected_reference_count']}")
    print(f"total_development_updates={report['total_development_updates']}")
    print(f"json={args.output.resolve()}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
