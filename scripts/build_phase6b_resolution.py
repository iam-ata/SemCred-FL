"""Build or resume the Phase 6B train-only feature and trigger resolution."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.resolution import (  # noqa: E402
    Phase6BResolutionRequest,
    build_phase6b_resolution,
)
from semcred_fl.config import DatasetName  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--client-partition-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Phase6BResolutionRequest(
        dataset=DatasetName(args.dataset),
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        output_directory=args.output_directory,
        work_database=args.work_database,
        batch_size=args.batch_size,
    )
    report = build_phase6b_resolution(request)
    print(f"resolution_id={report['resolution_id']}")
    json_path = request.output_directory / f"{args.dataset}_phase6b_resolution.json"
    print(f"json={json_path.resolve()}")
    print(f"resume_database={request.work_database.resolve()}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
