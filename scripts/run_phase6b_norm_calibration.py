"""Build or resume one Phase 6B.2 benign update-norm reference."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.norm_calibration import (  # noqa: E402
    Phase6BNormCalibrationRequest,
    run_phase6b_norm_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--client-partition-manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("iid", "dirichlet_0_1", "dirichlet_0_3", "dirichlet_1_0"),
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cache-work-database", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache-scan-batch-size", type=int, default=100_000)
    parser.add_argument("--max-rounds-this-call", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Phase6BNormCalibrationRequest(
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        profile=args.profile,
        output_directory=args.output_directory,
        cache_directory=args.cache_directory,
        cache_work_database=args.cache_work_database,
        device=args.device,
        cache_scan_batch_size=args.cache_scan_batch_size,
        max_rounds_this_call=args.max_rounds_this_call,
    )
    report = run_phase6b_norm_calibration(request)
    print(f"status={report.get('status', 'complete')}")
    print(f"calibration_input_id={report['calibration_input_id']}")
    print(f"federated_run_id={report['federated_run_id']}")
    print(f"cache_id={report['cache_id']}")
    if report.get("status") == "interrupted":
        print(f"next_round={report['next_round']}")
    else:
        print(f"calibration_id={report['calibration_id']}")
        reference = report["reference"]
        if not isinstance(reference, dict):
            raise TypeError("Completed reference record is malformed")
        print(f"reference_id={reference['reference_id']}")
        print(f"update_count={reference['update_count']}")
        print(f"lower_norm={reference['lower_norm']}")
        print(f"upper_norm={reference['upper_norm']}")
        print(f"json={(request.output_directory / 'benign_norm_reference.json').resolve()}")
    print(f"resume_cache_database={request.cache_work_database.resolve()}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
