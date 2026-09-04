"""Run or resume a sequential Phase 6C attack matrix to an absolute round target."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.matrix import (  # noqa: E402
    Phase6CMatrixRequest,
    inspect_phase6c_matrix,
    run_phase6c_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--client-partition-manifest", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--norm-reference-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("edge_iiotset", "cic_iot_2023"), required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("iid", "dirichlet_0_1", "dirichlet_0_3", "dirichlet_1_0"),
        required=True,
    )
    parser.add_argument(
        "--malicious-ratios",
        nargs="+",
        type=float,
        choices=(0.1, 0.2, 0.3, 0.4),
        required=True,
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--matrix", choices=("all_types", "full_grid"), default="all_types")
    parser.add_argument("--target-round", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache-scan-batch-size", type=int, default=100_000)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Phase6CMatrixRequest(
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        resolution_path=args.resolution,
        output_root=args.output_root,
        cache_root=args.cache_root,
        norm_reference_root=args.norm_reference_root,
        dataset=args.dataset,
        profiles=tuple(args.profiles),
        malicious_ratios=tuple(args.malicious_ratios),
        seeds=tuple(args.seeds),
        matrix=args.matrix,
        target_round=args.target_round,
        device=args.device,
        cache_scan_batch_size=args.cache_scan_batch_size,
    )
    if args.plan_only:
        report = inspect_phase6c_matrix(request)
        conditions = report["conditions"]
        if not isinstance(conditions, list):
            raise TypeError("Phase 6C matrix plan conditions are malformed")
        for value in conditions:
            if not isinstance(value, dict):
                raise TypeError("Phase 6C matrix plan condition is malformed")
            print(
                f"condition={value['position']}/{report['conditions_total']} "
                f"attack={value['attack_name']} poison_fraction={value['poison_fraction']} "
                f"strength={value['strength']} scale={value['scale']} "
                f"completed_rounds={value['completed_rounds']} "
                f"at_target={str(value['at_target']).lower()}"
            )
        print("status=plan_only")
        print(f"conditions_total={report['conditions_total']}")
        print(f"conditions_at_target={report['conditions_at_target']}")
        print(f"conditions_pending={report['conditions_pending']}")
        print(f"target_round={report['target_round']}")
        print("test_partition_accessed=false")
        print("performance_claims_allowed=false")
        return 0

    report = run_phase6c_matrix(request)
    print(f"status={report['status']}")
    print(f"matrix_id={report['matrix_id']}")
    print(f"matrix_run_id={report['matrix_run_id']}")
    print(f"conditions_total={report['conditions_total']}")
    print(f"conditions_at_target={report['conditions_at_target']}")
    print(f"target_round={report['target_round']}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
