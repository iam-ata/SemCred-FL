"""Build or resume the Phase 12 Step 5 clean utility matrix."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.experiments.phase12_clean_matrix import (  # noqa: E402
    CONFIRMATORY_SEEDS,
    DATASETS,
    METHODS,
    PROFILES,
    Phase12CleanMatrixRequest,
    inspect_phase12_clean_matrix,
    run_phase12_clean_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the Step 5 command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Dataset subset; the default runs both independently.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
        help="Client-partition subset; the default runs all four profiles.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(CONFIRMATORY_SEEDS),
        help="Matched experiment seeds; the default is the ten-seed registry.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        help="Development-only method subset; the default runs all five methods.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/experiments/phase_12/measured_final_clean"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/client_cache/phase_12/measured_final_clean"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-rounds-this-call", type=int)
    parser.add_argument("--max-conditions-this-call", type=int)
    parser.add_argument("--total-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clients-per-round", type=int, default=20)
    parser.add_argument("--burn-in-rounds", type=int, default=3)
    parser.add_argument("--attribution-interval", type=int, default=1)
    parser.add_argument("--trim-count", type=int, default=4)
    parser.add_argument("--krum-byzantine-count", type=int, default=4)
    parser.add_argument("--krum-selected-count", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact registry and artifact readiness without writing or training.",
    )
    parser.add_argument(
        "--standard-adam",
        action="store_true",
        help="Disable fused CUDA Adam for diagnostic parity runs.",
    )
    parser.add_argument(
        "--stream-batches",
        action="store_true",
        help="Disable one-transfer materialized client batches for diagnostics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected Step 5 registry and print its resume state."""

    args = build_parser().parse_args(argv)
    overrides: dict[str, object] = {
        "clients_per_round": args.clients_per_round,
        "total_rounds": args.total_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "burn_in_rounds": args.burn_in_rounds,
        "attribution_interval": args.attribution_interval,
        "trim_count": args.trim_count,
        "krum_byzantine_count": args.krum_byzantine_count,
        "krum_selected_count": args.krum_selected_count,
        "prefer_fused_adam": not args.standard_adam,
        "materialize_client_batches": not args.stream_batches,
    }
    if args.methods is not None:
        overrides["methods"] = tuple(args.methods)
    request = Phase12CleanMatrixRequest(
        project_root=PROJECT_ROOT,
        output_directory=args.output_directory,
        cache_root=args.cache_root,
        datasets=tuple(args.datasets),
        profiles=tuple(args.profiles),
        seeds=tuple(args.seeds),
        device=args.device,
        max_rounds_this_call=args.max_rounds_this_call,
        max_conditions_this_call=args.max_conditions_this_call,
        config_overrides=overrides,
    )
    if args.dry_run:
        report = inspect_phase12_clean_matrix(request)
        print(f"status={report['status']}")
        print(f"matrix_input_id={report['matrix_input_id']}")
        print(f"conditions_total={report['conditions_total']}")
        print(f"method_runs_total={report['method_runs_total']}")
        print(f"federated_rounds_per_method={report['federated_rounds_per_method']}")
        print(f"local_epochs_per_client={report['local_epochs_per_client']}")
        missing = report["missing_artifacts"]
        if not isinstance(missing, list):
            raise TypeError("Step 5 dry-run artifact list is malformed")
        print(f"missing_artifacts={len(missing)}")
        for path in missing:
            print(f"missing={path}")
        print(f"campaign_directory={report['campaign_directory']}")
        print("test_partition_accessed=false")
        print("performance_claims_allowed=false")
        return 0 if report["status"] == "ready" else 2
    report = run_phase12_clean_matrix(request)
    print(f"status={report['status']}")
    print(f"matrix_input_id={report['matrix_input_id']}")
    default_total = len(request.datasets) * len(request.profiles) * len(request.seeds)
    print(f"conditions_total={report.get('conditions_total', default_total)}")
    if report["status"] == "complete":
        print(f"matrix_result_id={report['matrix_result_id']}")
        report_path = args.output_directory / str(report["matrix_input_id"]) / "matrix_report.json"
        print(f"json={report_path.resolve()}")
    else:
        print(f"conditions_complete={report.get('conditions_complete', 0)}")
        active = report.get("active_condition")
        if isinstance(active, dict):
            print(
                f"active={active.get('dataset')}/{active.get('profile')}/seed_{active.get('seed')}"
            )
            methods = active.get("methods")
            if isinstance(methods, list):
                for item in methods:
                    if isinstance(item, dict):
                        print(
                            f"method={item.get('method')} status={item.get('status')} "
                            f"next_round={item.get('next_round', 'complete')}"
                        )
        print(f"campaign_directory={report['campaign_directory']}")
    print("validation_partition_accessed=completion_only")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
