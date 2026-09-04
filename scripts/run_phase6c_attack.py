"""Build or resume one train-only Phase 6C attack condition."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.attacks.execution import (  # noqa: E402
    Phase6CAttackRequest,
    run_phase6c_attack,
)
from semcred_fl.attacks.execution_contracts import AttackRunSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--client-partition-manifest", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--norm-reference", type=Path)
    parser.add_argument("--dataset", choices=("edge_iiotset", "cic_iot_2023"), required=True)
    parser.add_argument(
        "--profile",
        choices=("iid", "dirichlet_0_1", "dirichlet_0_3", "dirichlet_1_0"),
        required=True,
    )
    parser.add_argument(
        "--attack-name",
        choices=(
            "clean",
            "label_flip",
            "clean_label",
            "backdoor",
            "scaling",
            "sign_reversal",
            "model_replacement",
            "adaptive_norm",
        ),
        required=True,
    )
    parser.add_argument("--malicious-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--poison-fraction", type=float, default=0.0)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--aggregation", choices=("fedavg", "coordinate_median"), default="fedavg")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cache-work-database", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cache-scan-batch-size", type=int, default=100_000)
    parser.add_argument("--max-rounds-this-call", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = AttackRunSpec(
        dataset=args.dataset,
        profile=args.profile,
        attack_name=args.attack_name,
        malicious_ratio=args.malicious_ratio,
        seed=args.seed,
        poison_fraction=args.poison_fraction,
        strength=args.strength,
        scale=args.scale,
        aggregation=args.aggregation,
    )
    request = Phase6CAttackRequest(
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        resolution_path=args.resolution,
        norm_reference_path=args.norm_reference,
        output_directory=args.output_directory,
        cache_directory=args.cache_directory,
        cache_work_database=args.cache_work_database,
        spec=spec,
        device=args.device,
        cache_scan_batch_size=args.cache_scan_batch_size,
        max_rounds_this_call=args.max_rounds_this_call,
    )
    report = run_phase6c_attack(request)
    print(f"status={report.get('status', 'complete')}")
    print(f"experiment_input_id={report['experiment_input_id']}")
    print(f"attack_id={report['attack_id']}")
    print(f"realization_id={report['realization_id']}")
    print(f"selection_manifest_id={report['selection_manifest_id']}")
    print(f"federated_run_id={report['federated_run_id']}")
    if report.get("status") == "interrupted":
        print(f"next_round={report['next_round']}")
    else:
        print(f"attack_run_id={report['attack_run_id']}")
        print(f"final_parameters_sha256={report['final_parameters_sha256']}")
        print(f"json={(request.output_directory / 'phase6c_attack_run.json').resolve()}")
    print(f"resume_cache_database={request.cache_work_database.resolve()}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
