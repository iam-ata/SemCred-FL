"""Build or resume the accelerated Phase 12 Step 3 attack comparison."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.experiments.phase12_attack import (  # noqa: E402
    Phase12AttackConfig,
    Phase12AttackRequest,
    run_phase12_attack_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessing-report",
        type=Path,
        default=Path(
            "artifacts/preprocessing/phase_3c/edge_iiotset/edge_iiotset_phase3c_preprocessed.json"
        ),
    )
    parser.add_argument(
        "--client-partition-manifest",
        type=Path,
        default=Path(
            "artifacts/client_partitions/phase_3d/edge_iiotset/edge_iiotset_phase3d_clients.json"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=Path,
        default=Path(
            "artifacts/attacks/phase_6b/edge_iiotset/edge_iiotset_phase6b_resolution.json"
        ),
    )
    parser.add_argument(
        "--semantic-prior",
        type=Path,
        default=Path("artifacts/semantic_priors/phase_7/edge_iiotset/semantic_prior_bundle.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/experiments/phase_12/step_3_label_flip_edge_dirichlet_0_3"),
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path("artifacts/client_cache/phase_12/step_3/edge_iiotset/dirichlet_0_3"),
    )
    parser.add_argument(
        "--cache-work-database",
        type=Path,
        default=Path(
            "artifacts/client_cache/phase_12/step_3/edge_iiotset/dirichlet_0_3/client_cache.sqlite3"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-rounds-this-call", type=int)
    parser.add_argument(
        "--standard-adam",
        action="store_true",
        help="Disable fused CUDA Adam for diagnostic parity runs.",
    )
    parser.add_argument(
        "--stream-batches",
        action="store_true",
        help="Disable one-transfer materialized attacked batches for diagnostics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Phase12AttackConfig(
        prefer_fused_adam=not args.standard_adam,
        materialize_client_batches=not args.stream_batches,
    )
    request = Phase12AttackRequest(
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        resolution_path=args.resolution,
        semantic_prior_path=args.semantic_prior,
        output_directory=args.output_directory,
        cache_directory=args.cache_directory,
        cache_work_database=args.cache_work_database,
        device=args.device,
        max_rounds_this_call=args.max_rounds_this_call,
        config=config,
    )
    report = run_phase12_attack_comparison(request)
    print(f"status={report['status']}")
    print(f"comparison_input_id={report['comparison_input_id']}")
    print(f"attack_id={report['attack_id']}")
    print(f"realization_id={report['realization_id']}")
    methods = report.get("methods")
    if isinstance(methods, list):
        for item in methods:
            if isinstance(item, dict):
                print(
                    f"method={item.get('method')} status={item.get('status')} "
                    f"next_round={item.get('next_round', 'complete')}"
                )
    else:
        print(f"comparison_result_id={report['comparison_result_id']}")
        reporting = report.get("reporting")
        if isinstance(reporting, dict):
            print(f"summary_bundle_id={reporting['summary_bundle_id']}")
            print(f"macro_f1_artifact_id={reporting['macro_f1_artifact_id']}")
        print(f"json={(args.output_directory / 'comparison_report.json').resolve()}")
    print(f"resume_cache_database={args.cache_work_database.resolve()}")
    print("validation_partition_accessed=completion_only")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
