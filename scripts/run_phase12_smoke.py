"""Run the tiny synthetic Phase 12 end-to-end gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.experiments.phase12_smoke import (  # noqa: E402
    Phase12SmokeConfig,
    run_phase12_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/experiments/phase_12/smoke"),
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_phase12_smoke(
        Phase12SmokeConfig(
            output_directory=args.output_directory,
            seed=args.seed,
            rounds=args.rounds,
            device=args.device,
        )
    )
    metrics = report["synthetic_test_metrics"]
    reporting = report["reporting"]
    rounds = report["rounds"]
    if (
        not isinstance(metrics, dict)
        or not isinstance(reporting, dict)
        or not isinstance(rounds, list)
    ):
        raise TypeError("smoke report metrics or reporting record is malformed")
    print(f"status={report['status']}")
    print(f"smoke_run_id={report['smoke_run_id']}")
    print(f"result_id={report['result_id']}")
    print(f"rounds_completed={len(rounds)}")
    print(f"synthetic_macro_f1={metrics['macro_f1']}")
    print(f"summary_bundle_id={reporting['summary_bundle_id']}")
    print(f"macro_f1_artifact_id={reporting['macro_f1_artifact_id']}")
    print("synthetic_only=true")
    print("real_test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
