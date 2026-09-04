"""Run the validation-only Phase 4 centralized detector sanity baseline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.training.centralized import (  # noqa: E402
    CentralizedSanityConfig,
    run_centralized_sanity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--client-partition-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-class-weight", type=float, default=10.0)
    parser.add_argument("--train-record-limit", type=int, default=100_000)
    parser.add_argument("--validation-record-limit", type=int, default=50_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every-batches", type=int, default=100)
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Use the complete train and validation partitions instead of bounded sanity subsets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_limit = None if args.full_corpus else args.train_record_limit
    validation_limit = None if args.full_corpus else args.validation_record_limit
    config = CentralizedSanityConfig(
        preprocessing_report=args.preprocessing_report,
        client_partition_manifest=args.client_partition_manifest,
        output_directory=args.output_directory,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maximum_class_weight=args.maximum_class_weight,
        train_record_limit=train_limit,
        validation_record_limit=validation_limit,
        device=args.device,
        progress_every_batches=args.progress_every_batches,
    )
    report = run_centralized_sanity(config)
    print(f"sanity_id={report['sanity_id']}")
    print(f"output_directory={(config.output_directory / str(report['sanity_id'])).resolve()}")
    print(f"best_checkpoint={report['best_checkpoint']}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
