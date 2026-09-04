"""Plan or materialize a reproducible Phase 3B split and anchor manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.config import DatasetName  # noqa: E402
from semcred_fl.data.split_manifest import (  # noqa: E402
    SplitBuildRequest,
    build_phase3b_split,
    plan_phase3b_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path)
    parser.add_argument("--base-seed", type=int, default=20260816)
    parser.add_argument("--train-basis-points", type=int, default=7000)
    parser.add_argument("--validation-basis-points", type=int, default=1500)
    parser.add_argument("--test-basis-points", type=int, default=1500)
    parser.add_argument("--calibration-anchors-per-class", type=int, default=32)
    parser.add_argument("--scoring-anchors-per-class", type=int, default=32)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument("--phase3a-database", type=Path)
    parser.add_argument("--phase3a1-report", type=Path)
    parser.add_argument("--edge-raw-mitm-csv", type=Path)
    parser.add_argument("--edge-mitm-pcap", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def _resolved(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    output = args.output_directory.resolve()
    work = (
        args.work_database.resolve()
        if args.work_database is not None
        else output / f"{dataset.value}_phase3b.sqlite3"
    )
    request = SplitBuildRequest(
        dataset=dataset,
        canonical_manifest=args.canonical_manifest.resolve(),
        canonical_database=args.canonical_database.resolve(),
        output_directory=output,
        work_database=work,
        base_seed=args.base_seed,
        train_basis_points=args.train_basis_points,
        validation_basis_points=args.validation_basis_points,
        test_basis_points=args.test_basis_points,
        calibration_anchors_per_class=args.calibration_anchors_per_class,
        scoring_anchors_per_class=args.scoring_anchors_per_class,
        parquet_batch_size=args.parquet_batch_size,
        phase3a_database=_resolved(args.phase3a_database),
        phase3a1_report=_resolved(args.phase3a1_report),
        edge_raw_mitm_csv=_resolved(args.edge_raw_mitm_csv),
        edge_mitm_pcap=_resolved(args.edge_mitm_pcap),
    )
    report = (
        plan_phase3b_split(request)
        if args.plan_only
        else build_phase3b_split(request, progress_every=args.progress_every)
    )
    identifier = report["plan_id"] if args.plan_only else report["split_id"]
    print(f"{'plan_id' if args.plan_only else 'split_id'}={identifier}")
    print(f"output_directory={output}")
    if not args.plan_only:
        print(f"resume_database={work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
