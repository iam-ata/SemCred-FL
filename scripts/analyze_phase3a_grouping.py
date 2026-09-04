"""Audit Phase 3A grouping and split feasibility without assigning a split."""

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
from semcred_fl.data.group_feasibility import (  # noqa: E402
    GroupFeasibilityRequest,
    analyze_group_feasibility,
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
    parser.add_argument("--calibration-anchors-per-class", type=int, default=32)
    parser.add_argument("--scoring-anchors-per-class", type=int, default=32)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument("--connectivity-batch-size", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    output = args.output_directory.resolve()
    work = (
        args.work_database.resolve()
        if args.work_database is not None
        else output / f"{dataset.value}_phase3a.sqlite3"
    )
    report = analyze_group_feasibility(
        GroupFeasibilityRequest(
            dataset=dataset,
            canonical_manifest=args.canonical_manifest.resolve(),
            canonical_database=args.canonical_database.resolve(),
            output_directory=output,
            work_database=work,
            calibration_anchors_per_class=args.calibration_anchors_per_class,
            scoring_anchors_per_class=args.scoring_anchors_per_class,
            parquet_batch_size=args.parquet_batch_size,
            connectivity_batch_size=args.connectivity_batch_size,
        ),
        progress_every=args.progress_every,
    )
    print(f"analysis_id={report['analysis_id']}")
    print(f"json={output / f'{dataset.value}_phase3a.json'}")
    print(f"markdown={output / f'{dataset.value}_phase3a.md'}")
    print(f"resume_database={work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
