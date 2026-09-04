"""Plan or build leakage-safe Phase 3C preprocessing artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.config import DatasetName  # noqa: E402
from semcred_fl.data.preprocessing_artifact import (  # noqa: E402
    PreprocessingBuildRequest,
    build_phase3c_preprocessing,
    plan_phase3c_preprocessing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path)
    parser.add_argument("--base-seed", type=int, default=20260816)
    parser.add_argument("--lower-quantile", type=float, default=0.001)
    parser.add_argument("--upper-quantile", type=float, default=0.999)
    parser.add_argument("--quantile-sample-target", type=int, default=1_000_000)
    parser.add_argument("--standard-deviation-epsilon", type=float, default=1e-12)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    output = args.output_directory.resolve()
    work = (
        args.work_database.resolve()
        if args.work_database is not None
        else output / f"{dataset.value}_phase3c.sqlite3"
    )
    request = PreprocessingBuildRequest(
        dataset=dataset,
        canonical_manifest=args.canonical_manifest.resolve(),
        split_manifest=args.split_manifest.resolve(),
        anchor_manifest=args.anchor_manifest.resolve(),
        output_directory=output,
        work_database=work,
        base_seed=args.base_seed,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        quantile_sample_target=args.quantile_sample_target,
        standard_deviation_epsilon=args.standard_deviation_epsilon,
        parquet_batch_size=args.parquet_batch_size,
    )
    report = (
        plan_phase3c_preprocessing(request)
        if args.plan_only
        else build_phase3c_preprocessing(request, progress_every=args.progress_every)
    )
    key = "plan_id" if args.plan_only else "preprocessed_corpus_id"
    print(f"{key}={report[key]}")
    print(f"output_directory={output}")
    if not args.plan_only:
        preprocessor = cast(dict[str, object], report["preprocessor"])
        print(f"preprocessor_id={preprocessor['preprocessor_id']}")
        print(f"resume_database={work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
