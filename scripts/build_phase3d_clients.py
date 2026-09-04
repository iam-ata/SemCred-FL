"""Plan or build exact Phase 3D IID and Dirichlet client assignments."""

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
from semcred_fl.data.client_partition import (  # noqa: E402
    ClientPartitionBuildRequest,
    build_phase3d_clients,
    plan_phase3d_clients,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path)
    parser.add_argument("--base-seed", type=int, default=20260816)
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--minimum-samples-per-client", type=int, default=128)
    parser.add_argument("--dirichlet-alpha", type=float, action="append", dest="dirichlet_alphas")
    parser.add_argument("--maximum-partition-attempts", type=int, default=1_000)
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
        else output / f"{dataset.value}_phase3d.sqlite3"
    )
    alphas = tuple(args.dirichlet_alphas) if args.dirichlet_alphas is not None else (0.1, 0.3, 1.0)
    request = ClientPartitionBuildRequest(
        dataset=dataset,
        preprocessing_report=args.preprocessing_report.resolve(),
        split_manifest=args.split_manifest.resolve(),
        anchor_manifest=args.anchor_manifest.resolve(),
        output_directory=output,
        work_database=work,
        base_seed=args.base_seed,
        num_clients=args.num_clients,
        minimum_samples_per_client=args.minimum_samples_per_client,
        dirichlet_alphas=alphas,
        maximum_partition_attempts=args.maximum_partition_attempts,
        parquet_batch_size=args.parquet_batch_size,
    )
    report = (
        plan_phase3d_clients(request)
        if args.plan_only
        else build_phase3d_clients(request, progress_every=args.progress_every)
    )
    key = "plan_id" if args.plan_only else "client_partition_id"
    print(f"{key}={report[key]}")
    print(f"output_directory={output}")
    if not args.plan_only:
        print(f"resume_database={work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
