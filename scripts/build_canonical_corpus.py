"""Build or resume a Phase 2D canonical Parquet corpus."""

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
from semcred_fl.data.canonical import (  # noqa: E402
    CanonicalBuildRequest,
    build_canonical_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 2D command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path)
    parser.add_argument("--row-batch-size", type=int, default=10_000)
    parser.add_argument("--parquet-batch-size", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute Phase 2D and print its artifact paths."""

    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    output_directory = args.output_directory.resolve()
    work_database = (
        args.work_database.resolve()
        if args.work_database is not None
        else output_directory / f"{dataset.value}_canonical.sqlite3"
    )
    report = build_canonical_corpus(
        CanonicalBuildRequest(
            dataset=dataset,
            root=args.root.resolve(),
            archive_sha256=args.archive_sha256,
            output_directory=output_directory,
            work_database=work_database,
            row_batch_size=args.row_batch_size,
            parquet_batch_size=args.parquet_batch_size,
        ),
        progress_every=args.progress_every,
    )
    print(f"corpus_id={report['corpus_id']}")
    print(f"json={output_directory / f'{dataset.value}_canonical.json'}")
    print(f"markdown={output_directory / f'{dataset.value}_canonical.md'}")
    print(f"schema={output_directory / f'{dataset.value}_canonical_schema.json'}")
    print(f"resume_database={work_database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
