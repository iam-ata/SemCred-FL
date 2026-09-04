"""Aggregate Phase 11 run metrics and regenerate traceable table/figure artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.evaluation.reporting import (  # noqa: E402
    read_metric_jsonl,
    render_summary_artifacts,
    write_summary_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_metric_jsonl(args.input_jsonl)
    bundle = write_summary_bundle(
        records,
        args.output_directory,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = render_summary_artifacts(
        args.output_directory / "metric_summaries.json",
        args.output_directory,
        metric_name=args.metric,
    )
    print("status=complete")
    print(f"summary_bundle_id={bundle['summary_bundle_id']}")
    print(f"artifact_id={rendered['artifact_id']}")
    summaries = bundle["summaries"]
    if not isinstance(summaries, list):
        raise TypeError("summary bundle summaries are malformed")
    print(f"summaries={len(summaries)}")
    print(f"output_directory={args.output_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
