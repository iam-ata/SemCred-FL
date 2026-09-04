"""Plan, run, audit, or finalize the canonical Phase 12 clean experiment matrix."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from finalize_phase12 import main as finalize_main
from run_phase12_step5 import main as clean_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "audit", "finalize"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-rounds-this-call", type=int)
    parser.add_argument("--max-conditions-this-call", type=int)
    parser.add_argument("--campaign-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action in {"audit", "finalize"}:
        forwarded: list[str] = []
        if args.campaign_directory is not None:
            forwarded.extend(("--campaign-directory", str(args.campaign_directory)))
        if args.action == "audit":
            forwarded.append("--audit-only")
        return finalize_main(forwarded)

    forwarded = ["--device", args.device]
    if args.action == "plan":
        forwarded.append("--dry-run")
    if args.max_rounds_this_call is not None:
        forwarded.extend(("--max-rounds-this-call", str(args.max_rounds_this_call)))
    if args.max_conditions_this_call is not None:
        forwarded.extend(("--max-conditions-this-call", str(args.max_conditions_this_call)))
    return clean_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
