"""Command-line entry point for configuration inspection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from semcred_fl.config import ExperimentConfig
from semcred_fl.identity import canonical_json, run_id


def build_parser() -> argparse.ArgumentParser:
    """Construct the SemCred-FL command-line parser."""

    parser = argparse.ArgumentParser(prog="semcred-fl")
    parser.add_argument(
        "command",
        choices=("show-default", "validate-default"),
        help="Inspect or validate the frozen pilot configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    args = build_parser().parse_args(argv)
    config = ExperimentConfig()
    if args.command == "show-default":
        print(canonical_json(config))
        print(f"run_id={run_id(config)}")
    else:
        print(f"valid run_id={run_id(config)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
