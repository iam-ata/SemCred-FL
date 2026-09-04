"""Run or resume the SemCred-FL Phase 2C source diagnostics."""

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
from semcred_fl.data.phase2c import Phase2CRequest, run_phase2c  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 2C command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--work-database", type=Path)
    parser.add_argument("--max-exact-domain-values", type=int, default=100_000)
    parser.add_argument("--reported-top-values", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute Phase 2C and print its artifact paths."""

    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    output_directory = args.output_directory.resolve()
    work_database = (
        args.work_database.resolve()
        if args.work_database is not None
        else output_directory / f"{dataset.value}_phase2c.sqlite3"
    )
    report = run_phase2c(
        Phase2CRequest(
            dataset=dataset,
            root=args.root.resolve(),
            audit_json=args.audit_json.resolve(),
            audit_database=args.audit_database.resolve(),
            output_directory=output_directory,
            work_database=work_database,
            max_exact_domain_values=args.max_exact_domain_values,
            reported_top_values=args.reported_top_values,
        ),
        progress_every=args.progress_every,
    )
    print(f"analysis_id={report['analysis_id']}")
    print(f"json={output_directory / f'{dataset.value}_phase2c.json'}")
    print(f"markdown={output_directory / f'{dataset.value}_phase2c.md'}")
    print(f"resume_database={work_database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
