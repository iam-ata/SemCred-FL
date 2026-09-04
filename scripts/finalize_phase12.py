"""Audit and finalize a genuine Phase 12 measured clean campaign."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.data.contracts import DataContractError  # noqa: E402
from semcred_fl.experiments.phase12_completion import (  # noqa: E402
    audit_phase12_clean_campaign,
    finalize_phase12_clean_campaign,
)


def _default_campaign() -> Path:
    root = PROJECT_ROOT / "artifacts" / "experiments" / "phase_12" / "measured_final_clean"
    candidates = sorted(path for path in root.glob("*") if (path / "matrix_report.json").is_file())
    if len(candidates) != 1:
        raise ValueError(
            "The default campaign requires exactly one completed measured matrix; "
            "pass --campaign-directory explicitly"
        )
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--allow-development-registry",
        action="store_true",
        help="Audit a reduced development registry; never enables claims or test access.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign = (args.campaign_directory or _default_campaign()).resolve()
        require_confirmatory = not args.allow_development_registry
        if args.audit_only:
            audit, records = audit_phase12_clean_campaign(
                campaign,
                require_confirmatory_registry=require_confirmatory,
            )
            print("status=passed")
            print(f"audit_id={audit['audit_id']}")
            print(f"metric_records={len(records)}")
        else:
            result = finalize_phase12_clean_campaign(
                campaign,
                output_directory=args.output_directory,
                require_confirmatory_registry=require_confirmatory,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
            print(f"status={result['status']}")
            print(f"finalization_id={result['finalization_id']}")
            print(f"summary_bundle_id={result['summary_bundle_id']}")
            print(f"statistics_id={result['statistics_id']}")
            output = args.output_directory or campaign / "final_reporting"
            print(f"json={(output / 'final_manifest.json').resolve()}")
    except (DataContractError, ValueError) as error:
        print("status=blocked")
        print(f"error={error}")
        return 2
    print("manuscript_modified=false")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
