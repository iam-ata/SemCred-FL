"""Audit bounded Edge-IIoTset MITM PCAP grouping candidates for Phase 3A.1."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.data.group_refinement import (  # noqa: E402
    EdgeMitmRefinementRequest,
    analyze_edge_mitm_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-database", type=Path, required=True)
    parser.add_argument("--phase3a-report", type=Path, required=True)
    parser.add_argument("--phase3a-database", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--calibration-anchors-per-class", type=int, default=32)
    parser.add_argument("--scoring-anchors-per-class", type=int, default=32)
    parser.add_argument("--time-block-seconds", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--minimum-ip-alignment-rows", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_directory.resolve()
    report = analyze_edge_mitm_refinement(
        EdgeMitmRefinementRequest(
            canonical_manifest=args.canonical_manifest.resolve(),
            canonical_database=args.canonical_database.resolve(),
            phase3a_report=args.phase3a_report.resolve(),
            phase3a_database=args.phase3a_database.resolve(),
            raw_csv=args.raw_csv.resolve(),
            pcap=args.pcap.resolve(),
            output_directory=output,
            calibration_anchors_per_class=args.calibration_anchors_per_class,
            scoring_anchors_per_class=args.scoring_anchors_per_class,
            time_block_seconds=tuple(args.time_block_seconds),
            minimum_ip_alignment_rows=args.minimum_ip_alignment_rows,
        )
    )
    print(f"analysis_id={report['analysis_id']}")
    print(f"json={output / 'edge_iiotset_phase3a1.json'}")
    print(f"markdown={output / 'edge_iiotset_phase3a1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
