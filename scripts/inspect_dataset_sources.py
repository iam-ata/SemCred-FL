"""Run a bounded, read-only inspection of SemCred-FL dataset source files."""

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
from semcred_fl.data.ciciot2023 import CICIoT2023Adapter  # noqa: E402
from semcred_fl.data.edge_iiotset import EdgeIIoTsetAdapter  # noqa: E402
from semcred_fl.data.validation import DatasetAdapter, inspect_adapter  # noqa: E402
from semcred_fl.identity import canonical_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Construct the bounded source-inspection command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(item.value for item in DatasetName), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-rows-per-file", type=int, default=3)
    return parser


def _adapter(dataset: DatasetName, root: Path) -> DatasetAdapter:
    if dataset is DatasetName.CIC_IOT_2023:
        return CICIoT2023Adapter(root)
    return EdgeIIoTsetAdapter(root)


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect bounded source prefixes and print a canonical JSON summary."""

    args = build_parser().parse_args(argv)
    dataset = DatasetName(args.dataset)
    summary = inspect_adapter(
        _adapter(dataset, args.root),
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
