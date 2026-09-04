"""Freeze a public-only Phase 7 prompt without calling an external LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.semantic import (  # noqa: E402
    load_invocation_metadata,
    load_public_metadata,
    prepare_semantic_prompt,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--public-metadata", type=Path, required=True)
    parser.add_argument("--invocation-metadata", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate public inputs and persist the immutable prompt declaration."""

    arguments = _arguments()
    report = prepare_semantic_prompt(
        preprocessing_report=arguments.preprocessing_report,
        metadata=load_public_metadata(arguments.public_metadata),
        invocation=load_invocation_metadata(arguments.invocation_metadata),
        output_directory=arguments.output_directory,
    )
    print(f"prompt_id={report['prompt_id']}")
    print(f"public_metadata_id={report['public_metadata_id']}")
    print(f"invocation_id={report['invocation_id']}")
    print(f"prompt={arguments.output_directory.resolve() / 'prompt.txt'}")
    print("llm_called=false")
    print("private_records_accessed=false")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")


if __name__ == "__main__":
    main()
