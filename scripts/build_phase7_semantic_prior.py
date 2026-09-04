"""Build an immutable Phase 7 bundle from a saved LLM response and calibration anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.semantic import build_semantic_prior_bundle  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--prompt-directory", type=Path, required=True)
    parser.add_argument("--raw-response", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--alpha-l", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    """Validate, calibrate, fuse, and persist the no-LLM runtime bundle."""

    arguments = _arguments()
    report = build_semantic_prior_bundle(
        preprocessing_report=arguments.preprocessing_report,
        anchor_manifest=arguments.anchor_manifest,
        prompt_directory=arguments.prompt_directory,
        raw_response=arguments.raw_response,
        output_directory=arguments.output_directory,
        alpha_l=arguments.alpha_l,
        epsilon=arguments.epsilon,
    )
    print(f"bundle_id={report['bundle_id']}")
    print(f"prompt_id={report['prompt_id']}")
    print(f"response_id={report['response_id']}")
    print(f"anchor_prior_id={report['anchor_prior_id']}")
    print(f"bundle={arguments.output_directory.resolve() / 'semantic_prior_bundle.json'}")
    print("online_llm_call_required=false")
    print("validation_partition_accessed=false")
    print("test_partition_accessed=false")
    print("scoring_anchor_accessed=false")
    print("performance_claims_allowed=false")


if __name__ == "__main__":
    main()
