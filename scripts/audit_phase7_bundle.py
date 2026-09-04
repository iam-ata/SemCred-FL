"""Identity-check one completed Phase 7 bundle without contacting an LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.semantic import load_semantic_prior_bundle  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    return parser.parse_args()


def main() -> None:
    """Load the immutable artifact and report its offline dimensions."""

    arguments = _arguments()
    bundle = load_semantic_prior_bundle(
        arguments.bundle,
        expected_bundle_id=arguments.expected_bundle_id,
    )
    print(f"dataset={bundle.dataset}")
    print(f"bundle_id={bundle.bundle_id}")
    print(f"class_count={len(bundle.class_names)}")
    print(f"feature_group_count={len(bundle.feature_group_names)}")
    print(f"alpha_l={bundle.alpha_l}")
    print("identity_valid=true")
    print("online_llm_call_required=false")
    print("performance_claims_allowed=false")


if __name__ == "__main__":
    main()
