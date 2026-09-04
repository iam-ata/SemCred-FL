"""Invoke OpenAI once for a frozen public-only Phase 7 prompt."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.semantic import invoke_openai_semantic_prior  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    return await invoke_openai_semantic_prior(
        prompt_directory=arguments.prompt_directory,
        output_directory=arguments.output_directory,
        timeout_seconds=arguments.timeout_seconds,
    )


def main() -> None:
    """Run the guarded provider call and print its credential-free receipt summary."""

    arguments = _arguments()
    report = asyncio.run(_run(arguments))
    usage = report["usage"]
    if not isinstance(usage, dict):
        raise RuntimeError("Validated OpenAI receipt has no usage object")
    print(f"receipt_id={report['receipt_id']}")
    print(f"provider_response_id={report['provider_response_id']}")
    print(f"model={report['resolved_model']}")
    print(f"api_called={str(report['api_called']).lower()}")
    print(f"reused={str(report['reused']).lower()}")
    print(f"input_tokens={usage['input_tokens']}")
    print(f"output_tokens={usage['output_tokens']}")
    print(f"total_tokens={usage['total_tokens']}")
    print(f"output_directory={arguments.output_directory.resolve()}")
    print("store_requested=false")
    print("private_records_accessed=false")
    print("client_updates_accessed=false")
    print("validation_partition_accessed=false")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")


if __name__ == "__main__":
    main()
