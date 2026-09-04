"""Validate the SemCred-FL Python and CUDA execution environment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "PyYAML",
    "matplotlib",
    "seaborn",
    "pytest",
    "ruff",
    "black",
    "mypy",
)


def package_versions() -> dict[str, str | None]:
    """Return installed versions for the reproducibility-critical packages."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def collect_report(require_cuda: bool) -> tuple[dict[str, Any], list[str]]:
    """Collect environment metadata and return validation failures."""
    failures: list[str] = []
    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_required": require_cuda,
    }

    try:
        import torch
    except ImportError as exc:
        failures.append(f"PyTorch import failed: {exc}")
        report["torch"] = {"importable": False}
        return report, failures

    cuda_available = torch.cuda.is_available()
    torch_report: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
    }

    if cuda_available:
        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)
        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        smoke_value = float((tensor * tensor).sum().item())
        torch.cuda.synchronize(device)
        torch_report.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": list(capability),
                "device_count": torch.cuda.device_count(),
                "gpu_smoke_value": smoke_value,
            }
        )
        if smoke_value != 14.0:
            failures.append(f"Unexpected CUDA smoke-test value: {smoke_value}")
    elif require_cuda:
        failures.append("CUDA is required but torch.cuda.is_available() is false")

    report["torch"] = torch_report
    missing = [name for name, installed in report["packages"].items() if installed is None]
    if missing:
        failures.append("Missing required packages: " + ", ".join(missing))
    return report, failures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    """Run validation and print a machine-readable JSON report."""
    args = parse_args()
    report, failures = collect_report(require_cuda=not args.allow_cpu)
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
