"""Benchmark baseline versus optimized Step 2 local training on one real client."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.data.client_cache import load_selected_client_cache  # noqa: E402
from semcred_fl.experiments.phase12_clean import (  # noqa: E402
    _MaterializedClientBatchSource,
    _tensor_sha256,
)
from semcred_fl.federation.client import (  # noqa: E402
    LocalTrainingConfig,
    train_local_client,
)
from semcred_fl.federation.parameters import (  # noqa: E402
    flatten_parameters,
    parameter_manifest,
)
from semcred_fl.identity import run_id  # noqa: E402
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-cache-manifest",
        type=Path,
        default=Path("artifacts/client_cache/phase_6c/edge_iiotset/iid/client_cache.json"),
    )
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/phase_12/performance/step2_client00.json"),
    )
    return parser


def _run_variant(
    *,
    name: str,
    cache_manifest: Path,
    client_index: int,
    batch_size: int,
    seed: int,
    device: str,
    fused: bool,
    materialized: bool,
    prevalidated: bool,
) -> dict[str, object]:
    cache = load_selected_client_cache(cache_manifest)
    if not 0 <= client_index < len(cache.selected_client_indices):
        raise ValueError("Benchmark client index is outside the cache")
    detector = DetectorConfig(cache.input_dimension, cache.num_classes)
    torch.manual_seed(seed)
    model = IntrusionMLP(detector)
    manifest = parameter_manifest(model)
    global_parameters = flatten_parameters(model, manifest).detach().cpu()
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    update = train_local_client(
        detector=detector,
        manifest=manifest,
        global_parameters=global_parameters,
        round_id=0,
        client_id=f"client-{client_index:02d}",
        declared_sample_count=cache.client_total(client_index),
        class_counts=cache.class_counts(client_index),
        batch_source=_MaterializedClientBatchSource(
            cache,
            client_index,
            batch_size,
            device,
            materialize=materialized,
        ),
        config=LocalTrainingConfig(
            local_epochs=1,
            learning_rate=1e-3,
            maximum_class_weight=10.0,
            seed=seed,
            device=device,
            optimizer_backend="fused" if fused else "standard",
            prevalidated_batches=prevalidated,
        ),
        is_malicious=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    return {
        "name": name,
        "duration_seconds": duration,
        "delta_sha256": _tensor_sha256(update.delta),
        "update_norm": update.update_norm,
        "mean_local_loss": update.mean_local_loss,
        "records": update.declared_sample_count,
        "batch_size": batch_size,
        "optimizer_backend": "fused" if fused else "standard",
        "materialized_client_batches": materialized,
        "prevalidated_batches": prevalidated,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but unavailable")
    baseline = _run_variant(
        name="baseline_streamed_standard_adam",
        cache_manifest=args.client_cache_manifest,
        client_index=args.client_index,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fused=False,
        materialized=False,
        prevalidated=False,
    )
    prevalidated = _run_variant(
        name="streamed_standard_adam_prevalidated",
        cache_manifest=args.client_cache_manifest,
        client_index=args.client_index,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fused=False,
        materialized=False,
        prevalidated=True,
    )
    materialized = _run_variant(
        name="materialized_standard_adam",
        cache_manifest=args.client_cache_manifest,
        client_index=args.client_index,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fused=False,
        materialized=True,
        prevalidated=True,
    )
    optimized = _run_variant(
        name="materialized_fused_adam",
        cache_manifest=args.client_cache_manifest,
        client_index=args.client_index,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fused=args.device == "cuda",
        materialized=True,
        prevalidated=True,
    )
    baseline_seconds = float(baseline["duration_seconds"])
    prevalidated_seconds = float(prevalidated["duration_seconds"])
    materialized_seconds = float(materialized["duration_seconds"])
    optimized_seconds = float(optimized["duration_seconds"])
    standard_hashes = {
        baseline["delta_sha256"],
        prevalidated["delta_sha256"],
        materialized["delta_sha256"],
    }
    if len(standard_hashes) != 1:
        raise RuntimeError("Non-mathematical Step 2 optimizations changed the client update")
    stable = {
        "schema": "semcred_phase12_step2_training_benchmark_v1",
        "client_cache_manifest": str(args.client_cache_manifest.resolve()),
        "client_index": args.client_index,
        "seed": args.seed,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "baseline": baseline,
        "prevalidated": prevalidated,
        "materialized": materialized,
        "optimized": optimized,
        "speedup": baseline_seconds / optimized_seconds,
        "gates": {
            "standard_backend_output_preserved_across_delivery_optimizations": True,
            "real_training_partition_only": True,
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    report = {"benchmark_id": run_id(stable, length=64), **stable}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"benchmark_id={report['benchmark_id']}")
    print(f"baseline_seconds={baseline_seconds:.6f}")
    print(f"prevalidated_seconds={prevalidated_seconds:.6f}")
    print(f"materialized_seconds={materialized_seconds:.6f}")
    print(f"optimized_seconds={optimized_seconds:.6f}")
    print(f"speedup={baseline_seconds / optimized_seconds:.4f}")
    print(f"json={args.output.resolve()}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
