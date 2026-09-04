"""Deterministic centralized detector sanity baseline for Phase 4."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from semcred_fl.data.training_corpus import (
    PreprocessedCorpus,
    iter_partition_batches,
    load_preprocessed_corpus,
    load_training_class_counts,
)
from semcred_fl.identity import run_id
from semcred_fl.modeling.checkpoint import save_checkpoint
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP
from semcred_fl.modeling.losses import (
    ClassWeightPolicy,
    class_weighted_cross_entropy,
    class_weights_from_counts,
)
from semcred_fl.modeling.metrics import ClassificationMetrics, ConfusionMatrixAccumulator
from semcred_fl.seeding import derive_seed

_SCHEMA_VERSION: Final = 2


@dataclass(frozen=True, slots=True)
class CentralizedSanityConfig:
    """Frozen controls for one centralized validation-only sanity run."""

    preprocessing_report: Path
    client_partition_manifest: Path
    output_directory: Path
    seed: int = 20260816
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    maximum_class_weight: float = 10.0
    train_record_limit: int | None = 100_000
    validation_record_limit: int | None = 50_000
    device: str = "auto"
    progress_every_batches: int = 100

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.maximum_class_weight < 1.0:
            raise ValueError("learning_rate must be positive and class-weight cap at least one")
        if self.train_record_limit is not None and self.train_record_limit < 2:
            raise ValueError("train_record_limit must be at least two")
        if self.validation_record_limit is not None and self.validation_record_limit < 2:
            raise ValueError("validation_record_limit must be at least two")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.progress_every_batches < 1:
            raise ValueError("progress_every_batches must be positive")


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_reproducibility(seed: int) -> None:
    np = importlib.import_module("numpy")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _config_record(
    config: CentralizedSanityConfig,
    corpus: PreprocessedCorpus,
    detector: DetectorConfig,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "phase": "phase_4_centralized_sanity",
        "dataset": corpus.dataset,
        "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
        "preprocessor_id": corpus.preprocessor_id,
        "split_id": corpus.split_id,
        "anchor_id": corpus.anchor_id,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "adam",
        "learning_rate": config.learning_rate,
        "class_weight_strategy": "capped_inverse_frequency_observed_classes",
        "maximum_class_weight": config.maximum_class_weight,
        "train_record_limit": config.train_record_limit,
        "validation_record_limit": config.validation_record_limit,
        "bounded_sampling_order": "phase3c_manifest_prefix",
        "progress_every_batches": config.progress_every_batches,
        "detector": detector.to_dict(),
        "test_partition_accessed": False,
    }


def _write_json(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _evaluate(
    model: IntrusionMLP,
    corpus: PreprocessedCorpus,
    class_weights: Tensor,
    config: CentralizedSanityConfig,
    device: torch.device,
) -> tuple[float, ClassificationMetrics]:
    model.eval()
    total_loss = 0.0
    records = 0
    accumulator = ConfusionMatrixAccumulator(corpus.num_classes)
    with torch.no_grad():
        for batch in iter_partition_batches(
            corpus,
            "validation",
            batch_size=config.batch_size,
            record_limit=config.validation_record_limit,
        ):
            features = batch.features.to(device)
            targets = batch.targets.to(device)
            logits = model(features)
            loss = class_weighted_cross_entropy(logits, targets, class_weights)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite validation loss")
            total_loss += float(loss.item()) * batch.records
            records += batch.records
            predictions = torch.argmax(logits, dim=1)
            accumulator.update(targets.detach().cpu().tolist(), predictions.detach().cpu().tolist())
    if records < 1:
        raise RuntimeError("Validation stream produced no records")
    return total_loss / records, accumulator.compute()


def run_centralized_sanity(config: CentralizedSanityConfig) -> dict[str, object]:
    """Train the Phase 4 centralized sanity model without touching the test set."""

    corpus = load_preprocessed_corpus(config.preprocessing_report.resolve())
    class_counts = load_training_class_counts(config.client_partition_manifest.resolve(), corpus)
    detector_config = DetectorConfig(corpus.input_dimension, corpus.num_classes)
    stable = _config_record(config, corpus, detector_config)
    sanity_id = run_id(stable, length=64)
    run_directory = config.output_directory.resolve() / sanity_id
    final_manifest = run_directory / "centralized_sanity.json"
    if final_manifest.exists():
        return json.loads(final_manifest.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_directory / "resolved_config.json", {"sanity_id": sanity_id, **stable})

    _set_reproducibility(config.seed)
    device = _device(config.device)
    model = IntrusionMLP(detector_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    weight_values = class_weights_from_counts(
        class_counts, ClassWeightPolicy(config.maximum_class_weight)
    )
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    history: list[dict[str, object]] = []
    log_path = run_directory / "epochs.jsonl"
    best_macro_f1 = -math.inf
    best_checkpoint: dict[str, object] | None = None
    final_metrics: ClassificationMetrics | None = None
    started = time.perf_counter()

    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_records = 0
            epoch_started = time.perf_counter()
            shuffle_seed = derive_seed(config.seed, f"phase4-centralized-epoch:{epoch}")
            for batch_index, batch in enumerate(
                iter_partition_batches(
                    corpus,
                    "train",
                    batch_size=config.batch_size,
                    record_limit=config.train_record_limit,
                    shuffle_seed=shuffle_seed,
                ),
                start=1,
            ):
                features = batch.features.to(device)
                targets = batch.targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = class_weighted_cross_entropy(logits, targets, class_weights)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite training loss")
                torch.autograd.backward(loss)
                optimizer.step()
                if any(
                    not torch.isfinite(parameter).all().item() for parameter in model.parameters()
                ):
                    raise RuntimeError("Non-finite detector parameter after optimizer step")
                epoch_loss += float(loss.item()) * batch.records
                epoch_records += batch.records
                if batch_index % config.progress_every_batches == 0:
                    print(
                        f"phase4_train epoch={epoch}/{config.epochs} "
                        f"batches={batch_index} records={epoch_records}",
                        flush=True,
                    )
            if epoch_records < 1:
                raise RuntimeError("Training stream produced no records")
            validation_loss, metrics = _evaluate(model, corpus, class_weights, config, device)
            final_metrics = metrics
            epoch_record: dict[str, object] = {
                "epoch": epoch,
                "train_records": epoch_records,
                "train_loss": epoch_loss / epoch_records,
                "validation_loss": validation_loss,
                "validation": metrics.to_dict(),
                "duration_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            print(
                f"phase4_epoch={epoch}/{config.epochs} "
                f"train_loss={epoch_record['train_loss']:.8f} "
                f"validation_macro_f1={metrics.macro_f1:.8f}",
                flush=True,
            )
            log.write(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True) + "\n")
            log.flush()
            checkpoint_path = run_directory / "checkpoints" / f"epoch-{epoch:03d}.pt"
            checkpoint_hash = save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                optimizer=optimizer,
                metadata={
                    "sanity_id": sanity_id,
                    "dataset": corpus.dataset,
                    "preprocessed_corpus_id": corpus.preprocessed_corpus_id,
                },
            )
            if metrics.macro_f1 > best_macro_f1:
                best_macro_f1 = metrics.macro_f1
                best_checkpoint = {
                    "path": checkpoint_path.relative_to(run_directory).as_posix(),
                    "sha256": checkpoint_hash,
                    "epoch": epoch,
                }

    if best_checkpoint is None or final_metrics is None:
        raise RuntimeError("No checkpoint was produced")
    report: dict[str, object] = {
        "sanity_id": sanity_id,
        **stable,
        "device": str(device),
        "class_counts": list(class_counts),
        "class_weights": list(weight_values),
        "history": history,
        "best_checkpoint": best_checkpoint,
        "environment": {
            "python_torch": torch.__version__,
            "numpy": importlib.metadata.version("numpy"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
        "duration_seconds": time.perf_counter() - started,
        "gates": {
            "checkpoint_created": True,
            "all_losses_finite": True,
            "all_parameters_finite": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
            "validation_all_classes_observed": all(count > 0 for count in final_metrics.support),
        },
    }
    _write_json(final_manifest, report)
    return report
