"""Run or resume the tiny synthetic Phase 5 federated-engine gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from semcred_fl.data.training_corpus import TrainingBatch  # noqa: E402
from semcred_fl.federation.client import LocalTrainingConfig, train_local_client  # noqa: E402
from semcred_fl.federation.parameters import ParameterManifest  # noqa: E402
from semcred_fl.federation.sampling import generate_client_schedule  # noqa: E402
from semcred_fl.federation.server import FederatedRunConfig, run_federated_simulation  # noqa: E402
from semcred_fl.federation.updates import ClientUpdate  # noqa: E402
from semcred_fl.modeling.detector import DetectorConfig, IntrusionMLP  # noqa: E402
from semcred_fl.seeding import derive_seed  # noqa: E402


class _SyntheticBatches:
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, batch_size: int) -> None:
        self.features = features
        self.targets = targets
        self.batch_size = batch_size

    def __call__(self, epoch: int, seed: int) -> Iterator[TrainingBatch]:
        del epoch
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.targets), generator=generator)
        for start in range(0, len(order), self.batch_size):
            selected = order[start : start + self.batch_size]
            yield TrainingBatch(self.features[selected], self.targets[selected])


class _SyntheticLocalProvider:
    def __init__(self, detector: DetectorConfig, device: str, seed: int) -> None:
        self.detector = detector
        self.local_config = LocalTrainingConfig(
            local_epochs=1,
            learning_rate=0.01,
            maximum_class_weight=10.0,
            seed=seed,
            device=device,
        )
        self.data = self._build_data(seed)

    @staticmethod
    def _build_data(seed: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        class_splits = ((24, 8), (20, 12), (12, 20), (8, 24))
        for client_index, (negative_count, positive_count) in enumerate(class_splits):
            generator = torch.Generator().manual_seed(
                derive_seed(seed, f"phase5-synthetic-data:{client_index}")
            )
            negative = torch.randn(negative_count, 2, generator=generator) * 0.15 - 1.0
            positive = torch.randn(positive_count, 2, generator=generator) * 0.15 + 1.0
            features = torch.cat((negative, positive))
            targets = torch.cat(
                (
                    torch.zeros(negative_count, dtype=torch.long),
                    torch.ones(positive_count, dtype=torch.long),
                )
            )
            result[f"client-{client_index:02d}"] = (features, targets)
        return result

    def __call__(
        self,
        round_id: int,
        selected_client_ids: tuple[str, ...],
        global_parameters: torch.Tensor,
        manifest: ParameterManifest,
    ) -> Sequence[ClientUpdate]:
        updates: list[ClientUpdate] = []
        for client_id in selected_client_ids:
            features, targets = self.data[client_id]
            counts = tuple(int(torch.sum(targets == class_id).item()) for class_id in range(2))
            updates.append(
                train_local_client(
                    detector=self.detector,
                    manifest=manifest,
                    global_parameters=global_parameters,
                    round_id=round_id,
                    client_id=client_id,
                    declared_sample_count=len(targets),
                    class_counts=counts,
                    batch_source=_SyntheticBatches(features, targets, batch_size=8),
                    config=self.local_config,
                )
            )
        return updates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/federation/phase_5/synthetic"),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--aggregation", choices=("fedavg", "coordinate_median"), default="fedavg")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-rounds-this-call", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1 or not 1 <= args.clients_per_round <= 4:
        raise ValueError("rounds or clients-per-round is invalid")
    torch.manual_seed(args.seed)
    detector = DetectorConfig(2, 2, (8,), 0.0)
    model = IntrusionMLP(detector)
    client_ids = tuple(f"client-{index:02d}" for index in range(4))
    schedule = generate_client_schedule(
        client_ids,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        seed=args.seed,
    )
    report = run_federated_simulation(
        initial_model=model,
        schedule=schedule,
        update_provider=_SyntheticLocalProvider(detector, args.device, args.seed),
        config=FederatedRunConfig(
            args.output_directory,
            total_rounds=args.rounds,
            aggregation=args.aggregation,
            seed=args.seed,
        ),
        max_rounds_this_call=args.max_rounds_this_call,
    )
    print(f"run_id={report['run_id']}")
    print(f"status={report['status']}")
    if report["status"] == "interrupted":
        print(f"next_round={report['next_round']}")
    else:
        print(f"rounds_completed={report['rounds_completed']}")
        print(f"final_parameters_sha256={report['final_parameters_sha256']}")
    print("test_partition_accessed=false")
    print("performance_claims_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
