"""Resumable sequential orchestration for Phase 6C attack matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from semcred_fl.attacks.execution import Phase6CAttackRequest, run_phase6c_attack
from semcred_fl.attacks.execution_contracts import (
    AttackRunSpec,
    Phase6CExecutionConfig,
    phase6c_execution_config,
)
from semcred_fl.data.contracts import DataContractError
from semcred_fl.identity import run_id


@dataclass(frozen=True, slots=True)
class Phase6CMatrixRequest:
    """Inputs for one sequential, target-round-bounded Phase 6C matrix."""

    preprocessing_report: Path
    client_partition_manifest: Path
    resolution_path: Path
    output_root: Path
    cache_root: Path
    norm_reference_root: Path
    dataset: str
    profiles: tuple[str, ...]
    malicious_ratios: tuple[float, ...]
    seeds: tuple[int, ...]
    matrix: str = "all_types"
    target_round: int = 1
    device: str = "auto"
    cache_scan_batch_size: int = 100_000
    config: Phase6CExecutionConfig = phase6c_execution_config()

    def __post_init__(self) -> None:
        if self.matrix not in {"all_types", "full_grid"}:
            raise ValueError("Phase 6C matrix must be all_types or full_grid")
        if not 1 <= self.target_round <= self.config.total_rounds:
            raise ValueError("Phase 6C matrix target round is outside the frozen run")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Phase 6C matrix device is invalid")
        if self.cache_scan_batch_size < 1:
            raise ValueError("Phase 6C matrix cache scan batch size must be positive")
        if (
            not self.profiles
            or len(set(self.profiles)) != len(self.profiles)
            or any(profile not in self.config.profiles for profile in self.profiles)
        ):
            raise ValueError("Phase 6C matrix profile selection is invalid")
        if (
            not self.malicious_ratios
            or len(set(self.malicious_ratios)) != len(self.malicious_ratios)
            or any(
                ratio not in self.config.malicious_client_ratios for ratio in self.malicious_ratios
            )
        ):
            raise ValueError("Phase 6C matrix malicious-ratio selection is invalid")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(seed < 0 for seed in self.seeds)
        ):
            raise ValueError("Phase 6C matrix seed selection is invalid")


def _attack_specs(
    *,
    dataset: str,
    profile: str,
    malicious_ratio: float,
    seed: int,
    matrix: str,
    config: Phase6CExecutionConfig,
) -> tuple[AttackRunSpec, ...]:
    def make(
        attack_name: str,
        *,
        poison_fraction: float = 0.0,
        strength: float | None = None,
        scale: float | None = None,
    ) -> AttackRunSpec:
        return AttackRunSpec(
            dataset=dataset,
            profile=profile,
            attack_name=attack_name,
            malicious_ratio=malicious_ratio,
            seed=seed,
            poison_fraction=poison_fraction,
            strength=strength,
            scale=scale,
        )

    if matrix == "all_types":
        fraction = config.poison_fractions[0]
        return (
            make("clean"),
            make("label_flip", poison_fraction=0.0),
            make("label_flip", poison_fraction=fraction),
            make(
                "clean_label",
                poison_fraction=fraction,
                strength=config.clean_label_strengths[0],
            ),
            make("backdoor", poison_fraction=fraction),
            make("scaling", scale=config.scaling_factors[0]),
            make("sign_reversal", scale=config.reversal_factors[0]),
            make("model_replacement", poison_fraction=fraction),
            make("adaptive_norm", poison_fraction=fraction),
        )

    specs: list[AttackRunSpec] = [
        make("clean"),
        make("label_flip", poison_fraction=0.0),
    ]
    specs.extend(
        make("label_flip", poison_fraction=fraction) for fraction in config.poison_fractions
    )
    specs.extend(
        make(
            "clean_label",
            poison_fraction=fraction,
            strength=strength,
        )
        for fraction in config.poison_fractions
        for strength in config.clean_label_strengths
    )
    specs.extend(make("backdoor", poison_fraction=fraction) for fraction in config.poison_fractions)
    specs.extend(make("scaling", scale=scale) for scale in config.scaling_factors)
    specs.extend(make("sign_reversal", scale=scale) for scale in config.reversal_factors)
    specs.extend(
        make("model_replacement", poison_fraction=fraction) for fraction in config.poison_fractions
    )
    specs.extend(
        make("adaptive_norm", poison_fraction=fraction) for fraction in config.poison_fractions
    )
    return tuple(specs)


def phase6c_matrix_specs(
    *,
    dataset: str,
    profile: str,
    malicious_ratio: float,
    seed: int,
    matrix: str,
    config: Phase6CExecutionConfig | None = None,
) -> tuple[AttackRunSpec, ...]:
    """Return the ordered, validated conditions for one matrix slice."""

    if matrix not in {"all_types", "full_grid"}:
        raise ValueError("Phase 6C matrix must be all_types or full_grid")
    resolved_config = config or phase6c_execution_config()
    specs = _attack_specs(
        dataset=dataset,
        profile=profile,
        malicious_ratio=malicious_ratio,
        seed=seed,
        matrix=matrix,
        config=resolved_config,
    )
    for spec in specs:
        spec.validate(resolved_config)
    return specs


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "_")


def condition_relative_directory(spec: AttackRunSpec) -> Path:
    """Return the stable human-readable directory for one attack condition."""

    path = (
        Path(spec.dataset)
        / spec.profile
        / spec.attack_name
        / (f"mr_{_number_token(spec.malicious_ratio)}")
    )
    if spec.attack_name in {
        "label_flip",
        "clean_label",
        "backdoor",
        "model_replacement",
        "adaptive_norm",
    }:
        path /= f"pf_{_number_token(spec.poison_fraction)}"
    if spec.strength is not None:
        path /= f"strength_{_number_token(spec.strength)}"
    if spec.scale is not None:
        path /= f"scale_{_number_token(spec.scale)}"
    return path / f"seed_{spec.seed}"


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def completed_rounds(output_directory: Path, total_rounds: int) -> int:
    """Return a condition's validated persisted round cursor without advancing it."""

    final_path = output_directory / "phase6c_attack_run.json"
    if final_path.is_file():
        completed = _load_json(final_path, "completed Phase 6C attack run")
        if completed.get("attack_run_id") is None:
            raise DataContractError("Completed Phase 6C run has no attack identity")
        return total_rounds
    latest_files = sorted((output_directory / "federation").glob("*/latest_checkpoint.json"))
    if not latest_files:
        return 0
    if len(latest_files) != 1:
        raise DataContractError("Phase 6C condition contains multiple federated runs")
    latest = _load_json(latest_files[0], "Phase 6C latest checkpoint")
    next_round = latest.get("next_round")
    if not isinstance(next_round, int) or not 0 <= next_round <= total_rounds:
        raise DataContractError("Phase 6C persisted round cursor is invalid")
    return next_round


def _matrix_plan(request: Phase6CMatrixRequest) -> tuple[AttackRunSpec, ...]:
    specs = tuple(
        spec
        for profile in request.profiles
        for ratio in request.malicious_ratios
        for seed in request.seeds
        for spec in phase6c_matrix_specs(
            dataset=request.dataset,
            profile=profile,
            malicious_ratio=ratio,
            seed=seed,
            matrix=request.matrix,
            config=request.config,
        )
    )
    for spec in specs:
        spec.validate(request.config)
    if len({spec.attack_id(request.config) for spec in specs}) != len(specs):
        raise DataContractError("Phase 6C matrix contains duplicate conditions")
    return specs


def inspect_phase6c_matrix(request: Phase6CMatrixRequest) -> dict[str, object]:
    """Describe current persisted progress without creating or advancing a run."""

    conditions: list[dict[str, object]] = []
    for index, spec in enumerate(_matrix_plan(request), start=1):
        relative = condition_relative_directory(spec)
        rounds = completed_rounds(request.output_root / relative, request.config.total_rounds)
        conditions.append(
            {
                "position": index,
                "attack_id": spec.attack_id(request.config),
                "attack_name": spec.attack_name,
                "poison_fraction": spec.poison_fraction,
                "strength": spec.strength,
                "scale": spec.scale,
                "profile": spec.profile,
                "malicious_ratio": spec.malicious_ratio,
                "seed": spec.seed,
                "path": relative.as_posix(),
                "completed_rounds": rounds,
                "at_target": rounds >= request.target_round,
            }
        )
    at_target = sum(record["at_target"] is True for record in conditions)
    return {
        "status": "plan_only",
        "dataset": request.dataset,
        "matrix": request.matrix,
        "target_round": request.target_round,
        "conditions_total": len(conditions),
        "conditions_at_target": at_target,
        "conditions_pending": len(conditions) - at_target,
        "conditions": conditions,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }


def run_phase6c_matrix(request: Phase6CMatrixRequest) -> dict[str, object]:
    """Advance every selected condition sequentially to one absolute round target."""

    specs = _matrix_plan(request)
    stable_plan: dict[str, object] = {
        "schema_version": 1,
        "phase": "6C-attack-matrix",
        "phase6c_contract_id": request.config.contract_id,
        "dataset": request.dataset,
        "profiles": list(request.profiles),
        "malicious_ratios": list(request.malicious_ratios),
        "seeds": list(request.seeds),
        "matrix": request.matrix,
        "target_round": request.target_round,
        "condition_ids": [spec.attack_id(request.config) for spec in specs],
        "condition_directories": [condition_relative_directory(spec).as_posix() for spec in specs],
        "test_partition_accessed": False,
        "validation_partition_accessed": False,
        "anchor_records_accessed": False,
        "performance_claims_allowed": False,
    }
    matrix_id = run_id(stable_plan, length=64)
    matrix_directory = request.output_root / "matrices" / matrix_id
    plan_path = matrix_directory / "phase6c_matrix_plan.json"
    if plan_path.exists():
        completed_plan = _load_json(plan_path, "Phase 6C matrix plan")
        plan_stable = {
            key: value
            for key, value in completed_plan.items()
            if key not in {"matrix_id", "created_at_utc"}
        }
        if completed_plan.get("matrix_id") != matrix_id or plan_stable != stable_plan:
            raise DataContractError("Persisted Phase 6C matrix plan differs")
    else:
        _write_json(
            plan_path,
            {
                "matrix_id": matrix_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                **stable_plan,
            },
        )

    progress_conditions: list[dict[str, object]] = []
    completed_conditions: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        relative = condition_relative_directory(spec)
        output_directory = request.output_root / relative
        before = completed_rounds(output_directory, request.config.total_rounds)
        print(
            f"phase6c_matrix={index}/{len(specs)} attack={spec.attack_name} "
            f"profile={spec.profile} ratio={spec.malicious_ratio:g} seed={spec.seed} "
            f"rounds={before}/{request.target_round}",
            flush=True,
        )
        report: dict[str, object] | None = None
        if before < request.target_round:
            norm_reference = (
                request.norm_reference_root
                / spec.dataset
                / spec.profile
                / "benign_norm_reference.json"
                if spec.attack_name == "adaptive_norm"
                else None
            )
            cache_directory = request.cache_root / spec.dataset / spec.profile
            report = run_phase6c_attack(
                Phase6CAttackRequest(
                    preprocessing_report=request.preprocessing_report,
                    client_partition_manifest=request.client_partition_manifest,
                    resolution_path=request.resolution_path,
                    norm_reference_path=norm_reference,
                    output_directory=output_directory,
                    cache_directory=cache_directory,
                    cache_work_database=cache_directory / "client_cache.sqlite3",
                    spec=spec,
                    device=request.device,
                    cache_scan_batch_size=request.cache_scan_batch_size,
                    max_rounds_this_call=request.target_round - before,
                    config=request.config,
                )
            )
        after = completed_rounds(output_directory, request.config.total_rounds)
        if after < request.target_round:
            raise DataContractError("Phase 6C condition stopped before the matrix target")
        progress_conditions.append(
            {
                "position": index,
                "attack_id": spec.attack_id(request.config),
                "attack_name": spec.attack_name,
                "profile": spec.profile,
                "malicious_ratio": spec.malicious_ratio,
                "seed": spec.seed,
                "poison_fraction": spec.poison_fraction,
                "strength": spec.strength,
                "scale": spec.scale,
                "path": relative.as_posix(),
                "rounds_before_call": before,
                "rounds_after_call": after,
                "skipped_at_target": before >= request.target_round,
                "federated_run_id": report.get("federated_run_id") if report else None,
            }
        )
        completed_conditions.append(
            {
                "position": index,
                "attack_id": spec.attack_id(request.config),
                "attack_name": spec.attack_name,
                "profile": spec.profile,
                "malicious_ratio": spec.malicious_ratio,
                "seed": spec.seed,
                "poison_fraction": spec.poison_fraction,
                "strength": spec.strength,
                "scale": spec.scale,
                "path": relative.as_posix(),
                "rounds_at_target": after,
            }
        )
        _write_json(
            matrix_directory / "phase6c_matrix_progress.json",
            {
                "matrix_id": matrix_id,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "conditions_total": len(specs),
                "conditions_at_target": len(progress_conditions),
                "conditions": progress_conditions,
                "test_partition_accessed": False,
                "performance_claims_allowed": False,
            },
        )

    stable_result: dict[str, object] = {
        **stable_plan,
        "matrix_id": matrix_id,
        "conditions_total": len(specs),
        "conditions_at_target": len(completed_conditions),
        "conditions": completed_conditions,
        "status": "target_reached",
    }
    result: dict[str, object] = {
        "matrix_run_id": run_id(stable_result, length=64),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        **stable_result,
    }
    result_path = matrix_directory / "phase6c_matrix_run.json"
    if result_path.exists():
        completed_result = _load_json(result_path, "completed Phase 6C matrix run")
        completed_stable = {
            key: value
            for key, value in completed_result.items()
            if key not in {"matrix_run_id", "completed_at_utc"}
        }
        if (
            completed_result.get("matrix_run_id") != result["matrix_run_id"]
            or completed_stable != stable_result
        ):
            raise DataContractError("Completed Phase 6C matrix result differs")
        return completed_result
    _write_json(result_path, result)
    return result
