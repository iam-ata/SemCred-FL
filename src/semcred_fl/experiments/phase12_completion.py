"""Fail-closed audit and final reporting for a measured Phase 12 clean matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from semcred_fl.data.contracts import DataContractError
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.evaluation.statistics import (
    PairedComparisonInput,
    analyze_comparison_family,
)
from semcred_fl.experiments.phase12_clean_matrix import (
    CONFIRMATORY_SEEDS,
    DATASETS,
    METHODS,
    PROFILES,
)
from semcred_fl.identity import canonical_json, run_id

_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "worst_class_recall",
    "macro_false_positive_rate",
)


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _write_immutable(path: Path, value: object) -> None:
    encoded = (canonical_json(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise DataContractError(f"Existing final artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(encoded)
    partial.replace(path)


def _verify_content_identity(
    value: Mapping[str, object],
    *,
    identity_key: str,
    ignored_keys: Iterable[str],
    description: str,
) -> None:
    stable = {key: item for key, item in value.items() if key not in set(ignored_keys)}
    if value.get(identity_key) != run_id(stable, length=64):
        raise DataContractError(f"{description} content identity is invalid")


def _reject_forbidden_true(value: object, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in {"test_partition_accessed", "performance_claims_allowed"} and item is True:
                raise DataContractError(f"Forbidden scientific gate is true at {child}")
            _reject_forbidden_true(item, path=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_true(item, path=f"{path}[{index}]")


def _resolved_controls(resolved: Mapping[str, object]) -> tuple[int, int, int, int]:
    overrides = resolved.get("config_overrides")
    if not isinstance(overrides, dict):
        raise DataContractError("Resolved matrix config_overrides is malformed")

    def integer(name: str, default: int) -> int:
        value = overrides.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DataContractError(f"Resolved matrix {name} is malformed")
        return cast(int, value)

    return (
        integer("total_rounds", 100),
        integer("local_epochs", 3),
        integer("batch_size", 128),
        integer("clients_per_round", 20),
    )


def _registry_entries(resolved: Mapping[str, object]) -> tuple[tuple[str, str, int], ...]:
    raw = resolved.get("registry")
    if not isinstance(raw, list) or not raw:
        raise DataContractError("Resolved matrix registry is empty or malformed")
    entries: list[tuple[str, str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise DataContractError("Resolved matrix registry entry is malformed")
        dataset, profile, seed = item.get("dataset"), item.get("profile"), item.get("seed")
        if not isinstance(dataset, str) or not isinstance(profile, str):
            raise DataContractError("Resolved matrix dataset or profile is malformed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DataContractError("Resolved matrix seed is malformed")
        entries.append((dataset, profile, seed))
    if len(entries) != len(set(entries)):
        raise DataContractError("Resolved matrix registry contains duplicate conditions")
    return tuple(entries)


def _read_metric_records(path: Path) -> tuple[RunMetricRecord, ...]:
    records: list[RunMetricRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataContractError(f"Cannot read condition metrics: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("record is not an object")
            record = RunMetricRecord.from_dict(value)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise DataContractError(
                f"Malformed condition metric at {path}:{line_number}: {error}"
            ) from error
        if record.test_partition_accessed or record.performance_claims_allowed:
            raise DataContractError("Development metric record violates scientific gates")
        records.append(record)
    return tuple(records)


def audit_phase12_clean_campaign(
    campaign_directory: Path,
    *,
    require_confirmatory_registry: bool = True,
) -> tuple[dict[str, object], tuple[RunMetricRecord, ...]]:
    """Verify a completed measured campaign and return its audit plus metric records."""

    campaign = campaign_directory.resolve()
    resolved = _load_json(campaign / "resolved_matrix.json", "resolved Phase 12 matrix")
    matrix = _load_json(campaign / "matrix_report.json", "completed Phase 12 matrix")
    matrix_input_id = resolved.get("matrix_input_id")
    if not isinstance(matrix_input_id, str) or len(matrix_input_id) != 64:
        raise DataContractError("Resolved matrix input identity is malformed")
    resolved_stable = {key: item for key, item in resolved.items() if key != "matrix_input_id"}
    if run_id(resolved_stable, length=64) != matrix_input_id:
        raise DataContractError("Resolved matrix input identity is invalid")
    if campaign.name != matrix_input_id or matrix.get("matrix_input_id") != matrix_input_id:
        raise DataContractError("Campaign directory and matrix identity differ")
    if matrix.get("status") != "complete":
        raise DataContractError("Phase 12 matrix is not complete")
    _verify_content_identity(
        matrix,
        identity_key="matrix_result_id",
        ignored_keys=("matrix_result_id", "status", "completed_at_utc"),
        description="Phase 12 matrix report",
    )
    _reject_forbidden_true(resolved)
    _reject_forbidden_true(matrix)

    registry = _registry_entries(resolved)
    methods = resolved.get("methods")
    if not isinstance(methods, list) or not methods or any(not isinstance(x, str) for x in methods):
        raise DataContractError("Resolved matrix method list is malformed")
    method_names = tuple(cast(list[str], methods))
    rounds, local_epochs, batch_size, clients_per_round = _resolved_controls(resolved)
    if require_confirmatory_registry:
        expected = {
            (dataset, profile, seed)
            for dataset in DATASETS
            for profile in PROFILES
            for seed in CONFIRMATORY_SEEDS
        }
        if set(registry) != expected:
            raise DataContractError("Campaign does not contain the full confirmatory registry")
        if method_names != METHODS:
            raise DataContractError("Campaign does not contain all five standardized methods")
        if (rounds, local_epochs, batch_size, clients_per_round) != (100, 3, 128, 20):
            raise DataContractError("Campaign does not use the frozen full-budget controls")

    result_entries = matrix.get("condition_results")
    if not isinstance(result_entries, list) or len(result_entries) != len(registry):
        raise DataContractError("Matrix condition result inventory is incomplete")
    result_keys = {
        (item.get("dataset"), item.get("profile"), item.get("seed"))
        for item in result_entries
        if isinstance(item, dict)
    }
    if result_keys != set(registry):
        raise DataContractError("Matrix result inventory differs from the resolved registry")

    all_records: list[RunMetricRecord] = []
    condition_ids: list[str] = []
    method_result_ids: list[str] = []
    for dataset, profile, seed in registry:
        condition_directory = campaign / "conditions" / dataset / profile / f"seed_{seed}"
        condition = _load_json(
            condition_directory / "condition_report.json",
            f"condition {dataset}/{profile}/seed_{seed}",
        )
        if condition.get("status") != "complete":
            raise DataContractError("Campaign contains an incomplete condition")
        _verify_content_identity(
            condition,
            identity_key="condition_result_id",
            ignored_keys=("condition_result_id", "status", "reporting", "completed_at_utc"),
            description=f"condition {dataset}/{profile}/seed_{seed}",
        )
        _reject_forbidden_true(condition)
        condition_result_id = condition.get("condition_result_id")
        if not isinstance(condition_result_id, str):
            raise DataContractError("Condition result identity is malformed")
        condition_ids.append(condition_result_id)
        raw_method_results = condition.get("method_results")
        if not isinstance(raw_method_results, list):
            raise DataContractError("Condition method inventory is malformed")
        found_methods = tuple(
            item.get("method") for item in raw_method_results if isinstance(item, dict)
        )
        if found_methods != method_names:
            raise DataContractError("Condition method inventory differs from the matrix")
        inventory_by_method = {
            cast(str, item["method"]): item
            for item in raw_method_results
            if isinstance(item, dict) and isinstance(item.get("method"), str)
        }
        for method in method_names:
            method_directory = condition_directory / "methods" / method
            report = _load_json(method_directory / "method_report.json", f"{method} report")
            if report.get("status") != "complete" or report.get("rounds_completed") != rounds:
                raise DataContractError(f"{method} did not complete the expected rounds")
            _verify_content_identity(
                report,
                identity_key="method_result_id",
                ignored_keys=("method_result_id", "status", "completed_at_utc"),
                description=f"{dataset}/{profile}/seed_{seed}/{method}",
            )
            _reject_forbidden_true(report)
            method_result_id = report.get("method_result_id")
            if not isinstance(method_result_id, str):
                raise DataContractError("Method result identity is malformed")
            inventory = inventory_by_method[method]
            if (
                inventory.get("method_run_id") != report.get("method_run_id")
                or inventory.get("method_result_id") != method_result_id
                or inventory.get("final_parameters_sha256") != report.get("final_parameters_sha256")
                or inventory.get("validation_metrics") != report.get("validation_metrics")
            ):
                raise DataContractError("Condition and method result inventories differ")
            method_result_ids.append(method_result_id)
            previous_round_id: str | None = None
            for round_id in range(rounds):
                round_report = _load_json(
                    method_directory / "rounds" / f"round-{round_id:04d}.json",
                    f"{method} round {round_id}",
                )
                if (
                    round_report.get("round_id") != round_id
                    or round_report.get("previous_round_record_id") != previous_round_id
                ):
                    raise DataContractError(f"{method} round hash chain is invalid")
                _verify_content_identity(
                    round_report,
                    identity_key="round_record_id",
                    ignored_keys=("round_record_id",),
                    description=f"{dataset}/{profile}/seed_{seed}/{method}/round_{round_id}",
                )
                previous_round_id = cast(str, round_report["round_record_id"])
            round_files = tuple((method_directory / "rounds").glob("round-*.json"))
            if (
                len(round_files) != rounds
                or report.get("last_round_record_id") != previous_round_id
            ):
                raise DataContractError(f"{method} round audit count or final link differs")
            pointer = _load_json(method_directory / "latest_checkpoint.json", "latest checkpoint")
            if pointer.get("next_round") != rounds:
                raise DataContractError(f"{method} checkpoint cursor is incomplete")
        records = _read_metric_records(condition_directory / "run_metrics.jsonl")
        if len(records) != len(method_names) * len(_METRICS):
            raise DataContractError("Condition metric inventory is incomplete")
        all_records.extend(records)

    stable: dict[str, object] = {
        "schema": "semcred_phase12_clean_campaign_audit_v1",
        "matrix_input_id": matrix_input_id,
        "matrix_result_id": matrix["matrix_result_id"],
        "registry": [
            {"dataset": dataset, "profile": profile, "seed": seed}
            for dataset, profile, seed in registry
        ],
        "methods": list(method_names),
        "controls": {
            "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "clients_per_round": clients_per_round,
        },
        "condition_result_ids": condition_ids,
        "method_result_ids": method_result_ids,
        "metric_record_count": len(all_records),
        "gates": {
            "registry_complete": True,
            "content_identities_verified": True,
            "round_and_checkpoint_inventories_complete": True,
            "metric_inventory_complete": True,
            "validation_partition_accessed": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    audit = {
        **stable,
        "audit_id": run_id(stable, length=64),
        "status": "passed",
    }
    return audit, tuple(all_records)


def _paired_statistics(records: Sequence[RunMetricRecord]) -> dict[str, object]:
    values: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for record in records:
        key = (record.dataset, record.condition_id, record.metric_name, record.method)
        if record.seed in values[key]:
            raise DataContractError("Duplicate seed metric in final campaign")
        values[key][record.seed] = record.value

    families: list[dict[str, object]] = []
    datasets_conditions_metrics = sorted(
        {(dataset, condition, metric) for dataset, condition, metric, _ in values}
    )
    for dataset, condition, metric in datasets_conditions_metrics:
        semcred = values.get((dataset, condition, metric, "semcred_fl"))
        if semcred is None:
            raise DataContractError("Final campaign has no SemCred-FL paired observations")
        family_id = f"{dataset}/{condition}/{metric}/semcred_vs_standard"
        comparisons: list[PairedComparisonInput] = []
        for baseline in METHODS[:-1]:
            other = values.get((dataset, condition, metric, baseline))
            if other is None or set(other) != set(semcred):
                raise DataContractError("Final campaign paired seed sets differ")
            comparisons.append(
                PairedComparisonInput(
                    comparison_id=f"semcred_fl_vs_{baseline}",
                    comparison_family_id=family_id,
                    left_by_pair={str(seed): semcred[seed] for seed in sorted(semcred)},
                    right_by_pair={str(seed): other[seed] for seed in sorted(other)},
                )
            )
        corrected = analyze_comparison_family(comparisons)
        families.append(
            {
                "comparison_family_id": family_id,
                "dataset": dataset,
                "condition_id": condition,
                "metric_name": metric,
                "comparisons": [
                    {
                        "comparison_id": item.comparison_id,
                        "holm_adjusted_p_value": item.holm_adjusted_p_value,
                        "wilcoxon": asdict(item.result),
                    }
                    for item in corrected
                ],
            }
        )
    stable: dict[str, object] = {
        "schema": "semcred_phase12_paired_statistics_v1",
        "families": families,
        "test_partition_accessed": False,
        "performance_claims_allowed": False,
    }
    return {**stable, "statistics_id": run_id(stable, length=64)}


def finalize_phase12_clean_campaign(
    campaign_directory: Path,
    *,
    output_directory: Path | None = None,
    require_confirmatory_registry: bool = True,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260904,
) -> dict[str, object]:
    """Audit and regenerate final clean-matrix reports outside the manuscript."""

    audit, records = audit_phase12_clean_campaign(
        campaign_directory,
        require_confirmatory_registry=require_confirmatory_registry,
    )
    output = (output_directory or campaign_directory / "final_reporting").resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(output / "campaign_audit.json", audit)

    metric_text = "".join(canonical_json(asdict(record)) + "\n" for record in records)
    metrics_path = output / "run_metrics.jsonl"
    if metrics_path.exists() and metrics_path.read_text(encoding="utf-8") != metric_text:
        raise DataContractError("Existing consolidated metrics differ")
    if not metrics_path.exists():
        metrics_path.write_text(metric_text, encoding="utf-8")

    bundle = write_summary_bundle(
        records,
        output,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    rendered: dict[str, str] = {}
    for metric in _METRICS:
        artifact = render_summary_artifacts(
            output / "metric_summaries.json",
            output,
            metric_name=metric,
        )
        rendered[metric] = artifact["artifact_id"]
    statistics = _paired_statistics(records)
    _write_immutable(output / "paired_statistics.json", statistics)

    stable: dict[str, object] = {
        "schema": "semcred_phase12_clean_final_manifest_v1",
        "audit_id": audit["audit_id"],
        "matrix_input_id": audit["matrix_input_id"],
        "matrix_result_id": audit["matrix_result_id"],
        "summary_bundle_id": bundle["summary_bundle_id"],
        "statistics_id": statistics["statistics_id"],
        "rendered_artifact_ids": rendered,
        "metric_record_count": len(records),
        "status": "clean_matrix_reporting_complete",
        "scope": "validation_only_clean_utility",
        "gates": {
            "manuscript_modified": False,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
            "final_test_opening_allowed": False,
        },
    }
    manifest = {
        **stable,
        "finalization_id": run_id(stable, length=64),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = output / "final_manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path, "existing Phase 12 final manifest")
        existing_stable = {
            key: item
            for key, item in existing.items()
            if key not in {"finalization_id", "created_at_utc"}
        }
        if existing_stable != stable or existing.get("finalization_id") != run_id(
            stable, length=64
        ):
            raise DataContractError("Existing Phase 12 final manifest differs")
        return existing
    _write_immutable(manifest_path, manifest)
    return manifest
