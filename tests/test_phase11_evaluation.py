"""Known fixtures and deterministic artifact gates for Phase 11."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from semcred_fl.evaluation import (
    CommunicationRecord,
    OfflineLLMCost,
    OnlineTiming,
    PairedComparisonInput,
    analyze_comparison_family,
    describe,
    eligibility_metrics,
    holm_adjust,
    malicious_client_ranking,
    paired_wilcoxon_exact,
    rate,
    robustness_metrics,
    threshold_crossing_delay,
    trust_separation,
    trust_volatility,
)
from semcred_fl.evaluation.reporting import (
    RunMetricRecord,
    aggregate_run_metrics,
    load_summary_bundle,
    read_metric_jsonl,
    render_summary_artifacts,
    write_summary_bundle,
)
from semcred_fl.identity import run_id


def test_rate_preserves_counts_and_does_not_replace_undefined_with_zero() -> None:
    defined = rate(3, 4, empty_reason="unused")
    assert defined.value == 0.75
    assert defined.to_dict()["denominator"] == 4
    empty = rate(0, 0, empty_reason="no eligible records")
    assert empty.value is None and empty.undefined_reason == "no eligible records"


@pytest.mark.parametrize("numerator,denominator", [(-1, 2), (3, 2), (1, 0)])
def test_rate_rejects_impossible_counts(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError):
        rate(numerator, denominator, empty_reason="empty")


def test_rate_record_rejects_inconsistent_defined_and_undefined_states() -> None:
    from semcred_fl.evaluation import RateMetric

    with pytest.raises(ValueError, match="undefined reason"):
        RateMetric(0, 0, 0.0)
    with pytest.raises(ValueError, match="must equal"):
        RateMetric(1, 2, 0.4)


def test_robustness_metrics_match_manuscript_equations() -> None:
    result = robustness_metrics(
        0.8,
        0.6,
        attack_successes=3,
        attack_eligible_records=5,
        backdoor_successes=1,
        backdoor_triggered_source_records=4,
    )
    assert result.absolute_degradation == pytest.approx(0.2)
    assert result.relative_degradation == pytest.approx(0.25)
    assert result.attack_success_rate.value == 0.6
    assert result.backdoor_success_rate.value == 0.25


@pytest.mark.parametrize(
    "clean,attacked,epsilon",
    [(float("nan"), 0.5, 1e-12), (0.5, 1.2, 1e-12), (0.5, 0.4, 0.0)],
)
def test_robustness_metrics_reject_invalid_inputs(
    clean: float, attacked: float, epsilon: float
) -> None:
    with pytest.raises(ValueError):
        robustness_metrics(
            clean,
            attacked,
            attack_successes=0,
            attack_eligible_records=1,
            backdoor_successes=0,
            backdoor_triggered_source_records=1,
            epsilon=epsilon,
        )


def test_malicious_client_ranking_perfect_and_reversed_fixtures() -> None:
    labels = [False, False, True, True]
    perfect = malicious_client_ranking(labels, [0.1, 0.2, 0.8, 0.9])
    reversed_result = malicious_client_ranking(labels, [0.9, 0.8, 0.2, 0.1])
    assert perfect.auroc == 1.0 and perfect.auprc == 1.0
    assert reversed_result.auroc == 0.0
    assert reversed_result.auprc == pytest.approx(5.0 / 12.0)


def test_malicious_client_ranking_is_tie_order_invariant() -> None:
    first = malicious_client_ranking([True, False, True, False], [0.5, 0.5, 0.2, 0.2])
    second = malicious_client_ranking([False, True, False, True], [0.5, 0.5, 0.2, 0.2])
    assert first == second


def test_malicious_client_ranking_marks_one_class_population_undefined() -> None:
    result = malicious_client_ranking([False, False], [0.2, 0.3])
    assert result.auroc is None and result.auprc is None and result.undefined_reason


def test_ranking_and_eligibility_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="same positive length"):
        malicious_client_ranking([], [])
    with pytest.raises(ValueError, match="finite"):
        malicious_client_ranking([True, False], [float("nan"), 0.1])
    with pytest.raises(ValueError, match="same positive length"):
        eligibility_metrics([True], [])


def test_eligibility_rates_use_participating_identity_only_after_decision() -> None:
    result = eligibility_metrics(
        malicious=[False, False, True, True], eligible=[True, False, True, False]
    )
    assert result.benign_false_rejection.value == 0.5
    assert result.malicious_false_acceptance.value == 0.5


def test_trust_separation_reports_signed_mean_and_median_gaps() -> None:
    result = trust_separation([False, False, True, True], [0.8, 1.0, 0.1, 0.3])
    assert result.mean_gap == pytest.approx(0.7)
    assert result.median_gap == pytest.approx(0.7)


def test_trust_separation_undefined_and_invalid_contracts() -> None:
    assert trust_separation([False], [0.8]).undefined_reason
    with pytest.raises(ValueError, match="same positive length"):
        trust_separation([], [])
    with pytest.raises(ValueError, match="finite"):
        trust_separation([False, True], [1.1, 0.2])


def test_detection_and_recovery_delays_count_participation_events() -> None:
    detection = threshold_crossing_delay([0.8, 0.5, 0.2], [0.4, 0.4, 0.4], crossing="below")
    recovery = threshold_crossing_delay([0.2, 0.35, 0.5], [0.4, 0.4, 0.4], crossing="at_or_above")
    assert detection.delay == 3 and not detection.censored
    assert recovery.delay == 3 and not recovery.censored


def test_threshold_delay_preserves_right_censoring() -> None:
    result = threshold_crossing_delay([0.8, 0.7], [0.4, 0.4], crossing="below")
    assert result.delay is None and result.censored and result.participating_events == 2


def test_threshold_delay_rejects_invalid_contracts_and_marks_empty_window() -> None:
    assert threshold_crossing_delay([], [], crossing="below").undefined_reason
    with pytest.raises(ValueError, match="same length"):
        threshold_crossing_delay([0.5], [], crossing="below")
    with pytest.raises(ValueError, match="crossing"):
        threshold_crossing_delay([0.5], [0.4], crossing="sideways")
    with pytest.raises(ValueError, match="finite"):
        threshold_crossing_delay([float("nan")], [0.4], crossing="below")


def test_trust_volatility_is_rms_over_participation_transitions() -> None:
    value, transitions = trust_volatility([[0.9, 0.7, 0.8], [0.5]])
    assert value == pytest.approx((0.025) ** 0.5)
    assert transitions == 2
    assert trust_volatility([[], [0.5]]) == (None, 0)
    with pytest.raises(ValueError, match="finite"):
        trust_volatility([[0.5, 1.1]])


def test_efficiency_records_separate_online_and_offline_costs() -> None:
    timing = OnlineTiming(1.0, 0.2, 0.3, 0.1, 0.05, 0.05)
    communication = CommunicationRecord(100, 200, 8, 4)
    llm = OfflineLLMCost("model-snapshot", 10, 4, 0.5, None, True)
    assert timing.total_seconds == pytest.approx(1.7)
    assert communication.total_bytes == 312
    assert llm.billed_cost_usd is None


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: OnlineTiming(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        lambda: CommunicationRecord(-1, 0, 0),
        lambda: OfflineLLMCost("", 0, 0, 0.0, None, True),
    ],
)
def test_efficiency_records_reject_invalid_measurements(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_offline_llm_cost_rejects_tokens_latency_and_cost() -> None:
    with pytest.raises(ValueError, match="token"):
        OfflineLLMCost("model", -1, 0, 0.0, None, True)
    with pytest.raises(ValueError, match="latency"):
        OfflineLLMCost("model", 0, 0, -1.0, None, True)
    with pytest.raises(ValueError, match="cost"):
        OfflineLLMCost("model", 0, 0, 0.0, -0.1, True)


def test_descriptive_statistics_match_known_fixture_and_are_deterministic() -> None:
    first = describe([1.0, 2.0, 3.0], bootstrap_resamples=100, bootstrap_seed=7)
    second = describe([1.0, 2.0, 3.0], bootstrap_resamples=100, bootstrap_seed=7)
    assert first == second
    assert first.mean == 2.0 and first.sample_standard_deviation == 1.0
    assert first.median == 2.0
    assert first.bootstrap_lower <= first.mean <= first.bootstrap_upper


@pytest.mark.parametrize(
    "values,resamples,confidence",
    [([], 10, 0.95), ([1.0], 0, 0.95), ([1.0], 10, 1.0)],
)
def test_descriptive_statistics_reject_invalid_contracts(
    values: list[float], resamples: int, confidence: float
) -> None:
    with pytest.raises(ValueError):
        describe(
            values,
            bootstrap_resamples=resamples,
            bootstrap_seed=1,
            confidence_level=confidence,
        )


def test_exact_paired_wilcoxon_known_no_tie_fixture() -> None:
    result = paired_wilcoxon_exact(
        {"s1": 2.0, "s2": 4.0, "s3": 6.0},
        {"s1": 1.0, "s2": 2.0, "s3": 3.0},
    )
    assert result.positive_rank_sum == 6.0 and result.negative_rank_sum == 0.0
    assert result.statistic == 0.0
    assert result.exact_two_sided_p_value == 0.25
    assert result.rank_biserial_correlation == 1.0


def test_exact_paired_wilcoxon_logs_zero_and_tied_absolute_ranks() -> None:
    result = paired_wilcoxon_exact(
        {"a": 2.0, "b": 0.0, "c": 4.0},
        {"a": 1.0, "b": 0.0, "c": 5.0},
    )
    assert result.zero_differences == 1
    assert result.tied_absolute_ranks
    assert result.positive_rank_sum == result.negative_rank_sum == 1.5
    assert result.exact_two_sided_p_value == 1.0


def test_exact_paired_wilcoxon_all_zero_fixture_is_neutral() -> None:
    result = paired_wilcoxon_exact({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
    assert result.nonzero_pairs == 0
    assert result.exact_two_sided_p_value == 1.0
    assert result.rank_biserial_correlation == 0.0


def test_exact_paired_wilcoxon_rejects_unmatched_pair_ids() -> None:
    with pytest.raises(ValueError, match="identical"):
        paired_wilcoxon_exact({"a": 1.0}, {"b": 1.0})


def test_exact_paired_wilcoxon_rejects_nonfinite_difference() -> None:
    with pytest.raises(ValueError, match="finite"):
        paired_wilcoxon_exact({"a": float("inf")}, {"a": 1.0})


def test_holm_adjust_matches_known_family_and_is_monotone() -> None:
    result = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert result == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_holm_and_comparison_inputs_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        holm_adjust({})
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust({"a": 1.1})
    with pytest.raises(ValueError, match="identifiers"):
        PairedComparisonInput("", "family", {"s": 1.0}, {"s": 0.0})
    with pytest.raises(ValueError, match="declared family"):
        analyze_comparison_family(())


def test_declared_comparison_family_applies_one_holm_correction() -> None:
    comparisons = (
        PairedComparisonInput(
            "semcred_vs_a",
            "clean_macro_f1",
            {"s1": 3.0, "s2": 4.0, "s3": 5.0},
            {"s1": 1.0, "s2": 2.0, "s3": 3.0},
        ),
        PairedComparisonInput(
            "semcred_vs_b",
            "clean_macro_f1",
            {"s1": 3.0, "s2": 2.0, "s3": 1.0},
            {"s1": 1.0, "s2": 2.0, "s3": 3.0},
        ),
    )
    result = analyze_comparison_family(comparisons)
    assert [item.comparison_id for item in result] == ["semcred_vs_a", "semcred_vs_b"]
    assert all(item.comparison_family_id == "clean_macro_f1" for item in result)
    assert all(item.holm_adjusted_p_value >= item.result.exact_two_sided_p_value for item in result)


def test_comparison_family_rejects_mixed_or_duplicate_declarations() -> None:
    base = PairedComparisonInput("a", "family-a", {"s": 1.0}, {"s": 0.0})
    mixed = PairedComparisonInput("b", "family-b", {"s": 1.0}, {"s": 0.0})
    duplicate = PairedComparisonInput("a", "family-a", {"s": 2.0}, {"s": 0.0})
    with pytest.raises(ValueError, match="one correction family"):
        analyze_comparison_family((base, mixed))
    with pytest.raises(ValueError, match="unique"):
        analyze_comparison_family((base, duplicate))


def _record(
    run_hex: str,
    seed: int,
    value: float,
    *,
    method: str = "semcred_fl",
    metric: str = "macro_f1",
) -> RunMetricRecord:
    return RunMetricRecord(
        run_id=run_hex * 8,
        experiment_family_id="pilot_clean",
        dataset="edge_iiotset",
        method=method,
        condition_id="iid_clean",
        seed=seed,
        metric_name=metric,
        value=value,
        source_artifact_id=f"artifact-{run_hex}",
        test_partition_accessed=False,
        performance_claims_allowed=False,
    )


def test_run_metric_claim_gate_requires_test_access() -> None:
    with pytest.raises(ValueError, match="final-test"):
        RunMetricRecord(
            run_id="a" * 64,
            experiment_family_id="confirmatory",
            dataset="edge_iiotset",
            method="semcred_fl",
            condition_id="iid_clean",
            seed=1,
            metric_name="macro_f1",
            value=0.8,
            source_artifact_id="artifact-a",
            test_partition_accessed=False,
            performance_claims_allowed=True,
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"run_id": "not-hex"}, "hexadecimal"),
        ({"method": "bad method"}, "unsupported"),
        ({"seed": -1}, "seed"),
        ({"value": float("nan")}, "seed"),
    ],
)
def test_run_metric_record_rejects_invalid_identity_or_value(
    change: dict[str, object], match: str
) -> None:
    values = asdict(_record("a", 1, 0.8))
    values.update(change)
    with pytest.raises(ValueError, match=match):
        RunMetricRecord(**values)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("seed", True, "seed"),
        ("value", True, "numeric"),
        ("test_partition_accessed", "false", "Boolean"),
        ("method", 7, "strings"),
    ],
)
def test_run_metric_from_dict_rejects_wrong_json_types(
    field: str, value: object, match: str
) -> None:
    payload = asdict(_record("a", 1, 0.8))
    payload[field] = value
    with pytest.raises(TypeError, match=match):
        RunMetricRecord.from_dict(payload)


def test_aggregate_run_metrics_is_order_invariant_and_tracks_every_run() -> None:
    records = (_record("a", 1, 0.8), _record("b", 2, 0.9), _record("c", 3, 0.7))
    forward = aggregate_run_metrics(records, bootstrap_resamples=50, bootstrap_seed=3)
    reverse = aggregate_run_metrics(
        tuple(reversed(records)), bootstrap_resamples=50, bootstrap_seed=3
    )
    assert forward == reverse
    assert forward[0].run_ids == ("a" * 8, "b" * 8, "c" * 8)
    assert forward[0].seeds == (1, 2, 3)


def test_aggregate_rejects_duplicate_seed_within_summary() -> None:
    with pytest.raises(ValueError, match="one run per seed"):
        aggregate_run_metrics(
            (_record("a", 1, 0.8), _record("b", 1, 0.9)),
            bootstrap_resamples=10,
        )


def test_aggregate_rejects_empty_duplicate_metric_and_mixed_gates() -> None:
    record = _record("a", 1, 0.8)
    with pytest.raises(ValueError, match="at least one"):
        aggregate_run_metrics(())
    with pytest.raises(ValueError, match="only once"):
        aggregate_run_metrics((record, record))
    mixed = replace(
        _record("b", 2, 0.9),
        test_partition_accessed=True,
        performance_claims_allowed=False,
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        aggregate_run_metrics((record, mixed), bootstrap_resamples=10)


def test_summary_bundle_and_rendered_artifacts_regenerate_byte_identically(
    tmp_path: Path,
) -> None:
    records = (
        _record("a", 1, 0.8),
        _record("b", 2, 0.9),
        _record("c", 1, 0.6, method="fedavg"),
        _record("d", 2, 0.7, method="fedavg"),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    bundle_first = write_summary_bundle(records, first, bootstrap_resamples=40, bootstrap_seed=5)
    bundle_second = write_summary_bundle(records, second, bootstrap_resamples=40, bootstrap_seed=5)
    assert bundle_first == bundle_second
    assert (first / "metric_summaries.json").read_bytes() == (
        second / "metric_summaries.json"
    ).read_bytes()

    rendered_first = render_summary_artifacts(
        first / "metric_summaries.json", first, metric_name="macro_f1"
    )
    rendered_second = render_summary_artifacts(
        second / "metric_summaries.json", second, metric_name="macro_f1"
    )
    assert rendered_first["artifact_id"] == rendered_second["artifact_id"]
    for suffix in ("table.csv", "table.tex", "plot.svg", "artifact_manifest.json"):
        assert (first / f"macro_f1_{suffix}").read_bytes() == (
            second / f"macro_f1_{suffix}"
        ).read_bytes()

    svg = (first / "macro_f1_plot.svg").read_text(encoding="utf-8")
    assert "data-summary-id=" in svg and "run_ids=" in svg
    table = (first / "macro_f1_table.csv").read_text(encoding="utf-8")
    assert "summary_id" in table and "run_ids" in table


def test_summary_bundle_identity_detects_content_tampering(tmp_path: Path) -> None:
    write_summary_bundle((_record("a", 1, 0.8),), tmp_path, bootstrap_resamples=10)
    path = tmp_path / "metric_summaries.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["summaries"][0]["statistics"]["mean"] = 0.1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        load_summary_bundle(path)


def test_summary_bundle_loader_and_renderer_reject_malformed_clean_input(tmp_path: Path) -> None:
    invalid_schema = tmp_path / "invalid-schema.json"
    invalid_schema.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_summary_bundle(invalid_schema)

    summaries_not_list = {
        "schema": "semcred_phase11_summary_bundle_v1",
        "summaries": {},
    }
    summaries_not_list["summary_bundle_id"] = run_id(summaries_not_list, length=64)
    wrong_type = tmp_path / "wrong-type.json"
    wrong_type.write_text(json.dumps(summaries_not_list), encoding="utf-8")
    with pytest.raises(TypeError, match="list"):
        load_summary_bundle(wrong_type)

    malformed_entry = {
        "schema": "semcred_phase11_summary_bundle_v1",
        "summaries": ["not-an-object"],
    }
    malformed_entry["summary_bundle_id"] = run_id(malformed_entry, length=64)
    malformed = tmp_path / "malformed-entry.json"
    malformed.write_text(json.dumps(malformed_entry), encoding="utf-8")
    with pytest.raises(TypeError, match="malformed"):
        load_summary_bundle(malformed)

    write_summary_bundle((_record("a", 1, 0.8),), tmp_path, bootstrap_resamples=10)
    with pytest.raises(ValueError, match="no metric"):
        render_summary_artifacts(
            tmp_path / "metric_summaries.json", tmp_path, metric_name="accuracy"
        )


def test_jsonl_reader_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    record = _record("a", 1, 0.8)
    malformed = (
        {**record.__dict__, "unexpected": True}
        if hasattr(record, "__dict__")
        else {
            "run_id": record.run_id,
            "unexpected": True,
        }
    )
    path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_metric_jsonl(path)


def test_jsonl_reader_success_and_empty_or_nonobject_failures(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text("\n" + json.dumps(asdict(_record("a", 1, 0.8))) + "\n", encoding="utf-8")
    assert read_metric_jsonl(valid_path)[0].run_id == "a" * 8

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        read_metric_jsonl(empty_path)

    nonobject_path = tmp_path / "nonobject.jsonl"
    nonobject_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        read_metric_jsonl(nonobject_path)
