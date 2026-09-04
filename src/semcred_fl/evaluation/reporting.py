"""Deterministic run aggregation and traceable table/figure generation."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape as xml_escape
from pathlib import Path

from semcred_fl.evaluation.statistics import DescriptiveStatistics, describe
from semcred_fl.identity import canonical_json, run_id

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
_RUN_ID = re.compile(r"^[0-9a-f]{8,64}$")


@dataclass(frozen=True, slots=True)
class RunMetricRecord:
    """One scalar metric from one completed experimental seed."""

    run_id: str
    experiment_family_id: str
    dataset: str
    method: str
    condition_id: str
    seed: int
    metric_name: str
    value: float
    source_artifact_id: str
    test_partition_accessed: bool
    performance_claims_allowed: bool

    def __post_init__(self) -> None:
        identifiers = (
            self.experiment_family_id,
            self.dataset,
            self.method,
            self.condition_id,
            self.metric_name,
            self.source_artifact_id,
        )
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a lowercase hexadecimal identifier")
        if any(not value or _IDENTIFIER.fullmatch(value) is None for value in identifiers):
            raise ValueError("reporting identifiers contain unsupported characters")
        if self.seed < 0 or not math.isfinite(self.value):
            raise ValueError("metric seed and value must be valid")
        if self.performance_claims_allowed and not self.test_partition_accessed:
            raise ValueError("performance claims require final-test access")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunMetricRecord:
        """Parse one exact-schema JSON record."""

        expected = {
            "run_id",
            "experiment_family_id",
            "dataset",
            "method",
            "condition_id",
            "seed",
            "metric_name",
            "value",
            "source_artifact_id",
            "test_partition_accessed",
            "performance_claims_allowed",
        }
        if set(value) != expected:
            raise ValueError("run metric record keys do not match the frozen schema")
        if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
            raise TypeError("seed must be an integer")
        if isinstance(value["value"], bool) or not isinstance(value["value"], int | float):
            raise TypeError("value must be numeric")
        bool_fields = ("test_partition_accessed", "performance_claims_allowed")
        if any(not isinstance(value[field], bool) for field in bool_fields):
            raise TypeError("partition and claim gates must be Boolean")
        text_fields = expected - {"seed", "value", *bool_fields}
        if any(not isinstance(value[field], str) for field in text_fields):
            raise TypeError("run metric identifiers must be strings")
        run_identifier = value["run_id"]
        family = value["experiment_family_id"]
        dataset = value["dataset"]
        method = value["method"]
        condition = value["condition_id"]
        metric = value["metric_name"]
        artifact = value["source_artifact_id"]
        test_accessed = value["test_partition_accessed"]
        claims_allowed = value["performance_claims_allowed"]
        assert isinstance(run_identifier, str)
        assert isinstance(family, str)
        assert isinstance(dataset, str)
        assert isinstance(method, str)
        assert isinstance(condition, str)
        assert isinstance(metric, str)
        assert isinstance(artifact, str)
        assert isinstance(test_accessed, bool)
        assert isinstance(claims_allowed, bool)
        return cls(
            run_id=run_identifier,
            experiment_family_id=family,
            dataset=dataset,
            method=method,
            condition_id=condition,
            seed=value["seed"],
            metric_name=metric,
            value=float(value["value"]),
            source_artifact_id=artifact,
            test_partition_accessed=test_accessed,
            performance_claims_allowed=claims_allowed,
        )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One displayed estimate and its complete run provenance."""

    summary_id: str
    experiment_family_id: str
    dataset: str
    method: str
    condition_id: str
    metric_name: str
    statistics: DescriptiveStatistics
    seeds: tuple[int, ...]
    run_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    test_partition_accessed: bool
    performance_claims_allowed: bool

    def to_dict(self) -> dict[str, object]:
        """Return the canonical summary schema."""

        return {
            "summary_id": self.summary_id,
            "experiment_family_id": self.experiment_family_id,
            "dataset": self.dataset,
            "method": self.method,
            "condition_id": self.condition_id,
            "metric_name": self.metric_name,
            "statistics": asdict(self.statistics),
            "seeds": list(self.seeds),
            "run_ids": list(self.run_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "test_partition_accessed": self.test_partition_accessed,
            "performance_claims_allowed": self.performance_claims_allowed,
        }


def aggregate_run_metrics(
    records: Sequence[RunMetricRecord],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260901,
) -> tuple[MetricSummary, ...]:
    """Aggregate run metrics with deterministic grouping and bootstrap streams."""

    if not records:
        raise ValueError("aggregation requires at least one run metric")
    duplicate_keys = [(record.run_id, record.metric_name) for record in records]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError("a run may contribute a metric only once")
    grouped: dict[tuple[str, str, str, str, str], list[RunMetricRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.experiment_family_id,
            record.dataset,
            record.method,
            record.condition_id,
            record.metric_name,
        )
        grouped[key].append(record)

    summaries: list[MetricSummary] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: (item.seed, item.run_id))
        seeds = tuple(item.seed for item in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError("each summary requires one run per seed")
        access_flags = {item.test_partition_accessed for item in group}
        claim_flags = {item.performance_claims_allowed for item in group}
        if len(access_flags) != 1 or len(claim_flags) != 1:
            raise ValueError("partition and claim gates cannot be mixed within one summary")
        group_seed = bootstrap_seed ^ int(run_id(key, length=8), 16)
        statistics = describe(
            [item.value for item in group],
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=group_seed,
        )
        identity = {
            "schema": "semcred_phase11_metric_summary_v1",
            "key": key,
            "statistics": asdict(statistics),
            "seeds": seeds,
            "run_ids": [item.run_id for item in group],
            "source_artifact_ids": [item.source_artifact_id for item in group],
            "test_partition_accessed": group[0].test_partition_accessed,
            "performance_claims_allowed": group[0].performance_claims_allowed,
        }
        summaries.append(
            MetricSummary(
                summary_id=run_id(identity, length=64),
                experiment_family_id=key[0],
                dataset=key[1],
                method=key[2],
                condition_id=key[3],
                metric_name=key[4],
                statistics=statistics,
                seeds=seeds,
                run_ids=tuple(item.run_id for item in group),
                source_artifact_ids=tuple(item.source_artifact_id for item in group),
                test_partition_accessed=group[0].test_partition_accessed,
                performance_claims_allowed=group[0].performance_claims_allowed,
            )
        )
    return tuple(summaries)


def read_metric_jsonl(path: Path) -> tuple[RunMetricRecord, ...]:
    """Read exact-schema metric records from JSON Lines."""

    records: list[RunMetricRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} must contain a JSON object")
            records.append(RunMetricRecord.from_dict(value))
    if not records:
        raise ValueError("metric JSONL contains no records")
    return tuple(records)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_summary_bundle(
    records: Sequence[RunMetricRecord],
    output_directory: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260901,
) -> dict[str, object]:
    """Write a canonical summary and run-provenance sidecar."""

    summaries = aggregate_run_metrics(
        records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_values = [summary.to_dict() for summary in summaries]
    stable = {
        "schema": "semcred_phase11_summary_bundle_v1",
        "summaries": summary_values,
    }
    bundle_id = run_id(stable, length=64)
    bundle: dict[str, object] = {**stable, "summary_bundle_id": bundle_id}
    _write_text_atomic(output_directory / "metric_summaries.json", canonical_json(bundle) + "\n")
    provenance = {
        "schema": "semcred_phase11_provenance_v1",
        "summary_bundle_id": bundle_id,
        "entries": {
            summary.summary_id: {
                "run_ids": list(summary.run_ids),
                "source_artifact_ids": list(summary.source_artifact_ids),
            }
            for summary in summaries
        },
    }
    _write_text_atomic(
        output_directory / "metric_provenance.json", canonical_json(provenance) + "\n"
    )
    return bundle


def load_summary_bundle(path: Path) -> tuple[str, tuple[MetricSummary, ...]]:
    """Load and verify a previously generated canonical summary bundle."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "summaries", "summary_bundle_id"}:
        raise ValueError("summary bundle schema is invalid")
    stable = {"schema": value["schema"], "summaries": value["summaries"]}
    if value["schema"] != "semcred_phase11_summary_bundle_v1" or value[
        "summary_bundle_id"
    ] != run_id(stable, length=64):
        raise ValueError("summary bundle identity does not match its content")
    if not isinstance(value["summaries"], list):
        raise TypeError("summaries must be a list")
    summaries: list[MetricSummary] = []
    for item in value["summaries"]:
        if not isinstance(item, dict) or not isinstance(item.get("statistics"), dict):
            raise TypeError("summary entry is malformed")
        statistics = DescriptiveStatistics(**item["statistics"])
        summary = MetricSummary(
            summary_id=item["summary_id"],
            experiment_family_id=item["experiment_family_id"],
            dataset=item["dataset"],
            method=item["method"],
            condition_id=item["condition_id"],
            metric_name=item["metric_name"],
            statistics=statistics,
            seeds=tuple(item["seeds"]),
            run_ids=tuple(item["run_ids"]),
            source_artifact_ids=tuple(item["source_artifact_ids"]),
            test_partition_accessed=item["test_partition_accessed"],
            performance_claims_allowed=item["performance_claims_allowed"],
        )
        summaries.append(summary)
    return value["summary_bundle_id"], tuple(summaries)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_summary_artifacts(
    summary_bundle_path: Path,
    output_directory: Path,
    *,
    metric_name: str,
) -> dict[str, str]:
    """Regenerate CSV, LaTeX, and deterministic SVG from clean summaries."""

    bundle_id, all_summaries = load_summary_bundle(summary_bundle_path)
    summaries = tuple(summary for summary in all_summaries if summary.metric_name == metric_name)
    if not summaries:
        raise ValueError(f"summary bundle has no metric named {metric_name}")
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream, lineterminator="\n")
    writer.writerow(
        [
            "summary_id",
            "dataset",
            "condition_id",
            "method",
            "mean",
            "sample_standard_deviation",
            "median",
            "bootstrap_lower",
            "bootstrap_upper",
            "run_ids",
        ]
    )
    for summary in summaries:
        stats = summary.statistics
        writer.writerow(
            [
                summary.summary_id,
                summary.dataset,
                summary.condition_id,
                summary.method,
                f"{stats.mean:.8g}",
                f"{stats.sample_standard_deviation:.8g}",
                f"{stats.median:.8g}",
                f"{stats.bootstrap_lower:.8g}",
                f"{stats.bootstrap_upper:.8g}",
                ";".join(summary.run_ids),
            ]
        )
    _write_text_atomic(output_directory / f"{metric_name}_table.csv", csv_stream.getvalue())

    latex_lines = [
        "% Generated from machine-readable Phase 11 summaries; do not edit numbers.",
        r"\begin{tabular}{lllll}",
        r"Dataset & Condition & Method & Mean $\pm$ SD [95\% CI] & Summary ID \\",
        r"\hline",
    ]
    for summary in summaries:
        stats = summary.statistics
        value = (
            f"{stats.mean:.4f} $\\pm$ {stats.sample_standard_deviation:.4f} "
            f"[{stats.bootstrap_lower:.4f}, {stats.bootstrap_upper:.4f}]"
        )
        latex_lines.append(
            " & ".join(
                (
                    _latex_escape(summary.dataset),
                    _latex_escape(summary.condition_id),
                    _latex_escape(summary.method),
                    value,
                    rf"\texttt{{{summary.summary_id[:12]}}}",
                )
            )
            + r" \\"
        )
    latex_lines.extend((r"\end{tabular}", f"% summary_bundle_id={bundle_id}"))
    _write_text_atomic(output_directory / f"{metric_name}_table.tex", "\n".join(latex_lines) + "\n")

    width = 920
    height = max(220, 90 + 34 * len(summaries))
    left = 265
    plot_width = width - left - 45
    means = [summary.statistics.mean for summary in summaries]
    lows = [summary.statistics.bootstrap_lower for summary in summaries]
    highs = [summary.statistics.bootstrap_upper for summary in summaries]
    minimum = min(0.0, *lows)
    maximum = max(1.0 if all(0.0 <= value <= 1.0 for value in means) else 0.0, *highs)
    span = maximum - minimum or 1.0

    def x_position(value: float) -> float:
        return left + ((value - minimum) / span) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f"<metadata>summary_bundle_id={bundle_id}</metadata>",
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="20" y="28" font-family="sans-serif" font-size="18">'
            f"{xml_escape(metric_name)}</text>"
        ),
        f'<line x1="{left}" y1="45" x2="{left + plot_width}" y2="45" stroke="#444"/>',
    ]
    for index, summary in enumerate(summaries):
        y = 75 + index * 34
        label = f"{summary.dataset} | {summary.condition_id} | {summary.method}"
        low_x = x_position(summary.statistics.bootstrap_lower)
        high_x = x_position(summary.statistics.bootstrap_upper)
        mean_x = x_position(summary.statistics.mean)
        title = (
            f"summary_id={summary.summary_id}; run_ids={','.join(summary.run_ids)}; "
            f"mean={summary.statistics.mean:.8g}"
        )
        svg.extend(
            (
                f'<text x="20" y="{y + 5}" font-family="sans-serif" font-size="12">'
                f"{xml_escape(label)}</text>",
                f'<g data-summary-id="{summary.summary_id}"><title>{xml_escape(title)}</title>',
                f'<line x1="{low_x:.3f}" y1="{y}" x2="{high_x:.3f}" y2="{y}" '
                'stroke="#355c7d" stroke-width="3"/>',
                f'<circle cx="{mean_x:.3f}" cy="{y}" r="5" fill="#c06c84"/></g>',
            )
        )
    svg.append("</svg>")
    _write_text_atomic(output_directory / f"{metric_name}_plot.svg", "\n".join(svg) + "\n")

    artifact_identity = {
        "schema": "semcred_phase11_rendered_artifacts_v1",
        "summary_bundle_id": bundle_id,
        "metric_name": metric_name,
        "summary_ids": [summary.summary_id for summary in summaries],
        "run_ids_by_summary": {summary.summary_id: list(summary.run_ids) for summary in summaries},
    }
    artifact_id = run_id(artifact_identity, length=64)
    manifest = {**artifact_identity, "artifact_id": artifact_id}
    _write_text_atomic(
        output_directory / f"{metric_name}_artifact_manifest.json",
        canonical_json(manifest) + "\n",
    )
    return {
        "artifact_id": artifact_id,
        "summary_bundle_id": bundle_id,
        "table_csv": str(output_directory / f"{metric_name}_table.csv"),
        "table_latex": str(output_directory / f"{metric_name}_table.tex"),
        "plot_svg": str(output_directory / f"{metric_name}_plot.svg"),
    }
