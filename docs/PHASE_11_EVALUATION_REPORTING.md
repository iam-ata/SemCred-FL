# Phase 11: evaluation and reporting

Status: implementation gate complete on 2026-09-01. No dataset, validation
partition, test partition, model checkpoint, or LLM endpoint was accessed by this
phase. The included fixtures are analytical/synthetic and are not article results.

## Implemented boundary

Phase 11 adds a leakage-isolated evaluation layer under
`src/semcred_fl/evaluation/`:

- `metrics.py` implements attack-rate evidence, absolute and relative utility
  degradation, malicious-client AUROC/AUPRC, benign false rejection, malicious
  false acceptance, benign-minus-malicious trust separation, participation-event
  detection/recovery delay, and benign trust volatility;
- the existing `modeling/metrics.py` remains the source for accuracy, macro
  precision/recall/F1, balanced accuracy, worst-class recall, one-versus-rest
  false-positive rates, per-class values, and the confusion matrix;
- `efficiency.py` keeps online timing, communication bytes, and the one-time LLM
  request record in separate typed structures;
- `statistics.py` implements deterministic descriptive summaries, percentile-
  bootstrap intervals for the seed-level mean, an exact two-sided paired
  Wilcoxon sign-permutation distribution, the paired rank-biserial effect size,
  and Holm correction over an explicitly declared comparison family;
- `reporting.py` aggregates exact-schema run records and regenerates CSV, LaTeX,
  SVG, and JSON provenance artifacts from the clean summary bundle;
- `scripts/aggregate_phase11_results.py` is the command-line entry point used by
  Phase 12 after completed run records exist.

The defense path never receives simulated malicious identity. Phase 12 must first
finish a defense decision and only then pass malicious labels, attack outcomes, and
test predictions to this evaluation package.

## Frozen metric conventions

- Macro-F1 is the primary clean and attacked utility endpoint.
- Absolute degradation is `clean - attacked`; relative degradation is
  `(clean - attacked) / max(clean, epsilon)`.
- Every attack success rate stores its numerator, denominator, value, and an
  undefined reason. An empty denominator is not converted to zero.
- Malicious-client suspiciousness is supplied after the defense decision. For
  trust-based methods, Phase 12 uses `1 - trust`. Empirical AUROC assigns half
  credit to score ties. AUPRC is threshold-grouped average precision, so input
  order cannot break tied scores.
- Trust separation reports both mean and median benign-minus-malicious gaps.
  Positive values indicate higher benign trust.
- Detection and recovery delay count participating events, including the event
  on which the threshold is crossed. An unobserved crossing is right-censored;
  it is not assigned the last round or an infinite value.
- Benign trust volatility is the root-mean-square change across consecutive
  benign participation events. Inactive calendar rounds are not inserted.
- The bootstrap interval is a two-sided percentile interval for the seed-level
  mean. Its resample count and derived seed are stored in every summary.
- The paired Wilcoxon implementation requires identical pair IDs. Exact zero
  differences use the Wilcox discard convention and are counted. Equal absolute
  differences use midranks; the sign-permutation distribution is enumerated by
  dynamic programming. Rank-biserial correlation is positive when the declared
  left method has larger values.
- Holm adjustment is applied once to one preregistered comparison family. Mixed
  or duplicate family declarations fail.

Five seeds remain a pilot. They support descriptive intervals and effect-size
estimation, not confirmatory significance claims. A confirmatory family requires
the existing simulation-based power-analysis record and at least ten matched
seeds.

## Run-record and artifact contract

Each scalar input row contains:

`run_id`, `experiment_family_id`, `dataset`, `method`, `condition_id`, `seed`,
`metric_name`, `value`, `source_artifact_id`, `test_partition_accessed`, and
`performance_claims_allowed`.

The schema rejects unknown/missing fields, nonfinite values, repeated
`(run_id, metric_name)` pairs, duplicate seeds inside one summary, mixed
partition/claim gates, and performance authorization without test access.

Every output row or plotted point carries a `summary_id`. The provenance sidecar
maps that identifier to all contributing run IDs and source artifact IDs. The SVG
also stores the mapping in each point's metadata. The summary bundle and rendered
artifact manifest are content-addressed. Re-running the same input, bootstrap
configuration, and metric produces byte-identical machine-readable artifacts.

## Validation gate

`tests/test_phase11_evaluation.py` contains known fixtures for rates, degradation,
ranking ties, eligibility errors, trust gaps and delays, timing/communication,
bootstrap determinism, exact Wilcoxon zeros/ties, Holm correction families,
schema rejection, bundle tamper detection, provenance completeness, and
byte-identical regeneration.

Closure evidence on the pinned `semcred-fl` environment:

- 55 focused Phase 11 tests passed;
- branch coverage for `semcred_fl.evaluation` is 99.72% (540 statements, one
  uncovered defensive branch);
- the complete repository suite passed 458 tests in 27.888 seconds;
- Ruff reported all checks passed and 146 files formatted;
- strict mypy completed without errors for `src` and the aggregation script;
- two `pdflatex` passes produced the synchronized 44-page manuscript PDF.

Use the lightweight gate:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python -m pytest -q -p no:cacheprovider tests\test_phase11_evaluation.py
conda run --no-capture-output -n semcred-fl python -m ruff check src tests scripts
conda run --no-capture-output -n semcred-fl python -m ruff format --check src tests scripts
conda run --no-capture-output -n semcred-fl python -m mypy --strict src scripts\aggregate_phase11_results.py
```

No user-run heavy command is required for Phase 11. Phase 12 begins with a tiny
synthetic smoke experiment and then a bounded one-dataset clean comparison before
any full matrix is authorized.
