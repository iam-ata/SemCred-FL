# Phase 12 Step 4 validation report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-02
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- Source: `bca0fb435639192f3ee25d74fbb566f911a73099c0cb33cf13f5c740621efce3`
- Overall Confidence: CAUTION
- Scope: one Edge-IIoTset `dirichlet_0_3` label-flip seed, three rounds, validation only

### Statistical findings

| Method | Accuracy | Macro-F1 | Balanced accuracy | Worst-class recall |
|---|---:|---:|---:|---:|
| SemCred-FL | 0.764158 | 0.244747 | 0.219391 | 0 |
| Coordinate median | 0.697210 | 0.237371 | 0.205102 | 0 |
| Trimmed mean | 0.697504 | 0.237251 | 0.205165 | 0 |
| Multi-Krum | 0.697226 | 0.237133 | 0.205106 | 0 |
| FedAvg | 0.757137 | 0.236565 | 0.212130 | 0 |

SemCred-FL has the largest observed macro-F1 in this bounded run. The report contains
one seed, no sampling distribution, no confidence interval, no effect size, and no
hypothesis test. The ordering therefore cannot identify a best method. Every method
has zero recall for four of six classes; the accuracy values are dominated by the
two largest classes.

### Reproducibility and integrity checks

- The stored matrix ID was recomputed from the stable report content and matched.
- The report contains all five methods and 35 exact-schema metric records.
- The three newly executed baselines have nine round records in total.
- Their runtime records show 50 trained and 40 exact-input-reused local updates.
- No malicious label or attack metadata occurs in a defense-facing round payload.
- Validation was accessed after method completion; the test partition was not read.
- Reproducibility verdict: CANNOT_VERIFY. The artifact identity is verified, but an
  independent stochastic rerun was not performed.

### Warnings

| Type | Detail | Affected results |
|---|---|---|
| Single-seed uncertainty | No between-seed variance or paired inference is available. | All method comparisons |
| Short training | Three rounds and one local epoch are an integration budget. | All utility metrics |
| Class imbalance | Four classes have zero recall under every method. | Accuracy and aggregate ranking |
| Multiple outcomes | Five methods and seven reported scalar metrics have no inferential correction. | Any winner claim |
| Validation-only | The result can guide development but cannot be presented as final test performance. | Manuscript claims |

### Fallacy scan

- Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | No subgroup reversal can be tested from one condition; class-level results nevertheless show that aggregate accuracy hides four zero-recall classes. |
| Ecological fallacy | NOTE | No individual-client claim is inferred from the aggregate validation metrics. |
| Berkson's paradox | NOTE | The frozen validation partition was not selected using model outcomes. |
| Collider bias | NOTE | No post-treatment covariate adjustment is used. |
| Base-rate neglect | CAUTION | Accuracy is high relative to macro-F1 because the validation support is strongly imbalanced. |
| Regression to the mean | NOTE | This is not an extreme-group pre/post comparison. |
| Survivorship bias | NOTE | All scheduled submissions were represented; no method was dropped from the matrix. |
| Look-elsewhere effect | CAUTION | Multiple methods and outcomes are inspected without an inferential family at this pilot stage. |
| Garden of forking paths | CAUTION | The condition was frozen before execution, but the run is exploratory and cannot authorize post-hoc method selection. |
| Correlation versus causation | NOTE | No causal manuscript claim is made from this pilot. |
| Reverse causality | NOTE | The comparison is an executed intervention, not a cross-sectional association. |

The next valid use of these results is to verify the full-budget clean runner and
check whether minority-class learning becomes nonzero. The Step 4 ranking itself is
not a tuning objective and must remain unchanged in the evidence record.
