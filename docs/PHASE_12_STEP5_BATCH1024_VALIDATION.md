# Phase 12 Step 5 batch-1024 learning diagnostic

Status: completed and audited on 2026-09-02. This is validation-only development
evidence. It is not eligible for a manuscript performance claim.

## Material passport

- Source artifacts: the frozen Edge-IIoTset Phase 3C corpus, Phase 3D
  `dirichlet_0_3` client partition, and Phase 7 semantic-prior bundle.
- Condition: seed `20260816`, 50 clients, 20 clients per round, 10 rounds, three
  local epochs, batch size 1024, Adam at `1e-3`, and the shared 256/128/64
  detector.
- Compared methods: FedAvg, coordinate median, trimmed mean, Multi-Krum, and
  SemCred-FL under one initialization and client schedule.
- Matrix input ID:
  `30b612d0be41ce33ba7ab744c7cdfb957b855cd32228339ebefda19bbdc79a66`.
- Matrix result ID:
  `591183ecafb0b821c2cb88a54f541fce5543f389325e6108ab9bce410e272f6f`.
- Access boundary: validation was read only after each method completed; the test
  partition was not accessed.

## Integrity audit

The matrix, condition, and five method result identities recompute from their
stable JSON payloads. Each method completed ten rounds, each latest checkpoint
points to round 10, and every checkpoint file matches the SHA-256 stored in its
pointer. The metric stream contains the expected 35 valid JSONL records. Across
1,000 method/client update requests, 860 updates were trained and 140 were reused
only through content-addressed cache matches.

The five method runtimes sum to 4,600.65 seconds (76 minutes 40.65 seconds). This
is an execution measurement on the development computer, not an algorithmic
efficiency claim.

## Validation observations

| Method | Accuracy | Macro-F1 | Balanced accuracy | Worst-class recall | Runtime |
|---|---:|---:|---:|---:|---:|
| Multi-Krum | 0.696849 | 0.237373 | 0.205631 | 0.000000 | 1,060.94 s |
| Coordinate median | 0.696673 | 0.236252 | 0.205176 | 0.000000 | 821.29 s |
| Trimmed mean | 0.696716 | 0.236244 | 0.205127 | 0.000000 | 942.09 s |
| SemCred-FL | 0.683614 | 0.234934 | 0.202574 | 0.000000 | 748.29 s |
| FedAvg | 0.646126 | 0.230274 | 0.194605 | 0.000000 | 1,028.04 s |

The observed ordering does not establish method superiority. It is one seed, one
clean non-IID profile, and a ten-round development budget. SemCred-FL is fourth by
macro-F1: 0.002439 below Multi-Krum and 0.004660 above FedAvg.

The class-level result fails the learning gate. Class IDs 2
(`information_gathering`) and 5 (`man_in_the_middle`) have zero recall under all
five methods. Recall for class 3 (`injection`) is at most 0.003609, and recall for
class 4 (`malware`) is at most 0.002088. The corresponding validation supports are
19,974, 19, 29,646, and 9,101 records for classes 2, 5, 3, and 4. The high overall
accuracy relative to balanced accuracy is therefore dominated by the two largest
classes and cannot be used as evidence of a useful six-class detector.

## Statistical validity review

- There is no p-value, confidence interval, or effect-size claim to interpret.
- The single seed prevents uncertainty estimation and paired confirmatory
  inference.
- Method ranking is exploratory because five methods and several metrics are
  inspected together.
- Accuracy is not used to override macro-F1, balanced accuracy, or per-class
  recall under the severe class imbalance.
- The result is not pooled with the earlier IID or attacked pilots because their
  conditions differ.
- No subgroup, seed, stopping round, or baseline is removed because of this
  result.
- Re-running or tuning on validation can support development selection only; it
  cannot convert this seed into confirmatory evidence.

## Decision

Batch size 1024 remains a provisional runtime setting: the run stayed finite,
completed, and retained valid identities, but it did not pass the class-coverage
gate. The 100-round matrix is not released on this evidence.

The next validation-only candidate keeps batch size 1024 and the same total short-
run local-epoch budget, but changes the synchronization schedule from 10 rounds ×
3 local epochs to 30 rounds × 1 local epoch. This isolates whether more frequent
global aggregation reduces non-IID client drift. The diagnostic runs only FedAvg
and SemCred-FL because the zero-recall failure occurred under every aggregation
method and the immediate question concerns the shared learner. The complete
six-class validation metrics remain visible, with the pass/fail gate focused on
classes 2 and 5. This remains a separate, non-claim-bearing content-addressed
campaign. If their recall remains zero, the next investigation must audit the
shared training objective and class weighting before any 100-round matrix is run.

The targeted runner command is:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step5.py `
  --device cuda `
  --datasets edge_iiotset `
  --profiles dirichlet_0_3 `
  --seeds 20260816 `
  --methods fedavg semcred_fl `
  --batch-size 1024 `
  --total-rounds 30 `
  --local-epochs 1
```

The command is resumable. Because the method subset is part of the content
identity, it cannot overwrite or be mistaken for the complete five-method matrix.

## Synchronization diagnostic result

The targeted 30-round × 1-local-epoch campaign completed under matrix input ID
`95a7dacede60e04bebb8e9622e0d36fb1176c71a34dffabed92f7895f218d1e2`
and matrix result ID
`3e3c0261013eace2b25822e62b40e6e819b0cde2b78d1dc98ebf8f2766f820b0`.
The matrix, condition, FedAvg, and SemCred-FL identities recompute. Both methods
have valid round-30 checkpoints, and all 14 expected metric records are present.
The aggregate method runtime was 2,465.18 seconds (41 minutes 5.18 seconds).

| Method | Accuracy | Macro-F1 | Balanced accuracy | Class 2 recall | Class 5 recall |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.697561 | 0.236391 | 0.205131 | 0.000000 | 0.000000 |
| SemCred-FL | 0.645744 | 0.229974 | 0.194463 | 0.000000 | 0.000000 |

The synchronization candidate fails the targeted gate. Both methods also had zero
injection recall; FedAvg had zero malware recall, while SemCred-FL malware recall
was 0.002088. More frequent aggregation therefore did not repair the shared
six-class learner and is not selected for the final matrix.

The next test removes federation entirely. One centralized model will train for a
single full-corpus epoch with the existing detector, preprocessing, loss, class-
weight cap, batch size 1024, and seed. Full validation is evaluated once after the
epoch. If class 2 remains at zero recall, the shared representation or training
objective requires repair. Class 5 has only 136 training and 19 validation records,
so its result is interpreted as a data-sufficiency diagnostic rather than a stable
performance estimate.
