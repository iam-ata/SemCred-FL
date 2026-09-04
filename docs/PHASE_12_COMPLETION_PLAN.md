# Phase 12 completion plan

Status: active on 2026-09-02. Steps 1-4 are complete. Step 5 is verified, and both
learning diagnostics are complete. The 10-round × 3-local-epoch five-method run and
the 30-round × 1-local-epoch FedAvg/SemCred-FL run retained zero recall for classes
2 and 5. All audited identities and checkpoint hashes pass. The synchronization
change is rejected, and the 100-round matrix remains held while a one-model
centralized control isolates the shared learner from federation.

The Step 5 runtime calibration measured batch 128, 1024, and 2048 under matched
round-zero conditions. Batch 1024 was fastest at 91.337 seconds, compared with
520.849 seconds for batch 128 and 99.476 seconds for batch 2048. Batch 1024 remains
provisional: its ten-round diagnostic completed in 4,600.65 aggregate method-
seconds, but did not pass the class-coverage gate.

## Objective and scientific boundary

The objective is to give SemCred-FL the strongest fair opportunity to demonstrate
its value. No experiment may be configured, removed, renamed, or selectively
reported to force SemCred-FL into first place. All compared parameter-update methods
receive the same detector, initialization, client schedule, local-training budget,
data artifacts, and compatible attack realization. Hyperparameters are selected
from validation evidence only. The final test partition stays sealed until the
registry, analysis code, and manuscript claim rules are frozen.

## Current diagnosis

The three-round Steps 2 and 3 verified the pipeline but did not train a useful
six-class detector. The completed ten-round Step 5 diagnostic improved only the two
largest classes: every method still had zero recall for information gathering and
man-in-the-middle, while injection and malware recall remained below 0.004. Step 3
also had only six malicious participation events, of which SemCred-FL accepted
four. These runs cannot establish that a longer defense succeeds or fails. Step 2
was IID and Step 3 was non-IID, so their difference cannot estimate attack
degradation.

The repair is a matched, adequately trained, multi-seed protocol with per-class and
client-assessment gates. It is not a post-hoc change to the final test evaluation.

## Execution gates

| Step | Deliverable | Status | Mandatory gate before advancing |
|---|---|---|---|
| 1 | Synthetic end-to-end smoke run | Complete | Components execute without leakage |
| 2 | Clean Edge-IIoTset FedAvg/SemCred-FL pilot | Complete | Completion-only validation; no test access |
| 3 | One attacked non-IID pilot | Complete | Attack truth excluded from defense inputs |
| 4 | Reduced five-method baseline matrix | Complete | Same Step 3 condition; identity and leakage audits passed |
| 5 | Full clean utility matrix | Two diagnostics failed class coverage; centralized control next | Resolve shared detector learning before the 100-round matrix |
| 6 | Full robustness matrix | Pending | Every attack has a matched clean condition and defined denominator |
| 7 | Trust analysis | Pending | Derived from immutable round logs; no simulator truth before decisions |
| 8 | Ablations | Pending | Complete method plus every required one-component removal |
| 9 | Sensitivity and anchor contamination | Pending | Validation-only selection; one-factor-at-a-time before joint study |
| 10 | Efficiency profiling | Pending | Separate one-time, client, server, communication, and memory costs |
| 11 | Statistical aggregation | Pending | Power record and at least ten matched seeds for confirmatory claims |
| 12 | Manuscript tables, figures, and evidence audit | Pending | One final test opening after registry freeze; every number provenance-linked |

## Full-budget controls for Steps 5-10

- 50 clients and 20 submitted updates per round.
- 100 federated rounds and 3 local epochs unless a preregistered validation-only
  development study replaces these values before test access.
- The fixed MLP detector and training objective already mapped to the manuscript.
- Edge-IIoTset and CICIoT2023 evaluated independently.
- IID plus Dirichlet concentrations 1.0, 0.3, and 0.1.
- FedAvg, coordinate median, trimmed mean, Multi-Krum, and SemCred-FL under compatible
  parameter-update protocols.
- At least ten matched seeds for any confirmatory Wilcoxon-Holm statement. Five-seed
  output remains pilot evidence.
- Macro-F1 is primary; accuracy cannot override macro-F1 or zero minority recall.

## Optimization policy for SemCred-FL

SemCred-FL development may search its declared validation-only parameters, including
semantic/numerical fusion, disagreement penalty, trust decrease/recovery, retention,
clipping, attribution frequency, and prior fusion. Each candidate is a separately
identified condition. Selection uses a predeclared composite gate: macro-F1 first,
nonzero supported-class recall, malicious false acceptance, benign false rejection,
and runtime feasibility. The test partition cannot choose a parameter, stopping
round, seed, baseline subset, figure, or manuscript claim.

The final article reports the actual ranking. If SemCred-FL is not first on every
metric, the contribution is stated narrowly—for example, a robustness/client-
assessment tradeoff—only when matched multi-seed evidence supports it.

## Execution responsibility

Implementation, lightweight synthetic tests, static checks, identity audits, report
generation, and documentation updates are performed in the repository. Long GPU
training commands are run by the project owner and their final status block is
returned for audit. Every long runner must be resumable and idempotent before it is
released.
