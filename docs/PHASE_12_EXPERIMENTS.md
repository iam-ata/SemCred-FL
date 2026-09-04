# Phase 12: experiments and article artifacts

Status: Steps 1 and 2 completed on 2026-09-01; Steps 3 and 4 completed on
2026-09-02. The Step 5 clean-utility runner is implementation-ready. Completed
results remain bounded, non-claim-bearing development evidence. No confirmatory
experiment has started.

## Step 1 purpose

The smoke gate checks that the implemented components can execute as one pipeline
before a long dataset run is authorized. It uses four generated binary clients,
three federated rounds, one sign-reversal event, one scoring anchor, and a separate
generated test fixture. Round 0 is burn-in. Semantic attribution is scheduled in
rounds 0 and 2, while the numerical/history score is evaluated in every round.

This fixture exercises:

- deterministic local training and update validation;
- attack realization outside the defense input contract;
- semantic and numerical credibility scoring;
- the SemCred-FL trust, eligibility, clipping, aggregation, and history transition;
- final classification and malicious-client assessment metrics; and
- Phase 11 JSONL aggregation plus CSV, LaTeX, SVG, and provenance regeneration.

The generated data are intentionally easy and are not a dataset surrogate. In
particular, the observed synthetic macro-F1 is an implementation check, not an
article result, model-selection signal, or performance claim.

## Frozen controls and identities

The runner stores every scientific control in `stable_config` inside the immutable
report. Local output paths and wall-clock timings are excluded from the content
identity. The completed canonical run has:

- seed: `20260816`;
- smoke run ID: `7e0677ef6191a4e551ab2240df3980ee4ce781ff0890f58aab05225ab94e849c`;
- result ID: `ef0e0c978df408c3726e3265754946fd32805ef60f50f0840fe11738ac9a5ad3`;
- summary bundle ID: `f1aec867ceb3ddd68a0ddb30e54318ef51c6ded993d150e6b88f160a019a0ef9`;
- macro-F1 artifact ID: `ef44f9d3b3c76819ab3c0cff029e3f70041912e1558763abf3a490d377a859d5`.

The report records `synthetic_only=true`, `real_test_partition_accessed=false`,
and `performance_claims_allowed=false`. Malicious identity is joined to trust and
eligibility only after the defense decision. Attack metadata is not passed to
semantic scoring, numerical scoring, trust update, or aggregation.

## Run command

The canonical command is lightweight and completed in about 16 seconds on the
development PC. Re-running it is idempotent: a matching completed report is read
without recomputation. The smoke run is atomic rather than round-checkpointed;
because it is small, an interruption is recovered by running the same command.

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_smoke.py `
  --output-directory "artifacts\experiments\phase_12\smoke"
```

Canonical artifacts are under:

`artifacts/experiments/phase_12/smoke/7e0677ef6191a4e551ab2240df3980ee4ce781ff0890f58aab05225ab94e849c/`

The directory contains the immutable smoke report, exact-schema run metrics,
metric summaries, provenance, and generated macro-F1 table/plot artifacts.

## Verification

- 14 focused Phase 12 tests passed.
- Focused experiment-module coverage is 97.24%.
- The complete repository suite passed: 472 tests.
- Ruff lint passed, and all 150 Python files are formatted.
- Strict mypy passed for 83 library modules and the Phase 12 runner.

An optional all-scripts mypy invocation also inspects older utility scripts and
currently requires the `types-PyYAML` environment stub for two pre-existing YAML
validators. This does not affect the smoke runner or its runtime evidence.

## Step 2: clean Edge-IIoTset comparison

The Step 2 runner freezes a non-claim pilot before the larger experiment matrices:

- Edge-IIoTset, IID profile, seed `20260816`;
- 50 clients, with 10 selected per round for 3 rounds;
- one local epoch, batch size 256, and learning rate `1e-3`;
- detector widths 256, 128, and 64 with dropout 0.2; and
- FedAvg and SemCred-FL with the same initialization, schedule, local optimizer,
  cached training records, and completion-only validation evaluation.

All records belonging to a selected client are used. The reduced round and local-
epoch counts define this pilot and must not be reported as the full 100-round,
20-client-per-round, 3-local-epoch protocol. The final test partition is never
loaded. The resulting validation metrics remain non-claim-bearing until the full
preregistered matrices and evidence audit are complete.

### Accelerated execution

The default CUDA path combines fused Adam, prevalidated immutable cache batches,
and one concatenated host-to-device transfer per client. The batch boundaries and
optimizer-step order remain unchanged. A shared content-addressed update cache can
reuse work across the two methods only when the dataset cache, detector, local
configuration, round, client, batch-delivery contract, and global-parameter SHA-256
are identical. Method checkpoints store the model, SemCred-FL client state, round
cursor, and latest-round identity. Interrupted execution resumes by rerunning the
same command; already verified updates and complete rounds are reused.

The staged benchmark used all 202,438 cached training records for Edge-IIoTset
client 00 on the NVIDIA GeForce RTX 3050 6GB Laptop GPU. Streamed standard Adam took
11.9102 seconds. Materialized fused Adam took 5.3253 seconds, a measured 2.2365x
training hot-path speedup. The streamed-prevalidated and materialized-standard
variants retained the baseline delta SHA-256 exactly. The benchmark ID is
`85b4405be873a0103d3902dd2cf8c555aefe20e6477024d2eb89ea65ef70c0e2`.
It accessed training data only and cannot support a detection-performance claim.

### PC command

This is the heavy command. Run it from PowerShell and paste its final output back
into the project log:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step2.py --device cuda
```

If execution is interrupted, rerun the exact command. To stop cleanly after at most
one new round per incomplete method, add `--max-rounds-this-call 1`; subsequent
invocations continue from the recorded cursor. `--standard-adam` and
`--stream-batches` are diagnostic controls and are not part of the accelerated
canonical pilot identity.

Artifacts are written under
`artifacts/experiments/phase_12/step_2_clean_edge_iid/`. The canonical run completed
with these identities:

- comparison input ID:
  `5702c65031340ab0a441e4f4e8d88a08f666b42e88c8d4f44cdfbace9cc3dad6`;
- comparison result ID:
  `4fe709aa51000eace41b957881ddba7ce4a9b2a0da8d38796170b50414ddc0be`;
- summary bundle ID:
  `d4f748a9ab080366a748fa321f749e0ef91e58b482ee60765ed12fa233575865`;
- macro-F1 artifact ID:
  `8105cab160b72531e8c96a6356b2ee39edde0811925a2adb3d7fed59cf07801d`.

Both method reports completed three rounds and their content identities recompute
exactly. The comparison gates record completion-only validation access, no test
access, and no authorization for performance claims.

### Pilot observations

FedAvg produced validation accuracy 0.8533, macro-F1 0.2469, balanced accuracy
0.2326, and worst-class recall 0. SemCred-FL produced validation accuracy 0.6878,
macro-F1 0.2279, balanced accuracy 0.1963, and worst-class recall 0. These values
describe one bounded three-round integration pilot. They do not estimate final
method performance and do not justify a claim that either method is better.

The confusion matrices show zero recall for classes 2 through 5 under both methods.
FedAvg emitted only class labels 0 and 1; SemCred-FL also emitted label 2, but none
of those predictions were correct. This is consistent with an undertrained pilot
under severe six-class imbalance, including one validation class with only 19
records. Attack experiments may use the same bounded settings only as implementation
gates. The full clean and robustness matrices require the preregistered training
budget and must demonstrate nonzero minority-class behavior before manuscript
performance claims are enabled.

### Step 2 verification

- 36 Step 2 and local-training focused tests passed; the Step 2 module has 92%
  branch-aware coverage.
- Ruff lint and formatting checks passed for the changed files.
- Strict mypy passed for the runner, benchmark, experiment module, federation client,
  and experiment exports.
- The real one-client speed benchmark accessed neither validation nor test data.

## Step 3: label-flip non-IID comparison

Step 3 freezes one bounded attacked condition:

- Edge-IIoTset with the `dirichlet_0_3` client partition;
- label flipping from the Phase 6B source family to its resolved target family;
- malicious-client ratio 0.2 and poison fraction 0.3;
- seed `20260816`, 50 clients, 10 selected per round, and 3 rounds;
- one local epoch, batch size 256, learning rate `1e-3`; and
- FedAvg and SemCred-FL under the same initialization, schedule, local optimizer,
  attack realization, and completion-only validation protocol.

The deterministic schedule contains 27 unique participating clients. Step 3 builds
a schedule-bound cache for those clients instead of rescanning or materializing all
50 client partitions. The cache build is resumable at source-shard boundaries.
Content-addressed local updates are reused across methods only when the attacked
selection, malicious status, client cache, local configuration, and global-model
parameter hash are identical. Each completed round has a checksum-verified method
checkpoint, so the same command resumes after interruption.

Simulation truth remains outside `DefenseUpdate`. The runner joins malicious-client
identity to trust and eligibility only after the SemCred-FL decision, then computes
malicious-client AUROC/AUPRC, benign false rejection, malicious false acceptance,
and final observed trust separation. It evaluates classification on validation only
after a method completes. The test partition is never loaded.

### Material passport

- Sources: the Phase 3C Edge-IIoTset preprocessed corpus, Phase 3D
  `dirichlet_0_3` assignments, Phase 6B attack resolution, and Phase 7 semantic
  prior bundle.
- Transformations: deterministic malicious coalition and round schedule,
  deterministic source-record selection, source-to-target label replacement during
  local batch delivery, local training, FedAvg/SemCred-FL aggregation, and
  completion-only validation evaluation.
- Outputs: immutable attack selection evidence, per-round audit records, update and
  method checkpoints, method reports, exact-schema metric JSONL, summary/provenance
  bundle, and macro-F1 CSV/LaTeX/SVG artifacts.
- Limitations: one attack, one non-IID profile, one seed, and a three-round pilot;
  outputs are implementation evidence and cannot support manuscript performance or
  method-ranking claims.

### PC command

This is a heavy, resumable command. Its default paths were checked against the
existing Phase 3C, 3D, 6B, and 7 artifacts:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step3.py --device cuda
```

Rerun the same command after an interruption. To limit one invocation to at most one
new round per incomplete method, add `--max-rounds-this-call 1`. Do not add
`--standard-adam` or `--stream-batches` to the canonical accelerated run.

Artifacts are written under
`artifacts/experiments/phase_12/step_3_label_flip_edge_dirichlet_0_3/`; the
schedule-bound client cache is under
`artifacts/client_cache/phase_12/step_3/edge_iiotset/dirichlet_0_3/`.

### Step 3 implementation verification

- 31 Step 3 tests passed, including interruption/resume, cross-method update reuse,
  corrupt-cache rejection, and post-decision attack-truth isolation.
- Focused Step 3 branch-aware coverage is 91%.
- An 84-test regression suite passed for Step 2, Step 3, local training, attack
  transformations, and Phase 6C contracts/execution.
- Ruff lint and formatting checks passed for the source, CLI, export, and tests.
- Strict mypy passed for the three Step 3 production source/CLI files. PyArrow's
  test fixture package lacks type metadata, so test-only PyArrow imports are outside
  that strict production gate.

### Completed run identities

- comparison input ID:
  `af5d78ce54da6a699b45a63dd276159b62d0d651c8e50ec522939b3d7b60d371`;
- attack ID:
  `bd2c7ab59e4f56b316abb9c56d4577551c5401e8cc6e585b2656a34d815e3d0b`;
- realization ID:
  `bbc7f6cb36d502f493879c38ae614afaebcbedc3897fcacd18e33eaa8ac0d5a9`;
- comparison result ID:
  `3c6ada958211a1b8cfde56d7e787573c50214a0b4ca36a8501ae4e20abc20a67`;
- summary bundle ID:
  `e4d567d5f5eec886eaf318e0bb267956a02fcaf74be0de77ac5ae8758f28bb45`;
- macro-F1 artifact ID:
  `79d0099a60e134c35c6ba878a5a9818b38de5835a2ed00f9908a7815d42d58c0`.

The comparison identity recomputes exactly. Both methods completed three rounds,
all six round records are present, and the metric stream contains 20 exact-schema
records. Of 60 local-update requests across both methods, 50 were trained and 10
were reused under identical inputs. The attack manifest contains five active
malicious clients and 1,714 selected source records. Every round has exactly two
active malicious clients. All three serialized SemCred-FL defense payloads were
checked and contain no malicious labels, poison metadata, or selection identities.

### Pilot observations

FedAvg produced validation accuracy 0.7571, macro-F1 0.2366, balanced accuracy
0.2121, and worst-class recall 0. SemCred-FL produced validation accuracy 0.7642,
macro-F1 0.2447, balanced accuracy 0.2194, and worst-class recall 0. The numerical
differences describe one seed and cannot establish method superiority.

Both methods again had zero recall for classes 2 through 5. SemCred-FL's
post-decision client assessment produced AUROC 0.5833 and AUPRC 0.3731 over 30
participation events. Benign false rejection was 4/24 (0.1667), while malicious
false acceptance was 4/6 (0.6667). The final observed mean and median trust gaps
were 0.1700 and 0.0912. These values indicate that this short pilot did not reliably
exclude malicious participants. They are diagnostic evidence for the experiment
pipeline, not article performance results.

The clean Step 2 values must not be subtracted from these values to estimate attack
impact because Step 2 used IID clients while Step 3 used `dirichlet_0_3`. Attack
effects require matched clean and attacked conditions in the later matrices.

Step 3 is closed. The same no-test-tuning and non-claim boundary remains in force.

## Step 4: reduced standard-baseline matrix

Step 4 extends the exact completed Step 3 condition with the three remaining
standard parameter-update controls:

- coordinate-wise median;
- coordinate-wise trimmed mean with two values removed from each tail; and
- Multi-Krum with Byzantine bound two and eight selected updates.

Together with the immutable Step 3 FedAvg and SemCred-FL results, the matrix has
five methods. It inherits the Edge-IIoTset `dirichlet_0_3` partition, label-flip
realization, malicious ratio 0.2, poison fraction 0.3, seed `20260816`, detector,
client schedule, local optimizer, and validation protocol. The Step 3 reports are
consumed as prerequisites and are not retrained or overwritten.

The local-update identity remains tied to the full Step 3 scientific input. Step 4
first checks its own shared cache and then the Step 3 shared cache. An update is
reused only when the client, round, attack selection, cache, detector, optimizer,
batch-delivery contract, and exact global-parameter SHA-256 match. New baseline
updates are stored under Step 4. Each method has independent checksum-verified round
checkpoints. Attack truth is joined only after aggregation for audit records.

### Step 4 PC command

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step4.py --device cuda
```

The command is resumable. Rerun it unchanged after interruption. To advance by at
most one new round per incomplete baseline, add `--max-rounds-this-call 1`. The
canonical accelerated run must not include `--standard-adam` or `--stream-batches`.
Outputs are written to
`artifacts/experiments/phase_12/step_4_reduced_baselines/`.

### Step 4 implementation verification

- Nine focused tests pass, including interruption/resume, Step 3 prerequisite
  enforcement, five-method assembly, exact-update reuse, and malformed-artifact
  rejection.
- The new module has 94% branch-aware coverage.
- A 145-test focused regression suite passes across the baseline cores, Steps 2-4,
  local training, attack transformations, and Phase 6C contracts/execution.
- Ruff lint/format and strict mypy pass for the production module, CLI, and exports.
- Every default artifact path exists. No validation or test data were read by the
  path/CLI checks; synthetic tests remain non-claim-bearing.

### Step 4 completed result

Step 4 completed with matrix result ID
`bca0fb435639192f3ee25d74fbb566f911a73099c0cb33cf13f5c740621efce3`.
The stored matrix identity recomputes exactly. It contains 35 scalar metric records,
nine newly executed baseline round records, 50 newly trained updates, and 40
exact-input update reuses. No defense-facing round payload contains malicious labels
or attack metadata.

Observed validation macro-F1 was 0.244747 for SemCred-FL, 0.237371 for coordinate
median, 0.237251 for trimmed mean, 0.237133 for Multi-Krum, and 0.236565 for FedAvg.
Every method had zero worst-class recall. These values describe one seed with a
three-round integration budget; they do not select a winner or support a manuscript
performance claim. The complete 11-item statistical fallacy scan is in
`docs/PHASE_12_STEP4_VALIDATION.md`.

## Step 5: full-budget clean utility development matrix

Step 5 uses 50 clients, 20 submissions per round, 100 rounds, three local epochs,
Adam at `1e-3`, and the 256/128/64 detector. The original batch-size candidate was
128; the validation-only runtime calibration below provisionally selects 1024.
FedAvg, coordinate median, trimmed mean, Multi-Krum, and
SemCred-FL share the initialization, schedule, local optimizer, and cached client
records within each dataset/profile/seed condition. The initial standard-baseline
budget is frozen to the primary malicious-ratio design: trimmed mean removes four
coordinates per tail and Multi-Krum uses `f=4` with 16 selected updates.

The registry covers Edge-IIoTset and CICIoT2023 independently under IID and
Dirichlet 1.0, 0.3, and 0.1 profiles. CICIoT2023 remains secondary and
non-confirmatory while its original-dataset license provenance is unresolved. A
single seed is a development gate; it cannot authorize paired inference.

### Step 5 PC command

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step5.py --device cuda --batch-size 1024
```

The same command resumes without repeating completed work.
`--max-rounds-this-call 1` advances each method in the active condition by at most
one round;
`--max-conditions-this-call 1` stops after one newly completed condition. Dataset,
profile, and seed subsets can be selected with `--datasets`, `--profiles`, and
`--seeds`. Changing scientific controls creates a different content-addressed
campaign and does not overwrite the canonical matrix.

The runner builds a selected-client cache once for each distinct
dataset/profile/client set and reuses it across seeds. Local updates are reused
across methods only when the client, round, cache, optimizer, and exact global-model
parameter hash match. Fused CUDA Adam and one-transfer materialized client batches
are enabled by default. Every method has checksum-verified per-round checkpoints.
Validation is loaded only after that method reaches round 100; the test partition is
never loaded.

### Step 5 implementation verification

- Twenty-six focused tests pass, including interruption/resume, completed-campaign
  reuse, content-identity failures, malformed-artifact rejection, and bounded
  condition execution.
- The Step 5 module has 94% branch-aware coverage.
- A 232-test regression suite passes across the Phase 10 baselines, Phase 12 Steps
  1-5, client caching, and federation components.
- Ruff and strict mypy pass for the production module, command-line entry point,
  exports, and related tests.
- All six required preprocessing, client-partition, and semantic-prior manifests
  exist in the active project. CUDA is available through PyTorch 2.12.0+cu130 on
  the NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Campaign directories include the matrix input ID. A changed scientific control
  therefore cannot collide with an earlier condition, while an unchanged command
  resumes the same checksum-verified campaign.

These checks exercised synthetic fixtures and artifact paths only. They did not
start the full training matrix, open validation data from the real datasets, or
access the test partition.

### Step 5 bounded execution checkpoint

The first real-data checkpoint for Edge-IIoTset `dirichlet_0_3`, seed `20260816`,
has matrix input ID
`23c9708743fcd3811833c3a165c8ea5166e72d9b9789412c0291123be1a0b12d`.
All five methods completed round 0 and report `next_round=1`. Each checkpoint file
matches the SHA-256 stored in its pointer. FedAvg trained the 20 selected client
updates; coordinate median, trimmed mean, Multi-Krum, and SemCred-FL reused those
80 method-level requests only because their round-zero update inputs and broadcast
parameters matched exactly. The bounded `interrupted` status is expected and means
the requested one-round limit was reached.

Validation remains completion-only, the test partition has not been accessed, and
performance claims remain disabled. No utility metric is available until a method
completes the frozen 100-round condition.

### Step 5 batch-size runtime calibration

Three isolated Edge-IIoTset `dirichlet_0_3` round-zero campaigns used the same
seed, schedule, client population, local epochs, optimizer, model, and hardware.
FedAvg trained the 20 client updates once in each campaign; the other methods reused
the exact round-zero updates within that campaign.

| Batch size | FedAvg round-zero time | Relative to batch 128 |
|---:|---:|---:|
| 128 | 520.849 s | 1.00x |
| 1024 | 91.337 s | 5.70x faster |
| 2048 | 99.476 s | 5.24x faster |

Every method checkpoint hash matched its pointer. Batch 2048 was 8.9% slower than
1024 on the measured hardware, so 1024 is the provisional runtime choice. This is
not yet a final scientific setting: a completion-based learning diagnostic must
show acceptable macro-F1 and minority-class recall before the manuscript or
confirmatory registry adopts it. No validation or test record was read during the
runtime calibration.

### Step 5 batch-1024 learning diagnostic

The separate ten-round Edge-IIoTset `dirichlet_0_3` diagnostic completed under
matrix input ID
`30b612d0be41ce33ba7ab744c7cdfb957b855cd32228339ebefda19bbdc79a66`
and matrix result ID
`591183ecafb0b821c2cb88a54f541fce5543f389325e6108ab9bce410e272f6f`.
It used seed `20260816`, three local epochs, and batch size 1024. The matrix,
condition, and five method result identities recompute, all method checkpoints are
valid at round 10, and the expected 35 metric records are present. Validation was
read at method completion; the test partition remained sealed.

Observed macro-F1 was 0.237373 for Multi-Krum, 0.236252 for coordinate median,
0.236244 for trimmed mean, 0.234934 for SemCred-FL, and 0.230274 for FedAvg.
All five methods had zero recall for information gathering and man-in-the-middle.
Injection and malware recall remained below 0.004 for every method. The diagnostic
therefore fails the declared class-coverage gate despite completing without a
numerical or artifact-integrity failure. Accuracy values from 0.646126 to 0.696849
are not treated as sufficient because they are dominated by the two largest
classes.

The next one-factor development candidate retains batch size 1024 and redistributes
the same short-run local-epoch budget from 10 rounds × 3 local epochs to 30 rounds ×
1 local epoch. It runs only FedAvg and SemCred-FL because all five methods shared
the zero-recall failure and the immediate question concerns the common learner.
The runner still evaluates and records all six classes; the development gate
focuses on nonzero recall for classes 2 and 5. This tests more frequent aggregation
under non-IID clients before a 100-round campaign is authorized. The full audit and
statistical-validity review are recorded in
`docs/PHASE_12_STEP5_BATCH1024_VALIDATION.md`.

The targeted synchronization diagnostic subsequently completed under matrix result
ID `3e3c0261013eace2b25822e62b40e6e819b0cde2b78d1dc98ebf8f2766f820b0`.
FedAvg and SemCred-FL macro-F1 values were 0.236391 and 0.229974. Both retained
zero recall for information gathering and man-in-the-middle, and both had zero
injection recall. The matrix, condition, and two method identities recompute; both
round-30 checkpoints and all 14 metric records pass. The changed synchronization
schedule therefore fails the targeted gate and is not selected.

No additional federated schedule is run at this stage. A one-epoch, full-corpus
centralized control with the same detector, preprocessing, weighted loss, batch
size, and seed will determine whether the failure exists before aggregation. This
is a validation-only diagnostic and cannot support a method-ranking claim.
