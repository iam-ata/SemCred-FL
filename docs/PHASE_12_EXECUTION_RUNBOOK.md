# Phase 12 measured execution runbook

This runbook covers the implemented measured clean-matrix workflow. It does not
edit files under `Manuscript`, open the final test partition, or convert synthetic
fixtures into measured evidence.

## Frozen defaults

The default registry contains both datasets, all four client-partition profiles,
ten matched seeds (`20260816` through `20260825`), and five methods: FedAvg,
coordinate median, trimmed mean, Multi-Krum, and SemCred-FL. Each method uses 50
clients, 20 client submissions per round, 100 rounds, three local epochs, and batch
size 128. The device is selected automatically; CUDA is used when PyTorch reports
it as available.

The full SemCred-FL aggregation and credibility configuration is part of every
condition identity. A change to fusion, disagreement, trust, retention, clipping,
history, or scoring coefficients therefore creates a different campaign instead
of reusing incompatible checkpoints.

## 1. Read-only plan

Run this first. It validates every resolved condition and checks the prepared
Phase 3C, Phase 3D, and Phase 7 artifacts without creating a campaign directory.

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py plan
```

The canonical plan is ready only when `status=ready`, `conditions_total=80`,
`method_runs_total=400`, and `missing_artifacts=0`.

## 2. Start or resume the measured run

The complete run is intentionally long:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py run
```

For controlled PC sessions, advance one round of one condition per invocation:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py run `
  --max-conditions-this-call 1 `
  --max-rounds-this-call 1
```

Both commands are resumable. Repeating the same command reuses the campaign ID,
validated client-update cache, method checkpoints, and immutable round records.
Do not change scientific arguments when resuming a campaign.

To require CUDA instead of automatic selection, add `--device cuda`. The command
then fails before training if CUDA is unavailable.

## 3. Audit a completed campaign

Use the campaign directory printed by the plan or run command:

```powershell
$campaign = "artifacts\experiments\phase_12\measured_final_clean\<matrix_input_id>"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py audit `
  --campaign-directory $campaign
```

The audit rejects reduced registries by default. It also rejects missing method
reports, partial round histories, broken record chains, mismatched identities,
noncanonical budgets, and any artifact that reports test access or permits
performance claims prematurely.

## 4. Generate reporting artifacts

After the audit passes:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py finalize `
  --campaign-directory $campaign
```

The finalizer derives metric summaries and paired SemCred-FL-versus-baseline exact
Wilcoxon-Holm results from the immutable method reports. It writes CSV, LaTeX, SVG,
JSON, and a content-addressed final manifest under the campaign directory. Running
it again is idempotent.

## 5. Colab fail-safe

Build a new fail-safe archive after code changes so the current runner, configs,
tests, prepared corpora, client partitions, attack resolutions, semantic priors,
and notebook are packaged together:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_colab_bundle.py
```

Upload `artifacts\colab_bundle\semcred_fl_colab_bundle.zip` and its `.sha256`
file, then follow `notebooks\00_colab_failsafe.ipynb`. The archive excludes raw
provider datasets and SQLite resume databases. Copy the active campaign and cache
directories separately when migrating an in-progress training run.

## Remaining scientific gates

Clean-matrix completion is not whole-project experimental completion. The full
attacked-condition matrix, trust analysis, required ablations, sensitivity and
anchor-contamination study, efficiency profiling, registry freeze, and one final
test evaluation remain distinct Phase 12 gates. Their required scope is frozen in
`docs\PHASE_12_COMPLETION_PLAN.md`; until those measured runs and audits exist,
manuscript performance claims remain disabled.
