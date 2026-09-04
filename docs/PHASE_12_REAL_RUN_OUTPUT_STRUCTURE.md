# Phase 12 Real Clean-Matrix Output Structure

This document defines how a genuine Phase 12 Step 5 clean utility matrix should
look during execution and after completion. It also provides the checks needed to
distinguish measured experiment artifacts from the separate synthetic fixture
suite.

## Expected experiment scope

The full clean matrix contains:

- two datasets: `edge_iiotset` and `cic_iot_2023`;
- four client profiles: `iid`, `dirichlet_1_0`, `dirichlet_0_3`, and
  `dirichlet_0_1`;
- ten matched seeds: `20260816` through `20260825`;
- five aggregation methods: FedAvg, coordinate median, trimmed mean, Multi-Krum,
  and SemCred-FL;
- 100 federated rounds per method;
- three local epochs per selected client; and
- 20 participating clients per round from a population of 50 clients.

This produces 80 dataset-profile-seed conditions and 400 method runs.

## Full execution command

Run the matrix from PowerShell:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

$env:PYTHONUNBUFFERED = "1"

conda run --no-capture-output -n semcred-fl python scripts\run_phase12_step5.py `
  --datasets edge_iiotset cic_iot_2023 `
  --profiles iid dirichlet_1_0 dirichlet_0_3 dirichlet_0_1 `
  --seeds 20260816 20260817 20260818 20260819 20260820 20260821 20260822 20260823 20260824 20260825 `
  --methods fedavg coordinate_median trimmed_mean multi_krum semcred_fl `
  --device auto `
  --total-rounds 100 `
  --local-epochs 3 `
  --batch-size 128 `
  --clients-per-round 20 `
  --burn-in-rounds 3 `
  --attribution-interval 1 `
  --trim-count 4 `
  --krum-byzantine-count 4 `
  --krum-selected-count 16 `
  --output-directory "artifacts\experiments\phase_12\measured_final_clean" `
  --cache-root "artifacts\client_cache\phase_12\measured_final_clean"
```

The command is resumable. After an interruption, run the exact same command
again. Changing any scientific or output setting creates a different matrix
identity and does not resume the existing campaign.

## Terminal output while incomplete

An interrupted or deliberately bounded invocation should print a status similar
to the following:

```text
status=interrupted
matrix_input_id=<64-character hash>
conditions_total=80
conditions_complete=<0-to-79>
active=edge_iiotset/iid/seed_20260816
method=fedavg status=interrupted next_round=<next round>
method=coordinate_median status=<status> next_round=<next round>
method=trimmed_mean status=<status> next_round=<next round>
method=multi_krum status=<status> next_round=<next round>
method=semcred_fl status=<status> next_round=<next round>
campaign_directory=<absolute campaign directory>
validation_partition_accessed=completion_only
test_partition_accessed=false
performance_claims_allowed=false
```

The precise active dataset, profile, seed, method, and next round depend on where
execution stopped.

## Terminal output after completion

A completed full clean matrix should print:

```text
status=complete
matrix_input_id=<64-character hash>
conditions_total=80
matrix_result_id=<64-character hash>
json=F:\...\measured_final_clean\<matrix_input_id>\matrix_report.json
validation_partition_accessed=completion_only
test_partition_accessed=false
performance_claims_allowed=false
```

The matrix input ID printed by the command must equal the campaign directory
name and the `matrix_input_id` stored in `matrix_report.json`.

## Root directory structure

```text
measured_final_clean/
└── <matrix_input_id>/
    ├── matrix_report.json
    ├── resolved_matrix.json
    └── conditions/
        ├── edge_iiotset/
        │   ├── iid/
        │   ├── dirichlet_1_0/
        │   ├── dirichlet_0_3/
        │   └── dirichlet_0_1/
        └── cic_iot_2023/
            ├── iid/
            ├── dirichlet_1_0/
            ├── dirichlet_0_3/
            └── dirichlet_0_1/
```

Each profile directory should contain all ten matched seeds:

```text
<profile>/
├── seed_20260816/
├── seed_20260817/
├── seed_20260818/
├── seed_20260819/
├── seed_20260820/
├── seed_20260821/
├── seed_20260822/
├── seed_20260823/
├── seed_20260824/
└── seed_20260825/
```

## Condition directory structure

Every dataset-profile-seed condition should contain:

```text
seed_<seed>/
├── client_schedule.json
├── condition_report.json
├── resolved_config.json
├── run_metrics.jsonl
├── shared_updates/
├── reporting/
│   ├── macro_f1_table.csv
│   ├── macro_f1_table.tex
│   ├── macro_f1_plot.svg
│   ├── macro_f1_artifact_manifest.json
│   ├── metric_summaries.json
│   └── metric_provenance.json
└── methods/
    ├── fedavg/
    ├── coordinate_median/
    ├── trimmed_mean/
    ├── multi_krum/
    └── semcred_fl/
```

`shared_updates/` contains the cached client updates used to give compatible
methods the same client schedule, initialization, optimizer settings, and local
training evidence.

## Method directory structure

Each method directory should contain its final report, resume state, round audit
records, runtime records, and model checkpoints:

```text
<method>/
├── method_report.json
├── latest_checkpoint.json
├── checkpoints/
│   ├── round-0001.pt
│   └── ...
├── rounds/
│   ├── round-0000.json
│   └── ...
└── runtime/
    ├── round-0000.json
    └── ...
```

Round filenames can reflect the implementation's checkpoint and event numbering
conventions. The authoritative completion field is `rounds_completed` in
`method_report.json`; it must equal 100 for every method in the full run.

## Required method report fields

A completed method report should have this general structure:

```json
{
  "schema": "semcred_phase12_step5_method_result_v1",
  "status": "complete",
  "method": "semcred_fl",
  "method_run_id": "<64-character hash>",
  "method_result_id": "<64-character hash>",
  "final_parameters_sha256": "<64-character hash>",
  "rounds_completed": 100,
  "validation_metrics": {
    "records": 0,
    "accuracy": 0.0,
    "balanced_accuracy": 0.0,
    "macro_precision": 0.0,
    "macro_recall": 0.0,
    "macro_f1": 0.0,
    "macro_false_positive_rate": 0.0,
    "worst_class_recall": 0.0,
    "per_class_precision": [],
    "per_class_recall": [],
    "per_class_f1": [],
    "per_class_false_positive_rate": [],
    "confusion_matrix": [],
    "support": []
  },
  "gates": {
    "clean_training_only": true,
    "shared_initialization_schedule_and_optimizer": true,
    "validation_partition_accessed": true,
    "test_partition_accessed": false,
    "performance_claims_allowed": false
  }
}
```

The numeric zeros and empty arrays above are structural examples only. A genuine
completed report contains values calculated from the experiment's validation
predictions.

## Required matrix completion checks

After completion, run:

```powershell
$root = "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\experiments\phase_12\measured_final_clean"

$campaign = Get-ChildItem -LiteralPath $root -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$report = Get-Content -LiteralPath (Join-Path $campaign.FullName "matrix_report.json") -Raw |
  ConvertFrom-Json

"campaign=$($campaign.FullName)"
"status=$($report.status)"
"folder_id=$($campaign.Name)"
"matrix_input_id=$($report.matrix_input_id)"
"conditions=$($report.condition_results.Count)"
$report.gates | Format-List
```

The expected full-clean output is:

```text
status=complete
folder_id=<same value as matrix_input_id>
conditions=80
test_partition_accessed=false
```

Verify every condition and method report with:

```powershell
$conditionReports = Get-ChildItem -LiteralPath (Join-Path $campaign.FullName "conditions") `
  -Filter "condition_report.json" -File -Recurse

$methodReports = Get-ChildItem -LiteralPath (Join-Path $campaign.FullName "conditions") `
  -Filter "method_report.json" -File -Recurse

"condition_reports=$($conditionReports.Count)"
"method_reports=$($methodReports.Count)"

$incomplete = foreach ($file in $methodReports) {
  $item = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  if ($item.status -ne "complete" -or $item.rounds_completed -ne 100) {
    [PSCustomObject]@{
      Path = $file.FullName
      Status = $item.status
      Rounds = $item.rounds_completed
    }
  }
}

$incomplete | Format-Table -AutoSize
```

Expected counts are 80 condition reports and 400 method reports. The final
`$incomplete` table should be empty.

## Measured-versus-synthetic distinction

A genuine Step 5 measured campaign has:

- `matrix_report.json` with schema
  `semcred_phase12_step5_clean_matrix_result_v1`;
- `resolved_matrix.json`;
- nested dataset, profile, and seed condition directories;
- real `.pt` model checkpoints;
- per-round JSON audit records;
- method reports derived from validation predictions; and
- no synthetic provenance fields.

The separate synthetic fixture suite has indicators such as:

```text
suite_manifest.json
export_guard.json
result_origin=synthetic_fixture
synthetic_fixture=true
measured_experiment=false
real_dataset_accessed=false
performance_claims_allowed=false
manuscript_export_allowed=false
```

Renaming or copying a synthetic directory into `measured_final_clean` does not
change its provenance. Synthetic fixture values cannot be used as measured
experimental evidence.

## Scientific boundary after Step 5

Completing this clean matrix does not open the final test partition and does not
automatically authorize manuscript performance claims. The measured clean matrix
must first pass its identity, class-coverage, fairness, and leakage checks. The
attacked-condition matrix, ablations, sensitivity studies, efficiency analysis,
matched-seed statistical analysis, registry freeze, and final test evaluation are
separate subsequent gates.
