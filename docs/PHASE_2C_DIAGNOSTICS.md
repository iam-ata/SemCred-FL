# Phase 2C: conflict and raw-domain diagnostics

Status: complete; evidence verified and preparation policies frozen.

## Purpose

Phase 2B established that exact model-feature duplicates are common in both datasets
and that some fingerprints occur with more than one fine label. Phase 2C performs the
additional scan needed to distinguish two cases:

- the same model-feature fingerprint occurs under different primary attack families;
- the fingerprint has different fine labels that map to the same primary family.

This distinction is required because the primary experiment predicts attack family
plus benign, while fine-label evaluation is secondary. Phase 2C does not silently
delete, merge, or relabel either case.

For Edge-IIoTset, the scan also stores exact raw values for declared categorical
fields, the Phase 2B constant column, and low-cardinality fields with text
representations. For high-cardinality text-bearing fields, only exceptional
text, missing, and non-finite tokens are stored. The complete counts remain in the
resume database; the JSON report includes bounded summaries.

The included Edge-IIoTset paper describes 61 extracted predictors and identifies
their protocol-level data types in Table VII. Its published preparation procedure
uses duplicate removal, dummy variables, one-hot encoding, standardization, a random
train/test split, and SMOTE. Those steps are evidence about the source authors'
pipeline, not instructions for SemCred-FL. In particular, random row splitting and
pre-split oversampling are not adopted because they do not satisfy this project's
duplicate-component and group-disjoint evaluation requirements.

## CICIoT2023 command

Run from the canonical `Code` directory in PowerShell:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

conda run --no-capture-output -n semcred-fl python scripts\analyze_phase2c_sources.py `
  --dataset cic_iot_2023 `
  --root "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\Datasets\CICIoT2023\Downloads\CSV\CSV" `
  --audit-json "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2b\cic_iot_2023\cic_iot_2023_audit.json" `
  --audit-database "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2b\cic_iot_2023\cic_iot_2023_audit.sqlite3" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2c\cic_iot_2023" `
  --progress-every 1
```

## Edge-IIoTset command

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

conda run --no-capture-output -n semcred-fl python scripts\analyze_phase2c_sources.py `
  --dataset edge_iiotset `
  --root "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\Datasets\Edge-IIoTset\Edge-IIoTset dataset" `
  --audit-json "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2b\edge_iiotset\edge_iiotset_audit.json" `
  --audit-database "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2b\edge_iiotset\edge_iiotset_audit.sqlite3" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\data_audit\phase_2c\edge_iiotset" `
  --progress-every 1
```

Each command writes a JSON report, a Markdown summary, and a SQLite resume database.
Rerunning the same command skips completed files after verifying their byte size and
modification timestamp. A changed source file or a mismatched Phase 2B audit causes a
hard failure.

## Verified results

| Dataset | Valid rows | Raw unique fingerprints | Primary-conflict fingerprints | Primary-conflict occurrences | Fine-only fingerprints | Fine-only occurrences |
|---|---:|---:|---:|---:|---:|---:|
| CICIoT2023 | 46,775,660 | 21,055,707 | 461,347 | 13,807,409 | 81,553 | 513,304 |
| Edge-IIoTset | 20,939,622 | 15,235,695 | 36,289 | 135,857 | 71 | 150 |

Both JSON analysis identifiers recomputed correctly, both resume databases passed
SQLite integrity checks, and every Phase 2B conflict occurrence was resolved. The
CICIoT2023 analysis ID is
`4ca1ad75e446bb98d7a1836d07d7331564040897527a6445523855b0ad824211`;
the Edge-IIoTset analysis ID is
`c023ee61bf7d21f465e14b5b30f255ae957e244599c5ca0726a41040af1fe766`.

## Frozen preparation policy

1. Validate raw values and canonicalize decimal and hexadecimal numeric forms before
   constructing the final model-feature fingerprint. The Phase 2C raw-string counts
   are therefore evidence and lower bounds, not final processed-corpus counts.
2. Exclude the entire final fingerprint component if its records map to more than
   one primary family. Do not resolve a feature-identical contradiction by majority
   vote or source priority.
3. A component with multiple fine labels but one primary family remains eligible for
   the primary task and is excluded from secondary fine-label analysis.
4. Keep one label-consistent representative using ascending `source_file` and then
   one-based `source_row`. Split construction occurs after this global deduplication.
5. CICIoT2023 retains 39 continuous source features. Edge-IIoTset uses four closed
   categorical domains: `http.request.method`, `http.request.version`,
   `mqtt.conack.flags`, and `mqtt.protoname`. `icmp.unused` is removed as a verified
   constant; the other retained fields use finite numeric parsing.
6. Edge fields that contain misplaced IP addresses, DNS names, payload fragments, or
   other text where a numeric value is required fail row validation. `dns.qry.name.len`
   is numeric despite the provider README's dummy encoding because the dataset paper
   defines it as an unsigned integer and the observed text values are DNS names.

The next phase builds a resumable canonical corpus and reports the final invalid-row,
post-canonicalization ambiguity, and deduplication counts. It does not fit scaling,
clipping, or category vocabularies; those remain training-partition-only operations.
