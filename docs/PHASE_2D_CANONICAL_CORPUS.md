# Phase 2D: canonical corpus construction

Status: implementation, synthetic validation, and both full dataset builds complete.

Phase 2D converts each audited raw dataset into a portable Parquet corpus without
fitting any learned preprocessing state. It applies the feature rules frozen after
Phase 2C, constructs fingerprints from canonical feature values, resolves duplicate
components over the complete dataset, and writes one deterministic representative
for every eligible component.

## Decision rules

The builder applies these rules in order:

1. Validate the exact source schema and explicit label map.
2. Canonicalize retained numerical and categorical values under the frozen dataset
   policy. Rows with malformed retained values are rejected and counted by reason.
3. Construct the model-feature fingerprint after canonicalization. Grouping fields,
   labels, and record identity do not enter the fingerprint.
4. Exclude the complete fingerprint component if it contains more than one primary
   family.
5. Retain a same-family component with multiple fine labels for the primary task,
   mark it `fine_label_ambiguous=true`, and exclude it from later fine-label analysis.
6. For every other duplicate component, retain the lowest
   `(source_file, source_row)` representative.

The output keeps canonical scalar values only. Scaling, clipping, imputation, and
categorical vocabulary fitting occur after the group-disjoint split and use only the
federated training partition.

## Resumability and artifacts

The command uses two passes over the source CSVs. The indexing pass stores fingerprint
counts, label masks, source-file membership, and representative coordinates in
SQLite. The emission pass rescans each source file and writes its eligible
representatives to a Zstandard-compressed Parquet shard. Each source file is one
transactional checkpoint in both passes. If a run stops, rerunning the same command
skips completed files; an interrupted file is repeated from its beginning.

The output directory contains:

- `<dataset>_canonical.sqlite3`: resumable work state and duplicate-component index;
- `shards/*.parquet`: portable canonical records;
- `<dataset>_canonical_schema.json`: ordered fields and retained feature types;
- `<dataset>_canonical.json`: provenance, counts, shard hashes, and corpus ID;
- `<dataset>_canonical.md`: concise audit summary.

The final gate requires the number of emitted rows to equal the number of eligible
canonical fingerprints. Source-file size and modification time are checked before a
completed checkpoint is reused, and a build identity mismatch stops the run.

## PowerShell commands

Run these commands from the canonical `Code` directory. They are full two-pass scans
and should be executed by the researcher locally or in Colab, not as a quick check.

Edge-IIoTset:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_canonical_corpus.py `
  --dataset edge_iiotset `
  --root "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\Datasets\Edge-IIoTset\Edge-IIoTset dataset" `
  --archive-sha256 "42DF619D4A222C5AE014A3B8AD2017536DBF49FBC2D35145349E6012A65B74AE" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\edge_iiotset" `
  --progress-every 1
```

CICIoT2023:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_canonical_corpus.py `
  --dataset cic_iot_2023 `
  --root "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\Datasets\CICIoT2023\Downloads\CSV\CSV" `
  --archive-sha256 "B18C8B58CE8C2EE5F058EB72FFAD1F2F9EE40AFE89B6D3B53A8F3189551CFE5D" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\cic_iot_2023" `
  --progress-every 1
```

The work database is created in the selected output directory by default. Do not
delete it until Phase 2D and the subsequent split-manifest phase have passed their
integrity checks.

## Full-corpus evidence

- Edge-IIoTset corpus ID:
  `ea83ce6fa2f0ad01a673792dffb29ba91f7b399fbaf5afef6d98e162f11effe9`;
  15,017,480 canonical records in 24 shards.
- CICIoT2023 corpus ID:
  `4d2185287b18333ccb89a31851c90ff5588864de0ccff18662074d8a76021f97`;
  20,594,360 canonical records in 309 shards.

The manifests, schemas, shard sizes and hashes, record totals, and resume databases
were checked after each completed build. These are corpus-construction results, not
model-performance results.

## Verified implementation evidence

Synthetic end-to-end tests cover cross-family exclusion, within-family fine-label
ambiguity, duplicate collapse after numerical canonicalization, invalid retained
values, dropped Edge-IIoTset fields, source-file membership counts, deterministic
representatives, resume behavior, and source-identity mismatch. These are
implementation tests, not model-performance results.
