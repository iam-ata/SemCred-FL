# Phase 3C leakage-safe preprocessing

Phase 3C fits one immutable preprocessing object per dataset and applies it to every
Phase 3B partition. It does not construct federated clients or train a detector. The
Edge-IIoTset and CICIoT2023 full runs are complete and verified. Neither
preprocessing artifact is a detector-performance result.

## Frozen fit population and controls

The fit population is the Phase 3B training partition after removal of both trusted
anchor subsets. Validation, test, calibration-anchor, and scoring-anchor records cannot
contribute to a clipping bound, mean, standard deviation, or categorical vocabulary.
The builder verifies the Phase 2D corpus ID, Phase 3B split ID, anchor ID, root seed,
file checksums, record order, anchor-record digest, and training-pool cardinality before
accepting a result.

The continuous-feature procedure is:

1. derive a dataset-specific sampling seed from root seed `20260816`;
2. select a deterministic SHA-256 threshold sample targeting 1,000,000 non-anchor
   training records;
3. estimate the 0.001 and 0.999 quantiles with NumPy's linear quantile method;
4. clip every non-anchor training record to those bounds;
5. compute the exact population mean and population standard deviation over the full
   clipped fit population; and
6. use scale `1.0` when a standard deviation is at most `1e-12`.

CICIoT2023 has 39 continuous input coordinates. Edge-IIoTset has 38 continuous source
features and four categorical source features. Each Edge vocabulary contains only
categories observed in the non-anchor training pool plus a dedicated `__UNKNOWN__`
coordinate. All one-hot coordinates derived from one categorical source feature share
one semantic-group identifier. The final Edge input dimension is known only after the
training-only vocabularies have been fitted.

Every transformed Parquet row contains `record_id`, its final partition, integer family
and fine-label identifiers, the fine-label ambiguity flag, and ordered `float32` model
coordinates named `x0000`, `x0001`, and so on. The five possible partitions are
`train`, `validation`, `test`, `anchor_calibration`, and `anchor_scoring`.

## Frozen real-data plans

| Dataset | Phase 3C plan ID | Fit records | Shards | Status |
|---|---|---:|---:|---|
| Edge-IIoTset | `a0d9418906b3b6a80f9ba2f9b99afc231c32ee0752569de5201772eb0ddb3046` | 10,121,915 | 24 | Full run verified |
| CICIoT2023 | `910eac34dc77f7de6d051a12c4ac3149d031f789d648c37cddac385154aa797a` | 14,414,670 | 309 | Full run verified |

The completed Edge corpus has preprocessed corpus ID
`a9a1cdb3ec5e5ec8846be236432a1e0c478b170478e3e581dc7441356c86941a`
and preprocessor ID
`c175c802c8c30049f0ddfcba72b44c1c020e52e7472b691b5488dc9d842bc7ff`.
Its deterministic threshold sample contains 998,889 records. The fitted schema has
38 standardized continuous coordinates and 18 categorical one-hot coordinates, for
a model input dimension of 56. All 24 transformed files passed size and SHA-256
verification, contain 15,017,480 rows under one Parquet schema, and total 652,452,670
bytes.

The completed CIC corpus has preprocessed corpus ID
`8bb930472baf37a4a0b66e479c99aa3701e8025bb654046d95a53cf7af9c76ce`
and preprocessor ID
`e39b411b302da0c97ef766c0d8c9e36caa4c718e65f702f40e78705fde497e8a`.
Its deterministic threshold sample contains 1,000,399 records. The fitted schema has
39 standardized continuous coordinates and no categorical coordinates. All 309
transformed files passed size and SHA-256 verification, contain 20,594,360 rows under
one Parquet schema, and total 1,159,467,637 bytes.

The plan-only command reads the JSON manifests but does not scan the Parquet corpora.
The full builder performs three resumable stages: quantile sampling, exact training
statistics, and transformation. Completion is recorded per shard in SQLite. Re-running
the same command verifies completed files and continues from the first incomplete
shard. Changing an identity-bearing input or fit control requires a new output
directory and resume database.

## Edge-IIoTset full run

This command produced the verified Edge artifact above. Re-running it with unchanged
inputs performs resume-integrity checks and returns the same stable identities.

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

conda run --no-capture-output -n semcred-fl python scripts\build_phase3c_preprocessing.py `
  --dataset edge_iiotset `
  --canonical-manifest "artifacts\canonical\phase_2d\edge_iiotset\edge_iiotset_canonical.json" `
  --split-manifest "artifacts\splits\phase_3b\edge_iiotset\edge_iiotset_phase3b_split.json" `
  --anchor-manifest "artifacts\splits\phase_3b\edge_iiotset\anchor_manifest.json" `
  --output-directory "artifacts\preprocessing\phase_3c\edge_iiotset" `
  --work-database "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c.sqlite3" `
  --progress-every 1
```

## CICIoT2023 full run

This command produced the verified CICIoT2023 artifact above. Re-running it with
unchanged inputs performs resume-integrity checks and returns the same stable
identities.

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

conda run --no-capture-output -n semcred-fl python scripts\build_phase3c_preprocessing.py `
  --dataset cic_iot_2023 `
  --canonical-manifest "artifacts\canonical\phase_2d\cic_iot_2023\cic_iot_2023_canonical.json" `
  --split-manifest "artifacts\splits\phase_3b\cic_iot_2023\cic_iot_2023_phase3b_split.json" `
  --anchor-manifest "artifacts\splits\phase_3b\cic_iot_2023\anchor_manifest.json" `
  --output-directory "artifacts\preprocessing\phase_3c\cic_iot_2023" `
  --work-database "artifacts\preprocessing\phase_3c\cic_iot_2023\cic_iot_2023_phase3c.sqlite3" `
  --progress-every 10
```

Do not delete the Phase 2D canonical shards, Phase 3B assignment shards, anchors, or
their manifests until Phase 3C has completed and its checksums have been backed up.
After successful Phase 3C verification, the transformed shards plus `preprocessor.json`,
the Phase 3C report, Phase 2D/3B manifests, and required checksummed inputs form the
portable package for Phase 3D and Colab. The raw provider CSV files are not read by
Phase 3C.

## Completion evidence

A successful command prints the preprocessed corpus ID, preprocessor ID, output
directory, and resume-database path. The JSON report must show exact agreement with
the Phase 3B partition counts, a fit cardinality equal to the non-anchor training
population, finite transformed coordinates, and `preprocessing_fitted: true`.
`client_partition_created` must remain false until Phase 3D.
