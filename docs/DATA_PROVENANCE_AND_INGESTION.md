# Dataset provenance and ingestion decision record

Status: both Phase 2D canonical corpora and both Phase 3A full audits complete;
dataset-specific split-policy refinement is in progress.

## Bottom line

The official CICIoT2023 shuffled CSV distribution and the Edge-IIoTset raw sources
are present, and both complete Phase 2B/2C audits have been verified. Neither source
is approved for confirmatory experiments yet. The locally computed CIC ZIP SHA-256 is
`b18c8b58ce8c2ee5f058eb72ffad1f2f9ee40afe89b6d3b53a8f3189551cfe5d`;
it identifies the downloaded bytes but has not been matched to an independently
published provider checksum. The resumable canonical-corpus builder passed synthetic
conflict, duplicate, invalid-value, and resume tests. The completed Edge-IIoTset
corpus contains 15,017,480 records in 24 shards; the CICIoT2023 corpus contains
20,594,360 records in 309 shards. The completed Phase 3A audits show that none of
the initial candidate policies satisfies the necessary three-way grouping condition
for every primary family. Phase 3A.1 therefore refines the evaluation unit without
claiming unavailable capture or session independence.

| Dataset | Local evidence | Intended use now | Confirmatory blocker |
|---|---|---|---|
| CICIoT2023 | Official 2,141,834,171-byte `CSV.zip`; completed 20,594,360-record canonical corpus | Deduplicated provider-shuffled CSV evaluation; derivative splits remain smoke-only | Freeze a record-level split policy and prohibit capture-generalization claims |
| Edge-IIoTset | 1,746,605,436-byte ZIP; completed 15,017,480-record canonical corpus with raw PCAPs present | Refine the primary-family grouping policy; ML subset remains smoke-only | Validate a source-file/run-block policy, including the single MITM capture |

The complete hashes and paths are stored in `configs/datasets/`.

## Evidence and provenance

### CICIoT2023

The official Canadian Institute for Cybersecurity page describes 33 attacks in
seven attack families and distributes PCAP, CSV, examples, and supplementary
material through a registration form. The official `CSV.zip` is now available at
`Datasets/CICIoT2023/Downloads/CSV.zip`. Its extracted source tree contains 309
non-empty CSV files in 34 label directories. All files share one 39-column numeric
feature header and have no embedded label column. The adapter therefore derives the
fine label from a strict source-directory map, preserves the original CSV identity
as provenance, and fails on any unknown directory. The provider README states that
the 309 CSV files constitute a combined and shuffled dataset. Their filenames and
row order are therefore not treated as capture identities or temporal boundaries.

The earlier derivative ZIP remains present and contains only:

- `CICIOT23/train/train.csv`
- `CICIOT23/validation/validation.csv`
- `CICIOT23/test/test.csv`

Its paths, 47-column schema, and published split shapes match the Kaggle derivative
maintained by `himadri07`, but the ZIP contains no manifest proving that origin.
The origin is therefore recorded as likely, not verified. The Kaggle page's MIT
metadata is not accepted as the license for the original CIC data. The public CIC
page and download form do not state an explicit dataset license; this remains an
access/terms item for the researcher to retain at download time.

A previous full local scan tied to the recorded ZIP hash found:

| Check | Measured result |
|---|---:|
| Duplicate rows in supplied train split | 134,565 (2.4502%) |
| Duplicate rows in supplied validation split | 6,297 |
| Duplicate rows in supplied test split | 6,384 |
| Validation unique rows also present in train | 57,744 (4.9330%) |
| Test unique rows also present in train | 57,847 (4.9422%) |
| Test unique rows also present in validation | 12,422 (1.0613%) |

These derivative splits must not be repaired and relabeled as a confirmatory
benchmark. The new adapter instead creates stable record IDs from dataset ID,
official relative source path, and original one-based row number before filtering.

### Edge-IIoTset

The local author notice grants academic research use in perpetuity and requires
author permission for commercial use. The included IEEE Access paper describes 14
attacks in five threat families, 61 selected features, and the two labels
`Attack_label` and `Attack_type`. The raw and selected CSV files have 63 columns:
61 predictors plus those labels.

The 157,800-row ML file contains the same 15 fine-grained values defined in the
dataset paper and is almost entirely contained in the DNN file (99.9968% in the
previous local comparison). It is therefore a smoke-test subset, not an independent
experiment corpus. The DNN file is useful for adapter development but is not a
predefined holdout.

The raw CSV timestamps are not reliable calendar timestamps in every file: most
retain only year and time-of-day, and some files contain numeric values such as
`0.0`. A split must not depend on timestamp alone. Candidate grouping evidence is
the source file, recoverable capture/run boundaries, unordered endpoint pair,
contiguous time block within a source file, and exact-record fingerprint components.
The group derivation must be versioned and tested before this source becomes
confirmatory eligible.

The complete raw-source audit verified that `OS_Fingerprinting_attack.csv` contains
1,001 rows whose provider label is `OS_Fingerprinting`. The ingestion contract maps
that exact raw spelling to the canonical secondary label `Fingerprinting`; both map
to the primary family `information_gathering`. This is an explicit source alias, not
an inferred relabeling.

## Frozen ingestion rules

1. Process the datasets independently. Never share fitted statistics, categorical
   vocabularies, feature groups, semantic priors, or anchor records across datasets.
2. Preserve raw files. Ingestion writes new manifests and processed artifacts; it
   never edits the source archives or extracted files.
3. Add `source_file` and any recoverable capture/scenario metadata before cleaning.
   Create a stable record ID from dataset ID, source identity, and source row index.
4. Normalize label spelling only through the explicit maps in the dataset YAML
   files. Unknown labels fail ingestion. The primary target is published attack
   family plus benign; fine-grained labels are retained for secondary outcomes.
5. Remove invalid labels and non-finite mandatory values with logged counts. Exact
   duplicate records belong to one fingerprint component and may not cross splits.
6. Create train, validation, and test assignments before fitting any transformation.
   Split assignment is group-disjoint and deterministic from the root seed.
7. Draw calibration and scoring anchors only from the federated training partition.
   Their record IDs must be disjoint from one another, all clients, validation, and
   test.
8. Fit categorical vocabularies, clipping bounds, imputation values, and scaling
   statistics on the remaining federated training records only. Apply the frozen
   preprocessor to every other partition without refitting.
9. Fields used for grouping or likely to expose device, scenario, endpoint, payload,
   URI, topic, or attack tooling are excluded from model inputs. The Edge-IIoTset
   exclusion list is explicit in its YAML. CIC exclusions will be finalized after
   the original distribution is present.
10. Client partitioning occurs after the global split and anchor reservation. No
    record is copied between clients. Every artifact records source hashes, config
    hash, schema version, label counts, group counts, and record-ID checksums.

## Remaining source verification

Before confirmatory work, retain any access/license terms shown by the CIC download
form. Preserve the completed canonical manifests and databases, complete Phase 3A.1,
freeze the dataset-specific evaluation-unit policies and split fractions, and then
construct and verify immutable split manifests. The CIC protocol must identify its
input as provider-shuffled feature windows and must not describe a record-level split
as capture-disjoint. The Edge protocol must distinguish capture-level groups from
any within-capture time-block or flow surrogate.

The hash was computed by the researcher because reading the full archive is a long
operation. No full dataset scan was run during Phase 2A.

## Phase 2A implementation evidence

- CICIoT2023 bounded smoke check: 309 files discovered; three files and nine rows
  sampled; no invalid rows in the bounded sample; schema ID
  `19b3e0f0217c3eb4`.
- Edge-IIoTset bounded smoke check: 24 raw files discovered; three files and nine
  rows sampled; no invalid rows in the bounded sample; schema ID
  `43dec6c0ee3a522c`.
- Edge source fields used for endpoints, time, payload, URI, topic, or attack
  tooling remain outside model features. Mixed decimal/hexadecimal retained fields
  remain explicitly type-unresolved until Phase 2B rather than being silently cast.

## Manuscript consequences

The manuscript currently states the correct leakage-safe protocol and does not claim
results. Before submission, the dataset subsection should name the exact source
archives, hashes, access dates, and terms; state that the CIC derivative splits were
excluded from confirmatory evaluation; and describe the final group key for each
dataset. No performance table or result language should be added until the frozen
pipeline and experiments have run.
