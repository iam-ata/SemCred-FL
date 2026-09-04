# Phase 3B split and anchor manifests

Status: full assignments validated for Edge-IIoTset and CICIoT2023.

## Frozen contract

Phase 3B operates only on the Phase 2D canonical corpora. The root seed is
`20260816`. The target proportions are 70% training, 15% validation, and 15% test.
For every primary family, 32 calibration anchors and 32 scoring anchors are selected
from the training partition. The two anchor subsets are disjoint and are removed from
the later federated client pool.

The construction order is fixed:

1. verify the canonical manifest and every shard used by the run;
2. assign each canonical record to one partition under the dataset policy;
3. verify primary-family coverage in every partition;
4. select disjoint calibration and scoring anchors from training records;
5. write immutable assignment, anchor, and summary manifests;
6. fit preprocessing only in the next phase, after anchors have been removed.

No model is trained in Phase 3B.

## Dataset-specific policies

CICIoT2023 uses seeded record-level hash thresholds after Phase 2D conflict exclusion
and canonical deduplication. Its 309 provider files were combined and shuffled by the
provider, and no defensible capture/session identity is available. The resulting
evaluation is record-level and cannot support a capture- or session-generalization
claim.

Edge-IIoTset uses each raw source capture as one group except MITM. The MITM records
use the Phase 3A.1 PCAP-derived, direction-normalized bidirectional flow. These groups
subdivide one capture and are not independent sessions. Whole-group assignment means
that realized class-specific proportions can differ substantially from 70/15/15 when
a family has only three or four large source groups. The implementation reports those
proportions and never breaks a group merely to improve the ratio.

## Validated plans

| Dataset | Plan ID | Policy | State |
|---|---|---|---|
| CICIoT2023 | `459997fb13595092a904ec6a3d08dfe4966a49d65e7c1c1486fab08f67e81a13` | `fingerprint_deduplicated_record_hash_v1` | full split validated |
| Edge-IIoTset | `205406da53c1848385666fc9ed7ac6a7d1e700640d0f6fd65713e66561d32d88` | `source_capture_plus_mitm_pcap_bidirectional_flow_v1` | full split validated |

The Edge plan contains 70 groups. Its frozen group-assignment hash is
`647eb0b9a8814fa745781c20ecd38221de085e05771b23cc34abb588eace4c82`.
Every primary family has at least one planned group in training, validation, and test.

## Completed Edge-IIoTset split

Split ID: `b6616d2030f682906d205b0a105234a78a86ec995588fd4bbfe100ba2b6ca886`.

The full builder assigned 15,017,480 records in 24 Parquet shards. Training contains
10,122,299 records (67.40%), validation contains 1,624,189 (10.82%), and test contains
3,270,992 (21.78%). These realized proportions preserve all frozen groups. The split
contains 384 unique anchors: 192 calibration and 192 scoring records, with 64 total
anchors for each of the six primary families. Removing them leaves 10,121,915 records
for the future federated client pool.

All recorded primary structural gates passed. Fine labels are not eligible for
confirmatory evaluation because several source-specific fine labels occur in only one
partition. Preprocessing and client partitioning remain incomplete.

## Completed CICIoT2023 split

Split ID: `590fb686703ef03c30314499629f13d8b083f24f45348f3cd289198abcac99ca`.

The full builder assigned 20,594,360 records in 309 Parquet shards. Training contains
14,415,182 records (69.996%), validation contains 3,088,931 (14.999%), and test
contains 3,090,247 (15.005%). The split contains 512 unique anchors: 256 calibration
and 256 scoring records, with 64 total anchors for each of the eight primary families.
Removing them leaves 14,414,670 records for the future federated client pool.

All recorded primary structural gates passed, and all 34 fine labels occur in each
partition. CICIoT2023 still supports only record-level evaluation over
provider-shuffled feature windows; the split does not establish capture- or
session-generalization. Preprocessing and client partitioning remain incomplete.

## Outputs and resumability

Each full run writes:

- one assignment Parquet file per canonical shard;
- `anchors.parquet` and `anchor_manifest.json`;
- a JSON and Markdown split report;
- a SQLite resume database.

Completed shards are checksum-verified on resume. The resume database is bound to the
plan ID and canonical shard inventory. Re-running the same command skips completed
shards. Changing the seed, policy, canonical corpus, or shard inventory requires a new
output directory and resume database.

The split report records target and realized proportions, family and fine-label
counts, assignment checksums, derived seeds, anchor identity, and all scientific
gates. `preprocessing_fitted` and `client_partition_created` remain false because
those are later phases.

## Full-run commands

Run from the `Code` directory. These commands scan tens of millions of records and
should be run locally only when adequate disk space is available, or in Colab after
copying the canonical Phase 2D artifacts and this repository.

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_phase3b_split.py `
  --dataset cic_iot_2023 `
  --canonical-manifest "artifacts\canonical\phase_2d\cic_iot_2023\cic_iot_2023_canonical.json" `
  --canonical-database "artifacts\canonical\phase_2d\cic_iot_2023\cic_iot_2023_canonical.sqlite3" `
  --output-directory "artifacts\splits\phase_3b\cic_iot_2023" `
  --progress-every 10
```

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_phase3b_split.py `
  --dataset edge_iiotset `
  --canonical-manifest "artifacts\canonical\phase_2d\edge_iiotset\edge_iiotset_canonical.json" `
  --canonical-database "artifacts\canonical\phase_2d\edge_iiotset\edge_iiotset_canonical.sqlite3" `
  --phase3a-database "artifacts\split_feasibility\phase_3a\edge_iiotset\edge_iiotset_phase3a.sqlite3" `
  --phase3a1-report "artifacts\split_feasibility\phase_3a1\edge_iiotset\edge_iiotset_phase3a1.json" `
  --edge-raw-mitm-csv "Datasets\Edge-IIoTset\Edge-IIoTset dataset\Attack traffic\MITM_attack.csv" `
  --edge-mitm-pcap "Datasets\Edge-IIoTset\Edge-IIoTset dataset\Attack traffic\MITM (ARP spoofing + DNS) Attack.pcap" `
  --output-directory "artifacts\splits\phase_3b\edge_iiotset" `
  --progress-every 1
```

For Colab, translate the paths to the mounted Drive location and run the same script.
The original multi-gigabyte raw datasets are not required for CICIoT2023 Phase 3B;
the Phase 2D canonical directory is required. Edge additionally needs the Phase 3A
database, the small Phase 3A.1 JSON report, and the MITM CSV and PCAP listed above.

## Completion gate

Both full commands have returned stable split IDs and all primary structural gates are
true. Phase 3B is therefore complete. Its outputs are approved inputs to preprocessing
and client partitioning, not detector-performance results.
