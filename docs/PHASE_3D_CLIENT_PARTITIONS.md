# Phase 3D client-partition manifests

Status: complete for Edge-IIoTset and CICIoT2023. Both final manifests passed their
record-conservation, quota, exclusion, client-range, and minimum-size gates.

## Frozen contract

Phase 3D allocates only the Phase 3C `train` records remaining after calibration and
scoring anchors are removed. Validation, test, calibration-anchor, and scoring-anchor
records are rejected by the builder and cannot enter a client partition. Allocation
uses the primary attack family; fine labels remain secondary metadata and do not
control the partition.

The client population contains 50 clients. Each accepted profile must give every
client at least 128 records, equal to one configured local batch. The frozen profiles
are:

- an IID family-balanced control;
- Dirichlet family allocation with `alpha = 0.1`;
- Dirichlet family allocation with `alpha = 0.3`;
- Dirichlet family allocation with `alpha = 1.0`.

For each family, the builder converts the sampled proportions to exact integer quotas
using largest-remainder apportionment. It then creates a deterministically shuffled
client sequence with exactly those quotas. A profile that violates the minimum client
size is retried with a seed derived from the root seed, profile, and attempt number.
The accepted attempt is stored in the plan and final manifest.

The full build scans each transformed Phase 3C shard once. It writes one compact row
for every eligible training record with the following columns:

- `record_id`;
- `family_id`;
- `iid_client_id`;
- `dirichlet_0_1_client_id`;
- `dirichlet_0_3_client_id`;
- `dirichlet_1_0_client_id`.

Transformed feature vectors remain in the Phase 3C shards and are not duplicated for
each client or profile. Training code will join or filter by `record_id` and the chosen
client column.

## Validation gates

Before planning or writing assignments, the builder verifies the Phase 3C corpus,
preprocessor, Phase 3B split, and anchor identities. It checksum-validates the input
inventory and anchor file, confirms the recorded family counts, and checks that the
eligible population contains only non-anchor training records.

The final manifest is produced only when all of the following are exact:

- every eligible record occurs in one output assignment row;
- each profile assigns every row to one client in `[0, 49]`;
- the realized client-by-family matrix equals the planned quota matrix;
- every client contains at least 128 records;
- output file sizes and SHA-256 hashes are recorded;
- input and output record counts are conserved.

Each profile records client totals, normalized label entropy, total-variation distance
from the global family distribution, and the number of observed families per client.
These values describe the prepared partition and are not detector-performance results.

## Frozen real-data plans

The metadata-only planner reads the completed manifests and family counts but does not
scan or rewrite the transformed corpus.

| Dataset | Plan ID | Eligible records | Accepted attempts: IID / 0.1 / 0.3 / 1.0 |
|---|---|---:|---|
| Edge-IIoTset | `8560b9cb63fa1391e3a349242bbd368f328da383ef4a3119c10132e050b4361c` | 10,121,915 | 0 / 1 / 0 / 0 |
| CICIoT2023 | `6c0788417a6fb5ceb4dc29f168245398716337140e3a72e05852f432be3e7138` | 14,414,670 | 0 / 0 / 0 / 0 |

| Dataset/profile | Minimum client size | Maximum client size | Mean normalized entropy | Mean total variation | Minimum observed families |
|---|---:|---:|---:|---:|---:|
| Edge IID | 202,437 | 202,440 | 0.553653 | 0.000006 | 6 |
| Edge Dirichlet 0.1 | 295 | 1,579,132 | 0.245782 | 0.571906 | 1 |
| Edge Dirichlet 0.3 | 1,867 | 1,034,089 | 0.375368 | 0.410194 | 4 |
| Edge Dirichlet 1.0 | 23,503 | 532,715 | 0.452114 | 0.297133 | 5 |
| CIC IID | 288,292 | 288,297 | 0.596522 | 0.000005 | 8 |
| CIC Dirichlet 0.1 | 326 | 1,874,534 | 0.275522 | 0.618266 | 3 |
| CIC Dirichlet 0.3 | 22,369 | 1,609,321 | 0.400612 | 0.496096 | 6 |
| CIC Dirichlet 1.0 | 54,463 | 737,915 | 0.547750 | 0.264665 | 8 |

The Edge `alpha = 0.1` candidate at attempt 0 did not meet the minimum. Attempt 1 is
the first accepted deterministic allocation and remains above the gate with a minimum
of 295 records.

## Completed assignment inventories

| Dataset | Client-partition ID | Shards | Assignment rows | Bytes |
|---|---|---:|---:|---:|
| Edge-IIoTset | `88ac4670d265e0e1b60f0e843c443e0d648a87dc9571b0c633ec52d0f40f67f8` | 24 | 10,121,915 | 376,688,181 |
| CICIoT2023 | `25e8770e51b12afc057d8e2346cdc0ef298a9a5574e377ee64a3882b6e79a0ed` | 309 | 14,414,670 | 543,579,503 |

Every manifest-listed file exists with its recorded byte size. The builders computed
and stored a SHA-256 digest for every assignment shard. Each profile exactly matches
its planned client-by-family quota matrix and assigns every eligible record once.
Neither build initialized or trained a detector.

## Full Edge-IIoTset build

Run from the canonical `Code` directory:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\build_phase3d_clients.py `
  --dataset edge_iiotset `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --split-manifest "artifacts\splits\phase_3b\edge_iiotset\edge_iiotset_phase3b_split.json" `
  --anchor-manifest "artifacts\splits\phase_3b\edge_iiotset\anchor_manifest.json" `
  --output-directory "artifacts\client_partitions\phase_3d\edge_iiotset" `
  --work-database "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d.sqlite3" `
  --base-seed 20260816 `
  --num-clients 50 `
  --minimum-samples-per-client 128 `
  --dirichlet-alpha 0.1 `
  --dirichlet-alpha 0.3 `
  --dirichlet-alpha 1.0 `
  --progress-every 1
```

## Full CICIoT2023 build

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl python scripts\build_phase3d_clients.py `
  --dataset cic_iot_2023 `
  --preprocessing-report "artifacts\preprocessing\phase_3c\cic_iot_2023\cic_iot_2023_phase3c_preprocessed.json" `
  --split-manifest "artifacts\splits\phase_3b\cic_iot_2023\cic_iot_2023_phase3b_split.json" `
  --anchor-manifest "artifacts\splits\phase_3b\cic_iot_2023\anchor_manifest.json" `
  --output-directory "artifacts\client_partitions\phase_3d\cic_iot_2023" `
  --work-database "artifacts\client_partitions\phase_3d\cic_iot_2023\cic_iot_2023_phase3d.sqlite3" `
  --base-seed 20260816 `
  --num-clients 50 `
  --minimum-samples-per-client 128 `
  --dirichlet-alpha 0.1 `
  --dirichlet-alpha 0.3 `
  --dirichlet-alpha 1.0 `
  --progress-every 10
```

Both commands are resumable at the input-shard boundary. A rerun verifies completed
assignment files against the SQLite record and continues from the first incomplete
shard. Do not delete the work database or completed `assignments` directory while a
run is incomplete. If a recorded output file is missing or its hash has changed, the
builder stops instead of silently replacing it.

A successful run prints `client_partition_id`, `output_directory`, and
`resume_database`. It also writes `<dataset>_phase3d_clients.json` and
`<dataset>_phase3d_clients.md`. No detector is initialized or trained by these
commands.
