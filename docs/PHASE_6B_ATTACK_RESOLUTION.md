# Phase 6B train-only attack resolution

Status: complete. Phase 6B.1 real-data feature and trigger artifacts are complete for
both datasets. All eight Phase 6B.2 dataset/profile benign norm references are
complete, independently reproduced, and included in the checksum-bound closure
index. Poisoning experiments remain blocked until the Phase 6C execution gate is
implemented and tested.

## Scope

Phase 6B.1 implements the transformations fixed in the Phase 6A contract and resolves
the dataset-specific values that could not be chosen before examining the federated
training partition. It does not train an attacked model, read validation or test
features, access anchors, or support a performance claim.

The resolver consumes only the Phase 3C transformed shards and the Phase 3D manifest.
PyArrow scanner predicates restrict both resolution passes to `partition == train`.
The median pass adds a source/benign family predicate; no evaluation or anchor feature
value is returned to the process.

## Exact feature resolution

Eligible coordinates are Phase 3C `continuous_standardized` coordinates whose
training-derived clipping interval has positive width. For every eligible coordinate,
the resolver computes the exact source-family and benign medians over the complete
federated training population after anchor removal. Coordinates are ordered by
descending absolute median gap and then by coordinate name. The first three are the
clean-label poisoning coordinates.

The implementation stores source and benign values in float32 memory-mapped work
matrices. Each completed transformed shard is committed to SQLite, so interruption
does not require rescanning completed shards. Work matrices can require roughly one
to two GB of temporary disk space; they are calibration state, not portable research
inputs.

## Zero-collision trigger

The twelve highest-ranked coordinates form the trigger candidate pool. A second
train-filtered pass stores those coordinates for all federated training records.
Lower and upper Phase 3C clipping endpoints are converted to packed bit masks. The
search examines coordinate tuples of size three through six, using coordinate-name
order and lower-before-upper endpoint order. The first zero-collision tuple is frozen
in the resolution artifact. The search fails closed if no permitted tuple has zero
collisions.

Natural collisions in clean source-family test data are not inspected during
resolution. The triggered-test generator identifies them only during evaluation,
reports their count, and excludes them from the attack-success-rate denominator.

## Implemented transformations

`attacks/transformations.py` implements:

- deterministic SHA-256 row selection without replacement;
- targeted source-to-benign label flipping;
- label-preserving movement toward benign medians with Phase 3C bound projection;
- train-poisoning backdoor insertion and the triggered-test generator;
- update scaling and sign reversal;
- coalition model replacement;
- radial projection into a frozen benign update-norm interval.

The zero poison fraction returns the original feature and target tensors unchanged.
Simulation-only selection and attack metadata do not enter the defense update
contract.

## Verification

Focused tests cover deterministic row selection, exact no-op identity, label behavior,
feature bounds, trigger insertion, natural-collision exclusion, every direct-model
formula, two-sided norm projection, immutable norm-reference construction, and a
complete synthetic Parquet resolution with validation and test rows excluded by the
scanner predicate.

## Real-data command

Run one dataset at a time. The command is resumable at transformed-shard boundaries.
For Edge-IIoTset:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_phase6b_resolution.py `
  --dataset edge_iiotset `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --output-directory "artifacts\attacks\phase_6b\edge_iiotset" `
  --work-database "artifacts\attacks\phase_6b\edge_iiotset\edge_iiotset_phase6b.sqlite3"
```

For CICIoT2023, replace the dataset and the two dataset directories/names with
`cic_iot_2023`.

Do not run both dataset resolutions concurrently on the local laptop. Preserve the
final JSON and Markdown files. The `_work` directory and SQLite database may be
removed only after the final artifact has been copied and its identity recorded.

## Phase 6B.2 benign-norm calibration

The calibration uses three benign-only FedAvg development rounds under the same
dataset, Phase 3D profile, detector, and local-training settings as the later attack
runs. Each round selects 20 of the 50 clients from a pre-generated schedule. Every
selected client trains for three local epochs with batch size 128, Adam, learning rate
`1e-3`, zero proximal penalty, and the frozen `256-128-64` MLP. The 60 identified
update norms define the per-profile interquartile interval. The artifact cannot read
current-round benign updates during a later attack, validation or test records, or
anchor records.

The runner first builds a schedule-bound Parquet cache. Its scanner returns only
`partition == train` rows assigned to clients appearing in the three-round schedule.
Cache construction resumes at completed source-shard boundaries. Federated training
resumes at completed round boundaries; interruption during a round repeats that round.

Run only one dataset/profile at a time. For the first Edge-IIoTset IID reference:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase6b_norm_calibration.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --profile iid `
  --output-directory "artifacts\attacks\phase_6b\norm_references\edge_iiotset\iid" `
  --cache-directory "artifacts\client_cache\phase_6b\edge_iiotset\iid" `
  --cache-work-database "artifacts\client_cache\phase_6b\edge_iiotset\iid\client_cache.sqlite3" `
  --device auto
```

The final `benign_norm_reference.json`, federated round log, resolved configuration,
and checkpoints are evidence artifacts. After their identities and backups have been
verified, the derived client-cache directory can be removed and rebuilt later from
the preserved Phase 3C and Phase 3D artifacts. Complete all four profiles for each
dataset before Phase 6C attack execution.

### Completed real references

| Dataset | Profile | Updates | IQR lower | IQR upper | Reference ID |
|---|---|---:|---:|---:|---|
| Edge-IIoTset | IID | 60 | 22.1170015335083 | 26.909245491027832 | `7dff694c94a63e03205ab2e5b5f97ed782612eb048e6aa40c7fccff92a80ef63` |
| Edge-IIoTset | Dirichlet 0.1 | 60 | 5.352854371070862 | 19.78455686569214 | `6b970ff7694dd35c48293adcdbabd987521e7fffff5b8adcc7c70d4f44d8df34` |
| Edge-IIoTset | Dirichlet 0.3 | 60 | 10.323951959609985 | 23.033565521240234 | `e4fef6ed9575796c34262e769c7c45d7a3b929d7dd7a551fefebe759ff9f4cc2` |
| Edge-IIoTset | Dirichlet 1.0 | 60 | 18.266846656799316 | 26.66264581680298 | `771c0ff94fafd9daf3fa662249989898f017fa6910b476fcc17cbf8a174b13e9` |
| CICIoT2023 | IID | 60 | 26.881851196289062 | 31.703227519989014 | `68d1429c4bf749fc76b22d9c6db0cf3cfbcf03b4e949f04b19921d8a76896c7f` |
| CICIoT2023 | Dirichlet 0.1 | 60 | 11.823124885559082 | 26.171101570129395 | `0ed10a25ec4bef31b8453cc33a6e70a3b1385bd74f2c555c9f77e92c313adcb3` |
| CICIoT2023 | Dirichlet 0.3 | 60 | 16.245837450027466 | 28.159836292266846 | `38dc6b815916cc29ead1b19971748647a6dcc74f330f710fb6920e0b65ed8740` |
| CICIoT2023 | Dirichlet 1.0 | 60 | 20.425262928009033 | 33.077284812927246 | `1b3b45e3877194185fbd4b6f9a0be5fab22ed97226b9f2cc19bc7c706e1de30a` |

The Edge-IIoTset IID calibration artifact has calibration ID
`b29b779fe9a2652f0f9d1f6cb6033dc9c1cc3f22f3f0876cc21308057f43d64a`.
Its calibration and reference identities, 60-update population, 20-update-per-round
schedule, quartiles, and train-only boundary gates were independently recomputed on
2026-08-29. These numbers are attack-calibration inputs, not detector-performance
measurements.

The Edge-IIoTset Dirichlet-0.1 calibration has calibration ID
`081bbe528bd58530f66778985ffc7f1ffb92d59bc2d0eda99db5bdc62f86acc8`.
Its identities, schedule cardinality, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30.

The CICIoT2023 Dirichlet-1.0 calibration has calibration ID
`e15769cc9030480edf4793a242119349d582645b8811b0e0a7de0d20edc05d35`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-31.

### Phase 6B.2 closure index

The closure auditor independently reads and reconstructs all eight reference
artifacts. It verifies 480 globally unique development-update identities, exact
per-round cardinalities, file hashes, unique profile-bound scientific identities,
shared training controls, dataset-consistent model/partition identities, and every
restricted-data flag. The completed closure ID is
`1461106f84993cb974e197c40373c37d1765821881bbdf6a771242e139da32e9`.
The authoritative index is
`artifacts/attacks/phase_6b/norm_references/phase6b_norm_reference_index.json`.

The CICIoT2023 Dirichlet-0.3 calibration has calibration ID
`2c49f767916aa4569282edea591f67aa062e1f0cf4890061b2758f3e05cdc4a6`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30.

The CICIoT2023 Dirichlet-0.1 calibration has calibration ID
`21cd9e397c9b5f453f2a0625bb195b5c212d45f1a982d93659a53b621271c092`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30.

The Edge-IIoTset Dirichlet-1.0 calibration has calibration ID
`24b773110e7b4311cd7a31bb77b23c03b21c10605eca8c936409d5806d0a5c42`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30. All four Edge-IIoTset Phase 6B.2 profiles are
therefore complete; CICIoT2023 remains pending.

The CICIoT2023 IID calibration has calibration ID
`da0ad1aa38d9ab2af338b2877160c91de0bb1c7d13a5334afbf1e10300adc1de`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30.

The Edge-IIoTset Dirichlet-0.3 calibration has calibration ID
`dd1e71357e965ec1ec75d1c10187262e0b4674f8eaf7e91be5d85a0daf574b9e`.
Its identities, 60-update population, exact quartiles, and boundary gates were
independently recomputed on 2026-08-30.

The generic Phase 5 `federated_run_id` hashes the initial model, client schedule, and
server-engine controls. It does not encode the Phase 3D profile and therefore repeats
when those engine inputs are deliberately shared. It is namespaced under the
profile-specific calibration directory and must not be used as the scientific
experiment identity. Phase 6B.2 uses the profile-bound `calibration_input_id`,
`calibration_id`, and `reference_id` for that purpose.
