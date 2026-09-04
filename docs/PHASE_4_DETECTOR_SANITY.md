# Phase 4 detector and centralized sanity baseline

Status: detector implementation and Phase 4 gates complete; bounded compatibility
runs complete for both datasets; full-corpus centralized runs not executed.

## Scope

Phase 4 establishes the detector, objective, evaluation metrics, artifact-bound data
loader, and checkpoint path required by the federated implementation. The
centralized runner is a pipeline sanity check. It uses training data for optimization
and validation data for evaluation, never reads the test partition, and always writes
`performance_claims_allowed=false`.

This phase does not evaluate SemCred-FL, an attack, or a federated baseline. Its
metric values are not manuscript results.

## Detector and objective

The default detector is

```text
input(d)
-> Linear(d, 256) -> LayerNorm -> GELU -> Dropout(0.2)
-> Linear(256, 128) -> LayerNorm -> GELU -> Dropout(0.2)
-> Linear(128, 64) -> LayerNorm -> GELU -> Dropout(0.2)
-> Linear(64, C)
```

Training consumes logits. Softmax is exposed only for requested probabilities. For
a training population with `N` records, observed class set `C+`, and class count
`n_c`, the implemented class weight is

```text
weight_c = min(10, N / (|C+| * n_c))  when n_c > 0
weight_c = 0                           when n_c = 0
```

The weighted cross-entropy is the record mean
`-(1/N) sum_i weight[y_i] log p(y_i|x_i)`. This differs from a reduction that divides
by the sum of selected weights. The local objective also supports the optional
proximal term `mu/2 * ||theta - theta_reference||^2` needed by later phases.

## Data and leakage boundary

The loader verifies the Phase 3C corpus, preprocessing, split, and anchor identities;
the Phase 3D client-partition identity; feature dimension and class count; manifest
cardinalities; shard sizes; and recorded quality gates. It streams selected Parquet
partitions and does not load a full corpus into memory.

Class weights for the centralized sanity runner use the complete non-anchor Phase 3D
training-population counts. Optimization batches come only from the Phase 3C training
partition. Evaluation batches come only from validation. The code contains no test
evaluation path.

A bounded run reads a deterministic prefix of the Phase 3C manifest. Shards and rows
are not globally class-shuffled, so a prefix can omit declared classes. Every report
therefore records class support and the `validation_all_classes_observed` gate. A
failed class-coverage gate makes all reported metrics wiring evidence only.

## Metrics and records

The metric module derives the following values from one declared-class confusion
matrix:

- accuracy;
- macro precision, recall, and F1;
- balanced accuracy;
- worst-class recall;
- per-class and macro false-positive rate;
- per-class support, precision, recall, and F1.

Classes absent from an evaluated subset remain in the declared coordinate system and
receive zero recall and F1. This keeps metric dimensions stable and makes incomplete
coverage visible.

Each run stores a resolved configuration, one JSON-lines epoch record per completed
epoch, an atomic PyTorch checkpoint with a SHA-256 hash, environment versions, source
artifact identities, finite-value gates, class support, and explicit test-access and
claim flags. An existing final report with the same deterministic run identity is
returned without retraining.

## Verification evidence

The focused Phase 4 suite has 48 passing tests. The tests cover exact loss arithmetic,
class-weight capping and absent classes, optional proximal loss, known confusion-matrix
fixtures, streaming metrics, checkpoint round-trip prediction identity, Phase 3C and
Phase 3D contract rejection, bounded partition streaming, deterministic training,
loss decrease on a learnable fixture, and test-partition exclusion.

Two one-epoch GPU compatibility runs used 4,096 training and 2,048 validation records:

| Dataset | Sanity ID | Checkpoint SHA-256 | All validation classes observed |
|---|---|---|---|
| Edge-IIoTset | `648adc7d0fc58c6b75c5af2b5078f4e6b44de4cde4e89198809e0ed712104815` | `f7747620bc65b6766d6b2bb5d7aa977bca6260f2173b30cbb451049bcddb51f6` | no |
| CICIoT2023 | `aa508408f94e0da382c36bbcb02aa6a4fa1491234332aaca702af9bd28514524` | `b21df16159aa8c8f2a0fa447776e5a28072cc17480abe38bb731af2d254a993f` | no |

Both runs passed finite-loss, finite-parameter, and checkpoint gates. Both set
`test_partition_accessed=false` and `performance_claims_allowed=false`. Their metric
values must not be cited as detector performance because the bounded validation
prefixes do not contain every declared class.

## Commands

The default command is bounded to 100,000 training and 50,000 validation records.
Replace the two manifest paths and dataset output directory as needed:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase4_centralized.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --output-directory "artifacts\centralized\phase_4\edge_iiotset" `
  --device cuda
```

The complete training and validation partitions are enabled only by the explicit
`--full-corpus` option. That run is time-consuming and is suitable for the Colab
environment. It is still a centralized sanity baseline, not a SemCred-FL result:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase4_centralized.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --output-directory "artifacts\centralized\phase_4\edge_iiotset_full" `
  --epochs 3 `
  --batch-size 256 `
  --device cuda `
  --full-corpus
```

The next implementation phase is Phase 5: deterministic client training, synchronous
round execution, update serialization and validation, FedAvg, coordinate median,
round logs, and resumable checkpoints.
