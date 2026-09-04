# Phase 5 basic federated-learning engine

Status: implementation and synthetic gates complete; full real-data federated runs
not executed.

## Scope

Phase 5 supplies the deterministic synchronous engine used by later attacks,
credibility estimators, and aggregators. It implements ordinary FedAvg and
coordinate median only. SemCred-FL scoring, trust, clipping, histories, poisoning,
and final evaluation remain outside this phase.

The Phase 5 synthetic run is an implementation gate. It is not a dataset experiment
and always records `performance_claims_allowed=false`.

## Parameter and update contract

Trainable parameters are ordered by PyTorch `named_parameters()` and stored in an
immutable manifest containing each name, shape, and element count. A client update is
one detached one-dimensional tensor in this order. Flattening, reconstruction,
assignment, and update application reject any layout or cardinality mismatch.

Local training starts from the exact broadcast vector, uses the Phase 4 detector and
weighted objective, derives a client-and-round-specific seed, and verifies that every
local epoch consumes the declared client population exactly once. Its output records
mean loss, epoch timings, update norm, finite status, missing local classes, and the
declared sample count.

`ClientUpdate` may contain simulation-only attacker fields. Before validation, it is
projected to `DefenseUpdate`, which has no attacker identity or attack metadata.
Aggregation modules accept only this restricted contract.

## Client schedules and real-data streaming

The complete selected-client sequence is generated before training. Client IDs are
sorted, selection is without replacement within a round, and every round uses a
namespace-derived seed. The schedule content determines its SHA-256 identity. The
server rejects a schedule whose root seed or round count differs from the run.

The real-data adapter loads one named Phase 3D profile, verifies the assignment
inventory and its SHA-256 hashes, and pairs every assignment shard with its Phase 3C
feature shard. For a requested client it streams only non-anchor training records and
checks record IDs, family IDs, total records, and per-family counts against the Phase
3D profile matrix.

The current reference adapter scans the paired shards for each client. This favors a
simple auditable join over speed. A full real-data FL run would be expensive on the
local laptop and has not been started; it should use the Colab environment after the
real-data execution wrapper and temporary round-cache policy are frozen.

## Aggregation and invalid-update behavior

FedAvg applies

```text
weight_k = declared_sample_count_k / sum_j declared_sample_count_j
global_delta = sum_k weight_k * client_delta_k
```

Coordinate median sorts each parameter coordinate independently. With an odd number
of clients it selects the central value; with an even number it averages the two
central values. Because coordinate median is not a linear weighted sum, its audit
record contains an empty weight dictionary.

The server rejects duplicate client IDs, wrong-round or unselected submissions,
missing selected clients, nonpositive sample counts, wrong vector shapes, non-finite
values, invalid losses or norms, and a declared update norm that differs from the
submitted tensor. If no valid update remains, it applies a zero delta, leaves the
global model unchanged, and marks the round failed.

## Logs and exact resume

Each canonical `rounds.jsonl` record stores selected and accepted clients, rejection
reasons, safe client statistics, aggregation details, aggregate and global-vector
hashes, the preceding record hash, and its own SHA-256 hash. Attacker metadata and
wall-clock fields are absent.

Epoch durations are written to `timings.jsonl`; start and completion timestamps are
written to `runtime.json`. These telemetry files are not part of the deterministic
round chain.

Every completed round writes a tensor-only atomic checkpoint containing the detector,
parameter manifest, next-round cursor, run and schedule identities, and last canonical
record hash. Resume verifies the checkpoint checksum, reconstructs the model, validates
the complete log chain, and refuses to continue if the cursor or final record hash
differs. A completed run is returned without being overwritten.

## Verification evidence

The focused Phase 5 suite has 95 passing tests and 92.51% branch coverage over the
federation, aggregation, and Phase 3D client-stream modules. It includes these gates:

- flattening and reconstruction preserve predictions exactly;
- one-client FedAvg equals the local update;
- weighted FedAvg and coordinate median match analytical fixtures;
- local optimization and multi-round execution are deterministic;
- interrupted and uninterrupted runs have identical canonical round logs and final
  parameter hashes;
- attacker metadata is absent from defense and round-log contracts;
- all-invalid rounds preserve the global model;
- client-stream cardinality and family counts match the Phase 3D profile.

The tiny synthetic command is resumable and safe to run locally:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase5_synthetic.py `
  --output-directory "artifacts\federation\phase_5\synthetic" `
  --rounds 3 `
  --clients-per-round 2 `
  --aggregation fedavg `
  --device cpu
```

Verified synthetic run:

- run ID: `6cd53c069e3aac23859dfb0425bd0bcdf53f3cd98bbe684a86251675da94390b`;
- three rounds, two selected clients from a four-client population;
- final parameter SHA-256:
  `0f7f866fd7c2e113b06731cb167d64b2358b3ef5d77229133abd4e1bc23cc6e9`;
- test partition accessed: no;
- performance claims allowed: no.

## Next phase

Phase 6 defines poisoning interfaces and attack audit metadata. Before confirmatory
attack implementation, the open source/target labels, clean-label perturbation rule,
backdoor trigger, coordinated objective, and direct model-poisoning transformations
must be resolved from the manuscript evidence and baseline papers rather than guessed.
