# Phase 6C train-only attack execution

Status: the execution wrapper, typed/YAML contract, CLI, Colab recovery cell, and
synthetic no-op/resume gates are implemented. No full real-data Phase 6C condition
has been completed yet, so this phase currently provides no performance result.

## Scope and evidence boundary

Phase 6C generates attacked client updates from the prepared Phase 3C/3D training
artifacts. It does not read validation, test, or anchor records. The output is an
attack-generation artifact for later defense experiments; it does not authorize
accuracy, F1, ASR, robustness, or comparative claims.

Each condition is bound to the dataset, Phase 3D profile, attack family, malicious
ratio, seed, attack parameters, Phase 6B resolution, optional norm reference,
full-profile client cache, fixed coalition, and exact client schedule. The experiment
input identity is also included in the underlying federated-run identity.

## Frozen execution contract

- 50 total clients; 20 clients selected per round; 100 rounds.
- Three local epochs, batch size 128, Adam learning rate `0.001`.
- Detector hidden widths 256/128/64 with dropout 0.2.
- Malicious ratios 0.1, 0.2, 0.3, and 0.4 produce fixed coalitions of 5, 10, 15,
  and 20 clients.
- Every round contains exactly 2, 4, 6, or 8 malicious participants, respectively.
  The malicious clients therefore remain a strict minority in every round.
- Clean and attacked conditions with the same ratio and seed share the same
  coalition and schedule.
- Poison fractions are selected once over each malicious client's complete eligible
  source-family population. The selected record IDs are then reused in every local
  epoch. Mini-batch boundaries cannot change the selected fraction.

The implemented families are `clean`, `label_flip`, `clean_label`, `backdoor`,
`scaling`, `sign_reversal`, `model_replacement`, and `adaptive_norm`. Scaling submits
`s Delta`; sign reversal submits `-s Delta`. Coalition model replacement forms the
declared-sample-weighted malicious target model and submits
`(m / c) (theta_star - theta_t)` from each of the `c` active malicious clients, where
`m=20`. Adaptive norm begins with the targeted label-flip objective and radially
projects the submitted update into the frozen Phase 6B.2 profile-specific IQR.

Simulation-only malicious labels and attack metadata are written to separate round
audit files. The server projects every accepted client object to `DefenseUpdate`
before aggregation, removing those fields.

## Resumability

The full-profile cache resumes by completed transformed shard. Federated execution
resumes only from completed round checkpoints. An interruption inside a round repeats
that round. Poison selections and completed attack-round audits are identity-checked
when reused. The same output directory must never be reused for a different
condition.

## First bounded PC gate

Run from the authoritative `Code` directory. This command builds or reuses the shared
Edge-IIoTset IID cache and completes at most one clean round:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase6c_attack.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --resolution "artifacts\attacks\phase_6b\edge_iiotset\edge_iiotset_phase6b_resolution.json" `
  --dataset edge_iiotset `
  --profile iid `
  --attack-name clean `
  --malicious-ratio 0.1 `
  --seed 20260816 `
  --output-directory "artifacts\attacks\phase_6c\edge_iiotset\iid\clean\mr_0_1\seed_20260816" `
  --cache-directory "artifacts\client_cache\phase_6c\edge_iiotset\iid" `
  --cache-work-database "artifacts\client_cache\phase_6c\edge_iiotset\iid\client_cache.sqlite3" `
  --device auto `
  --max-rounds-this-call 1
```

This may take time because the first invocation materializes the complete 50-client
profile cache before training the selected 20 clients. It is safe to interrupt and
rerun, but a partial current round will be repeated. Preserve the complete terminal
output. A successful bounded call reports `status=interrupted` and `next_round=1`;
that status means the requested one-round bound was honored, not that the gate failed.

Rerun the same command to advance one more round. Remove
`--max-rounds-this-call 1` only when intentionally completing all remaining rounds.

## Sequential matrix runner

`scripts/run_phase6c_matrix.py` removes the need to launch conditions individually.
It executes conditions sequentially, reuses the profile cache, writes a matrix plan
and progress artifact after each condition, and stops on the first real error. The
`--target-round` value is absolute: a condition already at that round is validated and
skipped. An interrupted invocation can therefore be rerun unchanged.

The `all_types` matrix contains nine conditions: the clean control, the zero-poison
label-flip equivalence control, and one low-intensity frozen condition for each attack
family. The `full_grid` matrix contains all 25 frozen attack/parameter combinations
for one dataset/profile/ratio/seed slice. Neither mode crosses into evaluation data.

Run the Edge-IIoTset IID all-types gate to round 1 with:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase6c_matrix.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --client-partition-manifest "artifacts\client_partitions\phase_3d\edge_iiotset\edge_iiotset_phase3d_clients.json" `
  --resolution "artifacts\attacks\phase_6b\edge_iiotset\edge_iiotset_phase6b_resolution.json" `
  --output-root "artifacts\attacks\phase_6c" `
  --cache-root "artifacts\client_cache\phase_6c" `
  --norm-reference-root "artifacts\attacks\phase_6b\norm_references" `
  --dataset edge_iiotset `
  --profiles iid `
  --malicious-ratios 0.1 `
  --seeds 20260816 `
  --matrix all_types `
  --target-round 1 `
  --device auto
```

Because the clean condition is already at round 1, this exact command skips it and
runs the eight missing conditions. It may take roughly eight times as long as one
training round. Closing the terminal stops the active condition; rerunning the same
command resumes from completed checkpoints and completed matrix entries.

### Runtime optimization verified on the local GPU

Local training retains the same batch order, batch size, optimizer steps, losses, and
parameter updates, but finite-value checks and loss transfers are synchronized once
per epoch instead of after every mini-batch. On the Edge-IIoTset IID cache,
client 00 at round 0 improved from 18.889 to 11.383 seconds for one epoch. The submitted
update SHA-256 remained
`f0e875540337807dd5af9b4f48de699b6245d60dd4ddc7fa715afdd7703fd2cd`, and the
mean loss and update norm were unchanged. The three-epoch optimized update also
matched the already persisted clean-round hash
`4bf492b0be5d7220d3e8f96fbf92f0a27c1c7d9ade00e5334b48bae875fb3312`.

Changing batch size, local epochs, mixed precision, or nondeterministic CUDA settings
would alter the frozen training contract or invalidate the Phase 6B norm references;
those shortcuts are not enabled.

## Attack-specific arguments

- `label_flip`, `backdoor`, `model_replacement`, and `adaptive_norm`: add
  `--poison-fraction` with 0.1, 0.3, or 0.5.
- `clean_label`: also add `--strength 0.5` or `--strength 1.0`.
- `scaling`: use `--scale 5` or `--scale 10` and leave poison fraction at zero.
- `sign_reversal`: use `--scale 1`, `--scale 5`, or `--scale 10` and leave poison
  fraction at zero.
- `adaptive_norm`: add the matching
  `artifacts/attacks/phase_6b/norm_references/<dataset>/<profile>/benign_norm_reference.json`
  through `--norm-reference`.

## Colab recovery

The portable bundle now requires and includes both Phase 6B.1 resolutions, all eight
audited Phase 6B.2 norm-reference artifacts, the closure index, Phase 3C transformed
corpora, and Phase 3D assignments. In `notebooks/00_colab_failsafe.ipynb`, keep
`RUN_PHASE6C_ATTACK = False` until bundle validation and tests pass. Then set one
condition and retain the one-round bound until its output has been copied to Drive.
