# SemCred-FL local-first and Colab fail-safe runbook

Status: local execution is primary. Colab is a transfer and recovery path. This
runbook covers the implemented gates through Phase 6C; it does not authorize claims
from bounded Phase 4 or synthetic Phase 5 runs.

## 1. Local preflight

Run commands sequentially from the project `Code` directory:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\validate_environment.py
conda run --no-capture-output -n semcred-fl python scripts\validate_dataset_contracts.py
conda run --no-capture-output -n semcred-fl python -m pytest -q --basetemp=.pytest_tmp_semcred_local
conda run --no-capture-output -n semcred-fl python -m ruff check src tests scripts
conda run --no-capture-output -n semcred-fl python -m ruff format --check src tests scripts
conda run --no-capture-output -n semcred-fl python -m mypy --strict src
```

Do not launch commands in parallel. Stop at the first failure and preserve its full
output.

## 2. Local implemented execution gates

The Phase 5 synthetic gate is small, resumable, and safe on CPU:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase5_synthetic.py `
  --output-directory "artifacts\federation\phase_5\local_recheck" `
  --rounds 3 `
  --clients-per-round 2 `
  --aggregation fedavg `
  --device cpu
```

The bounded Phase 4 GPU checks are optional because verified artifacts already exist.
They must remain bounded and must not be reported as detector-performance results.

## 3. Build the Colab fail-safe bundle

First inspect the selected file count and uncompressed size:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_colab_bundle.py `
  --project-root . `
  --dry-run
```

Then build the ZIP64 bundle. This streams and hashes the prepared artifacts and may
take several minutes:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_colab_bundle.py `
  --project-root . `
  --output "artifacts\colab_bundle\semcred_fl_colab_bundle.zip" `
  --progress-every 25
```

Verify the completed archive before uploading it:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_colab_bundle.py `
  --verify "artifacts\colab_bundle\semcred_fl_colab_bundle.zip"
```

The archive contains the source code, configs, tests, documentation, notebook,
Phase 3C transformed shards, Phase 3D client assignments, both Phase 6B.1 resolutions,
all eight norm-reference artifacts, and the Phase 6B closure index. It excludes raw
datasets, cache directories, and SQLite resume databases. Keep the adjacent
`.zip.sha256` file with the archive.

## 4. Colab recovery procedure

1. Upload the bundle and its `.zip.sha256` sidecar to Google Drive.
2. Open `notebooks/00_colab_failsafe.ipynb` in Colab.
3. Select a GPU runtime.
4. Run the cells in order and edit only the bundle and output Drive paths.
5. Do not continue if bundle verification, CUDA validation, dataset-contract
   validation, or tests fail.
6. Copy generated result directories back to Drive using the final notebook cell.

The notebook uses Colab's CUDA-enabled PyTorch instead of replacing it with the
Windows-specific wheel from `environment.yml`. It installs the remaining bounded
dependencies from `requirements-colab.txt` and records the runtime package report.

## 5. Phase 6B.1 resolution

The Phase 6A contract can be checked locally without scanning feature shards:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\validate_phase6a_attack_contract.py `
  --project-root "."
```

Resolve one dataset at a time using the commands in
`docs/PHASE_6B_ATTACK_RESOLUTION.md`. Each command scans only Phase 3C/3D prepared
artifacts, is resumable at shard boundaries, and can use one to two GB of temporary
work space. Do not run the two dataset commands concurrently on the local laptop.

## 6. Phase 6B.2 benign norm references

Use the first local command in `docs/PHASE_6B_ATTACK_RESOLUTION.md`. Run one
dataset/profile at a time. The cache resumes by completed source shard and federated
training resumes by completed round. An interruption within a round repeats that
round. The Colab notebook contains the same wrapper behind
`RUN_PHASE6B_NORM_CALIBRATION = False`; set the dataset and profile explicitly before
enabling it.

## 7. Phase 6B closure

All eight norm references are complete. Rebuild or verify their closure index with:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\audit_phase6b_norm_references.py `
  --root-directory "artifacts\attacks\phase_6b\norm_references" `
  --output "artifacts\attacks\phase_6b\norm_references\phase6b_norm_reference_index.json"
```

The expected closure ID is
`1461106f84993cb974e197c40373c37d1765821881bbdf6a771242e139da32e9`.

## 8. Phase 6C bounded gate

The Phase 6C wrapper and synthetic no-op, transformation, audit, and resume gates are
implemented. The Edge-IIoTset IID `all_types` matrix reached its absolute round-one
target for all nine conditions. Matrix identity
`06e12e61f1d147abf906c46d80df9dde91e9e94bb90c91b81664b8af66c95d14`
and run identity
`e9a94f31ab055cec747636d9ad0c8764682eb1d20b07a770a3a5af0e5fe7a561`
passed the closure audit. Do not advance this development matrix to 100 rounds or
start the full grid before Phases 7-11 pass. Phase 6C remains train-only and cannot
support performance claims.

## 9. Phase 7 public-input boundary

Phase 7 is complete for Edge-IIoTset and CICIoT2023. The provider declaration is
pinned to `gpt-5.4-mini-2026-03-17`; the API key remains outside the repository. Both
human-approved public prompts were sent once with `store=false`, and the resulting
responses and fused bundles passed identity and offline-loading checks. Re-running
the invocation command with these intact artifacts reports `api_called=false` and
`reused=true`; do not delete the provider receipts or raw responses unless a new,
explicitly approved semantic-prior generation is intended. See
`docs/PHASE_7_OFFLINE_SEMANTIC_PRIOR.md` for the frozen IDs and reproduction commands.
