# SemCred-FL environment record

Status: verified on 2026-08-21.

## Validated machine

- Windows 11 build 26200.
- NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- NVIDIA driver 591.86.
- GPU compute capability 8.6.
- Conda 25.11.1.
- Python 3.13.11.
- PyTorch 2.12.0+cu130 with CUDA 13.0.

The CUDA wheel used for the validated environment was
`torch-2.12.0+cu130-cp313-cp313-win_amd64.whl`, with SHA-256:

```text
6E728C5FDEFFA19B3FA6A759FF585147851772789F3DC84DEC5F8CBDE0F7A5B0
```

The environment validation imported every required package, detected one CUDA device,
identified the RTX 3050, and completed a CUDA tensor operation with the expected value.

## Environment files

- `environment.yml` is the readable environment contract. It explicitly requests the
  CUDA 13.0 PyTorch build so it cannot silently resolve to the CPU package.
- `environment-win-64.lock.yml` records the exact Windows package versions and Conda
  build identifiers used for validation.

## Validation commands

```powershell
conda run -n semcred-fl python scripts/validate_environment.py
conda run -n semcred-fl python -m ruff check src tests scripts
conda run -n semcred-fl python -m ruff format --check src tests scripts
conda run -n semcred-fl python -m mypy --strict src
conda run -n semcred-fl python -m pytest -q
```

Black 26.5.1 stalled without a diagnostic when invoked from this Windows automation
session, including with one worker. Ruff formatting is therefore the canonical automated
format gate. This does not relax formatting checks: `ruff format --check` must pass.

## Local Conda channel limitation

The installed system-level Conda configuration points to the `defaults` channel, which
requested an Anaconda authentication token during online environment creation. The
minimal Python 3.13 environment was created from already cached Conda packages, and the
declared Python packages were then installed with pip. No global Conda configuration was
modified.

