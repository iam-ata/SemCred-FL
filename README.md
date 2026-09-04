# SemCred-FL

**LLM-Guided Dual-View Credibility Aggregation for Robust Federated Intrusion
Detection under Data Poisoning**

Nahideh Derakhshanfard and Ata Mohammadi  
Department of Computer Engineering, Ta.C., Islamic Azad University, Tabriz, Iran

## Overview

SemCred-FL is a research implementation of a server-side aggregation method for
federated intrusion detection. The method combines semantic evidence obtained from
a frozen offline prior with numerical evidence derived from client-update geometry.
The fused credibility score updates persistent client trust and controls update
eligibility, clipping, and aggregation.

The repository contains the Python implementation, experiment configurations,
tests, reproducibility documentation, and a Colab fail-safe notebook. Raw datasets,
prepared data artifacts, model checkpoints, experimental outputs, baseline papers,
and manuscript sources are intentionally excluded from version control.

Version `0.1.0` implements the data contracts and preparation pipeline, detector and
local training, federated simulation, attack transformations, semantic-prior
construction, dual-view credibility scoring, persistent trust aggregation, standard
baselines, evaluation utilities, and the resumable Phase 12 clean experiment matrix.
The full confirmatory attacked matrix, ablation study, sensitivity analysis, and
efficiency campaign remain separate measured-execution gates. No unexecuted or
synthetic result is presented as measured evidence.

## Method

For each submitted client update, SemCred-FL evaluates two evidence views:

1. The semantic view measures degradation on a disjoint scoring anchor and compares
   grouped feature attribution with an immutable semantic-prior bundle.
2. The numerical view measures deviation from the round population in update
   location, norm, direction, and accepted client history.

The server applies missing-aware robust normalization, disagreement-aware fusion,
asymmetric trust updates, eligibility filtering, norm clipping, and trust-weighted
aggregation. Simulated malicious-client identity and attack metadata are retained
for post-decision evaluation only and are not provided to the defense.

The equations and their implementation locations are mapped in
[`docs/equation_traceability.md`](docs/equation_traceability.md). The complete
scientific and software contract is recorded in
[`docs/SEMCRED_FL_IMPLEMENTATION_MASTER_SPEC.md`](docs/SEMCRED_FL_IMPLEMENTATION_MASTER_SPEC.md).

## Repository structure

```text
SemCred-FL/
├── .github/                 GitHub CI and dependency-update configuration
├── configs/                 Dataset, attack, defense, and experiment contracts
├── docs/                    Reproducibility and implementation documentation
├── notebooks/               Colab fail-safe workflow
├── scripts/                 Audits, preparation stages, experiments, and reporting
├── src/semcred_fl/          Installable Python package
├── tests/                   Unit, integration, leakage, and resume tests
├── CITATION.cff             GitHub citation metadata
├── .zenodo.json             Zenodo release metadata
├── environment.yml          Cross-machine Conda environment
├── environment-win-64.lock.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Requirements

- Windows 11, Linux, or Google Colab
- Conda or Miniconda
- Python 3.13
- CUDA-capable PyTorch is recommended for full experiment matrices
- Approximately 16 GB RAM for preparation and bounded local runs; full runtime and
  storage depend on the selected dataset and experiment registry

The committed Windows lock file records the environment used for local validation.
`environment.yml` is the portable installation contract.

## Installation

From the repository root:

```powershell
conda env create --file environment.yml
conda activate semcred-fl
python scripts\validate_environment.py
python scripts\validate_dataset_contracts.py
```

If the environment already exists, update it without recreating it:

```powershell
conda env update --name semcred-fl --file environment.yml --prune
```

## Datasets

The source data are not redistributed. Obtain the datasets from their official
providers and review their terms before use:

- CICIoT2023
- Edge-IIoTset

Place local source files beneath `Datasets/` according to
[`docs/DATA_PROVENANCE_AND_INGESTION.md`](docs/DATA_PROVENANCE_AND_INGESTION.md).
The directory is ignored by Git. Generated corpora and experiment artifacts are
written beneath `artifacts/`, which is also ignored.

Dataset preparation is deliberately staged. Source auditing, canonicalization,
split assignment, preprocessing, client partitioning, attack resolution, and
semantic-prior construction each produce content-addressed manifests. A later stage
rejects an artifact when its upstream identity differs.

## Validation

Run the repository checks before changing an experiment or creating a release:

```powershell
conda run --no-capture-output -n semcred-fl ruff check src scripts tests
conda run --no-capture-output -n semcred-fl ruff format --check src scripts tests
conda run --no-capture-output -n semcred-fl mypy --strict src
conda run --no-capture-output -n semcred-fl python -m pytest -q
conda run --no-capture-output -n semcred-fl python scripts\audit_public_release.py
```

The release audit examines the exact Git candidate set, rejects datasets and
generated artifacts, rejects oversized files, scans for high-confidence credential
patterns, and verifies version and author consistency across `pyproject.toml`,
`CITATION.cff`, and `.zenodo.json`.

## Phase 12 clean experiment

Inspect the canonical registry without writing files or starting training:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py plan
```

The default registry contains two datasets, four partition profiles, ten matched
seeds, and five aggregation methods. Each method uses 50 clients, 20 submissions per
round, 100 federated rounds, three local epochs, and batch size 128. The run selects
CUDA automatically when available.

Start or resume the matrix:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py run
```

For bounded sessions, advance one condition by one round:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py run `
  --max-conditions-this-call 1 `
  --max-rounds-this-call 1
```

Audit and finalize a completed campaign using the directory printed by the plan:

```powershell
$campaign = Read-Host "Paste the completed campaign directory"
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py audit `
  --campaign-directory $campaign
conda run --no-capture-output -n semcred-fl python scripts\run_phase12.py finalize `
  --campaign-directory $campaign
```

The finalizer verifies the registry, checkpoints, round-record hash chains, leakage
gates, full method budget, and paired-seed statistics before creating machine-
readable summaries, LaTeX fragments, and SVG figures. It does not edit the
manuscript or open the final test partition.

Detailed commands are provided in
[`docs/PHASE_12_EXECUTION_RUNBOOK.md`](docs/PHASE_12_EXECUTION_RUNBOOK.md).

## Reproducibility and evidence boundaries

- Data splits, anchor identities, client allocations, attack realizations, client
  schedules, configurations, checkpoints, and reports are content-addressed.
- Resume operations validate upstream identities and immutable round records before
  continuing.
- Training-only, validation, and test access are recorded explicitly.
- The final test partition remains sealed until the experiment registry and analysis
  rules are frozen.
- Synthetic fixtures test execution and reporting. They are not measured performance
  evidence and cannot pass the measured-campaign finalizer.
- Dataset licenses and provider files remain outside this repository.

See [`docs/SCIENTIFIC_DECISIONS.md`](docs/SCIENTIFIC_DECISIONS.md) for frozen
decisions and [`docs/PHASE_12_COMPLETION_PLAN.md`](docs/PHASE_12_COMPLETION_PLAN.md)
for the remaining experimental gates.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub will expose
these metadata through its **Cite this repository** panel. Zenodo metadata are in
`.zenodo.json` and will be applied when a tagged GitHub release is archived.

Until the first DOI-bearing release is published, cite version `0.1.0` as:

> Derakhshanfard, N., & Mohammadi, A. (2026). *SemCred-FL: LLM-Guided Dual-View
> Credibility Aggregation for Robust Federated Intrusion Detection under Data
> Poisoning* (Version 0.1.0) [Computer software].

After Zenodo assigns the DOI, use the citation displayed on the Zenodo record. The
release procedure is documented in
[`docs/GITHUB_ZENODO_RELEASE_GUIDE.md`](docs/GITHUB_ZENODO_RELEASE_GUIDE.md).

## Authors

- Nahideh Derakhshanfard
- Ata Mohammadi

Affiliation: Department of Computer Engineering, Ta.C., Islamic Azad University,
Tabriz, Iran.

## License

Copyright © 2026 Nahideh Derakhshanfard and Ata Mohammadi. The current release is
source-available under the terms in [`LICENSE`](LICENSE); no open-source reuse grant
is implied. Third-party datasets and publications are governed by their respective
providers and are not included.
