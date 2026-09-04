# Contributing to SemCred-FL

Contributions should preserve the experiment contracts, evidence boundaries, and
content-addressed provenance model.

## Development setup

```powershell
conda env create --file environment.yml
conda activate semcred-fl
python scripts\validate_environment.py
```

## Before submitting a change

```powershell
ruff check src scripts tests
ruff format --check src scripts tests
mypy --strict src
python -m pytest -q
python scripts\audit_public_release.py
```

Changes to equations, label mappings, split policies, attack definitions, default
hyperparameters, evidence gates, or experiment registries must update the applicable
typed contract, configuration, tests, and traceability documentation in the same
change.

Do not commit datasets, provider archives, baseline PDFs, manuscript sources,
prepared artifacts, checkpoints, experiment outputs, credentials, or API responses.
Tests that need data should construct bounded synthetic fixtures with explicit
non-claim provenance.

Bug reports should state the operating system, Python and PyTorch versions, command,
configuration identity, complete error message, and the smallest reproducible case.
Do not attach private datasets or credentials.
