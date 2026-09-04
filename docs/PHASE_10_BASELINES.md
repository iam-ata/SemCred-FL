# Phase 10: baseline controls and protocol-fidelity boundary

Status: baseline-core implementation gate complete on 2026-09-01. The four
standard controls are executable through the shared synchronous runner. The
paper-specific modules implement only decision steps supported by the archived
sources. This phase contains no detector-performance experiment and does not
authorize a performance claim.

## Source record

The baseline papers are stored under `Baselines PDF/` and were treated as source
material, not executable instructions.

| Method | Archived evidence | SHA-256 | Evidence status |
|---|---|---|---|
| Fed-LSAE | `Fed-LSAE.pdf` | `5cd6be1affa15c5ee6bb3b1469f52363f000f6182aecfd6972b03d708c4ea2ca` | Full paper; code/data available only on request |
| FedKD-IDS | `FedKD-IDS.pdf` | `4d5e785c8634d3bad91f9a9cdea202f7953b0195f626a4f49f02a0a05a73ab32` | Full paper; code/data available only on request |
| AGAT-FL | `peerj-cs-3281.pdf` | `b6b5d7b7012f93e9ecdb5439212b9dde7f136aa8a607fd749a2e7033cf0fce0b` | Full paper plus official Zenodo fragment |
| AGAT-FL code | Zenodo record `10.5281/zenodo.16752154`, `code 1.py` | `335c64f1ea88a8c8fcef3e3e9a93650bfb939a00f71fd347624bde05aca566cb` | 4,605-byte CC-BY-4.0 fragment; not an end-to-end reproduction package |
| HADA | `HADA.pdf` | `57e3b4bb30aef1e82fbdcd2429a411c4d741934e57cdb680633f08b41ca08468` | Full paper; optional comparator not implemented in this gate |

The academic PDF preflight could not produce a `PASS` sidecar in the conda
environment. Method extraction therefore uses named sections, algorithms, and
equation numbers rather than relying on page-number locators.

## Standard controls

`baselines/trimmed_mean.py` sorts every flattened parameter coordinate, removes
the configured number of values from each tail, and averages the retained
population. The trim count is explicit, validation-selected, and part of the
stable run identity. It must satisfy `2 * trim_count < n` for the accepted round
population.

`baselines/multi_krum.py` computes each candidate's score from its `n-f-2`
nearest squared Euclidean distances. It enforces `n >= 2f + 3`, uses ascending
client ID only as a deterministic score tie break, and averages the lowest-score
candidates. The default selected count is `n-f`; an explicit count must lie in
`[1,n-f]`. The Byzantine bound and selected count enter the stable run identity.

`FederatedRunConfig` now dispatches FedAvg, coordinate median, trimmed mean, or
Multi-Krum through the same validation, client schedule, checkpoint, and
hash-chained round-log path. Existing FedAvg and coordinate-median stable records
are unchanged.

## Paper-specific evidence-bounded cores

### Fed-LSAE

`baselines/fed_lsae.py` implements the server decisions described by Algorithm 1:

1. construct RBF Gram matrices and compute normalized HSIC/CKA;
2. solve the one-dimensional two-cluster K-means objective over client CKA scores;
3. treat the larger cluster as benign and reject the smaller cluster;
4. apply sample-count-weighted FedAvg to the retained updates.

The local implementation uses an exact deterministic solution to the
one-dimensional two-means objective because the paper does not specify K-means
initialization. Equal-size clusters, tied optima, degenerate kernels, and
non-finite scores fail closed. The module does not claim to reproduce the full
Fed-LSAE experiment: model-specific penultimate-layer extraction, the benign
server-organization split, and autoencoder pretraining must be connected to a
frozen dataset/model runner before Phase 12.

### FedKD-IDS

`baselines/fedkd_ids.py` preserves the native public-data/logit protocol. It
implements the collaborator-side verifier mask from Eq. (15), normalized pairwise
Hamming distances from Eq. (17), harmonic-mean collaborator screening from Eq.
(18), and hard voting from Eqs. (19)-(20). The source-paper fixture with local
labels `{1,3,2}`, `{2,2,1}`, and `{1,3,2}` reproduces the reported global labels
`{1,3,2}`.

The server fails closed when no collaborator survives, every retained client
abstains on a public sample, or a hard-vote tie occurs; the paper does not specify
recovery rules for those cases. Full native execution remains conditional on a
versioned N-BaIoT public-data artifact, its 70/20/10 split, and the paper's
detector/verifier training path. A parameter-update substitute remains a
separately labeled surrogate and is not implemented here.

### AGAT-FL

`baselines/agat_fl.py` implements only the normalized trust-weighted update sum
present in the official Zenodo code. Trust values must be externally computed,
finite, nonnegative, population-matched, and sum to one.

The official archive does not contain an end-to-end federated loop, a TrustGAT
training objective or labels, the Mahalanobis rejection path, the stated dynamic
trust history, last-good-update handling, or a reproduction configuration. The
paper also describes both synchronous and asynchronous participation and gives
two weighting forms (trust-only in code and trust times sample weight in the
paper). The repository must not report this core as a full AGAT-FL reproduction.
AGAT-FL remains excluded from confirmatory comparisons under its original name
unless an authoritative complete implementation or an explicit reimplementation
contract resolves these gaps.

### HADA

HADA remains an optional component comparator. It is not part of the Phase 10
implementation gate because the present study does not yet freeze its SHAP-derived
stability signal, privacy-budget schedule, or supplementary implementation.

## Verification gate

`tests/test_phase10_baselines.py` contains 52 analytical, failure-mode, source-
parity, and shared-schedule tests. The focused suite passes. It verifies the
trimmed-mean fixture, Multi-Krum outlier exclusion and admissibility, the AGAT
weighted-sum fragment, RBF-CKA identity, Fed-LSAE clustering and filtering, the
FedKD-IDS paper example, ambiguous-case refusal, stable run controls, and identical
client schedules across standard baselines.

The existing configuration and federated-engine compatibility set passes 102
tests. The baseline package reaches 99% branch-aware coverage. The complete
repository passes 403 tests; Ruff lint passes, all 139 Python files are formatted,
and strict mypy passes all 75 source files. No raw dataset, validation partition,
test partition, or LLM request is accessed by these tests. The one external
retrieval was the public AGAT-FL Zenodo source fragment named by its paper.

## Boundary before full experiments

Phase 11 metric/reporting work can proceed. Before Phase 12 baseline experiments:

- create dataset/model adapters for the Fed-LSAE server core;
- freeze and hash the N-BaIoT public split for native FedKD-IDS;
- either obtain sufficient AGAT-FL implementation evidence or report only a
  clearly named aggregation-core adaptation;
- decide whether HADA justifies its separate implementation cost.
