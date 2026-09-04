# Phase 6A attack contract

Status: complete. Attack execution remains blocked until the Phase 6B train-only
resolution artifacts and transformations are implemented and tested.

## Purpose and evidence boundary

Phase 6A fixes the objectives, attacker knowledge, parameter grids, deterministic
selection rules, and dataset-specific class targets before any poisoning result is
observed. It does not execute attacks, access test records, or support a performance
claim.

The targeted label-flip objective follows Lavaur, Busnel, and Autrel's FIDS protocol:
selected records from one attack class are relabeled as benign, the local poisoning
rate is defined over eligible target records, and colluding attackers share the target
and poisoning rate. Their study also uses ten independent seeds and shows why IID-only
attack evaluation is insufficient. See DOI `10.1016/j.cose.2025.104462` and the author
postprint at `https://orbilu.uni.lu/handle/10993/65906`.

The model-replacement family is kept distinct from data poisoning. Its construction is
based on the model-replacement and constrained-scaling attack class introduced by
Bagdasaryan et al. (`https://proceedings.mlr.press/v108/bagdasaryan20a.html`) and the
targeted update-boosting threat studied by Bhagoji et al.
(`https://proceedings.mlr.press/v97/bhagoji19a.html`). The numerical factors in this
project are declared study-design choices; they are not attributed to either paper.

## Frozen dataset profiles

The source family is selected using federated-training metadata only: among non-benign
primary families with at least 10,000 records after anchor removal, choose the least
frequent family. The floor avoids a nominal fraction that yields almost no eligible
records on most clients while retaining a minority-family objective.

| Dataset | Source | Target | Eligible federated-training records |
|---|---|---|---:|
| Edge-IIoTset | `injection` (ID 3) | `benign` (ID 0) | 46,519 |
| CICIoT2023 | `web_based` (ID 7) | `benign` (ID 0) | 17,060 |

The count is the Phase 3B training count minus 64 class-balanced calibration/scoring
anchors. Neither validation nor test counts participate in the selection rule.

## Data-level attacks

- Poison fractions are `0.1`, `0.3`, and `0.5`; `0.0` is the exact clean no-op.
- Eligible records are ranked without replacement by SHA-256 over the run seed,
  attack ID, client ID, and record ID. The selected count is the floor of the fraction
  times the eligible local source-row count.
- Label flipping changes only the primary label from the dataset source family to
  `benign`.
- Clean-label poisoning preserves labels. Phase 6B will select three nonconstant
  continuous Phase 3C coordinates by the absolute standardized median gap between the
  source and benign training populations. Selected values move 50% or 100% toward the
  benign median and are projected to the Phase 3C training-derived clipping bounds.
- The backdoor uses the same source and target. Phase 6B will resolve a three-to-six
  coordinate endpoint tuple from the twelve highest-ranked candidate coordinates. It
  must have zero exact collisions in clean federated training data. Triggered test
  records are generated only during evaluation, from clean source-family test rows;
  natural test collisions are reported and excluded from the ASR denominator.

Feature and trigger resolution may inspect the federated training partition after
anchor removal. It may not inspect anchors, validation outcomes, or test records.

## Coordinated and direct-model attacks

Coordinated attackers share the attack family, source, target, poisoning fraction,
resolved feature/trigger artifact, and active rounds. Record selection remains
client-specific. Coalition members may share their malicious updates with one another,
but they cannot observe current-round benign updates or any anchor record.

Direct update manipulations are separate conditions:

- scaling: `delta_prime = scale * delta`, with `scale` in `{5, 10}`;
- sign reversal: `delta_prime = -scale * delta`, with `scale` in `{1, 5, 10}`;
- coalition model replacement:
  `delta_prime = (participants_per_round / coalition_size) * (theta_star - theta_t)`,
  where `theta_star` is trained using the frozen backdoor objective.

The replacement factor is a FedAvg-oriented construction. It is not described as an
exact replacement under coordinate median, Multi-Krum, or SemCred-FL weighting.
Declared sample counts are not manipulated in these conditions.

## Adaptive constrained-norm attack

The adaptive condition starts from the targeted label-flip objective and radially
clips its submitted L2 update norm to a frozen interquartile interval. That interval is
computed from benign-only development runs using the same dataset, model, and client
partition, then stored as an immutable Phase 6B artifact. The attacker receives the
fixed interval but cannot observe current-round benign updates or anchors.

## Matrix and readiness

Malicious-client ratios are `{0.1, 0.2, 0.3, 0.4}`. Heterogeneity profiles are IID and
Dirichlet alpha `{0.1, 0.3, 1.0}`. Five seeds are retained for the reduced pilot matrix;
confirmatory paired analysis requires at least ten seeds and the separate power record
required by `StatisticalConfig`.

The authoritative files are:

- `configs/attacks/phase6a.yaml` — immutable experiment configuration;
- `src/semcred_fl/attacks/contracts.py` — typed invariant enforcement;
- `scripts/validate_phase6a_attack_contract.py` — Phase 3 identity/count validation;
- `tests/test_attack_contracts.py` — contract and privacy-boundary tests.

Phase 6B must produce the clean-label feature, backdoor trigger, and benign update-norm
reference artifacts. No poisoning run is authorized until those artifacts and the
attack transformations pass their exact tests.
