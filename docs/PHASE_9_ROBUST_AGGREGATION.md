# Phase 9: fusion, trust, and robust aggregation

Status: implementation gate complete on 2026-09-01. This phase provides the
deterministic server-side state transition that consumes Phase 8 scores. It does not
contain a real-data defense experiment and does not authorize a performance claim.

## Implemented procedure

`credibility/fusion.py` implements the disagreement-aware fused score

```text
c_k = (s_k + epsilon)^alpha_S (u_k + epsilon)^(1-alpha_S)
      exp(-lambda_D |s_k-u_k|), clipped to [0,1].
```

If exactly one complete view is available, that view is used directly and the
disagreement term is omitted with an explicit missing-view reason. Two missing views
fail closed.

`federation/state.py` and `aggregation/trust.py` initialize a new client's trust from
its first fused score and apply the asymmetric update to later participation. The
defaults are `eta_down=0.6` and `eta_up=0.1`. Inactive clients are absent from the
transition input and retain their stored trust and history without decay.

After burn-in, `aggregation/eligibility.py` computes

```text
tau = max(tau_min, median(T) - 1.4826 kappa_T MAD(T)).
```

Clients at or above `tau` are initially eligible. If fewer than
`max(1, ceil(q_min |S_t|))` remain, excluded clients are added by descending trust
with ascending client ID as the deterministic tie break.

`aggregation/clipping.py` computes `B=c_B median_k ||Delta_k||` over the final
eligible set and applies the manuscript clipping formula. `aggregation/weighting.py`
then computes finite normalized weights from capped sample counts and persistent
trust. The confirmatory default is `xi=0`, so declared counts do not affect weights
and the sample cap is inactive. Sensitivity runs with `xi>0` use
`n_cap=c_n median_k n_k`; the default `c_n` is 2.0.

## Burn-in and accepted history

For rounds `t<R_B`, `aggregation/burn_in.py` applies the coordinate-wise midpoint
median to every accepted update for the global step. The preliminary history set
contains clients at or above the participating median fused credibility, followed by
the same minimum-retention rule. Only clipped updates from this set enter history.

After burn-in, only final eligible clipped updates enter history. A first admission
initializes the history exactly. Later admissions apply

```text
history_next = rho_delta history_previous + (1-rho_delta) clipped_update.
```

Rejection and inactivity do not change history. The server never stores an unclipped
submission in the accepted-history field.

## Server integration

`federation/server.py::process_semcred_round` performs the Phase 9 order explicitly:

1. require Phase 8 score populations to match the validated `DefenseUpdate` population;
2. fuse the available credibility views;
3. update participating trust;
4. choose the burn-in or post-burn-in aggregation path;
5. admit only the selected clipped updates to history;
6. return the aggregate delta, updated client states, eligibility, clipping, cap, and
   weight audit fields.

The existing Phase 5 runner remains unchanged for baseline and engine-gate runs.
Phase 9 adds the SemCred state transition without changing FedAvg or coordinate-median
baseline semantics.

## Frozen defaults

| Control | Value |
|---|---:|
| Burn-in rounds | 3 |
| `alpha_S` | 0.5 |
| `lambda_D` | 1.0 |
| `eta_down` / `eta_up` | 0.6 / 0.1 |
| `gamma` | 2.0 |
| Confirmatory `xi` | 0 |
| Sensitivity count-cap multiplier `c_n` | 2.0 |
| `kappa_T` / `tau_min` | 2.5 / 0.1 |
| `q_min` / `c_B` | 0.5 / 2.5 |
| `rho_delta` | 0.8 |

These are prespecified starting values, not measured optima. Any tuning remains
restricted to the declared validation procedure.

## Verification gate

`tests/test_phase9_aggregation.py` provides 26 analytical, fail-closed, and state-transition
tests. They cover fusion symmetry and disagreement monotonicity, missing-view use,
asymmetric trust, inactivity, history initialization and EMA, deterministic minimum
retention, clipping, conditional sample capping, normalized weights, an exact weighted
sum, controlled burn-in, and deterministic two-round server integration.

The Phase 9 aggregation package reached 98% branch-aware coverage. The complete
repository passed 351 tests; Ruff lint passed, all 133 Python files were formatted,
and strict mypy passed all 70 source files. No dataset, validation partition, test
partition, or provider API is accessed by this phase's tests.
