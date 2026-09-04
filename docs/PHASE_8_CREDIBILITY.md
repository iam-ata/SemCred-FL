# Phase 8: semantic and numerical credibility

Status: implementation gate complete on 2026-08-31. This phase contains equation
fixtures and synthetic scientific safeguards. It does not contain a real-data defense
experiment or authorize a detector-performance claim.

## Scope and boundaries

Phase 8 converts the accepted updates from one participating population into two
scores in `(0, 1]`. It does not fuse those scores, update persistent trust, select
eligible clients, clip updates, or aggregate a global update; those operations belong
to Phase 9.

The defense-facing interface accepts `DefenseUpdate`, which has no `is_malicious` or
`attack_metadata` field. Semantic evaluation reads only the frozen
`anchor_scoring` partition and the immutable Phase 7 prior bundle. It does not read
calibration anchors, validation records, test records, or make an online LLM request.

## Robust population normalization

For each available indicator population, `credibility/robust_stats.py` implements

```text
MAD(v) = median_k |v_k - median_j(v_j)|
z+(v_k) = min(z_max, max(0, (v_k - median(v)) / (1.4826 MAD(v) + epsilon)))
```

The defaults are `z_max=20` and `epsilon=1e-12`. An unavailable indicator is omitted
before its population statistic is computed. Its configured coefficient is removed
for that client and the remaining positive coefficients are renormalized to sum to
one. No neutral score and no stale measurement is substituted.

## Semantic credibility

`credibility/semantic_score.py` reconstructs each submitted client model as
`theta_k = theta_global + Delta_k` without mutating the global model. It measures
scoring-anchor loss degradation

```text
e_k = L(theta_k; A_scoring) - L(theta_global; A_scoring).
```

When attribution is scheduled for the current round,
`semantic/attribution.py` computes class-specific gradient-times-input values on
`A_scoring`. Coordinates are reduced to the frozen Phase 3C semantic groups by the
root mean square and then averaged over that class's scoring anchors. Each class row
is L1-normalized before its cosine discrepancy from the corresponding Phase 7 prior
is measured. A class with a negligible attribution or prior norm is unavailable.

The semantic score is

```text
s_k = exp[-beta_loss z+(e_k) - beta_attr z+(d_sem,k)].
```

The default coefficients are `0.5` and `0.5`. If attribution is not scheduled or no
class has two non-negligible vectors, semantic scoring uses loss only with an active
coefficient of one. A previous round's attribution is never reused.

## Numerical credibility

`credibility/numerical_score.py` uses the ordinary midpoint coordinate median for an
even population, matching the Phase 5 convention. It implements the manuscript's
center-distance, log-norm, half-cosine direction, and half-cosine accepted-history
discrepancies.

Direction is unavailable when the update or round center has negligible norm.
History is unavailable for a client without a non-negligible accepted history. It is
disabled for the complete participating population when fewer than three clients
have valid histories; historical robust normalization then uses only valid-history
participants. The numerical score is

```text
u_k = exp[-sum_i beta_i z+(d_k,i)].
```

All four default coefficients are `0.25`. Without history, the remaining three are
therefore `1/3` each. Other missing indicators are handled by the same per-client
renormalization rule.

## Verification gate

`tests/test_phase8_credibility.py` checks the equations and scientific failure modes:

- midpoint median, MAD scaling, one-sided clipping, and coefficient renormalization;
- finite grouped attribution and restoration of the detector's training mode;
- lower semantic discrepancy for prior-aligned attribution and unavailable zero rows;
- non-mutating client-model reconstruction and exact uniform-logit cross entropy;
- scores that remain finite in `(0, 1]` and decrease for adverse synthetic evidence;
- global history disabling below three valid participants and per-client direction
  unavailability for a negligible round center.

The Phase 8 focused gate passed 40 tests. Ruff passed for all changed files, and strict
mypy passed for the five changed source modules using a workspace-local cache. The
full repository suite is recorded separately in the completion handoff. None of
these checks accessed a test partition or produced a performance estimate.
