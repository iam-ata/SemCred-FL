# Equation-to-code traceability

Status: synchronized through Phase 12 Step 3, including the completed clean and
label-flip non-IID execution pilots, on 2026-09-02.
Rows marked **future** are not implemented and cannot be inferred from a frozen
contract or unresolved artifact.

| Manuscript equation or concept | Current implementation | Verification |
|---|---|---|
| Detector mapping, `eq:global-model` | `modeling/detector.py` | detector shape and probability tests |
| Local objective, `eq:local-objective` | `modeling/losses.py`, `federation/client.py` | exact weighted-loss, proximal-term, and deterministic local-training tests |
| Class weight, `eq:class-weight` | `modeling/losses.py` | cap, absent-class, and normalization tests |
| Local update, `eq:local-update` | `federation/parameters.py`, `federation/client.py` | exact flatten/unflatten and one-client FedAvg tests |
| Submitted update contract | `federation/updates.py` | shape, round, duplicate, finite-value, and safe-projection tests |
| Coordinate-wise center, `eq:robust-update-center` | `baselines/coordinate_median.py` | odd and even analytical fixtures |
| Global weighted update, `eq:global-update` | `baselines/fedavg.py`, `federation/server.py` | analytical weighted fixture and deterministic multi-round integration test |
| Client selection and broadcast loop | `federation/sampling.py`, `federation/server.py` | immutable schedule, shared broadcast, and round-order tests |
| Checkpoint and audit state | `federation/checkpoint.py`, `federation/server.py` | checksum rejection, hash-chain validation, and resumed-versus-uninterrupted equality |
| Attack objectives, grids, attacker knowledge, and formulas | `attacks/contracts.py`, `configs/attacks/phase6a.yaml` | typed/YAML identity and invariant tests |
| Train-only feature/trigger resolution | `attacks/artifacts.py`, `attacks/resolution.py` | exact-median ranking, zero-collision search, resume, and scanner-boundary tests |
| Poisoning transformations and triggered test generator | `attacks/transformations.py` | exact no-op, label, bounds, trigger, collision, and formula tests |
| Benign update-norm reference generation | `attacks/norm_calibration.py`, `data/client_cache.py`, `attacks/norm_reference_audit.py` | all eight references and 480 development updates independently reconstructed in closure index `1461106f84993cb974e197c40373c37d1765821881bbdf6a771242e139da32e9` |
| Fixed coalition, exact malicious-minority schedule, and attacked local execution | `attacks/execution_contracts.py`, `attacks/execution.py`, `scripts/run_phase6c_attack.py` | YAML identity, coalition cardinality, population-selection, exact no-op, resume, and synthetic end-to-end tests |
| LLM relevance, `eq:llm-relevance` | `semantic/contracts.py`, `semantic/pipeline.py` | duplicate, missing, unknown, non-integer, Boolean, and out-of-range response rejection |
| Anchor-derived relevance, `eq:data-relevance` | `semantic/prior.py` | hand-calculated grouped RMS fixture; calibration IDs reconciled against the anchor manifest |
| Prior normalization and fusion, `eq:normalized-llm-prior`, `eq:normalized-data-prior`, `eq:fused-semantic-prior` | `semantic/prior.py`, `semantic/pipeline.py` | L1-plus-epsilon and exact `alpha_L=0/1` endpoint tests; immutable no-LLM bundle load |
| Robust MAD and one-sided anomaly score | `credibility/robust_stats.py` | midpoint-median, MAD, constant-population, cap, and monotonic fixtures |
| Scoring-anchor loss degradation and grouped attribution | `credibility/semantic_score.py`, `semantic/attribution.py` | exact uniform-logit loss, non-mutating reconstruction, group shape, finite-value, and unavailable-class tests |
| Semantic discrepancy and credibility, `eq:semantic-discrepancy`, `eq:semantic-credibility` | `credibility/semantic_score.py` | prior alignment, adverse-indicator monotonicity, score range, and missing-attribution renormalization |
| Numerical discrepancies and credibility, `eq:center-distance` through `eq:numerical-credibility` | `credibility/numerical_score.py` | midpoint coordinate median, analytical cosine, adverse-update monotonicity, negligible-direction, and three-history gate tests |
| Dual-view fusion, `eq:credibility-fusion` | `credibility/fusion.py` | half-coefficient symmetry, disagreement monotonicity, clipping, and missing-view tests |
| Persistent trust, `eq:asymmetric-trust`, `eq:trust-initialization` | `aggregation/trust.py`, `federation/state.py` | faster fall than recovery, initialization, and inactive-state preservation |
| Adaptive threshold and minimum retention, `eq:adaptive-trust-threshold` through `eq:minimum-eligible-count` | `aggregation/eligibility.py` | threshold floor, nonempty retention, and deterministic client-ID tie break |
| Update clipping, `eq:clipping-bound`, `eq:clipped-update` | `aggregation/clipping.py` | analytical bound, norm tolerance, direction, and zero-vector tests |
| Count cap and aggregation weights, `eq:capped-sample-count`, `eq:aggregation-weight` | `aggregation/weighting.py` | conditional median cap, confirmatory count independence, finite nonnegative unit-sum weights |
| Global robust update, `eq:global-update` | `aggregation/semcred.py` | exact analytical weighted sum |
| Burn-in aggregation and admission, `eq:burn-in-aggregation`, `eq:burn-in-history-set` | `aggregation/burn_in.py` | midpoint coordinate median and credibility-controlled clipped admission |
| Accepted history, `eq:update-history` | `federation/state.py` | first initialization, exact EMA, rejection, and inactivity tests |
| Complete server procedure, `alg:semcred-fl` | `federation/server.py::process_semcred_round` | deterministic burn-in-to-post-burn-in state-transition scenario |
| Detection utility and confusion counts | `modeling/metrics.py` | exact binary/multiclass confusion fixtures |
| Absolute/relative degradation, `eq:absolute-degradation`, `eq:relative-degradation` | `evaluation/metrics.py` | analytical clean/attacked fixture and empty-denominator checks |
| Client-assessment and temporal-trust metrics | `evaluation/metrics.py` | perfect/reversed/tied ranking, eligibility, delay, censoring, and volatility fixtures |
| Percentile-bootstrap summaries and paired statistical protocol | `evaluation/statistics.py` | deterministic bootstrap, exact Wilcoxon zero/tie, rank-biserial, and Holm-family fixtures |
| Run-to-table/figure traceability | `evaluation/reporting.py`, `scripts/aggregate_phase11_results.py` | order invariance, content identity, tamper rejection, provenance, and byte-identical regeneration |

The Phase 5 runner retains its original FedAvg and coordinate-median baseline
semantics. The separate Phase 9 server transition implements SemCred-FL fusion,
state, eligibility, clipping, weighting, burn-in, and accepted history without
changing those baseline paths.
