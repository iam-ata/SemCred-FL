# SemCred-FL: Article and Implementation Master Specification

## 0. Purpose of this document

This file is the single handoff document for implementing and evaluating the article:

> **SemCred-FL: LLM-Guided Dual-View Credibility Aggregation for Robust Federated Intrusion Detection under Data Poisoning**

It defines the scientific scope, algorithm, software architecture, experiments, tests, outputs, and completion criteria. It is written for a coding agent that has not seen the earlier conversation.

The implementation must reproduce the method described here. It must not invent experimental values, silently change equations, use the final test set for tuning, or transmit private client data or online model updates to an LLM.

---

## 1. Research objective

Build and evaluate a robust synchronous federated-learning intrusion-detection system for heterogeneous IoT clients. Each submitted client update is assessed from two complementary views:

1. **Semantic credibility:** whether the reconstructed client model behaves correctly on a trusted server anchor set and relies on feature groups that agree with a class-specific cybersecurity semantic prior.
2. **Numerical credibility:** whether the submitted update is consistent with robust current-round update geometry and the client's accepted historical behavior.

The two scores are fused with an explicit disagreement penalty. A persistent trust state is updated asymmetrically, so suspicious behavior lowers trust faster than credible behavior restores it. Trust controls eligibility and aggregation weight. Eligible updates are norm-clipped, sample counts are capped, and only accepted clipped updates enter client history.

### Central hypothesis

Dual-view credibility should suppress poisoned client updates more reliably than a single semantic or numerical view while preserving useful updates from benign non-IID clients.

### Non-negotiable scientific constraints

- Raw traffic records remain at their originating clients.
- The server owns a small, trusted, class-balanced anchor set that is excluded from client training, validation, and final testing.
- The LLM is used **only once, offline**, with public feature definitions and public attack/class descriptions.
- The LLM never receives private records, client identities, model parameters, gradients, submitted updates, trust values, or online decisions.
- The LLM output cannot directly accept, reject, or weight a client.
- The LLM prior is calibrated with anchor-derived evidence before online use.
- Online credibility, trust, eligibility, clipping, and aggregation are deterministic and auditable.
- Benign statistical heterogeneity must not automatically be labeled malicious.
- A malicious minority is assumed in every evaluated round: fewer than half of participating clients are malicious.
- Only eligible, clipped updates may update a client's history.
- Inactive and ineligible clients retain their stored trust/history unchanged.
- Minimum retention must guarantee at least one eligible update.
- All claims in the article must be supported by generated results. Do not insert synthetic or guessed outcomes.

---

## 2. Article structure and required evidence

The final article should contain the following sections.

### 2.1 Introduction

Cover:

- Federated intrusion detection for distributed IoT/edge environments.
- The privacy value of keeping raw traffic local.
- The vulnerability of federated aggregation to data and model poisoning.
- The difficulty of distinguishing malicious behavior from benign non-IID updates.
- Limitations of purely numerical robust aggregation and purely explanation-based defenses.
- The role of an offline, auditable semantic prior.
- The SemCred-FL contribution: dual-view scoring, disagreement-aware fusion, persistent trust, adaptive eligibility, clipping, and trust-aware aggregation.

Do not make numerical performance claims in the introduction until experiments support them.

### 2.2 Related work

Preserve these categories:

1. Federated intrusion-detection architectures.
2. IDS-specific poisoning attacks and defenses.
3. Robust aggregation and trust assessment.
4. LLM-assisted and semantic cybersecurity methods.
5. Research gap.

The gap should state that prior approaches do not jointly test whether an update remains behaviorally correct on trusted traffic and whether its feature dependence is consistent with affected attack-class semantics, while also using numerical geometry, temporal trust, robust clipping, and non-IID-aware retention.

Citation requirements:

- Cite each source once at the claim it supports unless repeated citation is genuinely necessary.
- Do not place duplicate citations to the same paper in one sentence or adjacent sentences.
- Citation numbering is determined by first appearance in the compiled paper; do not manually number references.
- Keep the category order unchanged.

### 2.3 Proposed method

Required subsections:

1. Overview and design objectives.
2. System model and problem formulation.
3. Local intrusion-detection backbone.
4. Threat model.
5. Trusted semantic anchor set.
6. Offline LLM-guided semantic prior.
7. Semantic credibility estimation.
8. Numerical update credibility.
9. Disagreement-aware dual-view fusion.
10. Persistent client trust.
11. Adaptive eligibility and robust update clipping.
12. Credibility-aware robust aggregation.
13. Initialization and burn-in.
14. Operational safeguards and auditability.
15. Complete SemCred-FL procedure.
16. Computational and communication complexity.
17. Initial reproducible configuration.
18. Ablation configuration.

Required proposed-method artifacts:

- End-to-end SemCred-FL framework figure.
- Server-side decision-workflow figure.
- Principal-notation table.
- Initial-configuration table.
- Ablation-variants table.
- Algorithm: offline semantic-prior construction.
- Algorithm: server-side dual-view credibility estimation.
- Algorithm: complete SemCred-FL training and aggregation.

### 2.4 Experimental protocol

Required subsections:

1. Research questions.
2. Datasets and preprocessing.
3. Baselines and standard controls.
4. Attack configuration.
5. Metrics and statistical protocol.
6. Result-reporting requirements.
7. Ablation and sensitivity analysis.
8. Efficiency and communication overhead.

### 2.5 Results and discussion

Create this section only from generated artifacts. It must answer:

- **RQ1:** Does SemCred-FL preserve clean-data detection utility?
- **RQ2:** How robust is it to the five poisoning families?
- **RQ3:** Does it distinguish malicious clients from benign non-IID clients?
- **RQ4:** Are credibility and trust calibrated and temporally stable?
- **RQ5:** Which components drive robustness, and what do they cost?

Discuss failure cases, confidence intervals, corrected significance tests, effect sizes, runtime, communication, and limits. Do not describe a result as superior when its interval/effect does not support that claim.

### 2.6 Conclusion

Summarize the verified contribution and limits. Explicitly retain these boundaries:

- Malicious-minority assumption.
- Trusted-anchor dependence.
- Evaluated attack families only.
- Centralized server orchestration.
- No claim of defense against server compromise, Sybil attacks, privacy inference, or communication-layer attacks unless separately implemented and tested.

---

## 3. Mathematical specification

### 3.1 System model

Let clients be \(\mathcal K=\{1,\ldots,K\}\). Client \(k\) owns

\[
\mathcal D_k=\{(\mathbf x_{k,i},y_{k,i})\}_{i=1}^{n_k},
\]

with \(\mathbf x\in\mathbb R^d\), labels \(y\in\{1,\ldots,C\}\), and non-IID distributions \(P_k(\mathbf x,y)\neq P_j(\mathbf x,y)\). Input coordinates are mapped into \(G\) non-overlapping semantic feature groups \(\mathcal G_j\). One-hot coordinates from one original categorical variable form one group.

At round \(t\), the server selects \(\mathcal S_t\), broadcasts \(\boldsymbol\theta^t\), and each selected client minimizes class-weighted cross-entropy with optional proximal regularization:

\[
F_k(\boldsymbol\theta;\boldsymbol\theta^t)
=-\frac{1}{n_k}\sum_i\omega_{k,y_i}\log p_{\boldsymbol\theta}(y_i\mid\mathbf x_i)
+\frac{\mu}{2}\|\boldsymbol\theta-\boldsymbol\theta^t\|_2^2.
\]

The submitted update is

\[
\boldsymbol\Delta_k^t=\boldsymbol\theta_k^{t+1}-\boldsymbol\theta^t.
\]

The participant set is partitioned into honest and malicious clients, with

\[
|\mathcal M_t|<|\mathcal S_t|/2.
\]

### 3.2 Trusted anchor and semantic prior

The server owns

\[
\mathcal A=\{(\mathbf a_m,y_m)\}_{m=1}^{M},
\]

with class subsets \(\mathcal A_c\). The set is class-balanced, trusted, server-only, and disjoint from all validation/test data and client partitions.

The offline LLM assigns each class/group pair an integer relevance:

\[
q_{c,j}^{\mathrm{LLM}}\in\{0,1,2,3\}.
\]

The anchor-derived relevance is

\[
q_{c,j}^{\mathrm{data}}
=\left[\frac{1}{|\mathcal G_j|}\sum_{h\in\mathcal G_j}
\left(\frac{\mu_{c,h}-\mu_{\neg c,h}}{\sigma_h+\varepsilon}\right)^2\right]^{1/2}.
\]

Normalize each source by its L1 norm:

\[
\bar{\mathbf q}_c^{x}=\frac{\mathbf q_c^{x}}{\|\mathbf q_c^{x}\|_1+\varepsilon},
\qquad x\in\{\mathrm{LLM},\mathrm{data}\}.
\]

Fuse them as

\[
\mathbf r_c=\alpha_L\bar{\mathbf q}_c^{\mathrm{LLM}}
+(1-\alpha_L)\bar{\mathbf q}_c^{\mathrm{data}}.
\]

Persist the prompt, model identifier, decoding parameters, feature/class metadata, raw response, parsed response, schema-validation log, anchor manifest hash, configuration hash, and final \(\mathbf r_c\).

### 3.3 Robust anomaly transform

For scalar values \(v_k\) over the relevant participating population:

\[
\operatorname{MAD}(\mathbf v)=\operatorname{median}_k
|v_k-\operatorname{median}_j(v_j)|.
\]

Use the one-sided robust anomaly transform

\[
z^+(v_k)=\min\left(z_{\max},\max\left(0,
\frac{v_k-\operatorname{median}_j(v_j)}
{1.4826\operatorname{MAD}(\mathbf v)+\varepsilon}\right)\right).
\]

This must be implemented centrally and unit-tested for constant, near-constant, negative, finite, and extreme inputs.

### 3.4 Semantic credibility

Reconstruct the submitted client model:

\[
\boldsymbol\theta_k^{t+1}=\boldsymbol\theta^t+\boldsymbol\Delta_k^t.
\]

Compute anchor losses:

\[
\mathcal L_k^t=-\frac{1}{M}\sum_m\log p_{\boldsymbol\theta_k^{t+1}}(y_m\mid\mathbf a_m),
\]

\[
\mathcal L_g^t=-\frac{1}{M}\sum_m\log p_{\boldsymbol\theta^t}(y_m\mid\mathbf a_m),
\qquad
e_k^t=\mathcal L_k^t-\mathcal L_g^t.
\]

Do not clamp \(e_k^t\) directly; the one-sided \(z^+\) already treats below-median behavior as non-adverse.

For class \(c\), feature group \(j\), use RMS gradient-times-input attribution:

\[
g_{k,c,j}^t=\frac{1}{|\mathcal A_c|}\sum_{\mathbf a_m\in\mathcal A_c}
\left[\frac{1}{|\mathcal G_j|}\sum_{h\in\mathcal G_j}
\left(a_{m,h}\frac{\partial\log p_{\boldsymbol\theta_k^{t+1}}(c\mid\mathbf a_m)}{\partial a_{m,h}}\right)^2\right]^{1/2}.
\]

Normalize \(\mathbf g_{k,c}^t\) by its L1 norm. Measure class-wise cosine discrepancy and average valid classes:

\[
d_{k,\mathrm{sem}}^t=\frac{1}{C}\sum_c\left[1-
\frac{(\bar{\mathbf g}_{k,c}^t)^\top\mathbf r_c}
{\|\bar{\mathbf g}_{k,c}^t\|_2\|\mathbf r_c\|_2+\varepsilon}\right].
\]

If either vector has negligible norm, mark that class unavailable. Average only valid classes. If attribution is unavailable for the client/round, disable the attribution term and renormalize active semantic coefficients.

Semantic credibility is

\[
s_k^t=\exp[-\beta_{\mathrm{loss}}z^+(e_k^t)
-\beta_{\mathrm{attr}}z^+(d_{k,\mathrm{sem}}^t)],
\]

where active nonnegative coefficients sum to one.

### 3.5 Numerical credibility

Compute the coordinate-wise median update:

\[
\mathbf m^t=\operatorname{coordinate\text{-}median}_{k\in\mathcal S_t}(\boldsymbol\Delta_k^t).
\]

Indicators:

\[
d_{k,\mathrm{ctr}}^t=
\frac{\|\boldsymbol\Delta_k^t-\mathbf m^t\|_2}
{\operatorname{median}_j\|\boldsymbol\Delta_j^t-\mathbf m^t\|_2+\varepsilon},
\]

\[
\nu^t=\operatorname{median}_j\|\boldsymbol\Delta_j^t\|_2,
\]

\[
d_{k,\mathrm{norm}}^t=\left|\log
\frac{\|\boldsymbol\Delta_k^t\|_2+\varepsilon}{\nu^t+\varepsilon}\right|,
\]

\[
d_{k,\mathrm{dir}}^t=\frac12\left[1-
\frac{(\boldsymbol\Delta_k^t)^\top\mathbf m^t}
{\|\boldsymbol\Delta_k^t\|_2\|\mathbf m^t\|_2+\varepsilon}\right],
\]

\[
d_{k,\mathrm{hist}}^t=\frac12\left[1-
\frac{(\boldsymbol\Delta_k^t)^\top\bar{\boldsymbol\Delta}_k^t}
{\|\boldsymbol\Delta_k^t\|_2\|\bar{\boldsymbol\Delta}_k^t\|_2+\varepsilon}\right].
\]

Historical discrepancy is unavailable for clients without accepted history. Disable it federation-wide when fewer than three participating clients have valid histories. Normalize historical values only over valid-history participants.

Numerical credibility is

\[
u_k^t=\exp[-\beta_{\mathrm{ctr}}z^+(d_{k,\mathrm{ctr}}^t)
-\beta_{\mathrm{norm}}z^+(d_{k,\mathrm{norm}}^t)
-\beta_{\mathrm{dir}}z^+(d_{k,\mathrm{dir}}^t)
-\beta_{\mathrm{hist}}z^+(d_{k,\mathrm{hist}}^t)].
\]

Renormalize active coefficients to sum to one.

### 3.6 Fusion and persistent trust

Fuse the views using

\[
c_k^t=(s_k^t+\varepsilon)^{\alpha_S}
(u_k^t+\varepsilon)^{1-\alpha_S}
\exp[-\lambda_D|s_k^t-u_k^t|],
\]

then clip \(c_k^t\) to \([0,1]\).

If one complete view is unavailable, use the available view and omit the cross-view disagreement term. Log this event; do not silently substitute a neutral score.

For an existing client:

\[
T_k^t=
\begin{cases}
(1-\eta_\downarrow)T_k^{t-1}+\eta_\downarrow c_k^t,&c_k^t<T_k^{t-1},\\
(1-\eta_\uparrow)T_k^{t-1}+\eta_\uparrow c_k^t,&c_k^t\ge T_k^{t-1},
\end{cases}
\]

with \(0<\eta_\uparrow<\eta_\downarrow\le1\). Initialize a new client's trust as \(T_k^t=c_k^t\). Do not decay trust for inactivity.

### 3.7 Eligibility, clipping, and aggregation

Adaptive threshold:

\[
\tau^t=\max\left[\tau_{\min},
\operatorname{median}_k(T_k^t)-1.4826\kappa_T\operatorname{MAD}(\{T_k^t\})\right].
\]

Initial eligible set:

\[
\mathcal E_t=\{k\in\mathcal S_t:T_k^t\ge\tau^t\}.
\]

Minimum retention:

\[
N_{\min}^t=\max(1,\lceil q_{\min}|\mathcal S_t|\rceil).
\]

If necessary, add excluded clients in descending trust order, with deterministic client-ID tie breaking, until \(|\mathcal E_t|=N_{\min}^t\).

Clipping bound and clipped update:

\[
B^t=c_B\operatorname{median}_{k\in\mathcal E_t}\|\boldsymbol\Delta_k^t\|_2,
\]

\[
\widehat{\boldsymbol\Delta}_k^t=\boldsymbol\Delta_k^t
\min\left(1,\frac{B^t}{\|\boldsymbol\Delta_k^t\|_2+\varepsilon}\right).
\]

For sensitivity experiments with \(\xi>0\), cap sample counts using a declared
multiplier of the eligible-client median. The confirmatory configuration uses
\(\xi=0\), so declared counts do not affect its weights:

\[
\widetilde n_k^t=\min(n_k,n_{\mathrm{cap}}^t).
\]

Weights and global update:

\[
w_k^t=\frac{(\widetilde n_k^t)^\xi(T_k^t+\varepsilon)^\gamma}
{\sum_{j\in\mathcal E_t}(\widetilde n_j^t)^\xi(T_j^t+\varepsilon)^\gamma},
\]

\[
\boldsymbol\theta^{t+1}=\boldsymbol\theta^t+
\sum_{k\in\mathcal E_t}w_k^t\widehat{\boldsymbol\Delta}_k^t.
\]

Weights must be finite, nonnegative, and sum to one within numerical tolerance.

### 3.8 Burn-in and accepted history

For \(0\le t<R_B\), aggregate with the coordinate-wise median:

\[
\boldsymbol\theta^{t+1}=\boldsymbol\theta^t+\mathbf m^t.
\]

Construct preliminary history eligibility:

\[
\mathcal E_t^B=\{k\in\mathcal S_t:c_k^t\ge
\operatorname{median}_{j\in\mathcal S_t}(c_j^t)\}.
\]

Enforce the same minimum-retention count, compute a median-based clipping bound over \(\mathcal E_t^B\), and admit only those clipped updates to history.

After burn-in, admit only \(\mathcal E_t\). The history update is event-driven:

\[
\bar{\boldsymbol\Delta}_k^{t+1}=
\rho_\Delta\bar{\boldsymbol\Delta}_k^t+(1-\rho_\Delta)
\widehat{\boldsymbol\Delta}_k^t.
\]

The first admitted clipped update initializes history directly. Rejected/inactive clients keep their prior histories unchanged.

---

## 4. Default reproducible configuration

These are starting values, not claimed optima. Tune only on validation data.

| Parameter | Default |
|---|---:|
| Clients per round | 10–20 |
| Local epochs | 3 |
| Batch size | 128 or 256 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Hidden widths | 256, 128, 64 |
| Activation | GELU |
| Normalization | LayerNorm |
| Dropout | 0.2 |
| Anchor records per class | 64 |
| Burn-in rounds | 3 |
| `alpha_L` | 0.5 |
| `beta_loss` | 0.5 |
| `beta_attr` | 0.5 |
| Current numerical coefficients | 1/3 each |
| All numerical coefficients when history is valid | 0.25 each |
| `alpha_S` | 0.5 |
| `lambda_D` | 1.0 |
| `eta_down` | 0.6 |
| `eta_up` | 0.1 |
| `gamma` | 2.0 |
| `xi` | 0 |
| `kappa_T` | 2.5 |
| `tau_min` | 0.1 |
| `q_min` | 0.5 |
| `c_B` | 2.5 |
| `rho_delta` | 0.8 |
| Sample-count cap | `2 x` eligible-client median when `xi > 0` |
| Attribution interval `r_sem` | every round initially |
| `z_max` | 20 |
| `epsilon` | `1e-12` |

Configuration validation must reject:

- `eta_up >= eta_down`;
- coefficients outside their allowed ranges;
- active coefficient vectors that cannot be normalized;
- `q_min <= 0` or `q_min > 1`;
- fewer than three selected clients for robust experiments;
- malicious ratio `>= 0.5`;
- anchor overlap with validation, test, or client partitions.

---

## 5. Technology and architecture decisions

### 5.1 Initial stack

- Python 3.11 or later.
- PyTorch for the detector and automatic differentiation.
- NumPy, pandas, and scikit-learn for preprocessing and metrics.
- PyYAML or OmegaConf for configuration.
- PyArrow/Parquet for processed tabular data.
- pytest for tests.
- Ruff for linting and formatting; mypy for typed core modules.
- MLflow is optional; canonical results must also exist as plain JSON/CSV/Parquet files.

Use a custom synchronous FL simulator first. The server needs direct, deterministic access to flattened updates, reconstructed client models, anchor evaluation, attributions, client trust/history, and custom aggregation. Add a Flower adapter only after the reference implementation passes parity tests.

### 5.2 Reproducibility rules

- One root seed controls derived seeds for data splitting, client partitioning, client sampling, local batches, model initialization, and attacks.
- Persist every derived seed.
- Enable deterministic PyTorch behavior when supported.
- Record device, Python/package versions, Git commit, dataset hashes, configuration hash, semantic-prior artifact ID, and start/end timestamps.
- Store train/validation/test/anchor/client split manifests by immutable record ID.
- Run all compared methods on the same manifests, sampled-client schedule, initialization, local batches, and attack realizations.
- Never overwrite completed runs; use content-addressed or unique run directories.

---

## 6. Repository structure

Create this structure:

```text
SemCred-FL/
├── README.md
├── pyproject.toml
├── requirements-colab.txt
├── LICENSE
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   │   ├── edge_iiotset.yaml
│   │   └── ciciot2023.yaml
│   ├── attacks/
│   │   ├── none.yaml
│   │   ├── label_flip.yaml
│   │   ├── clean_label.yaml
│   │   ├── backdoor.yaml
│   │   ├── coordinated.yaml
│   │   ├── model_poison.yaml
│   │   └── adaptive_norm.yaml
│   ├── defenses/
│   │   ├── semcred.yaml
│   │   ├── fedavg.yaml
│   │   ├── coordinate_median.yaml
│   │   ├── trimmed_mean.yaml
│   │   ├── multi_krum.yaml
│   │   ├── agat_fl.yaml
│   │   ├── fed_lsae.yaml
│   │   ├── fedkd_ids.yaml
│   │   └── hada.yaml
│   └── experiments/
│       ├── smoke.yaml
│       ├── clean_utility.yaml
│       ├── robustness.yaml
│       ├── trust_analysis.yaml
│       ├── ablation.yaml
│       ├── sensitivity.yaml
│       └── efficiency.yaml
├── src/semcred_fl/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── preprocessing.py
│   │   ├── edge_iiotset.py
│   │   ├── ciciot2023.py
│   │   ├── splits.py
│   │   ├── partition.py
│   │   ├── anchor.py
│   │   └── manifests.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── losses.py
│   │   └── parameters.py
│   ├── federation/
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── simulator.py
│   │   ├── sampling.py
│   │   ├── state.py
│   │   └── events.py
│   ├── attacks/
│   │   ├── base.py
│   │   ├── label_flip.py
│   │   ├── clean_label.py
│   │   ├── backdoor.py
│   │   ├── coordinated.py
│   │   ├── model_poison.py
│   │   └── adaptive_norm.py
│   ├── semantic/
│   │   ├── feature_groups.py
│   │   ├── prompt.py
│   │   ├── schema.py
│   │   ├── llm_prior.py
│   │   ├── data_prior.py
│   │   ├── prior_fusion.py
│   │   ├── attribution.py
│   │   └── audit.py
│   ├── credibility/
│   │   ├── robust_stats.py
│   │   ├── semantic_score.py
│   │   ├── numerical_score.py
│   │   ├── fusion.py
│   │   └── missing.py
│   ├── aggregation/
│   │   ├── trust.py
│   │   ├── eligibility.py
│   │   ├── clipping.py
│   │   ├── weighting.py
│   │   ├── semcred.py
│   │   └── burn_in.py
│   ├── baselines/
│   │   ├── fedavg.py
│   │   ├── coordinate_median.py
│   │   ├── trimmed_mean.py
│   │   ├── multi_krum.py
│   │   ├── agat_fl.py
│   │   ├── fed_lsae.py
│   │   ├── fedkd_ids.py
│   │   └── hada.py
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── attack_metrics.py
│   │   ├── trust_metrics.py
│   │   ├── efficiency.py
│   │   ├── statistics.py
│   │   ├── reporting.py
│   │   └── plots.py
│   ├── portability/
│   │   └── colab_bundle.py
│   └── utils/
│       ├── hashing.py
│       ├── logging.py
│       ├── reproducibility.py
│       └── validation.py
├── scripts/
│   ├── build_colab_bundle.py
│   ├── prepare_dataset.py
│   ├── build_semantic_prior.py
│   ├── run_experiment.py
│   ├── run_matrix.py
│   ├── aggregate_results.py
│   └── make_paper_artifacts.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── notebooks/
│   └── 00_colab_failsafe.ipynb
├── artifacts/
│   ├── manifests/
│   ├── semantic_priors/
│   └── audit/
├── results/
│   ├── runs/
│   ├── summaries/
│   ├── tables/
│   └── figures/
└── docs/
    ├── LOCAL_AND_COLAB_RUNBOOK.md
    ├── data_cards/
    ├── experiment_registry.md
    └── equation_traceability.md
```

Do not commit raw datasets, secrets, temporary checkpoints, or large generated run directories.

Local execution is primary. The Colab fail-safe bundle contains only source code,
configuration, validation assets, Phase 3C transformed shards, and Phase 3D client
assignments. It excludes raw datasets and SQLite work databases, records a SHA-256
digest for every included file, and must pass full manifest verification before any
Colab command runs. Colab's CUDA-enabled PyTorch remains runtime-provided; the
Windows-specific CUDA wheel is not installed there. The notebook exposes only
implemented execution gates and must stop before real-data FL or attack commands when
their wrappers and contracts are not frozen.

---

## 7. Core interfaces and data contracts

Use typed dataclasses or validated models.

```python
@dataclass(frozen=True)
class ClientUpdate:
    round_id: int
    client_id: str
    delta: Tensor
    declared_sample_count: int
    is_malicious: bool | None  # simulation/evaluation only; defense must not read it
    attack_metadata: dict[str, Any] | None

@dataclass
class ClientState:
    trust: float | None
    accepted_history: Tensor | None
    first_seen_round: int
    last_seen_round: int
    accepted_rounds: list[int]

@dataclass(frozen=True)
class CredibilityRecord:
    semantic_score: float | None
    numerical_score: float | None
    fused_score: float
    semantic_indicators: dict[str, float | None]
    numerical_indicators: dict[str, float | None]
    active_coefficients: dict[str, float]
    missing_reasons: dict[str, str]

@dataclass(frozen=True)
class AggregationRecord:
    threshold: float
    eligible_client_ids: tuple[str, ...]
    clipping_bound: float
    sample_count_cap: float
    weights: dict[str, float]
```

Critical rule: the defense code must never access `is_malicious` or `attack_metadata`. Those fields are reserved for evaluation.

The canonical client update must be a single deterministic flattened parameter vector with a stored parameter-name/shape manifest. Provide reversible `flatten_parameters` and `unflatten_parameters` utilities and test exact round trips.

---

## 8. Equation-to-code traceability

Create `docs/equation_traceability.md` and keep it synchronized with this table.

| Paper label/concept | Required implementation | Minimum test |
|---|---|---|
| Client/data/model sets | `data/schemas.py`, `federation/state.py` | schema and dimensionality validation |
| Local objective/update | `models/losses.py`, `federation/client.py` | zero-step update, proximal term, deterministic training |
| Anchor set | `data/anchor.py` | class balance and zero split overlap |
| LLM relevance | `semantic/schema.py`, `semantic/llm_prior.py` | reject non-integer/out-of-range/missing scores |
| Data relevance | `semantic/data_prior.py` | hand-calculated fixture |
| Prior normalization/fusion | `semantic/prior_fusion.py` | L1 normalization and endpoint tests |
| Reconstructed model | `models/parameters.py` | parameter equality after delta application |
| Anchor degradation | `credibility/semantic_score.py` | known client/global losses |
| Group attribution | `semantic/attribution.py` | shape, grouping, gradients, finite outputs |
| Semantic discrepancy | `credibility/semantic_score.py` | identical vector gives near zero discrepancy |
| MAD and robust z | `credibility/robust_stats.py` | constant/near-constant/outlier cases |
| Semantic credibility | `credibility/semantic_score.py` | range and monotonicity |
| Coordinate median | `credibility/robust_stats.py` | odd/even client counts |
| Center/norm/direction/history indicators | `credibility/numerical_score.py` | analytical vector fixtures |
| Numerical credibility | `credibility/numerical_score.py` | missing-history coefficient renormalization |
| Fusion | `credibility/fusion.py` | symmetry at `alpha_S=.5`, disagreement monotonicity, clipping |
| Asymmetric trust | `aggregation/trust.py` | faster fall than recovery, inactive unchanged |
| Threshold/eligibility | `aggregation/eligibility.py` | floor, MAD, deterministic tie break, minimum retention |
| Clipping | `aggregation/clipping.py` | bound respected, direction preserved, zero update finite |
| Sample cap/weights | `aggregation/weighting.py` | cap enforced, weights finite/nonnegative/sum to one |
| Global update | `aggregation/semcred.py` | analytical weighted sum |
| Burn-in | `aggregation/burn_in.py` | global coordinate median and controlled history admission |
| History EMA | `federation/state.py` | first admission, subsequent EMA, rejection/inactivity unchanged |
| Complete procedure | `federation/server.py` | deterministic multi-round integration scenario |

---

## 9. Dataset pipeline

### 9.1 Datasets

Implement separate adapters for:

- **Edge-IIoTset**
- **CICIoT2023**

Do not assume identical labels or feature schemas. Each adapter must expose:

- source files and expected hashes;
- canonical record IDs;
- original and mapped labels;
- categorical and continuous columns;
- feature descriptions and units when available;
- semantic feature-group mapping;
- invalid-row and duplicate-removal report;
- final class distribution.

If downloads are automated, use official dataset sources and record URLs/checksums. If licensing blocks automation, fail with clear placement instructions rather than substituting another dataset.

### 9.2 Leakage-safe order

For every seed:

1. Load and validate raw rows.
2. Create stable record IDs before filtering.
3. Remove duplicates, invalid labels, and non-finite rows; log counts.
4. Create stratified train/validation/test partitions.
5. Select the trusted anchor from the training partition only.
6. Remove anchor rows from the federated training pool.
7. Fit categorical vocabulary, clipping quantiles, and scaling statistics on the remaining federated training pool only.
8. Transform validation, test, anchor, and training data with the frozen preprocessing object.
9. Allocate the federated training pool to clients.
10. Persist record-ID manifests and verify pairwise disjointness.

### 9.3 Non-IID partitioning

Support:

- IID control.
- Dirichlet label allocation with `alpha_D` in `{0.1, 0.3, 1.0}`.
- A frozen population of 50 clients, with 20 submitted updates per communication round.

Requirements:

- No record duplication across clients.
- Every assigned record belongs to exactly one client.
- A minimum of 128 samples per client, equal to one configured local batch.
- Deterministic retry logic if a partition violates minimum size.
- Persist client-by-class count matrices.
- Report realized heterogeneity rather than only requested alpha.

### 9.4 Anchor construction

Default: 64 records per class, sampled from training before client allocation.

Persist:

- record IDs;
- label counts;
- feature schema version;
- preprocessing artifact hash;
- source dataset hash;
- seed;
- overlap validation report.

---

## 10. Detector and local training

Default MLP:

```text
input(d)
→ Linear(d, 256) → LayerNorm → GELU → Dropout(0.2)
→ Linear(256, 128) → LayerNorm → GELU → Dropout(0.2)
→ Linear(128, 64) → LayerNorm → GELU → Dropout(0.2)
→ Linear(64, C)
```

Return logits during training; apply softmax only when probabilities are required.
Use class-weighted cross-entropy and an optional FedProx-style proximal loss. For a
training population with \(N\) records, observed class set \(\mathcal C^+\), and class
count \(n_c\), define

\[
\omega_c=
\begin{cases}
\min\!\left(10,\dfrac{N}{|\mathcal C^+|n_c}\right), & n_c>0,\\
0, & n_c=0.
\end{cases}
\]

The weighted cross-entropy must retain the manuscript normalization
\(-N^{-1}\sum_i\omega_{y_i}\log p(y_i\mid\mathbf x_i)\); it must not silently use a
weight-normalized library reduction. Compute these weights from each client's
training data during federated training and record every missing class. The
centralized Phase 4 sanity runner uses the complete Phase 3D non-anchor training
population counts to construct the same deterministic weight vector.

Local-training output must include:

- flattened update;
- declared sample count;
- mean local loss;
- epoch timing;
- update norm;
- finite-value check.

Reject non-finite updates before scoring and log the rejection reason. Define a deterministic fallback when all received updates are invalid: keep the global model unchanged and record the round as failed; do not fabricate an update.

---

## 11. Federated simulator

Implement a synchronous round loop:

1. Select clients from a pre-generated schedule.
2. Copy/broadcast the same global state to each selected client.
3. Apply configured data poisoning before local optimization or model poisoning after optimization.
4. Collect `ClientUpdate` records.
5. Validate shape and finite values.
6. Compute semantic and numerical indicators over valid participants.
7. Apply missing-aware coefficient renormalization.
8. Fuse scores and update persistent trust.
9. During burn-in, use coordinate-median aggregation and controlled history admission.
10. After burn-in, calculate adaptive eligibility, enforce minimum retention, clip updates, cap counts, compute weights, and aggregate.
11. Update histories only from admitted clipped updates.
12. Evaluate the global model on validation/test only at configured checkpoints.
13. Append immutable round logs and checkpoints.

Partial participation is required. Client selection must be independent of malicious status unless the experiment explicitly defines an availability scenario.

### Required round log columns

- run ID, seed, dataset, defense, attack, round;
- selected client ID;
- simulated malicious label (evaluation namespace only);
- update norm and declared/capped sample count;
- every raw semantic/numerical indicator;
- every robust z-score;
- active/missing indicators and reasons;
- semantic, numerical, fused credibility;
- previous and updated trust;
- threshold and eligibility;
- clipping bound and clipped norm;
- aggregation weight;
- whether history was initialized/updated;
- component timings and bytes;
- global validation metrics at evaluation rounds.

Use a long-form Parquet file plus a small JSON round summary.

---

## 12. Attack implementation

All attacks require a common interface, deterministic seeded behavior, audit metadata, and a clean no-op configuration.

### 12.1 Label flipping

- Poison fractions: `{0.1, 0.3, 0.5}`.
- Map a fixed source class to a fixed target/benign class.
- Do not modify validation, test, or anchor data.
- Log selected record IDs and before/after labels.

### 12.2 Clean-label feature poisoning

- Preserve labels.
- Modify declared attack-relevant continuous features.
- Keep values within admissible or training-derived bounds.
- Provide explicit feature list, direction/magnitude rule, and record IDs.
- Ensure transformed values use the correct raw/preprocessed coordinate system.

### 12.3 Backdoor poisoning

- Define a fixed trigger, source population, and target behavior.
- Apply trigger only to selected malicious-client training rows.
- Keep a separate triggered test set derived from clean test records.
- Measure backdoor success rate on non-target source examples after trigger insertion.
- Also measure benign source accuracy/recall to detect degenerate target-class prediction.

### 12.4 Coordinated poisoning

- Multiple malicious clients use identical or complementary strategies.
- Coordination must be defined in configuration and reproducible.
- Log coalition membership and shared attack parameters.

### 12.5 Direct model poisoning

Implement separately:

- scaling: `delta' = scale * delta`;
- sign reversal: `delta' = -scale * delta`;
- model/update replacement using a precisely documented construction.

Do not call these equivalent attacks in analysis; report each manipulation.

### 12.6 Adaptive constrained-norm poisoning

Implement an attacker that targets a minority class while constraining its final update norm near the benign/round reference distribution. The attacker must not read the private anchor set. Document what defense details it knows. Use this to test whether semantic evidence adds value when numerical magnitude is camouflaged.

### 12.7 Attack matrix

- Malicious-client ratio: `{0.1, 0.2, 0.3, 0.4}`.
- Poisoned-record fraction for data attacks: `{0.1, 0.3, 0.5}`.
- Heterogeneity: IID plus Dirichlet `{0.1, 0.3, 1.0}`.
- Seeds: five independent seeds.

Begin with a reduced development matrix; run the full matrix only after phase gates pass.

---

## 13. Offline semantic-prior pipeline

This is a separate command and artifact from federated training.

### Input

- Public dataset feature dictionary.
- Feature groups after preprocessing.
- Public attack/class definitions.
- Trusted anchor set and preprocessing artifact.
- Fixed prompt template.
- Fixed LLM model identifier and deterministic decoding settings.

### Required response schema

```json
{
  "schema_version": "1.0",
  "dataset": "...",
  "classes": {
    "class_name": {
      "feature_group_name": {
        "score": 0,
        "reason": "short public-domain rationale"
      }
    }
  }
}
```

Every class must contain every feature group exactly once. Scores must be integers in `{0,1,2,3}`. Unknown groups, missing groups, duplicate keys, prose outside the schema, or invalid scores must fail validation. Do not silently repair an invalid response.

### Offline workflow

1. Build prompt from public metadata.
2. Save exact prompt before the request.
3. Perform deterministic LLM request.
4. Save raw response.
5. Validate schema strictly.
6. Compute anchor-derived relevance.
7. Normalize both sources.
8. Fuse with `alpha_L`.
9. Verify finite, nonnegative priors and valid dimensions.
10. Save an immutable semantic-prior bundle and its hash.

### No-LLM reproducibility mode

Training must accept a previously saved, validated semantic-prior bundle. Reproducing experiments must not require another LLM call. Include a clearly labeled anchor-only prior (`alpha_L=0`) for ablation, not as an undisclosed substitute.

---

## 14. Baselines

### Required standard controls

- FedAvg.
- Coordinate-wise median.
- Trimmed mean.
- Multi-Krum.

### Planned IDS-specific comparators, conditional on protocol fidelity

- AGAT-FL.
- Fed-LSAE.
- FedKD-IDS.
- HADA.

Implementation rule: reproduce each published decision mechanism faithfully. A
paper-specific method enters the confirmatory matrix under its original name only
when the available paper/code evidence defines its operative protocol. If essential
details are unavailable, implement only the verifiable core, record the boundary,
and label any runnable extension as a reimplementation, adaptation, or surrogate.
Do not create a simplified method under the original paper's name.

Fairness requirements:

- Same detector when the method permits.
- Same preprocessing and split manifests.
- Same sampled-client schedule and model initialization.
- Same local batches and attack realization when technically compatible.
- Independent validation tuning for each method.
- No test-set tuning.
- Report method-specific communication/runtime overhead.

Before implementing paper-specific baselines, obtain and archive the authoritative paper/code citation and an implementation note mapping its equations to local code.

---

## 15. Metrics

### 15.1 Detection utility

- Macro-F1 (**primary clean and attacked utility metric**).
- Accuracy.
- Macro precision.
- Macro recall.
- Balanced accuracy.
- Per-class recall.
- False-positive rate, with the positive/negative convention explicitly defined.
- Confusion matrix.

### 15.2 Robustness

- Attacked macro-F1.
- Absolute degradation: `clean_macro_f1 - attacked_macro_f1`.
- Relative degradation: `(clean - attacked) / max(clean, epsilon)`.
- Attack success rate with an attack-specific denominator.
- Backdoor success rate.
- Worst-class recall.

Always store numerator, denominator, and undefined-reason fields for rate metrics.

### 15.3 Client assessment

Use simulated malicious identity only after the defense has produced its scores.

- Malicious-client AUROC using `1 - score/trust` as suspiciousness where appropriate.
- Malicious-client AUPRC.
- Benign false-rejection rate.
- Malicious false-acceptance rate.
- Detection delay after attack onset.
- Trust-recovery delay after malicious behavior stops or benign participation resumes.
- Benign–malicious trust separation.
- Trust volatility for benign clients.
- Results stratified by client heterogeneity/sample size to expose bias against rare benign clients.

### 15.4 Efficiency

- Client training time.
- Global/anchor inference time.
- Attribution time.
- Numerical scoring time.
- Fusion/trust/eligibility/clipping/aggregation time.
- Peak CPU/GPU memory.
- Broadcast and upload bytes per round.
- Rounds/time to convergence under a predeclared criterion.
- One-time LLM request cost, token count, latency, and model identifier.

### 15.5 Statistical reporting

- Five independently generated seeds per condition.
- Mean, standard deviation, median, and percentile-bootstrap 95% confidence interval.
- Matched comparisons use identical seeds/manifests/attacks.
- Paired Wilcoxon signed-rank test for two-method contrasts.
- Holm correction across each declared family of comparisons.
- Rank-biserial correlation as effect size.
- Handle exact ties/zero differences explicitly and log the chosen convention.

The experimental registry must define the comparison families before final analysis.

---

## 16. Ablations and sensitivity analysis

### Required ablations

| Variant | Change |
|---|---|
| Without semantic credibility | numerical view only |
| Without numerical credibility | semantic view only |
| Without disagreement penalty | `lambda_D = 0` |
| Without persistent trust | set `T_k^t = c_k^t` each round |
| Without controlled burn-in | admit every clipped burn-in update to history |
| LLM-derived prior only | `alpha_L = 1` |
| Anchor-derived prior only | `alpha_L = 0` |
| Without robust clipping | `B^t = infinity` |
| Sample-size averaging | no eligibility, clipping, or trust-aware weighting |
| Complete SemCred-FL | all components active |

When only one credibility view is active, omit the disagreement term rather than assigning an artificial value to the missing view.

### Sensitivity variables

- Anchor records per class.
- `alpha_L`.
- `alpha_S`.
- `lambda_D`.
- `eta_down`, `eta_up`.
- `q_min`.
- `c_B`.
- `r_sem`.
- Attribution method: gradient-times-input versus Integrated Gradients on a fixed validation subset.
- Controlled anchor contamination fraction.

Use one-factor-at-a-time analysis for interpretability, followed by a limited joint sensitivity study for the most interacting parameters. Do not use the final test set to choose defaults.

---

## 17. Required paper tables and figures

Generate all result artifacts from machine-readable summaries; never type result numbers manually into LaTeX.

### Main tables

1. Dataset statistics after cleaning and splitting.
2. Clean utility across datasets and heterogeneity levels.
3. Robustness across attack families and malicious ratios.
4. Backdoor/targeted-attack results including ASR/BSR and source/target recall.
5. Client-assessment results: AUROC, AUPRC, false rejection/acceptance, delay.
6. Ablation results.
7. Efficiency and communication overhead.
8. Statistical comparisons with corrected p-values and effect sizes.

### Main figures

1. Existing end-to-end SemCred-FL framework.
2. Existing server-side workflow.
3. Macro-F1 versus malicious-client ratio.
4. Attack/backdoor success rate versus malicious-client ratio.
5. Representative trust trajectories for benign, malicious, intermittent, and newly joined clients.
6. Benign versus malicious credibility/trust distributions.
7. Per-class recall or confusion matrices for targeted attacks.
8. Ablation/sensitivity plot.
9. Runtime breakdown.

### Placement in the article

- Method framework: after the proposed-method overview.
- Notation table: after system model/problem formulation.
- Server workflow: immediately before the complete online algorithm.
- Initial configuration and ablation tables: at the end of the proposed-method section.
- Dataset table: dataset/preprocessing subsection.
- Main clean/attack tables: corresponding Results subsections after first textual reference.
- Trust figures/table: RQ3/RQ4 results.
- Efficiency table/figure: RQ5/overhead subsection.

Every table/figure must be referenced in the text before or at its appearance and must include a self-contained caption.

---

## 18. Implementation phases and gates

Complete phases in order. Do not start the full experiment matrix before all earlier gates pass.

### Phase 1 — Repository foundation

Tasks:

- Create package tree and `pyproject.toml`.
- Add typed configuration models and validation.
- Add structured logging, hashing, seed derivation, environment capture, and CLI skeleton.
- Add CI commands for lint, type check, unit tests, and smoke test.

Gate:

- Fresh environment installs successfully.
- `pytest`, Ruff, and mypy run.
- Same configuration produces the same run/config hash.

### Phase 2 — Dataset ingestion and preprocessing

Tasks:

- Implement both dataset adapters.
- Produce data cards and cleaning reports.
- Fit leakage-safe preprocessing.
- Persist preprocessing artifact and feature-group manifest.

Gate:

- No non-finite model input.
- Stable feature order/dimension.
- Duplicate/invalid removal counts reproducible.
- A fitted preprocessor transforms all splits without refitting.

### Phase 3 — Splits, anchor, and client partitioning

Tasks:

- Stratified splits.
- Anchor extraction.
- IID/Dirichlet partitioner.
- Immutable manifests and overlap checks.

Gate:

- Train/validation/test/anchor/client sets are disjoint as specified.
- All federated training records are assigned exactly once.
- Same seed reproduces byte-identical manifests.

### Phase 4 — Detector and centralized sanity baseline

Tasks:

- Implement MLP and loss.
- Train centralized model only as a pipeline sanity check.
- Verify metrics and checkpoints.

Gate:

- Loss decreases on a tiny learnable fixture.
- Save/load produces identical predictions.
- Metric tests match known scikit-learn fixtures.
- Validation-only real-data wiring reports record class support, test-set exclusion,
  and whether performance claims are permitted.

### Phase 5 — Basic FL engine

Tasks:

- Implement client training, update flattening, sampling, FedAvg, checkpoints, logs.
- Add coordinate median.

Gate:

- One-client FedAvg matches local training.
- Weighted FedAvg matches an analytical fixture.
- Multi-round run is deterministic.
- Resume-from-checkpoint matches uninterrupted run.

Implementation convention: coordinate median uses the ordinary midpoint definition
for an even participant count, applied independently to every flattened parameter
coordinate. Canonical round logs exclude nondeterministic duration fields and form a
SHA-256 chain. Durations remain required telemetry but are stored outside that
canonical chain. A resume checkpoint is accepted only when its next-round cursor and
last-record hash match the existing log.

### Phase 6 — Attacks

Phase 6A freezes the attack contract before any poisoning result is observed. The
authoritative contract is `configs/attacks/phase6a.yaml`; typed invariants are enforced
by `src/semcred_fl/attacks/contracts.py`. Dataset source families are selected from
federated-training metadata only: `injection -> benign` for Edge-IIoTset and
`web_based -> benign` for CICIoT2023. Feature, trigger, and benign-norm values are
resolved in Phase 6B under the train-only procedures fixed by that contract.

Tasks:

- Implement all attack classes and audit metadata.
- Build triggered test-set generator.

Gate:

- Poison fraction, label behavior, feature bounds, trigger presence, and update transformation pass tests.
- Clean no-op path is exactly unchanged.
- Defense code cannot access simulated attacker labels.

### Phase 7 — Offline semantic prior

Tasks:

- Implement prompt/schema/audit bundle.
- Compute anchor-derived prior.
- Fuse priors and add no-LLM loading mode.

Gate:

- Invalid response fails closed.
- Bundle is immutable/hashable.
- `alpha_L=0` and `alpha_L=1` exactly recover endpoint priors.
- No private/client/update data is present in prompt or audit payload.

### Phase 8 — Semantic and numerical credibility

Tasks:

- Implement robust statistics and missing-value policy.
- Implement anchor loss, grouped attribution, semantic discrepancy/score.
- Implement numerical indicators and score.

Gate:

- Every equation-level unit test passes.
- All scores are finite and in `(0,1]` within tolerance.
- Adverse synthetic indicators monotonically reduce credibility.
- Missing history/attribution renormalizes coefficients correctly.

### Phase 9 — Fusion, trust, and robust aggregation

Tasks:

- Implement fusion, asymmetric trust, threshold, eligibility, clipping, sample cap, weights, history, burn-in.
- Integrate into server.

Gate:

- Minimum retention never returns an empty set.
- Clipped norms never exceed the bound beyond tolerance.
- Weights are finite, nonnegative, and sum to one.
- Trust falls faster than it recovers for equal magnitude evidence.
- Only accepted clipped updates enter history.
- Inactivity/rejection leaves state unchanged.

### Phase 10 — Baselines

Tasks:

- Standard controls first.
- Paper-specific baselines with implementation notes and parity checks.

Gate:

- Shared inputs/schedules verified.
- Standard baselines match analytical or trusted small fixtures.
- Deviations from source methods documented.

Implementation record (2026-09-01): the standard-control and evidence-bounded
core gate is complete. FedAvg, coordinate median, trimmed mean, and Multi-Krum
share the frozen runner and client schedule. Fed-LSAE and native FedKD-IDS server
cores have analytical/source-fixture checks. The official AGAT-FL Zenodo fragment
supports only its normalized trust-weighted sum, so full AGAT-FL remains excluded
under its original name pending sufficient protocol evidence. See
`docs/PHASE_10_BASELINES.md`.

### Phase 11 — Evaluation and reporting

Tasks:

- Implement detection, attack, trust, efficiency, and statistical metrics.
- Implement result aggregation and plot/table generation.

Gate:

- Known metric fixtures pass.
- Aggregate script is deterministic.
- Every displayed value traces to run IDs.
- Figures/tables regenerate from clean summaries.

Implementation record (2026-09-01): the Phase 11 gate is complete. Detection
utility reuses the exact confusion-count accumulator; attack, client-assessment,
temporal-trust, efficiency, exact paired Wilcoxon, rank-biserial, bootstrap, and
Holm-family metrics are implemented under `evaluation/`. The deterministic
aggregator accepts exact-schema run records and emits content-addressed clean
summaries plus CSV, LaTeX, SVG, and JSON provenance artifacts. Every rendered
value maps through a summary ID to all contributing run IDs. See
`docs/PHASE_11_EVALUATION_REPORTING.md`.

### Phase 12 — Full experiments and article artifacts

Order:

1. Tiny synthetic smoke run.
2. One dataset, clean FedAvg/SemCred run.
3. One attack, one non-IID level, one seed.
4. Reduced baseline matrix.
5. Full clean utility matrix.
6. Full robustness matrix.
7. Trust analysis.
8. Ablations.
9. Sensitivity and anchor contamination.
10. Efficiency profiling.
11. Five-seed statistical aggregation.
12. Generate manuscript tables and figures.

Gate:

- All planned conditions accounted for as completed/failed with reasons.
- No test tuning.
- No missing provenance.
- Manuscript claims pass an evidence audit.

Implementation record (2026-09-01): Step 1 is complete. The deterministic
synthetic runner executes four clients for three rounds, realizes one sign-reversal
event outside the defense contract, evaluates semantic and numerical credibility,
performs SemCred-FL aggregation, computes synthetic-only metrics, and regenerates
the Phase 11 summary/table/plot/provenance bundle. The report is content-addressed,
immutable, idempotent, and explicitly forbids performance claims. No real dataset
or real test partition was accessed. Phase 12 as a whole remains in progress; Step
2 is the bounded Edge-IIoTset clean FedAvg/SemCred comparison. See
`docs/PHASE_12_EXPERIMENTS.md`.

Step 2 implementation record (2026-09-01): the clean Edge-IIoTset pilot completed.
FedAvg and SemCred-FL share the
same initialization, IID client schedule, cached training records, detector, and
local optimizer. The frozen pilot uses three rounds, ten of fifty clients per round,
one local epoch, and batch size 256. CUDA execution uses fused Adam, prevalidated
immutable cache batches, and one device transfer per client. Exact-input update
reuse and checksum-verified method checkpoints reduce repeated work without reading
malicious metadata or the final test partition. A training-only real-client
benchmark measured a 2.2365x hot-path speedup; it is not a detection-performance
result. The comparison result ID is
`4fe709aa51000eace41b957881ddba7ce4a9b2a0da8d38796170b50414ddc0be`.
Both method and comparison identities verify, validation was accessed only after
training, and the test partition was not accessed. Both methods had zero
worst-class recall and nonzero recall for only two of six validation classes, so this pilot
closes the integration gate but cannot support method ranking or manuscript claims.

Step 3 implementation record (2026-09-02): the attacked non-IID pilot completed.
The frozen condition is Edge-IIoTset with the
`dirichlet_0_3` partition, label flipping, malicious ratio 0.2, poison fraction
0.3, seed `20260816`, three rounds, ten of fifty clients per round, one local epoch,
and batch size 256. The deterministic schedule contains 27 unique clients; the
runner builds a resumable schedule-bound cache for only those clients. FedAvg and
SemCred-FL share initialization, schedule, attack realization, local optimizer, and
completion-only validation. Exact-input local updates may be reused across methods,
and method checkpoints verify their content identities and checksums. Simulation
truth is removed before semantic scoring, numerical scoring, trust update, and
aggregation, then joined only after each defense decision for client-assessment
metrics. The test partition is not read and the pilot cannot authorize performance
claims. The comparison result ID is
`3c6ada958211a1b8cfde56d7e787573c50214a0b4ca36a8501ae4e20abc20a67`.
FedAvg and SemCred-FL macro-F1 values were 0.2366 and 0.2447, respectively; both
had zero recall for classes 2 through 5. SemCred-FL malicious-client assessment
gave AUROC 0.5833, AUPRC 0.3731, benign false rejection 4/24, and malicious false
acceptance 4/6. These single-seed, three-round results close the Step 3 execution
gate but cannot support a method-ranking or performance claim. Because Step 2 was
IID and Step 3 was non-IID, their difference is not an estimate of attack impact.
See `docs/PHASE_12_EXPERIMENTS.md`.

Step 4 implementation record (2026-09-02): the reduced standard-baseline runner
completed. It consumed the immutable Step 3 FedAvg and SemCred-FL reports and
executed
coordinate median, trimmed mean, and Multi-Krum under the same Edge-IIoTset
`dirichlet_0_3` label-flip realization, initialization, client schedule, local
optimizer, and validation protocol. With ten submitted updates and malicious ratio
0.2, trimmed mean removes two values per coordinate tail; Multi-Krum uses `f=2` and
selects eight updates. Exact-match Step 3 local updates can be reused, while new
updates and round checkpoints remain isolated under Step 4. The matrix result ID is
`bca0fb435639192f3ee25d74fbb566f911a73099c0cb33cf13f5c740621efce3`.
Observed macro-F1 was 0.244747 for SemCred-FL, 0.237371 for coordinate median,
0.237251 for trimmed mean, 0.237133 for Multi-Krum, and 0.236565 for FedAvg. All
methods had zero worst-class recall, so the single-seed three-round ordering cannot
support a winner claim. Nine focused tests,
94% branch-aware module coverage, a 145-test regression suite, Ruff, formatting,
and strict mypy pass. The completed Step 4 run remains non-claim-bearing.
See `docs/PHASE_12_EXPERIMENTS.md`.

Step 5 implementation record (2026-09-02): the clean utility matrix runner is
resumable across methods, rounds, conditions, and repeated invocations. Its initial
full budget is 50 clients, 20 submissions per round, 100 rounds, three local epochs,
and batch size 128. It covers both prepared datasets, four heterogeneity profiles,
and the five standard parameter-update methods. Within a condition, all methods
share initialization, client schedule, local optimizer, and client cache. Exact
local updates are reused only when the complete update input and broadcast-parameter
hash match. Validation is completion-only and the test partition remains sealed.
CICIoT2023 is explicitly secondary while its original-dataset license provenance is
unresolved.
The implementation gate passes 26 focused tests, 94% branch-aware module coverage,
a 232-test regression suite, Ruff, and strict mypy. All six real input manifests
exist, and the configured PyTorch runtime detects the local RTX 3050 GPU. Each
matrix input ID owns an isolated condition tree, so a changed scientific control
cannot collide with a prior campaign. These checks did not execute the full-budget
matrix or access the test partition.

---

## 19. Test plan

### Unit tests

- Preprocessing fit/transform separation.
- Feature order and feature groups.
- Split overlap and record conservation.
- Dirichlet determinism.
- Flatten/unflatten.
- Proximal loss.
- Each attack transformation.
- LLM schema validation.
- Data-prior calculation.
- Prior normalization/fusion.
- Attribution grouping.
- MAD and robust z.
- All semantic/numerical indicators.
- Missing-indicator coefficient renormalization.
- Fusion and disagreement penalty.
- Trust update.
- Threshold/minimum retention.
- Clipping/sample cap/weights.
- History update and burn-in.
- Every metric and statistical correction.

### Integration tests

- Clean two-round FL.
- Burn-in to post-burn-in transition.
- New/inactive/rejoining client behavior.
- Mixed honest/malicious synthetic round.
- Reconstructed-model anchor scoring.
- Checkpoint resume.
- Artifact generation from a tiny run.

### Scientific regression tests

Use small fixed synthetic tensors, not claimed paper results:

- identical benign updates receive equal numerical scores;
- a very large reversed update receives worse numerical credibility;
- attribution aligned with its prior has lower semantic discrepancy;
- disagreement penalty lowers fusion when views diverge;
- minimum retention prevents federation collapse;
- one rejected update cannot contaminate its history;
- a declared huge sample count is capped;
- a benign client can recover gradually after an anomalous round.

### Leakage/privacy tests

- Assert no record-ID overlap among forbidden partitions.
- Scan the serialized LLM prompt/bundle for record values, client IDs, update tensors, model parameters, or trust logs.
- Verify the final test loader is unavailable to tuning code.
- Verify defense modules cannot import/read attack ground-truth fields.

---

## 20. Experiment registry

Each run must be represented by a resolved configuration with at least:

```yaml
run:
  seed: 0
  output_root: results/runs
dataset:
  name: edge_iiotset
  version: null
  raw_hash: null
  split_seed: 0
  dirichlet_alpha: 0.3
  num_clients: 50
  clients_per_round: 20
  minimum_samples_per_client: 128
  anchor_per_class: 64
model:
  hidden_dims: [256, 128, 64]
  dropout: 0.2
training:
  rounds: 100
  local_epochs: 3
  batch_size: 128
  optimizer: adam
  learning_rate: 0.001
  proximal_mu: 0.0
attack:
  name: label_flip
  malicious_ratio: 0.2
  poisoned_record_fraction: 0.3
  source_class: null
  target_class: null
defense:
  name: semcred
  semantic_prior_path: null
  burn_in_rounds: 3
  alpha_L: 0.5
  beta_loss: 0.5
  beta_attr: 0.5
  beta_ctr: 0.25
  beta_norm: 0.25
  beta_dir: 0.25
  beta_hist: 0.25
  alpha_S: 0.5
  lambda_D: 1.0
  eta_down: 0.6
  eta_up: 0.1
  gamma: 2.0
  xi: 0.0
  kappa_T: 2.5
  tau_min: 0.1
  q_min: 0.5
  clipping_multiplier: 2.5
  rho_delta: 0.8
  semantic_interval: 1
  z_max: 20.0
  epsilon: 1.0e-12
evaluation:
  checkpoint_interval: 1
  bootstrap_resamples: 10000
```

`null` values above are mandatory experiment-specific inputs, not permission to guess. Resolve and freeze them before a run.

---

## 21. Command-line contract

Implement equivalent commands:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/edge_iiotset.yaml

python scripts/build_semantic_prior.py \
  --dataset-config configs/datasets/edge_iiotset.yaml \
  --defense-config configs/defenses/semcred.yaml

python scripts/run_experiment.py \
  --config configs/experiments/smoke.yaml

python scripts/run_matrix.py \
  --config configs/experiments/robustness.yaml \
  --dry-run

python scripts/run_matrix.py \
  --config configs/experiments/robustness.yaml

python scripts/aggregate_results.py \
  --runs results/runs \
  --output results/summaries

python scripts/make_paper_artifacts.py \
  --summaries results/summaries \
  --tables results/tables \
  --figures results/figures
```

Every command must support `--help`. Matrix dry-run must print the exact number of runs and estimated combinations before execution.

---

## 22. Result and artifact schema

Each run directory should contain:

```text
results/runs/<run_id>/
├── resolved_config.yaml
├── provenance.json
├── status.json
├── metrics_round.parquet
├── client_round.parquet
├── final_metrics.json
├── confusion_matrix.npy
├── checkpoints/
└── logs/
```

`provenance.json` must include:

- code commit and dirty-state flag;
- dataset/source hashes;
- split/client/anchor manifest hashes;
- semantic-prior bundle hash;
- resolved configuration hash;
- Python, package, OS, CPU/GPU versions;
- seeds;
- command line;
- timestamps.

`status.json` values: `planned`, `running`, `completed`, or `failed`, with failure reason and traceback path. Aggregation must not treat failed runs as zero-valued results.

---

## 23. Quality and audit checklist

Before full execution:

- [ ] Raw data source and license documented.
- [ ] Dataset checksums recorded.
- [ ] No train/validation/test/anchor/client leakage.
- [ ] Semantic feature groups validated after encoding.
- [ ] LLM input contains public metadata only.
- [ ] LLM response and schema validation are versioned.
- [ ] Online runs can load a frozen prior without network access.
- [ ] Every equation has a code owner and test.
- [ ] Missing evidence is explicit and coefficient renormalization is logged.
- [ ] Malicious identity is inaccessible to defense logic.
- [ ] Minimum-retention, clipping, cap, trust, and history invariants pass.
- [ ] Same-seed reruns are deterministic within documented tolerance.
- [ ] All baselines receive matching inputs and schedules.
- [ ] Test data is not used for tuning or early stopping.
- [ ] Metrics include denominators and undefined cases.
- [ ] Result tables are generated, not hand-edited.
- [ ] Failed/missing runs are visible.
- [ ] Statistical comparison families are predeclared.
- [ ] Claims are linked to exact artifacts/run IDs.

---

## 24. Definition of done

The implementation is complete only when:

1. Both datasets can be prepared reproducibly from documented sources.
2. Leakage checks and immutable manifests pass.
3. The MLP, FL simulator, attacks, semantic-prior pipeline, dual-view scores, trust, burn-in, clipping, and aggregation are implemented.
4. Every mathematical component has unit coverage and the end-to-end system has integration coverage.
5. All required baselines are implemented or transparently documented as blocked/reimplemented.
6. The full experimental registry is executed for five seeds or every failure is explicitly accounted for.
7. Detection, robustness, trust, efficiency, and statistical outputs are produced from raw logs.
8. All manuscript tables/figures regenerate from machine-readable results.
9. No fabricated number or unsupported superiority statement appears in the article.
10. Another researcher can reproduce a selected run from its configuration, manifests, prior bundle, code revision, and documented dataset access.

---

## 25. First instruction to Codex

Use this document as the implementation contract. Start with **Phase 1 only**.

Before editing:

1. Inspect the repository and preserve existing manuscript/source files.
2. Report any conflict between the repository and this specification.
3. Create a short execution plan for Phase 1.
4. Implement the repository foundation, configuration validation, reproducibility utilities, and tests.
5. Run the Phase 1 gate commands and report exact results.
6. Do not download datasets, call an LLM, run large experiments, or fabricate result artifacts during Phase 1.

At the end of each phase, provide:

- files added/changed;
- commands run and their outcomes;
- gate status;
- deviations and unresolved decisions;
- the exact next phase.
