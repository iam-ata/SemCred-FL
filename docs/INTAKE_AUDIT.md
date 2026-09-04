# SemCred-FL manuscript and evidence intake audit

Date: 2026-08-16  
Mode: pre-implementation integrity intake; not a submission-ready integrity pass.

Post-intake update, 2026-08-21: local dataset archives, hashes, schemas, and
label maps are now recorded in `configs/datasets/` and
`DATA_PROVENANCE_AND_INGESTION.md`. This update does not retroactively change the
evidence-matrix DQS below.

## Bottom line

**Evidence-matrix DQS: 74/100 — useful for literature triage, not yet safe as the source of numerical claims or baseline implementation details.** The workbook has 59 structurally complete and unique records, but 47 exact publication dates are unavailable, 17 records have a stable publisher identifier instead of a DOI, 57 records have no confirmed official code repository, and every row instructs the author to verify numerical tables before manuscript use.

The manuscript is structurally coherent but not submission-ready. The abstract and conclusion intentionally remain incomplete because no experiments have been run. The imported source had 59 bibliography entries, all cited, no dangling citations, and 106 unique labels. Its last compiled log contained duplicate algorithm-line hyperlink destinations.

## DQS dimensions

| Dimension | Score | Evidence |
|---|---:|---|
| Completeness | 22/30 | No blank corpus cells, but placeholder text substitutes for many dates, DOI fields, code records, and numeric extractions. |
| Consistency | 22/25 | Stable 33-column schema, unique keys/DOIs/URLs, and internally consistent category counts; article baseline tiering differed from the workbook. |
| Validity | 7/20 | One DOI misdirection was confirmed and corrected; the remaining corpus has not received a 100% field-by-field publisher audit. |
| Uniqueness | 15/15 | No duplicate citation keys, normalized DOIs, or publisher URLs. |
| Timeliness | 8/10 | Search window ends 2026-08-07; current 2026 and online-first records remain liable to metadata changes. |

## Verified corrections made in the workspace copy

1. Corrected the AIDFL DOI from `10.1109/ACCESS.2024.0429000` to `10.1109/ACCESS.2025.3552168`.
2. Corrected the BibTeX accent encoding for García-Sáez.
3. Added the primary Edge-IIoTset and CICIoT2023 source papers and cited them in the data and experiment sections.
4. Reconciled the article with the evidence matrix: three main domain-specific baselines; HADA is secondary and conditional.
5. Distinguished native FedKD-IDS logit exchange from a non-equivalent parameter-update surrogate.
6. Added scientific guards for anchor separation, confirmatory repetitions, partial participation, Multi-Krum, declared sample counts, missing directions, skipped attribution rounds, and LLM artifact reproducibility.
7. Added uniform, permuted, loss-only, and attribution-only semantic ablations.
8. Added a unique algorithm-line hyperlink anchor definition; this must be confirmed by recompilation.

## Blocking integrity findings

- The 59-reference corpus has not received the required item-by-item publisher metadata and claim-context audit. It cannot yet receive a `PASS` verdict under the selected academic integrity workflow.
- Numerical results are not present, so experiment-backed claims, figure/table provenance, power, and statistical adequacy cannot be verified.
- The current attack descriptions are families, not executable specifications.
- AGAT-FL, Fed-LSAE, and FedKD-IDS lack confirmed official repositories in the workbook. Reimplementations will require full-text method extraction and explicit deviation logs.
- Dataset intake is incomplete for confirmatory work. Edge-IIoTset has an author
  archive and academic-use notice but still needs a validated session/group rule.
  The original CICIoT2023 CSV distribution and supplementary material are now
  present, while the earlier derivative pre-split mirror remains smoke-only; the
  official archive hash, access terms, and full source audit are still required.

## Confirmed authoritative records used in this intake

- Edge-IIoTset source: https://doi.org/10.1109/ACCESS.2022.3165809
- CICIoT2023 source: https://doi.org/10.3390/s23135941
- FedKD-IDS: https://doi.org/10.1016/j.inffus.2024.102807
- Fed-LSAE: https://doi.org/10.1016/j.jisa.2024.103916
- AGAT-FL: https://doi.org/10.7717/peerj-cs.3281
- HADA-containing IEEE Access paper: https://doi.org/10.1109/ACCESS.2025.3592481
- AIDFL corrected record: https://doi.org/10.1109/ACCESS.2025.3552168

## Material that must be located

1. The official CICIoT2023 archive hash and any access/license record shown during
   download; the provider distribution and supplementary material are now present.
2. Full texts and supplements for the three main baselines, plus any author-linked repositories or implementation artifacts.
3. The intended target journal or conference and its current LaTeX template.
4. The exact LLM provider/model to be used for the offline semantic prior and whether the account/API exposes a seed parameter.
5. Practical maximum parallel-run limits for the validated RTX 3050 Laptop GPU
   environment; hardware and package identity are recorded in `ENVIRONMENT.md`.

## Scope limitation

This intake used structural workbook inspection, manuscript/source comparison, selected publisher verification, and the existing LaTeX log. It is not a plagiarism report, a complete 59-reference semantic audit, or evidence that the proposed method works. Those gates require the source papers, datasets, implementation, and generated results.
