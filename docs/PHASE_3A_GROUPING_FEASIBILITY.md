# Phase 3A: grouping and split-feasibility audit

Status: implementation, synthetic validation, and both full corpus audits complete;
none of the initial policies passed all primary-family necessary conditions.

Phase 3A reads the Phase 2D Parquet corpora and tests whether candidate grouping
policies satisfy necessary conditions for later train/validation/test construction.
It does not assign records to a split. Split fractions and the final Edge-IIoTset
run/block rule remain open scientific decisions.

## Evidence computed

For each candidate policy, the audit records exact group counts, record counts,
largest-group concentration, mixed-label groups, and per-label support. A primary
family passes the Phase 3A necessary condition only when it has at least three
groups and at least 64 records for the two disjoint 32-record anchor subsets. The
fine-label check uses only records for which `fine_label_ambiguous=false` and requires
at least three groups per observed fine label.

The audited policies were:

- both datasets: retained representative `source_file` as a diagnostic key;
- both datasets: source files merged by the source-membership masks of exact
  duplicate components;
- CICIoT2023: `source_directory` as a diagnostic candidate;
- Edge-IIoTset: global unordered endpoint pair and source-file-plus-endpoint pair.

The CIC provider README was reviewed after the full audit and states that the 309
CSV files form a combined and shuffled dataset. CIC `source_file`,
`source_directory`, and contiguous row blocks therefore cannot support a capture-
or time-disjoint claim. They remain useful for provenance and diagnostics only.

The Edge scan also measures the shares of parseable, zero, IP-like, and other
`frame.time` values, nonmonotonic transitions among parseable values, and placeholder
endpoint pairs. It does not accept timestamp alone as a group.

## Resumability and boundaries

Each Parquet shard is one transactional checkpoint in the Phase 3A SQLite database.
Rerunning an interrupted command skips completed shards. Duplicate connectivity has
its own cursor and persists the union-find state after each batch. The resume
identity includes the Phase 2D corpus ID, complete shard inventory, anchor counts,
and candidate-policy version. Changed inputs stop the run instead of mixing states.

The report intentionally labels a passing policy `candidate_only`. The necessary
conditions do not prove that a class-stratified grouped split exists, and no model
features, fitted transforms, client assignments, or anchor record IDs are produced.

Outputs:

- `<dataset>_phase3a.sqlite3`: resumable group counts and connectivity state;
- `<dataset>_phase3a.json`: complete machine-readable evidence and analysis ID;
- `<dataset>_phase3a.md`: concise policy comparison.

## PowerShell commands

Run from the canonical `Code` directory. Both commands scan every retained canonical
record and should be run by the researcher locally or in Colab.

Edge-IIoTset:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\analyze_phase3a_grouping.py `
  --dataset edge_iiotset `
  --canonical-manifest "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\edge_iiotset\edge_iiotset_canonical.json" `
  --canonical-database "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\edge_iiotset\edge_iiotset_canonical.sqlite3" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\split_feasibility\phase_3a\edge_iiotset" `
  --progress-every 1
```

CICIoT2023:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\analyze_phase3a_grouping.py `
  --dataset cic_iot_2023 `
  --canonical-manifest "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\cic_iot_2023\cic_iot_2023_canonical.json" `
  --canonical-database "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\canonical\phase_2d\cic_iot_2023\cic_iot_2023_canonical.sqlite3" `
  --output-directory "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code\artifacts\split_feasibility\phase_3a\cic_iot_2023" `
  --progress-every 10
```

Do not delete either Phase 2D database or the Phase 3A resume database before the
final split manifest and overlap checks are complete. A Colab run needs the Phase 2D
manifest, schema, Parquet shards, and canonical SQLite database; it does not need the
raw CSV archives.

## Verified implementation evidence

Synthetic end-to-end tests cover duplicate components that merge source files,
policy-specific three-way feasibility, primary anchor capacity, exclusion of
fine-label-ambiguous records, deterministic resume identity, changed-contract
rejection, and the observed Edge time-format categories. These are implementation
tests, not dataset results.

## Full-corpus outcome

- CICIoT2023 analysis `05fcfbd3ad381080` scanned 20,594,360 canonical
  records. `source_file`, `source_directory`, and duplicate-connected source files
  all failed at least one primary-family necessary condition.
- Edge-IIoTset analysis `295711a5637cf925` scanned 15,017,480 canonical
  records. Every initial candidate failed because at least one primary family did
  not have three usable groups; the source-file policy's limiting class is the
  single MITM capture.

These failures do not invalidate either canonical corpus. They prevent Phase 3B
from assigning a split under the initial grouping assumptions. Phase 3A.1 must now
evaluate a declared record-level unit for CIC and a documented within-capture
surrogate for Edge MITM before any policy is frozen.

## Phase 3A.1 Edge refinement outcome

Analysis `e9d3fe40ed8d9b8f` verified the local MITM CSV against its raw PCAP.
Both contain 1,229 ordered observations. All 341 CSV rows with comparable IP
evidence matched the PCAP packet at the same ordinal, with no mismatches. The 365
canonical MITM representatives can therefore be mapped to PCAP metadata by their
one-based source-row index.

The PCAP-derived candidates all pass the primary-family necessary condition:

| MITM surrogate | MITM groups | Largest MITM group | Status |
|---|---:|---:|---|
| Direction-normalized bidirectional flow | 47 | 146 | candidate only |
| 30-second block | 18 | 47 | candidate only |
| 60-second block | 13 | 78 | candidate only |
| 120-second block | 8 | 113 | candidate only |

Bidirectional flow is the provisional primary-task choice because it keeps packets
from the same parsed communication flow together. Time blocks remain sensitivity
candidates because a fixed boundary can divide one flow. None of these keys creates
independent MITM capture sessions. Phase 3B must still prove that the provisional
hybrid policy can satisfy the frozen split fractions, label coverage, anchor
capacity, and overlap gates.
