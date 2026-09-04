# SemCred-FL scientific decision record

Status: implementation freeze v1.4, 2026-09-01. Items marked **open** must be resolved before confirmatory experiments.

## Frozen decisions

1. Edge-IIoTset and CICIoT2023 are processed independently. The primary task uses each dataset's published attack-family taxonomy plus benign; fine-grained labels are secondary outcomes.
2. Every seed has disjoint calibration and scoring anchor subsets. Calibration records construct the empirical prior; scoring records are used for anchor loss and attribution. Neither subset enters validation or testing.
3. Five seeds are a pilot only. A confirmatory run using exact paired Wilcoxon tests and Holm correction requires a simulation-based power record and at least ten seeds.
4. Confirmatory poisoning runs construct rounds by honest/malicious strata so the malicious-minority assumption holds in every round. Independent availability is a separate assumption-violation stress test.
5. The main aggregation configuration uses sample-count exponent `xi = 0`. Sensitivity runs with `xi > 0` cap declared counts at a validation-selected multiple of the eligible-client median.
6. Multi-Krum is run only when `n >= 2f + 3`. The main comparison uses 20 submitted updates per round; invalid `(n, f)` cases are omitted rather than coerced.
7. FedKD-IDS is implemented in its native public-data/logit-exchange protocol. Any parameter-update adaptation is labeled a surrogate and reported separately.
8. A skipped attribution round uses anchor loss only. A stale attribution value is never carried forward as current evidence.
9. Directional evidence is unavailable when either vector has negligible norm. Missing indicators are removed and active coefficients are renormalized.
10. The offline LLM response is schema-checked and stored as an immutable artifact. Reproducibility claims refer to reuse of that artifact, not to an unverified claim of provider-level deterministic generation.
11. The semantic ablation family includes numerical-only, semantic-only, LLM-only prior, anchor-only prior, uniform prior, permuted prior, loss-only, and attribution-only variants.
12. The paper claims raw-data locality. It does not claim formal privacy unless a separately specified and evaluated privacy mechanism is added.
13. Model-feature values are validated and canonicalized before the final duplicate
    fingerprint is constructed. Components that contain multiple primary families
    are excluded entirely. Same-family fine-label conflicts remain eligible only for
    the primary family task. Label-consistent components retain the first canonical
    `(source_file, source_row)` representative.
14. CICIoT2023 retains all 39 source predictors as continuous features. Edge-IIoTset
    removes the verified constant `icmp.unused`; four provider-declared fields remain
    categorical under closed raw domains, while `dns.qry.name.len` is parsed as the
    unsigned numeric field defined in the dataset paper. All other retained Edge
    fields require finite decimal or hexadecimal numeric input.
15. Phase 3B targets 70% training, 15% validation, and 15% testing. These are exact
    hash thresholds for CICIoT2023 and optimization targets for grouped Edge-IIoTset
    assignment. Edge realized proportions are reported without moving records across
    groups to make the fractions appear exact.
16. Each primary family contributes 32 calibration and 32 scoring anchors selected
    deterministically from its training partition. The subsets are disjoint, are
    excluded from the federated client pool, and use a seed derived from root seed
    `20260816`.
17. Phase 3C estimates each continuous feature's 0.001 and 0.999 clipping quantiles
    from a deterministic SHA-256 threshold sample targeting 1,000,000 records from
    the non-anchor training pool. Exact population means and standard deviations are
    then computed over every clipped record in that pool. A scale of `1.0` is used
    when the population standard deviation is at most `1e-12`.
18. Edge categorical vocabularies are fitted only on the non-anchor training pool.
    Categories absent from that pool map to one `__UNKNOWN__` coordinate. One-hot
    coordinates retain their source variable as a shared semantic group.
19. Phase 3D uses a fixed population of 50 clients. The IID control and Dirichlet
    profiles with concentration parameters `0.1`, `0.3`, and `1.0` allocate only the
    non-anchor Phase 3C training pool using primary-family labels.
20. Every accepted client profile has at least 128 records per client. Exact family
    quotas are obtained by largest-remainder apportionment; rejected profiles use a
    deterministic attempt-derived seed, and the accepted attempt is recorded.
21. Phase 3D stores one assignment row per eligible training record with one client-ID
    column for each profile. It does not duplicate transformed features. Realized
    client sizes, client-by-family counts, normalized label entropy, total-variation
    distance, and observed family counts are persisted with each profile.
22. The default detector is the fixed multilayer perceptron \(d\rightarrow256\rightarrow
    128\rightarrow64\rightarrow C\), with LayerNorm, GELU, and dropout `0.2` after
    each hidden linear layer. Training consumes logits; softmax is used only when
    probabilities are requested.
23. For a training population with \(N\) records and observed class set
    \(\mathcal C^+\), class \(c\) receives weight
    \(\min(10,N/(|\mathcal C^+|n_c))\) when \(n_c>0\). An absent class receives
    weight zero and is recorded. The loss normalization is exactly the record mean
    in the manuscript objective, not the weight-normalized mean used by the default
    reduction of some libraries.
24. Phase 4 centralized training is a validation-only pipeline sanity check. It never
    reads the test partition, does not tune or compare SemCred-FL, and cannot support
    a detector-performance claim. Its resolved configuration, epoch log, environment,
    and checksum-bound checkpoint are persisted.
25. A bounded Phase 4 run reads a deterministic prefix of the Phase 3C shard manifest.
    Because this prefix need not contain every class, the report records a
    `validation_all_classes_observed` gate. Metrics from a run that fails this gate
    are wiring evidence only.
26. The canonical client update is one detached one-dimensional tensor ordered by a
    stored trainable-parameter name/shape manifest. Flattening, reconstruction, and
    update application fail when model layout or vector cardinality differs.
27. A root seed generates an immutable client schedule before training. Selection is
    without replacement within each round, and compared methods must consume the
    same schedule identity.
28. Simulation-only `is_malicious` and `attack_metadata` fields are removed when a
    `ClientUpdate` is projected to the `DefenseUpdate` consumed by validation and
    aggregation. Phase 5 round records contain no attacker labels or attack metadata.
29. Phase 5 FedAvg weights each accepted update by its positive declared sample count.
    Coordinate median is computed independently per parameter; for an even number of
    updates, it is the midpoint of the two central coordinate values. A coordinate
    median has no linear client-weight map, so its recorded weight dictionary is empty.
30. Wrong-round, duplicate, unselected, missing, wrong-shaped, non-finite, and
    metadata-inconsistent updates are rejected with a recorded reason. If all
    submissions are invalid, the global model remains unchanged and the round is
    marked failed.
31. Canonical round records exclude wall-clock values and form a SHA-256 hash chain.
    Epoch durations and start/end timestamps are stored separately as telemetry.
    Every completed round has an atomic checksum-bound checkpoint whose resume cursor
    must match the canonical log before execution continues.

## Dataset intake decisions frozen in Step 3

- Exact fine-label-to-family maps for both datasets are stored in
  `configs/datasets/`; unknown labels fail ingestion.
- Available archive filenames, byte sizes, SHA-256 hashes, provenance status, and
  permitted roles are frozen in those contracts. The official CIC archive is locally
  hashed; no independently published provider checksum is available for comparison.
- The official CICIoT2023 source tree is now present: 309 CSV files in 34 label
  directories with a uniform 39-column feature schema. Labels are derived only
  through the frozen source-directory map; unknown directories fail ingestion.
- The CICIoT2023 derivative train/validation/test package is smoke-only because its
  capture identity is absent and duplicate rows cross supplied split boundaries.
- The Edge-IIoTset ML subset is smoke-only and is not an independent experiment
  corpus. Raw per-scenario CSVs are the confirmatory source candidate.
- Edge grouping-only fields and prohibited detector features are explicit in the
  dataset contract. Grouping metadata is removed before preprocessing.

## Phase 2D evidence now frozen

- Edge-IIoTset corpus ID
  `ea83ce6fa2f0ad01a673792dffb29ba91f7b399fbaf5afef6d98e162f11effe9`
  identifies 15,017,480 canonical records in 24 shards.
- CICIoT2023 corpus ID
  `4d2185287b18333ccb89a31851c90ff5588864de0ccff18662074d8a76021f97`
  identifies 20,594,360 canonical records in 309 shards.

These identifiers freeze prepared-corpus provenance. They are not experimental
results.

## Phase 3A evidence now frozen

- CICIoT2023 analysis `05fcfbd3ad381080` scanned all 20,594,360 canonical
  records. No initial source-based policy passed the necessary primary-family gate.
  The provider README describes the 309 CSVs as combined and shuffled; source files,
  source directories, and adjacent rows cannot be represented as capture or time
  groups.
- Edge-IIoTset analysis `295711a5637cf925` scanned all 15,017,480 canonical
  records. The initial policies fail the primary gate because MITM has only one
  source capture. The local PCAP is available, so Phase 3A.1 may evaluate explicitly
  labeled within-capture flow or time-block surrogates.
- Fine-grained labels remain secondary outcomes. Phase 3A does not establish
  capture-disjoint fine-label evaluation for either dataset.
- Edge-IIoTset Phase 3A.1 analysis `e9d3fe40ed8d9b8f` matched 1,229 raw MITM
  CSV rows to 1,229 PCAP packets and found 341 of 341 comparable IP rows aligned at
  the same ordinal. Direction-normalized bidirectional flow is the provisional MITM
  group because it keeps each observed flow within one split. Its 47 groups satisfy
  the primary-family necessary condition, but they remain within one capture and do
  not establish capture-session independence.

## Phase 3B plan and split evidence now frozen

- CICIoT2023 plan
  `459997fb13595092a904ec6a3d08dfe4966a49d65e7c1c1486fab08f67e81a13`
  assigns canonical fingerprints by a seeded record-level hash threshold. The
  provider-shuffled files do not support capture- or session-generalization claims.
- CICIoT2023 split
  `590fb686703ef03c30314499629f13d8b083f24f45348f3cd289198abcac99ca`
  assigns all 20,594,360 canonical records across 309 checksum-bound Parquet shards:
  14,415,182 training, 3,088,931 validation, and 3,090,247 test records. The realized
  proportions are 69.996%, 14.999%, and 15.005%. All primary structural gates passed.
  Its 512 unique anchors contain 32 calibration and 32 scoring records for each of
  eight primary families and leave 14,414,670 records for client partitioning. All
  34 fine labels occur in every split, but those secondary outcomes remain
  record-level rather than capture- or session-disjoint.
- Edge-IIoTset plan
  `205406da53c1848385666fc9ed7ac6a7d1e700640d0f6fd65713e66561d32d88`
  contains 70 source-capture or MITM bidirectional-flow groups. Its assignment hash is
  `647eb0b9a8814fa745781c20ecd38221de085e05771b23cc34abb588eace4c82`.
  Every primary family is represented in all three planned partitions. Realized
  family proportions differ from 70/15/15 where only three or four large source
  groups exist; this is a property of the evidence boundary, not a reason to split a
  capture group.

- Edge-IIoTset split
  `b6616d2030f682906d205b0a105234a78a86ec995588fd4bbfe100ba2b6ca886`
  assigns all 15,017,480 canonical records across 24 checksum-bound Parquet shards:
  10,122,299 training, 1,624,189 validation, and 3,270,992 test records. The realized
  overall proportions are 67.40%, 10.82%, and 21.78%. All primary-family coverage,
  canonical cardinality, group-disjointness, and anchor gates passed. The 384 trusted
  anchors contain 32 calibration and 32 scoring records for each of six primary
  families and leave 10,121,915 records for later client partitioning.

The Edge fine labels are source-specific and several occur in only one partition.
They therefore remain descriptive metadata and cannot be used for confirmatory
fine-grained performance evaluation under this split. Preprocessing and primary-
family client partitioning are complete.

## Phase 3C evidence now frozen

- Edge-IIoTset Phase 3C plan
  `a0d9418906b3b6a80f9ba2f9b99afc231c32ee0752569de5201772eb0ddb3046`
  binds its 24 canonical and assignment shards, split and anchor identities, 38
  continuous features, four categorical features, and 10,121,915-record fit pool.
  The completed preprocessed corpus
  `a9a1cdb3ec5e5ec8846be236432a1e0c478b170478e3e581dc7441356c86941a`
  uses preprocessor
  `c175c802c8c30049f0ddfcba72b44c1c020e52e7472b691b5488dc9d842bc7ff`.
  Its threshold sample realized 998,889 records, its model dimension is 56, and its
  24 output shards contain all 15,017,480 canonical records under one schema. File
  sizes and SHA-256 hashes match the final manifest.
- CICIoT2023 Phase 3C plan
  `910eac34dc77f7de6d051a12c4ac3149d031f789d648c37cddac385154aa797a`
  binds its 309 canonical and assignment shards, split and anchor identities, 39
  continuous features, and 14,414,670-record fit pool.
  The completed preprocessed corpus
  `8bb930472baf37a4a0b66e479c99aa3701e8025bb654046d95a53cf7af9c76ce`
  uses preprocessor
  `e39b411b302da0c97ef766c0d8c9e36caa4c718e65f702f40e78705fde497e8a`.
  Its threshold sample realized 1,000,399 records, its model dimension is 39, and its
  309 output shards contain all 20,594,360 canonical records under one schema. File
  sizes and SHA-256 hashes match the final manifest.

Both plans and transformed corpora passed validation. These preprocessing records do
not contain detector-performance measurements.

## Phase 3D plan and assignment evidence now frozen

- Edge-IIoTset plan
  `8560b9cb63fa1391e3a349242bbd368f328da383ef4a3119c10132e050b4361c`
  binds the completed Phase 3C corpus and 10,121,915-record non-anchor training pool.
  IID and Dirichlet `0.3` and `1.0` pass on attempt 0. Dirichlet `0.1` passes on
  deterministic attempt 1 with client sizes from 295 to 1,579,132 records. Completed
  client-partition manifest
  `88ac4670d265e0e1b60f0e843c443e0d648a87dc9571b0c633ec52d0f40f67f8`
  contains 10,121,915 assignment rows in 24 checksum-bound shards totaling
  376,688,181 bytes.
- CICIoT2023 plan
  `6c0788417a6fb5ceb4dc29f168245398716337140e3a72e05852f432be3e7138`
  binds the completed Phase 3C corpus and 14,414,670-record non-anchor training pool.
  All four profiles pass on attempt 0; Dirichlet `0.1` has the smallest accepted
  client at 326 records. Completed client-partition manifest
  `25e8770e51b12afc057d8e2346cdc0ef298a9a5574e377ee64a3882b6e79a0ed`
  contains 14,414,670 assignment rows in 309 checksum-bound shards totaling
  543,579,503 bytes.

The complete client-size and heterogeneity summaries are stored in
`docs/PHASE_3D_CLIENT_PARTITIONS.md`. Both completed manifests passed their recorded
conservation, exclusion, quota, client-range, and minimum-size gates. These artifacts
contain no model training or detector-performance measurements.

## Phase 4 implementation and compatibility evidence

- The detector, exact weighted objective, optional proximal penalty, declared-class
  metrics, streaming confusion matrix, Phase 3C/3D-bound loader, atomic checkpoint,
  and validation-only centralized runner are implemented.
- The focused Phase 4 suite contains 48 passing tests. It covers loss arithmetic,
  class-weight capping, absent classes, known metric fixtures, checkpoint round-trip,
  artifact-identity validation, partition streaming, the learnable-fixture gate, and
  the prohibition on test access.
- Edge-IIoTset bounded compatibility run
  `648adc7d0fc58c6b75c5af2b5078f4e6b44de4cde4e89198809e0ed712104815`
  used 4,096 training and 2,048 validation records for one epoch. Its checkpoint hash
  is `f7747620bc65b6766d6b2bb5d7aa977bca6260f2173b30cbb451049bcddb51f6`.
- CICIoT2023 bounded compatibility run
  `aa508408f94e0da382c36bbcb02aa6a4fa1491234332aaca702af9bd28514524`
  used the same bounds and one epoch. Its checkpoint hash is
  `b21df16159aa8c8f2a0fa447776e5a28072cc17480abe38bb731af2d254a993f`.

Both bounded validation prefixes omit one or more declared classes, and both reports
set `validation_all_classes_observed=false`, `test_partition_accessed=false`, and
`performance_claims_allowed=false`. Their metric values are therefore not scientific
results and must not be copied into the manuscript as detector performance.

## Phase 5 implementation evidence

- The Phase 5 core includes reversible parameter flattening, exact local Adam
  training from the broadcast state, immutable client schedules, defense-safe update
  validation, FedAvg, coordinate median, synchronous rounds, hash-chained JSONL logs,
  atomic per-round checkpoints, and validated exact resume.
- The Phase 3D client adapter binds a named heterogeneity profile to the completed
  assignment inventory, verifies its checksums, and streams only one client's
  non-anchor training records. It checks final record and family counts against the
  profile matrix.
- The focused suite contains 95 passing tests and measures 92.51% branch coverage
  over the Phase 5 federation, aggregation, and client-stream modules.
- Synthetic engine-gate run
  `6cd53c069e3aac23859dfb0425bd0bcdf53f3cd98bbe684a86251675da94390b`
  completed three FedAvg rounds with two of four clients selected per round. Its final
  parameter-vector hash is
  `0f7f866fd7c2e113b06731cb167d64b2358b3ef5d77229133abd4e1bc23cc6e9`.

The synthetic run uses generated two-dimensional records, sets
`test_partition_accessed=false` and `performance_claims_allowed=false`, and is not a
detector-performance or federated-comparison result. No full Edge-IIoTset or
CICIoT2023 federated run has been executed.

## Dataset-preparation gate outcome

- Edge-IIoTset has passed the technical preparation gates for the primary-family task
  under the frozen capture and within-capture MITM-flow claim boundary. Fine-grained
  confirmatory evaluation remains excluded.
- CICIoT2023 has passed the technical preparation gates for the primary-family
  record-level task. Its official-source license remains unresolved, and the prepared
  split does not support capture- or session-generalization claims.

The completed Phase 2B audits are identified by
`eb41b084eeb07a425e28985b989f9e4ba2f9966f5827ea33e57ebe4443514e24`
for CICIoT2023 and
`534caa215cbd6ca4fd668e803ad4b1519b6ce89de6bc9045215c8984f6b09973`
for Edge-IIoTset. These identifiers describe the local audited sources and contracts;
they are not performance results.

## Phase 6A attack decisions

- The targeted source family is selected from federated-training metadata only as the
  least frequent non-benign primary family with at least 10,000 records after anchor
  removal. This fixes `injection -> benign` for Edge-IIoTset (46,519 eligible records)
  and `web_based -> benign` for CICIoT2023 (17,060 eligible records).
- Data-poisoning fractions remain `{0.1, 0.3, 0.5}`. Eligible rows are selected without
  replacement by a deterministic SHA-256 rank over the run, attack, client, and record
  identities. Zero is the exact no-op path.
- Clean-label poisoning operates on Phase 3C standardized continuous coordinates,
  preserves labels, moves three train-resolved features toward the benign median at
  strength `0.5` or `1.0`, and projects to the Phase 3C training-derived bounds.
- The backdoor maps the same source family to benign using a train-resolved endpoint
  trigger with three to six continuous coordinates and zero clean-training collisions.
  Test records are not inspected while resolving the trigger.
- Coordinated attackers share their objective, parameters, resolution artifact, and
  active rounds. They use client-specific row selections and cannot inspect anchors or
  current-round benign updates.
- Direct model attacks are separate conditions: `delta'=s*delta` for `s in {5,10}`;
  `delta'=-s*delta` for `s in {1,5,10}`; and coalition replacement
  `delta'=(n/m)*(theta_star-theta_t)`. The numerical grid is a declared study-design
  choice, not a reproduced baseline parameter.
- Adaptive norm camouflage clips the malicious L2 norm to the interquartile interval
  from an immutable benign-only development artifact. It does not inspect the current
  round's benign updates or the trusted anchors.

Phase 6A freezes these rules in `configs/attacks/phase6a.yaml`. Attack execution is not
yet ready: Phase 6B must resolve and hash the feature, trigger, and benign-norm artifacts
from permitted training/development inputs and implement the transformations.

## Phase 6B.1 resolution decisions

- Source and benign medians are exact float32 medians over the complete Phase 3C
  federated training partition after anchor removal. Validation, test, and anchor
  features are excluded by the Parquet scanner predicate.
- Equal feature-ranking gaps are ordered by Phase 3C coordinate name.
- The trigger search uses the twelve highest-ranked coordinates, examines tuple sizes
  from three through six, and breaks zero-collision ties by coordinate name followed
  by lower-before-upper endpoint order.
- Resolution is shard-resumable through SQLite and memory-mapped work matrices. These
  work files are not result artifacts and do not support performance claims.
- Attack transformations and the triggered-test generator are implemented, but attack
  execution remains blocked until Phase 6B.2 produces immutable benign update-norm
  references for every dataset/profile condition.

## Phase 6B.2 benign norm-reference decisions

- Each dataset/profile reference uses three benign-only FedAvg development rounds,
  20 scheduled clients per round, and 60 identified update norms. A client selected in
  more than one round contributes a distinct round-specific update.
- The calibration shares the confirmatory detector and local optimizer settings:
  hidden widths `256-128-64`, dropout `0.2`, three local epochs, batch size 128, Adam,
  learning rate `1e-3`, zero proximal penalty, and maximum class weight 10.
- The reference interval is the linear 25th-to-75th percentile interval over the 60
  prior benign development updates for that exact dataset, Phase 3D profile, model,
  and training contract. It is not computed from the later attacked round.
- A schedule-bound client cache is permitted as derived execution state. Its scanner
  returns only federated-training rows after anchor removal, and its manifest binds
  every cache shard, selected client, class count, schedule, and source artifact by
  SHA-256 identity.
- Cache construction resumes at completed source-shard boundaries. Federated
  calibration resumes at completed round boundaries; a partial round is repeated.
  The cache may be deleted after the final reference and audit evidence are verified
  because it can be reconstructed from preserved Phase 3C/3D artifacts.
- The verified Edge-IIoTset IID reference contains 60 accepted benign development
  updates and fixes the interval to `[22.1170015335083, 26.909245491027832]`. Its
  reference ID is
  `7dff694c94a63e03205ab2e5b5f97ed782612eb048e6aa40c7fccff92a80ef63`.
  This is an attack-calibration artifact and is not a performance result.
- The verified Edge-IIoTset Dirichlet-0.1 reference also contains 60 accepted benign
  development updates and fixes the interval to
  `[5.352854371070862, 19.78455686569214]`. Its reference ID is
  `6b970ff7694dd35c48293adcdbabd987521e7fffff5b8adcc7c70d4f44d8df34`.
- The verified Edge-IIoTset Dirichlet-0.3 reference contains 60 accepted benign
  development updates and fixes the interval to
  `[10.323951959609985, 23.033565521240234]`. Its reference ID is
  `e4fef6ed9575796c34262e769c7c45d7a3b929d7dd7a551fefebe759ff9f4cc2`.
- The verified Edge-IIoTset Dirichlet-1.0 reference contains 60 accepted benign
  development updates and fixes the interval to
  `[18.266846656799316, 26.66264581680298]`. Its reference ID is
  `771c0ff94fafd9daf3fa662249989898f017fa6910b476fcc17cbf8a174b13e9`.
  This completes the four Edge-IIoTset Phase 6B.2 profiles.
- The verified CICIoT2023 IID reference contains 60 accepted benign development
  updates and fixes the interval to
  `[26.881851196289062, 31.703227519989014]`. Its reference ID is
  `68d1429c4bf749fc76b22d9c6db0cf3cfbcf03b4e949f04b19921d8a76896c7f`.
- The verified CICIoT2023 Dirichlet-0.1 reference contains 60 accepted benign
  development updates and fixes the interval to
  `[11.823124885559082, 26.171101570129395]`. Its reference ID is
  `0ed10a25ec4bef31b8453cc33a6e70a3b1385bd74f2c555c9f77e92c313adcb3`.
- The verified CICIoT2023 Dirichlet-0.3 reference contains 60 accepted benign
  development updates and fixes the interval to
  `[16.245837450027466, 28.159836292266846]`. Its reference ID is
  `38dc6b815916cc29ead1b19971748647a6dcc74f330f710fb6920e0b65ed8740`.
- The verified CICIoT2023 Dirichlet-1.0 reference contains 60 accepted benign
  development updates and fixes the interval to
  `[20.425262928009033, 33.077284812927246]`. Its reference ID is
  `1b3b45e3877194185fbd4b6f9a0be5fab22ed97226b9f2cc19bc7c706e1de30a`.
- The Phase 6B.2 closure audit reconstructs all eight references from 480 unique
  update records, verifies every artifact and file identity, and records closure ID
  `1461106f84993cb974e197c40373c37d1765821881bbdf6a771242e139da32e9`.
  Phase 6B is complete, but attack execution remains blocked until Phase 6C is
  implemented and tested.
- The generic Phase 5 `federated_run_id` describes the shared initial model, schedule,
  and server controls, so it can repeat across Phase 3D profiles. It is scoped by the
  calibration-specific output directory and is not the scientific run identity. The
  profile-bound calibration input, calibration, and reference IDs provide that
  identity.

## Phase 6C execution decisions

- The malicious coalition is fixed for one ratio/seed realization and is independent
  of attack family. Clean and attacked conditions therefore share the same selected
  clients and round schedule.
- Every round is sampled separately within the fixed malicious and honest strata.
  Ratios 0.1, 0.2, 0.3, and 0.4 yield exactly 2, 4, 6, and 8 malicious participants
  among 20 selected clients, preserving a malicious minority in every round.
- Poison fractions are resolved over each malicious client's complete eligible local
  source-family population before training. The immutable record-ID selection is
  reused across local epochs; batch size and shuffle order cannot change membership.
- Model replacement uses a shared, declared-sample-weighted malicious target model.
  Each active malicious client submits `(20/c)(theta_star - theta_t)`, where `c` is
  the active coalition size. Declared sample counts are unchanged.
- Adaptive norm uses targeted label flipping as its local objective and projects the
  resulting update into the matching Phase 6B.2 benign IQR. It does not inspect
  current-round benign updates.
- Phase 6C adds the complete experiment input ID to the federated engine identity.
  This prevents profile, attack, or realization context from being represented only
  by an output-directory convention.
- Malicious truth and transformation metadata remain in separate simulation audit
  files. They are removed before validation, defense processing, or aggregation.
- The synthetic Phase 6C gate is complete. The nine-condition Edge-IIoTset IID
  development matrix also reached round one under matrix ID
  `06e12e61f1d147abf906c46d80df9dde91e9e94bb90c91b81664b8af66c95d14`.
  This is a bounded transformation gate, not a completed 100-round condition, and
  the generated artifacts remain ineligible for performance claims.
- Local training batches GPU synchronization for loss reporting and finite-value
  validation at epoch boundaries. This changes neither batch order nor optimizer
  operations. A real Edge-IIoTset client benchmark reduced one epoch from 18.889 to
  11.383 seconds with identical update SHA-256, mean loss, and update norm; the
  three-epoch output also matched the persisted pre-optimization clean-round hash.

## Baseline evidence decisions

- Full papers for AGAT-FL, Fed-LSAE, FedKD-IDS, and HADA are archived locally with
  SHA-256 identities in `docs/PHASE_10_BASELINES.md`.
- The official AGAT-FL Zenodo record contains only a 4,605-byte code fragment. It
  verifies the normalized trust-weighted sum but does not define the end-to-end
  federated, TrustGAT-training, Mahalanobis-filtering, or trust-history protocol.
  The implemented aggregation core must not be reported as full AGAT-FL.
- Fed-LSAE and FedKD-IDS code/data are available only on request. Their source-
  specified server cores are implemented with fail-closed ambiguity handling.
  Full Fed-LSAE execution still needs model/dataset adapters; native FedKD-IDS
  still needs a frozen N-BaIoT public-data split and detector/verifier runner.
- HADA remains an optional component comparator until its SHAP signal,
  privacy-budget schedule, and supplementary implementation are frozen.
- Every departure from a paper's native model, dataset, payload, verifier, or
  aggregation protocol is recorded and labeled as a reimplementation, adaptation,
  or surrogate.

## Open decision before confirmatory semantic-prior use

- The frozen Phase 7 bundles use the manuscript default `alpha_L=0.5` without
  performance-based tuning. Confirmatory experiments must include the declared
  `alpha_L=0` and `alpha_L=1` endpoint ablations and a training-only or prespecified
  sensitivity rule; validation or test performance must not retroactively select the
  reported value.

## Phase 7 foundation decisions

- Phase 7 contains no provider SDK and never makes an implicit network request. Its
  explicit OpenAI adapter pins `gpt-5.4-mini-2026-03-17`, sets `store=false`, enables
  no tools or automatic retry, and obtains the credential only from
  `OPENAI_API_KEY`. Prompt preparation, provider invocation, response validation,
  anchor calibration, and no-LLM loading remain separate operations.
- Public metadata must have explicit reviewer approval and citations. Its class order
  must equal the Phase 3C model class-ID order, and its group order must equal the
  Phase 3C semantic-group layout. Missing definitions do not receive generated
  defaults.
- JSON parsing rejects duplicate, missing, and unknown keys. Relevance scores require
  Python integer semantics; Boolean values are rejected even though they are integer
  subclasses in Python.
- The empirical prior reads only manifested `anchor_calibration` records and reconciles
  their record IDs and family labels against the Phase 3B anchor file. Validation,
  test, and `anchor_scoring` records are unavailable to this computation.
- LLM and empirical rows use the manuscript L1-plus-epsilon normalization. Fusion
  special-cases `alpha_L=0` and `alpha_L=1` so the saved values exactly equal the
  selected normalized endpoint.
- A saved bundle includes prompt, metadata, invocation, response, anchor, and corpus
  identities. Online federation loads this bundle without contacting an LLM.
- Ata Mohammadi approved the public metadata at
  `2026-08-31T19:27:06.4641854Z`. The one-time requests used the pinned
  `gpt-5.4-mini-2026-03-17` snapshot with `store=false`, no tools, no automatic
  retry, and no API seed. The resulting frozen bundle IDs are
  `7e9a17e051f03d09ecad41a40f051c802bde3b07bd27ba16615f22aa800c1657`
  (Edge-IIoTset) and
  `cf5b579e6cb673f318d6a6ff61bfdec7f0975b58548aae20e8c601fc1b006589`
  (CICIoT2023).

## Phase 8 credibility decisions

- Semantic online scoring uses only `anchor_scoring`; Phase 7 calibration remains
  restricted to `anchor_calibration`. The two trusted partitions are not exchanged.
- Client models are reconstructed from defense-safe flattened updates. Phase 8 never
  receives malicious truth or attack metadata, and neither credibility path reads a
  validation or test partition.
- Current-round grouped gradient-times-input attribution is reduced by RMS within
  each frozen Phase 3C semantic group and averaged within each scoring-anchor class.
  A skipped attribution round uses loss alone. Stale attribution is not reused.
- A class-specific semantic discrepancy is unavailable if the current attribution or
  frozen prior has negligible norm. Available classes alone contribute to the class
  mean. If none remain, the attribution indicator is unavailable for that client.
- Numerical direction is unavailable for a negligible submitted update or round
  center. Historical discrepancy uses accepted history only and is disabled for the
  complete round when fewer than three participating clients have valid histories.
- Robust z-scores are computed only over clients for whom an indicator is available.
  Missing indicators are not replaced by zero or a neutral score. The active
  nonnegative coefficients are renormalized per client.
- Phase 8 ends at separate semantic and numerical scores. Fusion, persistent trust,
  eligibility, clipping, accepted-history admission, and aggregation remain Phase 9
  work. Synthetic equation checks do not support performance claims.

## Phase 9 robust-aggregation decisions

- The confirmatory aggregation uses `xi=0`; declared sample counts therefore have no
  effect on its weights. A sensitivity run with `xi>0` caps counts at a declared
  multiplier of the eligible-client median. The default multiplier is 2.0. The old
  master-spec entries for `xi=0.5` and a 90th-percentile cap were stale and have been
  synchronized with the manuscript and this decision record.
- Fusion uses both complete views with the disagreement penalty. If one entire view
  is unavailable, the available view is used directly and the omission is logged.
  The implementation never inserts a neutral score. Two unavailable views fail.
- Participating trust is updated before eligibility. New trust equals first fused
  credibility; subsequent decreases use `eta_down=0.6` and recoveries use
  `eta_up=0.1`. Inactive clients retain their exact prior state.
- Minimum retention is deterministic. Threshold-qualified clients are kept first;
  additional clients are ordered by descending current trust, then ascending client
  ID. Burn-in applies the same rule using fused credibility.
- The burn-in global step is the coordinate-wise midpoint median over all accepted
  updates. Burn-in history admission is separate: only the selected burn-in clients'
  clipped updates enter history.
- After burn-in, only final eligible clipped updates enter history. Rejected or
  inactive clients retain prior histories; a new history is initialized exactly and
  later admissions use the declared event-driven EMA.
- Phase 9 tests are equation and state-transition evidence. They access no dataset or
  held-out partition and do not authorize a defense-performance claim.

## Phase 11 evaluation and reporting decisions

- Macro-F1 remains the primary utility endpoint. Relative degradation uses
  `(clean - attacked) / max(clean, epsilon)`, synchronized with the master
  specification. Every success/error rate stores counts and an undefined reason;
  an empty denominator is never reported as zero.
- Simulated malicious identity is available only after the defense produces its
  score and eligibility decision. Trust-based suspiciousness is `1 - trust`.
  Ranking ties receive explicit tie handling.
- Detection and recovery delay count client participation events and retain
  right-censoring. Benign trust volatility is RMS change across consecutive
  participation events, excluding inactivity gaps.
- Bootstrap intervals describe the seed-level mean. Exact paired Wilcoxon uses
  identical pair IDs, discards and counts exact zero differences, assigns midranks
  to absolute ties, and reports paired rank-biserial correlation. Holm correction
  operates only over one preregistered family per call.
- Each displayed summary carries a content-derived summary ID. CSV/LaTeX/SVG
  artifacts and the provenance sidecar resolve that ID to every contributing run
  and source artifact. Mixed test-access or claim-authorization states fail.
- Phase 11 analytical fixtures are implementation evidence only. Five seeds remain
  pilot-only; confirmatory inference still requires a power-analysis record and at
  least ten matched seeds.

## Phase 12 experiment-execution decisions

- Phase 12 starts with a deterministic synthetic integration fixture: four clients,
  three rounds, one sign-reversal event, and separate generated anchor/test data.
  The fixture is permanently non-confirmatory and cannot support manuscript
  performance claims.
- Attack labels and metadata remain in the simulation/evaluation layer. They are
  joined to trust and eligibility only after each defense decision and are never
  inputs to semantic scoring, numerical scoring, aggregation, or history admission.
- The synthetic report and run metrics are content-addressed. Re-running a completed
  identity returns its immutable report; an interrupted tiny run is restarted rather
  than resumed from a round checkpoint.
- The next real-data gate uses Edge-IIoTset only and compares clean FedAvg with
  SemCred-FL under shared initialization, schedule, optimization, and data artifacts.
  Configuration and debugging must not read the final test partition.
- The Step 2 pilot is frozen at three rounds, ten of fifty IID clients per round,
  one local epoch, and batch size 256. It is an execution gate, not the full protocol
  used for article performance claims.
- Its accelerated CUDA path may prevalidate immutable cache batches, materialize one
  selected client's cached batches on the device, and use fused Adam. The preserved
  cache batch boundaries keep the optimizer-step sequence fixed. Standard-Adam
  benchmark variants must retain the exact update hash.
- A local update may be reused across FedAvg and SemCred-FL only when a content-derived
  identity covers the cache, detector, training configuration, round, client,
  delivery contract, and exact global-parameter hash. Round and method checkpoints
  are checksum-verified and restore SemCred-FL client history as well as model state.
- Validation is read only after a method completes. The test partition is not read,
  and Step 2 validation metrics remain non-claim-bearing pending the full matrices.
- The completed Step 2 pilot is retained as integration evidence. Both methods
  completed three rounds, but both had zero recall for four of the six validation
  classes; worst-class recall was zero for both. The pilot cannot support method
  ranking or manuscript performance claims. Later full-budget clean and robustness
  runs must check minority-class recall before claim authorization.
- The Step 3 implementation gate is frozen to Edge-IIoTset, `dirichlet_0_3`, label
  flipping, malicious ratio 0.2, poison fraction 0.3, seed `20260816`, three rounds,
  ten of fifty clients per round, one local epoch, and batch size 256. Changing any
  of these controls creates a different condition rather than a continuation.
- Step 3 caches only the clients present in its deterministic schedule. The selected
  27-client cache and local-update reuse are content-addressed by their complete
  scientific inputs; checkpoint restoration verifies identities and checksums.
- Simulated malicious identity and attack metadata are excluded from every defense
  input. They are joined to trust and eligibility records only after the aggregation
  decision for client-assessment metrics. Validation is completion-only, the test
  partition remains unopened, and the three-round result is non-claim-bearing.
- The Step 3 real-data pilot completed under its frozen identity. FedAvg and
  SemCred-FL macro-F1 values were 0.2366 and 0.2447, but both had zero recall for
  four of six classes. SemCred-FL malicious false acceptance was 4/6 participation
  events. These single-seed, three-round values are retained as execution evidence
  and do not authorize method ranking or manuscript claims.
- Step 2 and Step 3 are not a matched clean/attack pair: Step 2 used IID clients and
  Step 3 used `dirichlet_0_3`. Their metric difference must not be reported as an
  attack effect. Later robustness experiments require matched clean and attacked
  configurations.
- The Step 4 reduced matrix retains the exact Step 3 condition and adds coordinate
  median, trimmed mean, and Multi-Krum. With ten submitted updates and malicious
  ratio 0.2, trimmed mean removes two values per tail and Multi-Krum uses `f=2`
  with eight selected updates. These controls are frozen before the real Step 4
  run and are not tuned from its validation rankings.
- FedAvg and SemCred-FL Step 3 artifacts are immutable Step 4 inputs. Standard
  baselines may reuse a Step 3 local update only when its complete content identity,
  including the global-model parameter hash, matches. The reduced matrix remains a
  one-seed pilot and cannot select a best method.
- Step 4 completed under the frozen controls with matrix result ID
  `bca0fb435639192f3ee25d74fbb566f911a73099c0cb33cf13f5c740621efce3`.
  SemCred-FL had the largest observed macro-F1, but all five methods had zero
  worst-class recall. The ordering is retained as pilot evidence and is not used to
  remove baselines, tune the test evaluation, or claim that SemCred-FL is best.
- Step 5 uses 20 submissions per round. The initial clean utility matrix freezes
  trimmed mean at four removed values per coordinate tail and Multi-Krum at `f=4`
  with 16 selected updates, corresponding to the primary malicious-ratio design of
  0.2. Robustness conditions using a different declared Byzantine bound require a
  separately identified matched clean denominator.
- Step 5 first executes a one-seed, full-training-budget validation matrix. This is
  a development gate for class coverage and runtime feasibility, not confirmatory
  evidence. CICIoT2023 results remain secondary while original-dataset license
  provenance is unresolved.
- Step 5 runtime calibration compares batch sizes only in separately identified,
  round-zero, no-validation campaigns. On the local RTX 3050, FedAvg round zero took
  520.849 seconds at batch 128, 91.337 seconds at batch 1024, and 99.476 seconds at
  batch 2048. Batch 1024 is therefore the provisional validation-only candidate.
  It cannot enter the manuscript or confirmatory registry until a completed
  learning diagnostic checks macro-F1 and supported minority-class recall.
- The ten-round batch-1024 learning diagnostic completed under matrix result ID
  `591183ecafb0b821c2cb88a54f541fce5543f389325e6108ab9bce410e272f6f`.
  Its identities, checkpoints, and metric cardinality pass, but all five methods
  had zero recall for information gathering and man-in-the-middle; injection and
  malware recall remained below 0.004. It therefore fails the class-coverage gate
  and cannot release the 100-round matrix. The next validation-only candidate keeps
  batch 1024 and the same short-run local-epoch budget while changing 10 rounds × 3
  local epochs to 30 rounds × 1 local epoch. The diagnostic runs FedAvg and
  SemCred-FL only, while retaining complete six-class validation output and an
  explicit class-2/class-5 recall gate. This tests the synchronization schedule
  before changing the shared loss or class-weighting policy; omitted baseline
  methods are restored for the full matched matrix.
- The 30-round × 1-local-epoch diagnostic completed under matrix result ID
  `3e3c0261013eace2b25822e62b40e6e819b0cde2b78d1dc98ebf8f2766f820b0`.
  FedAvg and SemCred-FL both retained zero recall for classes 2 and 5, so the
  synchronization candidate fails and is not adopted. The next diagnostic removes
  federation and runs one full-corpus centralized epoch under the existing shared
  detector and loss. This isolates learner/data behavior before any loss,
  class-weight, taxonomy, or final-matrix change is considered.
