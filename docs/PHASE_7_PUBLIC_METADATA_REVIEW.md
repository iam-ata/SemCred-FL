# Phase 7 public-metadata review

Status: **approved by Ata Mohammadi at 2026-08-31T19:27:06.4641854Z**. The approved
runtime files are:

- `configs/semantic/edge_iiotset_public_metadata.json`
- `configs/semantic/cic_iot_2023_public_metadata.json`

## Scope and boundary

The candidates contain only public dataset descriptions, public feature semantics,
units, conservative admissible-range descriptions, and the frozen Phase 3C class and
semantic-group names. They contain no dataset rows, observed class counts, private
feature distributions, anchor values, client partitions, model updates, trust scores,
validation results, or test results.

The feature and class order was checked directly against each Phase 3C
`preprocessor.json`. Edge-IIoTset has 42 semantic groups and six primary classes.
CICIoT2023 has 39 semantic groups and eight primary classes. Both candidate orders
match the corresponding preprocessor exactly.

## Evidence inventory

### Edge-IIoTset

- Ferrag et al., *Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset
  of IoT and IIoT Applications for Centralized and Federated Learning*, IEEE Access
  10 (2022), 40281-40306, DOI `10.1109/ACCESS.2022.3165809`.
- Local source: `Datasets/Edge-IIoTset/Edge_IIoTset__DatasetFL.pdf`.
- Local SHA-256:
  `8C21D1339ECDC4DFA59FFBEC408C6C8FF3B148A255C53E67FE3C5AD1A1397A39`.
- Table V supports the attack-family descriptions. Table VII supports the retained
  protocol-field names and definitions.

### CICIoT2023

- Neto et al., *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks
  in IoT Environment*, Sensors 23(13) (2023), 5941, DOI
  `10.3390/s23135941`.
- Official University of New Brunswick CIC dataset page:
  `https://www.unb.ca/cic/datasets/iotdataset-2023.html`.
- Local CSV readme SHA-256:
  `76D79460E416B88A28E9DB3573255EA5FB5FCA836FC2E3D3F794A4DF2B9E7E98`.
- Local supplementary readme SHA-256:
  `A748399C72FBACC2CD2E6E2E4091561BC53B7D8B9A50C2B1D7ED3A17A073F581`.
- Local supplementary `pcap2csv.zip` SHA-256:
  `1AB24F2A4373ACBDAC08E6835F07BC87DA472390D5658524B63DE5E3E084B4AC`.
- The paper and official dataset page support the 33 attacks and seven attack
  categories. The supplied supplementary extractor supports the operational
  definitions of the 39 retained CSV fields.

## Decisions requiring explicit review

1. Feature ranges are protocol-level or mathematical ranges. Private observed
   minima, maxima, category frequencies, and preprocessing statistics are excluded
   from the LLM prompt.
2. CICIoT2023 uses the Phase 3C name `Time_To_Live`; the paper calls the corresponding
   feature `Duration`, while the supplied extractor reads the IP TTL field.
3. The CIC supplementary extractor labels a port-21 rule as `IRC`. The candidate
   preserves the dataset field name but states that it is not a protocol-perfect IRC
   detector.
4. The CIC supplementary extractor's `LLC` type-check expression appears
   constant-like. The candidate preserves the field and records that its semantic
   interpretation requires caution.
5. The eight CIC classes and six Edge classes use the Phase 3C family mappings; no
   fine-grained label is added to the LLM output schema.

## Approval gate

The reviewer approved both candidates and authorized the two one-time OpenAI requests.
The files were promoted to schema 1.0 with `review_status=approved`, reviewer
`Ata Mohammadi`, and the UTC timestamp above. Both prompts subsequently passed the
automated public-boundary and identity gates, and both authorized requests completed
without response repair or unknown-key acceptance.
