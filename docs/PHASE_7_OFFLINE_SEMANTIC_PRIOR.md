# Phase 7 offline semantic prior

Status: **complete for Edge-IIoTset and CICIoT2023**. Ata Mohammadi approved the
source-traceable metadata at `2026-08-31T19:27:06.4641854Z`; both public-only prompts
were frozen and audited, the two authorized OpenAI requests completed, and both
calibration-anchor priors and fused bundles passed their closing gates. Phase 8 may
load these bundles without an online LLM call. See
`docs/PHASE_7_PUBLIC_METADATA_REVIEW.md`.

## Frozen completion record

| Dataset | Prompt ID | Validated response ID | Anchor-prior ID | Bundle ID |
|---|---|---|---|---|
| Edge-IIoTset | `ba6f52c3f90e8b15f992ff3300686dc1d98c0796bd4e7361c62ee5156b4d9b38` | `3d4ee02f89675341f5dace0fd060e3c8d5bdf8480101aed957e6bb1fce7da7b6` | `7c638c644f83822cd6a74d9d9b6c8d6660760bccc83db9c233e612660cda05bb` | `7e9a17e051f03d09ecad41a40f051c802bde3b07bd27ba16615f22aa800c1657` |
| CICIoT2023 | `ae93db4c01759019ac322bac6328e8df7668899f1922fdfde506c70ae313b816` | `c8f00a3a8650aeaa6d61b69aabf7d9d7f74a613694cfbfac993a17d9353123f6` | `3e7c66c9b089bc40117dcd96536b0d108d4eb1c318687058342e6ec6e3e974fc` | `cf5b579e6cb673f318d6a6ff61bfdec7f0975b58548aae20e8c601fc1b006589` |

Both bundles use `alpha_L=0.5`, the manuscript default. It was not selected from
validation or test performance. Endpoint and sensitivity ablations remain required
before confirmatory performance claims.

The Edge provider receipt is
`dbb6eb35b1b620c2de87a139ec1bb3110765f0e5df0d341a23e24f09fef3f148`
(18,412 input and 5,230 output tokens). The CICIoT2023 receipt is
`65c10fd5653e226c93dbfaa3093aa0f5d18018d4680ac6ddf78ec0adb41ce499`
(21,439 input and 5,950 output tokens). Both receipts record `store=false`, no tools,
no automatic retry, no private/client/update data, and no validation or test access.
The strict response validator required no repair and accepted no unknown keys. The
focused Phase 7 regression suite passed 7 tests, and both bundles passed offline
identity loading with their expected bundle IDs.

## Implemented boundary

Phase 7 is divided into three explicit commands. The first command
validates reviewed public metadata against the exact Phase 3C class and feature-group
layout and writes an immutable prompt. The second command sends only that prompt to
OpenAI with a strict output schema and saves the validated response and a
credential-free receipt. The third command computes the
empirical prior from `anchor_calibration`, fuses both sources, and writes a bundle for
no-LLM runtime loading.

The adapter is dependency-free and does not contain an API key field, automatic retry,
response repair, or fallback-generated score. It requires `OPENAI_API_KEY` in the
process environment, sets `store=false`, enables no tools, and pins
`gpt-5.4-mini-2026-03-17`. A malformed or refused response stops the build. Provider
credentials remain outside the repository and are not part of invocation metadata or
receipts.

## Inputs still required

For Edge-IIoTset, the reviewed metadata must define the 42 semantic groups in the
exact order stored in the Phase 3C preprocessor and the six primary classes in model
ID order: benign, DoS/DDoS, information gathering, injection, malware, and
man-in-the-middle. The original Edge-IIoTset paper and an authoritative protocol-field
reference are required because the retained groups include ARP, ICMP, HTTP, TCP, UDP,
DNS, MQTT, and Modbus/TCP fields.

For CICIoT2023, the reviewed metadata must define the 39 Phase 3C groups and the eight
primary classes in model ID order: benign, brute force, DDoS, DoS, Mirai, recon,
spoofing, and web-based. The official CIC page, the dataset paper's feature table,
and the supplementary feature-extraction documentation are the preferred sources.

Each public-metadata file must record:

- dataset ID, reviewer, approval status, and review timestamp;
- source citation and URL;
- exact feature-group name, definition, unit, admissible-range description, and
  source IDs;
- exact primary-class name, public description, and source IDs.

The frozen provider declaration is `configs/semantic/llm_invocation.json`. It records
OpenAI, the exact model snapshot, declaration timestamp, temperature, top-p value,
maximum output tokens, a null seed, and the API seed limitation. It contains no API
key.

The initial manuscript value `alpha_L=0.5` is not yet a confirmatory selection. Before
confirmatory experiments it requires a calibration or sensitivity record that does
not use validation or test performance.

## Prompt preparation command

After the reviewed Edge-IIoTset metadata and provider declaration are present, run:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"

conda run --no-capture-output -n semcred-fl python scripts\prepare_phase7_prompt.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --public-metadata "configs\semantic\edge_iiotset_public_metadata.json" `
  --invocation-metadata "configs\semantic\llm_invocation.json" `
  --output-directory "artifacts\semantic_priors\phase_7\edge_iiotset"
```

This command is lightweight. It makes no LLM request and does not read anchor,
validation, test, or client data. Preserve `prompt.txt` exactly.

## Guarded OpenAI command

After inspecting the generated `prompt.txt`, run the one-time provider call:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\invoke_phase7_openai.py `
  --prompt-directory "artifacts\semantic_priors\phase_7\edge_iiotset" `
  --output-directory "artifacts\semantic_priors\provider_responses\edge_iiotset" `
  --timeout-seconds 180
```

The command refuses provider aliases, altered prompts, private-boundary terms,
non-OpenAI declarations, and incomplete existing artifacts. On success it writes
`raw_response.json` and `provider_receipt.json`. A later invocation with the same
valid artifacts performs no API call and reports `reused=true`. Never pass an API key
as a command-line argument or save it under the project directory.

## Bundle command

Save the provider's exact JSON response outside the prompt directory, then run:

```powershell
conda run --no-capture-output -n semcred-fl python scripts\build_phase7_semantic_prior.py `
  --preprocessing-report "artifacts\preprocessing\phase_3c\edge_iiotset\edge_iiotset_phase3c_preprocessed.json" `
  --anchor-manifest "artifacts\splits\phase_3b\edge_iiotset\anchor_manifest.json" `
  --prompt-directory "artifacts\semantic_priors\phase_7\edge_iiotset" `
  --raw-response "artifacts\semantic_priors\provider_responses\edge_iiotset\raw_response.json" `
  --output-directory "artifacts\semantic_priors\phase_7\edge_iiotset" `
  --alpha-l 0.5 `
  --epsilon 1e-12
```

This command reads only the 32-per-class calibration anchors. It verifies their IDs
against the Phase 3B anchor manifest. The 32-per-class scoring anchors remain reserved
for Phase 8 and are not read. The command must report that validation, test, and
scoring-anchor access are false and that no online LLM call is required.

## Gate

Phase 7 closes only when both datasets have:

- reviewed public metadata with exact class/group coverage;
- a frozen prompt and provider declaration;
- an unmodified raw response that passes strict schema validation;
- a checksum-bound empirical calibration prior;
- endpoint tests showing exact `alpha_L=0` and `alpha_L=1` recovery;
- an identity-valid bundle that reloads without an LLM call;
- a public-boundary audit finding no private record, client, update, model, trust, or
  online-decision data in the prompt.
