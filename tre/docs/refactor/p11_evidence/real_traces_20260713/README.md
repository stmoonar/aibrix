# A4 real-trace derivation evidence (2026-07-13 work item)

This directory freezes the reproducible derivation of `t8_azure_conv` and `t9_burstgpt`
from public production traces. The raw source CSV files are licensed datasets and are not
committed; their official release URLs and SHA-256 digests are pinned below. The derived
trace JSON files are committed here and under `replayer/traces_v2/`.

## Sources and attribution

| workload | official source | selected rows | license | downloaded | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| t8 | [Azure LLM Inference Trace 2024, conversation](https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024) | positive context/generated-token rows from `AzureLLMInferenceTrace_conv_1week.csv` | [CC-BY-4.0](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE) | 2026-07-10 | `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` |
| t9 | [BurstGPT v2.0](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0) | `BurstGPT_3.csv`: `Conversation log`, non-empty Session ID, positive request/response tokens | [CC-BY-4.0](https://github.com/HPMLL/BurstGPT/blob/main/LICENSE) | 2026-07-10 | `2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f` |

The server could query GitHub metadata but direct large-asset bodies stalled. The two bytesets
were transported through `gh-proxy.com`, then accepted only after matching the official
GitHub release SHA-256 values above. The proxy is not a provenance source.

Azure's published conversation schema contains only `TIMESTAMP`, `ContextTokens`, and
`GeneratedTokens`; it does not expose a session/conversation ID. Therefore t8 uses the
explicit stable fallback key `azure-request:<timestamp>:<row-number>`. This preserves the
locked deterministic weighted mapping but does not claim to reconstruct Azure sessions.
BurstGPT v2 exposes Session ID, so all requests in one selected conversation map to the same
TRE model.

## Locked transformation

- Duration: `1120 s`. The plan said to confirm and match the existing convention; t1-t7 are
  actually 740-1120 s, not 2 h, and t2/t7 establish the maximum 1120-second convention.
- Time bins: `5 s`; source timestamps are linearly compressed from the first to last selected
  timestamp. The official files are chronological and the generator rejects a regression.
- Peak target: `29.426667 RPS`, the measured aggregate peak of t4. For reference, t7 peaks at
  `47.773333 RPS`. One global rate multiplier preserves each source's relative time shape.
  The t7 target was evaluated and rejected: with the locked 0.40/0.35/0.25 mix it required
  9/8 integer GPUs at peak. The t4 target is the highest named reference that passes C1.
- Assignment: SHA-256 of `<seed>:<session-key>` with seed `20260713`; intervals are
  dsllama-8b `[0,.40)`, dsqwen-7b `[.40,.75)`, dsqwen-14b `[.75,1)`.
- Tokens: arithmetic mean per `(5 s bin, assigned model)` after bounding input to 8192 and
  output to 2048. This keeps every 14b request below its configured 12288-token context limit
  while retaining production-scale token variation. Invalid/zero-token requests are omitted.
- Serialization: fixed model order, chronological segments, JSON indent 2, LF newline.

## Derived outputs

| workload | valid rows | source span | rate scale | derived peak RPS | input/output clamps | trace SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| t8_azure_conv | 27,303,999 | 604,799.994 s | 0.000724466424 | 29.426667 | 0 / 0 | `f80c915ac3c8e90b44900d48d6337d9f571f81589642586b732de5ac21ab4403` |
| t9_burstgpt | 231,761 | 9,503,864 s | 0.037476651808 | 29.426667 | 930 / 1 | `7edc26bc08940ffd8ed173692fec3d56664e58712f7e02d4531f84c5bb30adeb` |

Realized source-row assignment counts were t8 `10,919,178 / 9,560,223 / 6,824,598`
and t9 `91,872 / 83,047 / 56,842` in dsllama-8b / dsqwen-7b / dsqwen-14b order.

`manifest.json` is the machine-readable record of all parameters, source spans, row counts,
clamp counts, realized model counts, rate scales, and output hashes.

## Regeneration

From the `tre/` repository root, with the two pinned source files present:

```bash
python3 replayer/scripts/derive_real_traces.py \
  --azure-source /tmp/tre-real-traces-src/AzureLLMInferenceTrace_conv_1week.csv \
  --burstgpt-source /tmp/tre-real-traces-src/BurstGPT_3.csv \
  --trace-root replayer/traces_v2 \
  --evidence-dir docs/refactor/p11_evidence/real_traces_20260713
```

Byte-reproducibility was checked without touching the canonical files:

```bash
rm -rf /tmp/a4-regen-traces /tmp/a4-regen-evidence
python3 replayer/scripts/derive_real_traces.py \
  --azure-source /tmp/tre-real-traces-src/AzureLLMInferenceTrace_conv_1week.csv \
  --burstgpt-source /tmp/tre-real-traces-src/BurstGPT_3.csv \
  --trace-root /tmp/a4-regen-traces \
  --evidence-dir /tmp/a4-regen-evidence
cmp replayer/traces_v2/t8_azure_conv/trace.json /tmp/a4-regen-traces/t8_azure_conv/trace.json
cmp replayer/traces_v2/t9_burstgpt/trace.json /tmp/a4-regen-traces/t9_burstgpt/trace.json
cmp docs/refactor/p11_evidence/real_traces_20260713/manifest.json /tmp/a4-regen-evidence/manifest.json
```

## Validation

- Both raw files matched their pinned official SHA-256 before derivation.
- `pytest replayer/tests/test_derive_real_traces.py -q`: 3 passed (weighted stable hash,
  Unix timestamp parsing, selected-source bounds, deterministic bytes, schema).
- `discover_trace_set` found 9 indexed cases. `build_poisson_schedule(seed=20260713)`
  accepted both traces: t8 = 672 segments / 19,625 requests, t9 = 672 segments /
  8,577 requests; both schedules extend through the final 1115-1120 s bin.
- Capacity audit against the frozen R3 surfaces: C1 passes for both traces at 7/8 peak
  integer GPUs and oracle violation fraction 0. C3 is not applicable because these traces
  have no synthetic headroom tier. Capacity confidence is low because production token
  shapes extend beyond the measured calibration grid; the Phase 3 live smoke remains required.
- Canonical and evidence copies compare byte-equal for t8 and t9.
- Independent full-source regeneration to `/tmp/a4-regen-*` compares byte-equal for both
  canonical trace files and the machine-readable manifest.
- `make check`: 492 passed.

The required 10-minute live t8 smoke is intentionally deferred until the Phase 3 controller
and service-manager rollout; doing it against the pre-freeze images would not validate the
campaign environment.