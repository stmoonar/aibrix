# v1-paper trace conversion (ICSE submission traces_v14 → tre_replayer traces_v1paper)

Branch `traces/v1-paper-20260924`, worktree `76:/tmp/wt-traces` (based on `main` at
`0290ec9b`). Converts the 7 ICSE-submission-final v1 traces to the v2 replayer's
`trace.json` schema under `tre/replayer/traces_v1paper/`, and reports where the v1
client's send semantics differ from the v2 replayer's (no replayer behaviour changed).

## 1. Source: what "the raw v1 trace" actually is

v1's generator config
(`76:/root/aibrix-main/CustomTraceGenerator/config/traces_v14/<trace>/{config.yaml,
trace.json|real_traces/*.json}`) only encodes a **rate schedule** (piecewise RPS + token
shape, or — for the two `real_trace` workloads — a 1s-binned real-corpus schedule). The
concrete per-request arrival plan v1's `client_dispatcher.py` actually sent is a product
of that schedule plus Python's `random` module consumed across **8 OS processes × 100
coroutines each**; it is not reproducible bit-for-bit from the config and seed alone.
This was confirmed directly: replaying the same config/seed for the TRE arm and the APA
arm of the submission run produced two different concrete request sequences (same
per-model request *counts*, different `request_id`↔timestamp↔prompt assignment — see
§3, "apa-arm cross-check").

What v1's generator *did* leave behind is the concrete, already-materialised plan it
sent for the submission runs, one file per trace per arm:

```
/data/nfs_shared_data/xxy/trace_output/output_traces_v14/{tre,apa}/<trace>/traces.json
```

Each entry is one `RequestRecord`: `request_id`, `timestamp` (s, relative to the 720s
window), `model_name`, `prompt` (the actual text sent), `prompt_length` (its real-
tokenizer length — the input token count), `max_output_tokens` (the `max_tokens` cap
sent to the API), `phase_type`. This has arrival time, model, input tokens and output
cap for every request v1 dispatched — it **is** the raw trace, no regeneration needed.
`performance_metrics.json` in the same directory is the corresponding dispatch log
(JSONL, one row per request, same `request_id`), used only to cross-check that
`traces.json` is indeed what was sent (§3).

This conversion uses the **tre/ arm** as the canonical source per trace (the arm the
paper's TRE numbers come from).

## 2. Conversion rule

v2's `trace.json` (`tre_replayer.traces.loader.load_trace_segments`) is a model-keyed
list of segments, each either fixed-length (`{start_time, end_time, rps, input_tokens,
max_tokens}`) or sampled (`{..., input_tokens_dist: {kind: log_uniform, low, high},
max_tokens_dist: {...}}`, every request draws its own length). There is no third option.

We bin the realised per-request plan (`traces.json`) into **1-second segments per
model** — matching the native granularity v1's own `real_trace_file` format already
uses, and finer than any v1 stable/transition phase (min 30s / min 2s respectively), so
no rate structure is smoothed away. Per `(model, 1s bin)`:

| trace | v1 `generate_mode` | rule applied | why |
| --- | --- | --- | --- |
| Alternating_hot_model_periodic_A | custom | **fixed**, bin's rounded mean | `max_output_tokens` is a config literal, exactly constant in every bin (0% multi-value bins, empirically). `prompt_length` targets a literal too but has small one-sided slop (see below) — the mean recovers the target. |
| Decode_heavy_burst | custom | fixed, rounded mean | same |
| Prefill_mixed_corner_decode_mix | custom | fixed, rounded mean | same (this trace also exercises v1's `mix_with` 50/50 segment blending; it shows up as extra spread inside the affected bins, still captured by the mean) |
| Simultaneous_spike_ramp_twice_tps1o2 | custom | fixed, rounded mean | same |
| Sinusoidal_demand | custom | fixed, rounded mean | same |
| Real_code_2024_slice_a_tok70 | real_trace | **sampled**, log-uniform `[min, max]` per bin | both input and output tokens vary genuinely and substantially per request (100% of bins have 2+ distinct values on both fields) — a real corpus, not a config literal. |
| Real_conv_2023_slice_a_tok70 | real_trace | sampled, log-uniform `[min, max]` per bin | same |

**Why the 5 "custom" traces are fixed, not sampled**: `prompt_length`'s per-bin spread
is not a real distribution — it is v1's sentence-level text expander
(`data_generator._expand_prompt_content`) stopping once the remaining deficit is ≤15
tokens, so realised length sits a few tokens *below* a literal target (e.g. one
`Decode_heavy_burst` bin: target 350, realised `{337, 338, 340, 341, 342, 342, 344, 344,
347, 348, 349, 350}`, stdev ≈3.9 over the trace). Representing that as a log-uniform
range would manufacture a fake "distribution" out of fitter noise; the rounded mean
recovers the actual design target, which is what matters for replay.

**Dropped by design, not by oversight**:
- The prompt text itself. v2 never stores prompt text in `trace.json` — see
  `tre_replayer.engine.prompt_store` — prompts are synthesised at send time from
  `(model, request_id)`. See §4.
- `phase_type` and v1's `request_id` ordering (v2 assigns its own ids at schedule build
  time, `<model>-<index>`).
- Requests generated during v1's inter-segment "transition" windows have
  `max_output_tokens: null` in `traces.json` (no segment override applies there); the
  dispatcher fell back to the trace's single per-model config default (verified against
  `performance_metrics.json`: e.g. `Sinusoidal_demand` `req_000039` has
  `max_output_tokens: null` in the plan and `output_tokens: 300` in the dispatch log,
  exactly the config default). The converter uses that same default
  (`TRANSITION_DEFAULT_MAX_TOKENS` in the script) rather than dropping these requests.

## 3. Verification

Script: `tre/replayer/scripts/convert_v1_paper_traces.py`. Verification driver (not
committed — needs the multi-hundred-MB uncommitted source files): the commands and
output below, reproduced by re-running the same script against
`/data/nfs_shared_data/xxy/trace_output/output_traces_v14`.

### 3.1 Request counts

`build_deterministic_schedule` (uniform 1/rps grid, no randomness) replayed against
every converted trace reproduces **exactly** v1's request count — the conversion's core
lossless property. `build_poisson_schedule` (what `run_trace.py` actually uses for a
live/dry run) resamples arrivals from the same per-bin rate and deviates by normal
Poisson-resampling noise, <1% on every trace:

| trace | v1 request count (tre arm) | deterministic replay | Δ | Poisson replay (seed 20260924) | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alternating_hot_model_periodic_A | 12330 | 12330 | 0 | 12216 | -0.92% |
| Decode_heavy_burst | 21742 | 21742 | 0 | 21751 | +0.04% |
| Prefill_mixed_corner_decode_mix | 10430 | 10430 | 0 | 10361 | -0.66% |
| Simultaneous_spike_ramp_twice_tps1o2 | 5120 | 5120 | 0 | 5134 | +0.27% |
| Sinusoidal_demand | 21632 | 21632 | 0 | 21554 | -0.36% |
| Real_code_2024_slice_a_tok70 | 43083 | 43083 | 0 | 42959 | -0.29% |
| Real_conv_2023_slice_a_tok70 | 45244 | 45244 | 0 | 45167 | -0.17% |

### 3.2 traces.json (plan) vs performance_metrics.json (v14 dispatch log), tre arm

For every one of the 7 traces, 100% of `request_id`s in `traces.json` are present in
`performance_metrics.json` with an **identical** `timestamp` and `model_name` —
confirming `traces.json` is exactly what v1 dispatched, not a separately-drawn plan:

| trace | n | timestamp match | model match |
| --- | ---: | --- | --- |
| all 7 traces | (5120–45244) | 100% | 100% |

`output_tokens == max_output_tokens` ("hit the cap", i.e. EOS never fired first) varies
a lot by trace — relevant to the ignore_eos discussion in §4:

| trace | output hits cap |
| --- | ---: |
| Alternating_hot_model_periodic_A | 100.0% |
| Decode_heavy_burst | 0.1% |
| Prefill_mixed_corner_decode_mix | 13.0% |
| Simultaneous_spike_ramp_twice_tps1o2 | 100.0% |
| Sinusoidal_demand | 100.0% |
| Real_code_2024_slice_a_tok70 | 99.9% |
| Real_conv_2023_slice_a_tok70 | 100.0% |

### 3.3 apa-arm cross-check (same config/seed, independently drawn)

Per-trace total request count matches the tre arm **exactly** (+0.00% on all 7); the
concrete per-request assignment (which `request_id` got which model/timestamp/prompt)
does not match — expected, see §1. This is evidence the two arms are the same
generator/config/seed run twice, not evidence they are literally the same trace.

### 3.4 Token-length quantiles: converted (deterministic replay) vs v1 source

| trace | field | src mean | src p50 | src p90 | converted mean | converted p50 | converted p90 | mean Δ% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alternating_hot_model_periodic_A | input_tokens | 492.3 | 493.0 | 499.0 | 492.3 | 493.0 | 496.0 | 0.00% |
| Alternating_hot_model_periodic_A | max_tokens | 400.0 | 400.0 | 400.0 | 400.0 | 400.0 | 400.0 | 0.00% |
| Decode_heavy_burst | input_tokens | 342.5 | 343.0 | 349.0 | 342.5 | 342.0 | 345.0 | 0.00% |
| Decode_heavy_burst | max_tokens | 649.4 | 760.0 | 950.0 | 649.4 | 760.0 | 950.0 | 0.00% |
| Prefill_mixed_corner_decode_mix | input_tokens | 547.5 | 493.0 | 1035.0 | 547.5 | 493.0 | 1038.0 | 0.00% |
| Prefill_mixed_corner_decode_mix | max_tokens | 429.9 | 400.0 | 700.0 | 429.9 | 400.0 | 700.0 | 0.00% |
| Simultaneous_spike_ramp_twice_tps1o2 | input_tokens | 991.9 | 993.0 | 999.0 | 991.9 | 993.0 | 997.0 | 0.00% |
| Simultaneous_spike_ramp_twice_tps1o2 | max_tokens | 600.0 | 600.0 | 600.0 | 600.0 | 600.0 | 600.0 | 0.00% |
| Sinusoidal_demand | input_tokens | 442.4 | 443.0 | 449.0 | 442.4 | 443.0 | 445.0 | 0.00% |
| Sinusoidal_demand | max_tokens | 300.0 | 300.0 | 300.0 | 300.0 | 300.0 | 300.0 | 0.00% |
| **Real_code_2024_slice_a_tok70** | input_tokens | 387.3 | 353.0 | 599.0 | 397.6 | 370.0 | 585.0 | **+2.66%** |
| **Real_code_2024_slice_a_tok70** | max_tokens | 80.4 | 80.0 | 89.0 | 80.7 | 79.0 | 90.0 | +0.27% |
| **Real_conv_2023_slice_a_tok70** | input_tokens | 332.8 | 303.0 | 460.0 | 346.5 | 327.0 | 477.0 | **+4.12%** |
| **Real_conv_2023_slice_a_tok70** | max_tokens | 109.2 | 91.0 | 177.0 | 119.8 | 105.0 | 183.0 | **+9.73%** |

The 5 fixed-value traces match exactly, as expected (§2). The 2 log-uniform-sampled
traces show the conversion's one real, quantified approximation: log-uniform sampling
over `[min, max]` pulls the mean (and p50) up relative to v1's actual right-skewed
distribution, most visibly on `Real_conv`'s output length (+9.7% mean). If a future use
of these two traces needs tighter fidelity than this, the fix is in `tre_replayer.engine
.schedule.TokenRange` (a second distribution kind, e.g. log-normal) — out of scope here
since the task is convert-and-report, not change replayer behaviour.

### 3.5 Replayer can load and run the converted traces

- `discover_trace_set("replayer/traces_v1paper")` loads all 7 cases, all 3 fleet models
  present in every trace, `INDEX.json` version `traceset-v1paper`.
- `replayer/tests/test_v1_paper_traces.py` (5 tests, committed): INDEX/segment-schema
  sanity, exact deterministic-schedule request counts, and one full
  `run_trace(..., dry_run=True, sleep=<instant fake>)` pass — schedule → dispatch →
  score, with a fake sender (no network) and an instant fake clock (no 720s of real
  waiting) — for `Simultaneous_spike_ramp_twice_tps1o2`.
- Full `replayer/tests` suite: 118 passed. Full `make check`: **1463 passed**.

## 4. v1 client vs v2 replayer: send-semantics differences (report only, no code changed)

| dimension | v1 (`CustomTraceGenerator/src/client_dispatcher.py`) | v2 (`tre_replayer/engine/http_sender.py`, `dispatcher.py`, `prompts.py`) | comparability impact |
| --- | --- | --- | --- |
| protocol / endpoint | `openai.AsyncOpenAI` **chat completions** (`/v1/chat/completions`), `messages=[{"role":"user","content":prompt}]` | raw HTTP POST to `/v1/completions`, `{"model", "prompt": <string>, ...}` (legacy completions, no chat template) | Chat-template wrapping adds token overhead v1 counts (observed: `prompt_length` 499 text tokens → 504 reported `input_tokens` after templating). v2's realised `prompt_tokens` will run a few tokens lower for the "same" nominal length. Minor for aggregate load, but exact per-request token counts are not bit-comparable across the two clients. |
| retries | `max_retries=2` (OpenAI SDK default retry-on-failure) | **zero retries, by design** ("a retried request would be counted once as offered and twice as sent... re-offer load the schedule never planned") | v1 numbers include some silently-retried requests; v2's open-loop guarantee does not. Under load/errors this makes v2 a *stricter* open loop — a v1-vs-v2 comparison at high error rates is not apples-to-apples on offered load. |
| timeout | flat `300.0s` for every request | `max(30.0, max_output_tokens / 4.0)` s, i.e. scales with output length (e.g. 75s at max_tokens=300, 175s at max_tokens=700) | v2 will time out long-output requests under sustained overload far sooner than v1 would have. For traces with `max_tokens` ≥ ~1200 v2's timeout exceeds v1's 300s; for the traces here (max_tokens 300–950 nominal, up to ~2048 in the sampled real-trace bins) v2 is uniformly *more impatient*. Recommend aligning if replaying these traces for a v1-comparable number. |
| streaming | yes (`stream=True`, `stream_options.include_usage`) | yes, same | comparable |
| routing | `routing-strategy: least-gpu-cache` header on every request → hits AIBrix's ext_proc gateway plugin, which explicitly load-balances by per-pod GPU-cache occupancy and reports which pod served it | **no routing strategy set by `run_trace.py`/`run_comparison.py`** (both leave `routing_strategy=None`) → sends the `model` header instead, hits the per-model HTTPRoute, Envoy load-balances across the model's endpoints (not GPU-cache-aware), and `target_pod` is always `None` on every row | This is the largest unresolved semantic gap. Replaying these traces through the current v2 drivers does **not** reproduce v1's GPU-cache-aware pod selection, and per-pod attribution is lost. `StreamingHttpSender` supports a `routing_strategy=` constructor arg, but `run_trace.py` has no CLI flag to set it — closing this gap needs a small (out-of-scope-here) code change if bit-comparable pod-selection is required. |
| output-length control | `max_tokens` cap only, **no `ignore_eos`** found anywhere in v1's client — natural EOS can (and empirically does, trace-dependently: §3.2) stop generation early | `ignore_eos: True` **always** — every request generates exactly `max_output_tokens` tokens, no early EOS | This is a real, trace-dependent confound, not just a detail: on `Decode_heavy_burst` only 0.1% of v1's requests hit the cap (EOS fires almost always), on `Prefill_mixed_corner_decode_mix` only 13%, but on the other 5 traces 100% hit the cap. Replaying with v2 (always `ignore_eos`) will send **strictly more decode work** than v1 actually served for the two traces where v1's models stopped early — output token volume, and therefore decode-bound load, is not comparable to v1's own realised load on those two traces without accounting for this. |
| concurrency model | 8 OS processes × 100 asyncio coroutines/process (≤800-way parallelism cap), 5s task-batching window (`task_batch_window: 5.0`) | single asyncio event loop; one `asyncio.Task` per scheduled request fired exactly on its open-loop schedule instant, backed by a `ThreadPoolExecutor` sized to `--max-in-flight` (default 512) for the blocking send | v2's `dispatch_open_loop` is a genuine, unbounded-arrival open loop (arrivals never wait for a free slot — only the *send* can queue, recorded as `pool_wait_ms`). v1's process/coroutine pool is itself an upper bound on in-flight requests (≤800) and batches dispatch decisions every 5s; at v1's own configured peaks (all 7 traces stay well under 800 concurrent) this is unlikely to bind, but it is a structurally different scheduler, not verified equivalent under saturation. |
| prompt content | Chinese template sentences (`data_generator._create_base_prompt` + `_expand_prompt_content`), **approximately** fit to the target token count (sentence-granularity, stops within ≤15 tokens of target — see §2) | English prose (`tre_replayer.engine.corpus`) or raw token ids, **exactly** fit to the target token count via the model's own tokenizer, deterministic per `(model, request_id)`, explicitly designed so no two requests share a prefix (defeats prefix caching) | Different language, different exactness, different repetition/caching behaviour. v1's prompts are not verified prefix-cache-safe; v2's explicitly are. Neither client's prompt content is preserved by this conversion (§2) — this row is about the *live* semantic difference if these traces are later replayed through v2 against a real cluster. |
| `temperature` | `0.0` (from `model_config.temperature`, same default in both) | `0` (hardcoded) | comparable |

**Recommended alignment, if a v1-comparable rerun is wanted** (not done here, report only):
set `StreamingHttpSender(routing_strategy="least-gpu-cache")` (needs a `run_trace.py`
CLI flag) to restore pod-attribution parity, and treat `ignore_eos=True` as a known,
trace-dependent load inflator for `Decode_heavy_burst` and
`Prefill_mixed_corner_decode_mix` specifically when interpreting any replayed numbers
against the original v1 paper figures for those two traces.

## 5. How to run

```bash
cd tre
PYTHONPATH=common:controller:service-manager:replayer:deploy:calibration \
  python3 -m tre_replayer.run_trace \
    --trace replayer/traces_v1paper/Sinusoidal_demand/trace.json \
    --gateway-url http://<gateway-host>:<port>/v1/completions \
    --out /tmp/sinusoidal_demand.raw.jsonl \
    --seed 20260924
# add --dry-run for a network-free smoke test (what the committed test does)
```

Regenerate the trace set (needs the uncommitted source under
`/data/nfs_shared_data/xxy/trace_output/output_traces_v14`):

```bash
cd tre
python3 replayer/scripts/convert_v1_paper_traces.py \
  --source-root /data/nfs_shared_data/xxy/trace_output/output_traces_v14 \
  --out-dir replayer/traces_v1paper --arm tre
```
