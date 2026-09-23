# traces_v1paper — ICSE-submission v1 traces, converted to the v2 replayer schema

Seven traces: the workloads v1's `CustomTraceGenerator` actually dispatched for the ICSE
submission's headline TRE-vs-APA comparison (`traces_v14`, ~720s each, 3 fleet models:
`dsllama-8b` / `dsqwen-7b` / `dsqwen-14b`). Converted so `tre_replayer` (`run_trace.py`,
`run_comparison.py`) can replay them directly — no v1 code, no GPU-less local run needed
to validate the conversion itself.

Full provenance, the exact conversion rule, and the verification table are in
[`../../docs/v1-paper-trace-conversion.md`](../../docs/v1-paper-trace-conversion.md).
Summary:

## Source

v1's generator config (`config/traces_v14/<trace>/config.yaml` on `76:/root/aibrix-main/
CustomTraceGenerator`) only encodes a *rate schedule*; the concrete per-request arrival
plan it actually sent for the submission runs is not reproducible bit-for-bit from the
config and seed alone (v1 dispatches from 8 OS processes x 100 coroutines each, so the
`random` module's draws interleave non-deterministically). The generator did leave the
realised plan behind, per experiment arm:

```
/data/nfs_shared_data/xxy/trace_output/output_traces_v14/{tre,apa}/<trace>/traces.json
```

Each entry is one `RequestRecord`: `request_id`, `timestamp` (s, relative to the 720s
window), `model_name`, `prompt_length` (input tokens), `max_output_tokens`. This
directory converts the **tre/ arm** (the arm the paper's TRE numbers come from) for all
7 traces; `docs/v1-paper-trace-conversion.md` reports how far the apa/ arm's independent
draw of the same config/seed diverges (request counts match exactly; individual request
composition does not — expected, see above).

## Conversion rule

v2's `trace.json` is a model-keyed list of `{start_time, end_time, rps, input_tokens,
max_tokens}` segments (fixed length) or `{..., input_tokens_dist, max_tokens_dist}`
(every request draws its own length from a log-uniform `[low, high]`). We bin the
realised per-request plan into 1s segments per model (v1's own `real_trace_file` format
is already 1s-native; this is finer than any v1 stable/transition phase, so no rate
structure is lost) and, per `(model, 1s bin)`:

- `Alternating_hot_model_periodic_A`, `Decode_heavy_burst`,
  `Prefill_mixed_corner_decode_mix`, `Simultaneous_spike_ramp_twice_tps1o2`,
  `Sinusoidal_demand` (v1 `generate_mode: custom`): emit **fixed** `input_tokens` /
  `max_tokens` = the bin's rounded mean. Both are config-literal targets in v1; the
  small per-request spread in `prompt_length` is v1's own sentence-level text-fitter
  slop (≤ a few tokens under target), not a real distribution — see the doc for the
  measurement that established this.
- `Real_code_2024_slice_a_tok70`, `Real_conv_2023_slice_a_tok70` (v1
  `generate_mode: real_trace`, backed by a real request corpus): emit
  `input_tokens_dist` / `max_tokens_dist` = the log-uniform range `[min, max]` observed
  in that bin. This is the conversion's one real approximation — the true distribution
  is not log-uniform — quantified in the verification table.

Dropped by design: the actual (Chinese, template-generated) prompt text — v2 never
stores prompt text in `trace.json`; see `tre_replayer.engine.prompt_store` — and v1's
`phase_type` label.

## Verification (see the doc for the full table)

- Deterministic replay (`build_deterministic_schedule`) of every converted trace
  reproduces **exactly** v1's per-trace request count (0 delta, all 7 traces).
- `traces.json` (the plan) matches the tre-arm `performance_metrics.json` (the dispatch
  log) 100% on `request_id` → `timestamp` and `model_name`, confirming traces.json is
  indeed what v1 sent.
- Token-length quantiles match exactly (0.00%) for the 5 fixed-value traces; the 2
  log-uniform-sampled real-trace traces show a documented +0.3%..+9.7% mean deviation
  (see doc).

## Regenerating

```bash
cd tre
python3 replayer/scripts/convert_v1_paper_traces.py \
  --source-root /data/nfs_shared_data/xxy/trace_output/output_traces_v14 \
  --out-dir replayer/traces_v1paper --arm tre
```

## Running a trace

```bash
cd tre
PYTHONPATH=common:controller:service-manager:replayer:deploy:calibration \
  python3 -m tre_replayer.run_trace \
    --trace replayer/traces_v1paper/Sinusoidal_demand/trace.json \
    --gateway-url http://<gateway>:<port>/v1/completions \
    --out /tmp/sinusoidal_demand.raw.jsonl \
    --seed 20260924
```

Add `--dry-run` to exercise the pipeline (schedule → dispatch → score) with a fake
sender and no network access, exactly as `replayer/tests/test_v1_paper_traces.py` does.
