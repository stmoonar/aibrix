# tre/calibration

`tre_calibration/` is the fitting library (windows, theta fit, delta margins, bootstrap,
ranking metrics). The campaign that produces its input lives in `tre/deploy/scripts/`:

| step | module |
|---|---|
| drive cells | `calibration_campaign.py` (ladder design: `calibration_ladder.py`), one `r3_grid.py` process per cell attempt |
| per-cell system-side capture | `calibration_capture.py` (called by `r3_grid.py`) |
| re-window raw captures | `rewindow_from_raw.py` |
| standard dataset | `calibration_dataset.py` |
| fit / freeze / accept | `dline_refit.py` (+ `alpha_fit.py`, `theta_verdict.py`) |
| ranking disclosure (AUROC, Kendall tau-b) | see `tre/docs/design/20260930-ranking-metrics.md` |

## Request endpoint: chat, fitted to the templated length

Since 2026-09-30 every calibration cell sends `/v1/chat/completions` (`r3_grid --api chat`,
`calibration_campaign --api chat`, both the default; `--gateway-url` must name the chat
path). Why chat and not `/v1/completions`:

* **It is what the experiments send.** v1 (calibration and experiments alike) and the E1
  client (`tre/loadgen_v1`) send one user message through the chat endpoint, so the engine
  prefills the model's chat template around the prompt. The completions endpoint prefills
  the bare string - on vLLM 0.30 without even a BOS. A theta calibrated on one prefill
  path is evidence about the other only by assumption.
* **The prefill path is then identical** - the same template (`<BOS><｜User｜>…<｜Assistant｜><think>\n`
  for all three DeepSeek-R1-Distill models, 5 tokens), the same BOS handling, the same
  tokenizer call in the engine. The one remaining difference is deliberate: calibration
  fixes the output length with `ignore_eos: true` + `max_tokens` (plus `temperature: 0`,
  `stream_options.include_usage`, and `seed` when `--request-seed` is given).

Exact length. `input_tokens` of a cell is the length of the prompt **after** the chat
template, i.e. what the engine reports as `usage.prompt_tokens`. The natural-prompt fitter
counts through the template (`tre_replayer.engine.model_tokenizer.for_api`: the template
rendered once with `apply_chat_template(add_generation_prompt=True)` into a prefix and a
suffix, checked on probe contents to render verbatim and to add a fixed number of tokens),
so the user content is `input_tokens` minus the template. The zh/en mix ratio applies to
that content (0.50 by token). The chat path needs `--prompt-mode natural`.

Gateway URL. There is no built-in default: pass `--gateway-url
http://<gateway>/v1/chat/completions` or set `TRE_CALIBRATION_GATEWAY_URL`; a run without
either is refused (a `--dry-run` / `--static-grid-list` needs none).

Checks, fail-closed:

* **Preflight** (`require_prompt_preflight`, every campaign entry point, next to the
  clock-domain check; standalone `r3_grid --prompt-preflight refuse`): per model and per
  input length - the run's shortest cell input, 512 and its longest (a sampled shape's
  range ends included; for the entry points that do not list their shapes, the span of
  every calibration shape: 128 / 512 / 4096) - one request of the run's exact kind
  (builder, tokenizer, corpus, endpoint, URL, routing header, seed) must come back with
  `usage.prompt_tokens` = the fitted length, `completion_tokens` = `max_tokens`, a first
  token in a recognised field and, for the mix, a content Chinese share within 0.01 of
  the target; otherwise the run is refused. Verdicts: `<out-dir>/prompt_preflight.json`.
  `--prompt-preflight skip` turns it off and is recorded. The same check as a standalone
  command (exit 0 only when every row is exact; one JSON row per model x target with
  `model, target, prompt_tokens, completion_tokens, zh_token_ratio, ok, reasons`):
  `cd tre/deploy && PYTHONPATH=../common:.:../replayer python3 -m scripts.calib_preflight
  --models dsqwen-7b,dsllama-8b,dsqwen-14b --gateway-url http://<gateway>/v1/chat/completions
  --out <file.jsonl> [--targets 128,512,2048]`.
* **Per request / per cell**: the raw log keeps `expected_prompt_tokens` (the fitted
  length) next to `input_tokens` (`usage.prompt_tokens`), on the schedule path and the
  closed-loop grid path alike; the guard artifact (`<cell>.guard.json`; the grid path:
  `<cell>.prompt_tokens_check.json`) records `api`, `chat_template_overhead`,
  `request_seed` and `prompt_tokens_check` (served requests whose two numbers differ, or
  without usage). The standard dataset carries `expected_prompt_tokens` per request and
  `api` / `prompt_tokens_mismatched` per cell, and **refuses to build** when a non-void
  cell has `prompt_tokens_mismatched > 0` (`--allow-prompt-token-mismatch` builds it for
  inspection and records the cells); `dline_refit trainset` never trains on such a cell.
* **Provenance**: `run_provenance["api"]` (endpoint, path, `ignore_eos`, seed, what
  `input_tokens` counts) is part of the load path (`scripts.prompt_corpus`). A dataset
  refuses campaigns of two APIs, a freeze records its training API and refuses a mixed
  training set, M / T14 / the training supplement / a `--reprobe-shapes` run (whose
  reused C_s comes from `--reprobe-source`; its load path goes to `reprobe_plan.json`)
  refuse another API - with **no** override flag. Records without an API are the
  completions captures from before 2026-09-30. A materialised prompt file records its
  `api` per row and a sender refuses a file of the other endpoint.
* **Errors inside a 200 stream**: an `{"error": ...}` SSE chunk sets `stream_error` on
  the request, which is then a model error (never a completion, never a latency sample).

TTFT / TPOT. The first token is the first SSE chunk carrying text in `text`
(completions), `delta.content`, or `delta.reasoning_content` / `delta.reasoning` (a
reasoning parser, not enabled on the fleet today); the role-only opening chunk and the
usage-only closing chunk are not tokens. Every request records this basis as
`ttft_basis: first_chunk_with_text` (sender row, raw log, guard artifact). TPOT stays the
client-side `(e2e - TTFT) / (completion_tokens - 1)` with `completion_tokens` from `usage`.

The E1 client (`tre/loadgen_v1`, profile `e1_v1`) reports v1's TTFT basis in its v1
columns: it stamps the first chunk whose `delta.content is not None`, and the role-only opening chunk
has `content: ""`, so its TTFT ends at the role chunk. vLLM emits that chunk in the same
engine iteration as the first token's text, so the two bases normally differ by the
serialisation of one chunk; they differ by one decode step (or more) whenever the first
token's text is empty - a byte of a multi-byte character (the mixed corpus makes Chinese
output likely), text the detokenizer holds back, or, with a reasoning parser, reasoning
that arrives outside `content`. The calibration TTFT is therefore never shorter than
E1's for the same request, and the TPOT of the two differs by the same amount spread over
`completion_tokens - 1`. Since 2026-09-30 both are profiles of one client
(`tre_replayer`, see `tre/replayer/README.md`): the E1 client's `performance_metrics.json`
keeps its v1-basis `ttft` / `tpot` and carries this basis next to them (`ttft_strict_s`,
`tpot_strict_s`, `success_strict`); compare TTFTs across the two only on the same basis.

The D6' label's idle-TTFT fit (`slo.ttft_idle_c_ms` / `_b_ms_per_token` in the registry)
was fitted on completions-era data; under chat each `L` includes the 5 template tokens
(`b * 5` is ~0.3 ms).

## Result directory layout

One campaign process per model writes `<run>/<model>/` (`--out-dir`). Capture layout
version 1 (2026-09-30) adds `manifest.json` and `cells/`; **every file that existed
before stays where it was**, so `rewindow_from_raw`, `calibration_dataset` and the fit
read old and new runs alike.

```
<run>/<model>/
  manifest.json                         run manifest (new): layout version, code commit/branch/dirty,
                                        registry path + sha256, model-pod and control-plane images,
                                        driver config, sha256 of plan.json / fit_plan.json / run_manifest.json
  plan.json  fit_plan.json  run_manifest.json  cells.jsonl  design_result.json  campaign_status.json
  <stem>.csv                            online windows (30 s window, 10 s grid)            [unchanged]
  raw/<stem>/<cell_id>.jsonl            per-request client log                             [unchanged]
  raw/<stem>/<cell_id>.instant.jsonl    1 Hz queue sidecar (sum over pods)                 [unchanged]
  raw/<stem>/<cell_id>.{guard.json,rps.csv,failures.jsonl}                                 [unchanged]
  prompts/<stem>/<cell_id>.prompts.jsonl  schedules/<model>/<stem>.json                    [unchanged]
  cells/<stem>/                         (new) one directory per cell attempt
    cell_meta.json                      identity, [start_ms, end_ms] (driver clock), the cell's redis-time
                                        span and clock-domain checks, pods (name, node, images), guard
                                        verdict, what was captured and how much, capture errors, and the
                                        RELATIVE paths of the unchanged files
    vllm_metrics_1hz/<ns>_<pod>.jsonl   per routable pod, 1 Hz (same scrape as the queue sidecar)
    gateway_redis_dump/hist/<ns>_<pod>.jsonl   tre:v2:hist:<ns>/<pod> docs covering the cell
    gateway_redis_dump/inst/<ns>_<pod>.jsonl   tre:v2:inst:<ns>/<pod> docs covering the cell
    controller_ticks.jsonl              tre:v2:decision:hist:<model> members covering the cell
  dataset/                              standard dataset (calibration_dataset)             [unchanged]
```

`<stem>` is the attempt name used by `raw/<stem>/` and `<stem>.csv`
(`<model>_<shape>_<primitive>_c<code>_a<n>`). `cells/` sits beside `raw/`, never inside
it: the re-window discovers cells by globbing `*.jsonl` under the raw root, and it also
skips any `cells/` directory in case a raw root is pointed at the run directory.
`calibration_capture.resolve_cell_artifacts(model_dir, stem)` returns one attempt's
files for either layout (old runs: the new entries are empty).

### File formats

* **`vllm_metrics_1hz/*.jsonl`** (`tre.vllm_metrics_1hz/v1`): a header line, then one row
  per sample. Kept: every `vllm:*_total` counter, the per-pod gauges (running, waiting, KV
  usage, swapped, sleep state) and the cumulative buckets + `_sum` + `_count` of the
  histograms TTFT, inter-token latency (0.10 name `time_per_output_token_seconds` too),
  e2e latency, request prompt tokens, request generation tokens (`r3_grid
  --vllm-histogram` to change). Series keys are Prometheus-style without `model_name`
  (`vllm:prompt_tokens_total{engine="0"}`); a histogram is `{"b": {le: count}, "s", "n"}`.
  Delta-encoded: a row with `"full": true` has everything; any other row only the series
  (for a histogram only the buckets / `s` / `n`) that changed since the previous row. A
  keyframe is written first, when the series set changes (pod restart) and every 300 rows.
  A failed scrape is `{"ts_ms", "error"}`. `calibration_capture.decode_vllm_metrics(lines)`
  yields the full state per second.
* **`gateway_redis_dump/{hist,inst}/*.jsonl`** (`tre.gateway_redis_dump/v1`): header, then
  `{"score": round stamp ms, "doc": <the doc as the gateway wrote it>}` for every doc of
  the model's pods with score in the cell's redis-time range (`clock.range_ms`, below). The
  pods are those of `tre:v2:pods:<model>` that still write in that range (the set is never
  pruned; sleeping residents are included - the controller reads their docs too), listed
  again at every backfill so a pod that came up in the tail is kept; `pods_in_set` counts
  the set.
* **`controller_ticks.jsonl`** (`tre.controller_ticks/v1`): header, then every member of
  the controller's decision history whose `window_end_ms` is in the same range: `trs`
  (TSS after the EMA), `trs_z_m` (`trs / theta_m`), `z_m` (active signal's Z), `state`
  (band), `window_end_ms`, `y_m`, `q_ctl`, replicas, plus `tss_raw` and `tss_raw_source`
  (like `trs`, a raw TSS of 0.0 also marks an idle / undefined window). The rescue and the
  fairness loop may both write a member for one window; dedup by `window_end_ms`.
* **Clocks: everything in redis time, checked, never shifted.** The driver reads redis
  `TIME` right before a cell's load and right after it drains; the dumps cover
  `[redis_start - margin, redis_end + margin]` (`clock.range_ms`), so a skewed driver
  cannot misplace a dump. The driver's own clock stays what the *client* files use
  (`start_ms` / `end_ms`, the per-request log, the window CSV, the vLLM 1 Hz files):
  to join those with the redis dumps, use the `redis_minus_local_ms` of
  `clock.cell_start.probe` / `cell_end.probe`. The gateway stamps its rounds
  (`now - now % 10 s`) with its node's clock and the controller's `window_end_ms` is a
  gateway round (the phase-aligned sampler publishes window `B` only once the gateway
  wrote its tick `B`), so both must be in redis's time domain; that is *checked*, not
  estimated:
  - gateway: its ticker started with the process, so it writes round `S` at `S + d` with
    a fixed write delay `d` in `[0, 10 s)` (the newest stamp alone trails redis by
    `[d, d + 10 s)`, which cannot tell a late ticker from a slow clock). The check waits
    for the next round (at most 12 s) and requires redis `TIME` at first sight minus its
    stamp in `[-tol, round + tol]` = `[-2, 12]` s;
  - controller: the two newest `window_end_ms` must be consecutive 10 s grid rounds (a
    free-running controller stamps its own clock - off the grid when sliding, 30 s apart
    when tumbling - and is refused) and the newest must trail redis `TIME` by
    `[-tol, round + read offset + tick + tol]` = `[-2, 31.5]` s (2-17 s at the base read
    offset of 2 s and the 5 s rescue loop; the offset adapts up to 9.5 s behind a late
    gateway, and with the fast loop disabled only the 10 s fairness loop writes);
  - gateway late writes: a doc may not appear more than one round + tol after its stamp
    (a second writer on a slow node, which the newest stamp hides). Every observation -
    the start mark, the end mark, the capture's dump, each backfill - fingerprints the
    docs it saw in the range; the next one fails the gateway on any older-stamped doc, of
    a pod it already listed, that the previous one did not see.

  Before a run every entry point of `calibration_campaign` checks each model and **refuses
  to start** on a failure, printing the reasons (a redis it cannot reach fails too: pass
  `--redis-url` from a host). A standalone `r3_grid --capture-dir` cell is its own run and
  refuses the same way (`--clock-domain-check refuse`); the campaign passes `flag` to its
  cells. Every cell is checked again at its start and end (`clock.cell_start` /
  `cell_end`); a failed source gets `clock_domain: clock_domain_mismatch` on its dump (the
  controller also when only the gateway failed: its windows are built from the gateway's
  docs), `cell_meta.json["clock_domain_mismatch"]` says why, the driver prints a warning,
  and the dump is kept unshifted but **never marked complete**. `clock.domain` is the
  verdict at the cell; a later backfill may mark a dump it re-dumps, so the per-dump
  `clock_domain` is authoritative. Labels and fits use the client's own records only, so
  a mismatch costs this supplementary evidence, nothing else.
  The check cannot see a gateway offset up to one round + tol (12 s: that far ahead looks
  like a late ticker, that far behind like an early one); the default margin
  (`--capture-margin-ms`) is one window + one round + that 12 s = 52 s at 30 s windows
  (a smaller one is refused) and the tail includes it too, so such an offset still lands
  inside the dump and before its tail. `--window-ms` must be the controller's
  `TRE_METRICS_WINDOW_MS` (the campaign's metrics window). Each check waits for one
  gateway round (~5 s, at most 12 s; with its two retries at worst 3 x 12 + 2 x 4 = 44 s
  per model, at each mark, in the pre-flight and in a backfill); the separate wait
  before the dump (`--gateway-flush-wait-s`) defaults to 0 since the end mark has just
  waited. Limit (controller, not capture): the phase-aligned sampler gives a window up
  0.5 s before the next round, so a gateway whose ticker writes more than ~9.5 s into its
  round leaves every controller window stale; the controller check then fails and the
  run is refused - restart the gateway plugin to re-roll its ticker phase.
* **Completeness and backfill.** A dump is `complete` when it reached `tail_ms` (the last
  10 s grid window end whose window can hold data of the cell, a blind-spot offset
  included: `floor((redis_end + window + 12 s - 1 ms) / 10 s) * 10 s`, i.e. redis end +
  32..41 s at 30 s windows), its `clock_domain` is `ok` and, for the gateway docs, its
  head was read before the gateway's 30 min retention could trim it
  (`head_within_retention`; a cell longer than ~29 min is never complete). Right after a
  cell the controller has not processed those windows yet, so both dumps are usually
  short (`reached_tail: false`) and the cell directory gets a `BACKFILL_PENDING` marker.
  The next cell's driver re-dumps every pending cell before it starts its load, and the
  campaign's finalize does the last one (waiting at most 90 s for redis time to pass the
  tail plus the controller's largest normal lag). Each backfill lists the model's pods
  again (a pod that came up in the tail is kept) and runs the clock-domain check again:
  rows added after a failed check, or late-written docs, make that dump
  `clock_domain_mismatch`. By hand:
  `python -m scripts.calibration_capture backfill <run dir> --redis-url <url>`
  (within 30 min for the gateway docs, ~24 h for the controller history). A re-dump
  replaces a file only when every row of the old file is still in redis (otherwise the old
  file is kept and the backfill record says why); every re-dump is logged in
  `cell_meta.json["backfills"]`. Because the backfill may run after a set is sealed, the
  M / T14 seals (`sha256sums`) exclude `cells/`.

### Controller ticks: no polling, one read-only field

The controller already keeps a per-model decision history in redis
(`tre:v2:decision:hist:<model>`, score `window_end_ms`, ~24 h retention) with the EMA'd
TSS, Z, band and window end of every tick - the same thing the console's timelines read.
The capture therefore reads that range once per cell instead of polling an API. What it
did **not** expose is the pre-EMA TSS: `y_m / q_ctl` misses the replica correction the
controller applies (the member's `assigned_replicas` is the bound count, not the factor
used). The controller now also writes `trs_raw` (read-only, no behaviour change;
`tre_controller/loops/tick.py`, `decision_snapshot.py`). Until a controller with it is
deployed, `tss_raw` is derived as `y_m / q_ctl` (factor 1, the serving-window path) and
`tss_raw_source` says so.

### Cost

Measured on a 60 s, 1 rps cell (one 7b pod, 8 pods with docs in the model's gateway set):
vLLM metrics 73 KB (median row 1.3 KB, keyframe 3.2 KB), gateway docs 209 KB (hist
~2.2 KB, inst ~0.33 KB per pod and round), controller ticks 4 KB, `cell_meta.json`
19 KB; parsing costs ~1.3 ms per scrape. A full 12 h, 3-model campaign adds roughly
0.4-0.7 GB (vLLM metrics 0.2-0.4 GB at 1.3-3.2 KB/s per pod, gateway docs ~0.3 GB for
~20 ready pods, the rest < 20 MB) against ~3.5 GB of existing output. Time: the clock
check at each cell's start and end waits for one gateway round (~5 s each, at most 12 s;
a failed check is retried twice, 4 s apart), the backfill takes well under a second per
cell.

### Switches

Every entry point of `calibration_campaign` (ladder, primitives, boundary / training
supplement, M acceptance set, T14) captures by default through `campaign.cell_command`;
`--no-capture-extras` turns it off for all of them. `r3_grid` alone captures only with
`--capture-dir` (`--no-vllm-metrics-capture`, `--no-gateway-dump`,
`--no-controller-ticks` drop one part). Nothing in the capture can fail or void a cell:
errors are recorded in `cell_meta.json["errors"]`. The only refusal is the clock-domain
check above, before any load is sent: once per campaign run, or per cell for a
standalone `r3_grid` (`--clock-domain-check refuse`, the default there).
