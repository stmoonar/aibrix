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
  - controller: the newest `window_end_ms` must be on the 10 s grid (a free-running
    controller stamps its own clock and is refused) and trail redis `TIME` by
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
  `TRE_METRICS_WINDOW_MS` (the campaign's metrics window). Each mark waits for one
  gateway round (~5 s, at most 12 s); the separate wait before the dump
  (`--gateway-flush-wait-s`) defaults to 0 since the end mark has just waited.
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
