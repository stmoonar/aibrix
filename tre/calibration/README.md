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
    cell_meta.json                      identity, [start_ms, end_ms], pods (name, node, images), guard
                                        verdict, what was captured and how much, redis clock probes,
                                        capture errors, and the RELATIVE paths of the unchanged files
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
  the model's pods (`tre:v2:pods:<model>`, sleeping residents included - the controller
  reads them too) with score in `[start - window - 10 s, end + window + 10 s]`. After the
  cell's load has drained the driver waits (at most one round + 2 s,
  `--gateway-flush-wait-s`) for the gateway's next write. The gateway stamps a round
  `now - now % 10 s` but its ticker runs at an arbitrary phase:
  `cell_meta.json.gateway_redis_dump.flush_wait.write_phase_ms` (redis TIME when the new
  round was first seen minus its stamp, resolution 250 ms) is that phase.
  `redis_clock_before/after` bracket redis `TIME` with the driver's clock.
* **`controller_ticks.jsonl`** (`tre.controller_ticks/v1`): header, then every member of
  the controller's decision history whose `window_end_ms` is in the same range: `trs`
  (TSS after the EMA), `trs_z_m` (`trs / theta_m`), `z_m` (active signal's Z), `state`
  (band), `window_end_ms`, `y_m`, `q_ctl`, replicas, plus `tss_raw` and `tss_raw_source`.
  The rescue and the fairness loop may both write a member for one window; dedup by
  `window_end_ms`.
* **Completeness and backfill.** Right after a cell the controller has not yet processed
  the windows ending up to one window later (`tail_ms` in `cell_meta.json`), so both dumps
  are usually cut short: they are marked `complete: false` and the cell directory gets a
  `BACKFILL_PENDING` marker. The next cell's driver re-dumps every pending cell before it
  starts its load, and the campaign's finalize does the last one (waiting at most 90 s);
  by hand: `python -m scripts.calibration_capture backfill <run dir> --redis-url <url>`
  (within 30 min for the gateway docs, ~24 h for the controller history). A re-dump only
  replaces a file with a superset of it; every re-dump is logged in `cell_meta.json["backfills"]`.

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

Measured on a 60 s, 1 rps cell (one 7b pod, 8 pods in the model's gateway set): vLLM
metrics 73 KB (median row 1.3 KB, keyframe 3.2 KB), gateway docs 209 KB (hist ~2.2 KB,
inst ~0.33 KB per pod and round), controller ticks 4 KB, `cell_meta.json` 19 KB; parsing
costs ~1.3 ms per scrape. A full 12 h, 3-model campaign adds roughly 0.4-0.7 GB (vLLM
metrics 0.2-0.4 GB at 1.3-3.2 KB/s per pod, gateway docs ~0.3 GB for ~20 ready pods,
the rest < 20 MB) against ~3.5 GB of existing output. The gateway flush wait adds on
average ~5 s (at most 12 s) per cell, the backfill well under a second.

Nothing in the capture can fail or void a cell: errors are recorded in
`cell_meta.json["errors"]`. `calibration_campaign --no-capture-extras` turns it off;
`r3_grid` captures only with `--capture-dir`.
