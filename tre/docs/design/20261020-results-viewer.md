# Results viewer (tre_eval.viewer), 2026-10-20

An offline, interactive page for one trace of an evaluation campaign: how load, replicas,
latency and GPU occupancy evolve over time, side by side for every arm. It complements
`index.html` of `tre_eval.report` (static figures, tables, validation); it does not replace it.
It is not the cluster console (`tre/ui`).

## Inputs and output

```
python -m tre_eval.viewer <trace_report_dir> [-o viewer.html] [--trace NAME] [--raw-root DIR]
```

- `<trace_report_dir>` is a `--out` directory of `tre_eval.report` (`comparison.csv`,
  `onsets.csv`, `decision_points.csv`, `validation.json`, `arms/<arm>/...`).
- Per arm it reads `timeseries_1s.csv`, `events.csv`, `gpu_intervals.csv`, `requests.csv`
  (only `t_send`, `model`, `ok`, `tpot_ms`, `viol` for the TPOT p95 series), `summary.json`,
  `manifest.json`.
- Raw arm directory (`manifest.dir`, else `<report>/../<arm>`, else `--raw-root/<arm>`), all
  optional: `live-registry.yaml` (TP size, SLO, τ, GPU topology), `safescale.json` (probe
  spans), `baseline-scaler.log` (SM refusals with reason), `sm.ts.log` (raw line attached to
  wake / sleep events), `components.json` + `clock_offsets.json` (clock correction of the
  controller / baseline-scaler / SM timestamps, same rule as `load.py`).
- Output: one self-contained HTML file. The data JSON, uPlot (MIT, vendored in
  `viewer_assets/`), CSS and JS are inlined; no network access at view time (works from
  `file://`). `tre_eval.report` writes `viewer.html` next to `index.html` (one file per trace
  when the report holds several traces: `viewer-<trace>.html`); `--no-viewer` skips it.
- A missing optional input never stops the export: the part that needs it is left out and a
  warning is shown in the page header.

## Embedded data model (`schema: 1`)

All times are seconds relative to load start (t = 0, reference clock), as in the report CSVs.

```
{schema, generated, tool, trace, seed, report_dir,
 models: [{name, tp, slo: {ttft_p95_ms, tpot_p95_ms}, tau: {crit, low, high} | null}],
 nodes:  [{name, short, gpus, pairs: [[0,1],[2,3]]}],          # GPU topology (registry, else intervals)
 onsets: [{arm, model, t_on, t_off, ...}], decision_points: [...],
 warnings: [str],                                               # global (export) warnings
 arms: [{name, label, family, decision_source, color: {light, dark}, dash: [..],
         summary: {model|ALL: {n, V_req_pct, V_req_ci_lo, V_req_ci_hi, ttft_p95_ms, tpot_p95_ms,
                                mean_gpus, gpu_s, gpu_s_per_good_req, goodput_rps, valid}},
         validation: {score_json, compare_json, mismatches}, warnings: [str],
         ts: {t0, n, models: {m: {field: [v | null] * n}}},     # 1 s grid t0 .. t0+n-1, no downsampling
         events: [{id, t, t_end, kind, model, binding, src, label, reason, detail, log}],
         probes: [{id, model, t0, t_hide, t1, outcome, reason, binding, detail}],
         gpus: {"node/gpu": [[t0, t1, model, binding, [[h0, h1], ...], open_end], ...]},
         frag: [{model, t0, t1, free}]}]}
```

- `ts` fields: `awake routable hidden target offered_rps rho viol_pct ttft_p95_ms tpot_p95_ms
  running waiting kv_mean z gpus_awake reissues`. Replica columns are copied unchanged from
  `timeseries_1s.csv` (sample-and-hold); `tpot_p95_ms` is computed here on the same 10 s
  send-time bins and percentile rule as the report's `ttft_p95_10s_ms`; `reissues` counts sidecar
  continuations per second (from `events.csv`, not drawn as markers).
- `events` kinds: `decision_up`, `decision_down` (TRE / APA / baseline scaler), `wake`
  (SM `wake_start` → `wake_done`, `t_end` = done), `sleep` (SM, `reason` = path), `refusal`
  (baseline scaler: SM refused with 409, else SM access log 409), `counterfactual` (what the
  TRE controller would have done on a baseline arm, observe mode). Layout `hidden/unhidden`,
  sidecar and SM polling lines are not markers (probe spans and Gantt hatching show hiding).
- `probes` from `safescale.json`: `t0` = probe start (controller wall clock), `t_hide` = hide
  anchor, `t1` = resolution (`null` = still probing at the end), `outcome` = commit / rollback
  / probing, `reason` = rollback code or commit kind (idle commit), `detail` = evidence summary.
- `gpus` copies every row of `gpu_intervals.csv` (a TP2 binding appears under both GPUs).
- `frag`: seconds where a TP ≥ 2 model has `awake < target` while at least `tp` GPUs are free
  (no awake binding) but no aligned free slot (`two_gpu_slots`) exists, merged into intervals.

Size: about 2–4 MB for a 20-minute trace × 4 arms (series rounded to 4 significant digits,
`null` for empty). Nothing is downsampled.

## Views

One shared time axis (seconds, shown as m:ss) for all three views; zoom state, cursor and arm
toggles are shared.

1. **Timeline** (`1`). Stacked panels per model (model selector, or *All models* = small
   multiples in columns): load ρ (offered req/s ÷ per-replica capacity, i.e. replica
   equivalents; offered req/s when no capacity file), awake replicas per arm (step lines;
   hidden replicas = hatched band between routable and awake; ρ as a grey demand area),
   SLO violation share (10 s bins), TTFT p95 and TPOT p95 (10 s bins, SLO line, log toggle),
   queue (waiting, optional running), KV cache, Z (TRE arms) with τ_crit / τ_high lines,
   sidecar reissues. Hot phases from `onsets.csv` are shaded. An event strip (one row per arm)
   sits on top. Hover: one tooltip with every arm's value at that second.
2. **Events** (`2`). Swimlanes per arm (decisions, wake / sleep, SafeScale probes, refusals,
   counterfactual) above the replica panel, plus a filterable event table. Filters: kind,
   model, arm, text. Click a marker or a row: side panel with time, model, binding / GPUs,
   reason, detail JSON, raw log line; *Zoom here* centres ±30 s around it.
3. **GPUs** (`3`). Gantt per arm: one lane per GPU grouped by node, bars coloured by the awake
   model, hidden (probe) periods hatched, a TP2 binding drawn as one bar over both lanes, free
   time blank; fragmentation intervals marked in a strip above the lanes.

Interactions: drag on any plot to zoom, double-click or `r` to reset, `←`/`→` pan, `+`/`-`
zoom, `1`/`2`/`3` switch view, `t` theme, `Esc` closes the side panel. Click an arm chip to
toggle it, shift-click to show only that arm. The URL hash keeps view, model, arms, panels,
zoom, log scale and theme (`#v=1&m=dsqwen-7b&a=tre,apa&x=100,400`), so a link opens the same
view.

## Design choices

- uPlot (canvas, ~50 KB, MIT) for the line panels: 1 s data for 4 arms × 3 models × ~10 panels
  stays fast; cursor sync is built in. The Gantt and event lanes are plain canvas on the same
  x scale (uPlot has no Gantt mark). ECharts would add ~1 MB for features we do not use.
- Colours: arms by family, models by registry order, the same hues as `style.py` (static
  report); dark mode uses the dark steps of the same hues. Arm identity is never colour-only:
  arms of one family get distinct dash patterns, every tooltip and legend names the arm.
- Event markers: shape = kind (▲ up, ▼ down, ● wake, ○ sleep, ✕ refusal, bar = probe),
  colour = model.
- Python does all joins and derived series (testable, one place); the page only draws.

## Tests (eval suite)

`tre/eval/tests/test_viewer.py`, invariants only: the replica series equal the
`timeseries_1s.csv` columns, every `gpu_intervals.csv` row appears in the Gantt data,
missing optional inputs give a page with a warning instead of an error, and the page JS
parses and its DOM-free core (hash state, alignment) runs under `node` when available.
