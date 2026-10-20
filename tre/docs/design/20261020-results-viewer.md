# Results viewer (tre_eval.viewer), 2026-10-20

An offline, interactive page for one trace of an evaluation campaign: how load, replicas,
latency and GPU occupancy evolve over time, side by side for every arm. It complements
`index.html` of `tre_eval.report` (static figures, tables, validation); it does not replace it.
It is not the cluster console (`tre/ui`).

## Inputs and output

```
python -m tre_eval.viewer <trace_report_dir> [-o viewer.html] [--trace NAME] [--raw-root DIR]
```

- `<trace_report_dir>` is a `--out` directory of `tre_eval.report`. From the top level only
  `comparison.csv` is read, filtered by trace and arm (a report can hold several traces).
- Per arm it reads `timeseries_1s.csv`, `events.csv`, `gpu_intervals.csv`, `requests.csv`
  (only `t_send`, `model`, `ok`, `tpot_ms`, `viol` for the TPOT p95 series), `onsets.csv`,
  `summary.json` (`validation`, `decision_points`, `warnings`) and `manifest.json` (trace,
  `load_s`, `decision_source`, `clock_offsets_s`, `component_nodes`, `registry_sha256`). Per-arm
  files keep traces apart; the top-level `onsets.csv` / `validation.json` are keyed by arm name
  only.
- Arm folders: the `report.py` hook passes the folders it has just written; the CLI takes every
  `arms/*` folder and warns about folders that are not in `comparison.csv` (left from an
  earlier report).
- Raw arm directory, the first that exists of `--raw-root/<arm>`, `manifest.dir`,
  `<report>/../<arm>`; all optional: `live-registry.yaml` (TP size, SLO, τ, GPU topology; from
  the first arm that has one, with a warning when `registry_sha256` differs across arms),
  `safescale.json` (probe spans), `baseline-scaler.log` (SM refusals with reason), `sm.ts.log`
  (raw line attached to wake / sleep events).
- Output: one self-contained HTML file. The data JSON, uPlot (MIT, vendored in
  `viewer_assets/`), CSS and JS are inlined; no network access at view time (works from
  `file://`). `tre_eval.report` writes `viewer.html` next to `index.html` (one file per trace
  when the report holds several traces: `viewer-<trace>.html`); `--no-viewer` skips it.
- A missing optional input never stops the export: the part that needs it is left out and a
  warning is shown in the page header.

## Embedded data model (`schema: 1`)

All times are seconds relative to load start (t = 0), as in the report CSVs.

### Clock rule

The viewer follows `load.py` exactly, so every marker lines up with the report's
`events.csv`:

- Corrected by the node's clock offset: SM events and the SM log (service-manager node),
  sidecar events (pod node), gateway events (redis node). These come from `events.csv`, except
  the raw `sm.ts.log` lines, which the viewer shifts by the same service-manager offset.
- Not corrected: controller ticks, decisions and `signal_log`; baseline-scaler decisions; APA
  decisions (runner host). The viewer adds SafeScale probes on the same uncorrected
  controller clock (`t0`, `t1`, and `t_hide` from `hide_anchor.controller_ts_ms`, not the Redis
  stamp), so a probe starts at the decision that opened it.
- Scaler refusals: `baseline-scaler.log` is stamped by the scaler host's clock, which `load.py`
  never reads. Each line is paired with its SM 409 access line (corrected SM clock) of the same
  model when scaler stamp − SM time is in [−1, 8 + `elapsed_s`] s, the closest first, and drawn
  at that SM time, which lines up with the decision markers. (In the E1 data of 10-19 the
  difference is −0.2 to +5 s, also for slow calls: the SM line marks the response. Adding
  `elapsed_s` to the upper bound keeps the pairing if it marked the request.)
- To pair them, the log is shifted by the offset of the scaler's node when `components.json`
  names it. Otherwise each candidate shift (none, or one of the node offsets) is tried, and the
  one with the most pairs is used only if it is clearly better: at least 3 pairs and at least
  twice the runner-up's count. The warning states the margin (for example "23/23 at −163.5 s,
  4/23 at +0.0 s"). If no shift is clearly better, every scaler line is "time uncertain": it is
  listed in the event table only, never drawn on a time axis, and the SM 409 access lines are
  drawn instead. In the E1 campaign of 10-19 the scaler log runs about 164 s ahead (the
  scaler's node), while its decisions (from the decision stream) are already on the reference
  clock.

Known limitation: `load.py` does not correct the controller and baseline-scaler clocks. When
their node is not the reference node (offset ≠ 0), or their node is unknown while offsets are
non-zero, decisions, probes and refusals can be shifted against the replica and latency
series. The page shows a warning for that arm.

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
         frag: [{tp, t0, t1, free}]}]}
```

- `ts` fields: `awake routable hidden target offered_rps rho viol_pct ttft_p95_ms tpot_p95_ms
  running waiting kv_mean z gpus_awake reissues`. Replica columns are copied unchanged from
  `timeseries_1s.csv` (sample-and-hold); `tpot_p95_ms` is computed here on the same 10 s
  send-time bins and percentile rule as the report's `ttft_p95_10s_ms`; `reissues` counts sidecar
  continuations per second (from `events.csv`, not drawn as markers).
- `events` kinds: `decision_up`, `decision_down` (TRE / APA / baseline scaler), `wake`
  (SM `wake_start` → `wake_done`, `t_end` = done; a start without done keeps reason
  `no wake_done`, and a second start on the same binding closes the first that way), `sleep`
  (SM, `reason` = path), `refusal` (baseline scaler: SM refused with 409, paired with the SM
  409 access line as described under the clock rule; without the scaler log, or with an
  uncertain scaler clock, the SM 409 lines; unpaired lines on either side are counted in a
  warning), `counterfactual` (what the
  TRE controller would have done on a baseline arm, observe mode). Layout `hidden/unhidden`,
  sidecar and SM polling lines are not markers (probe spans and Gantt hatching show hiding).
- `probes` from `safescale.json`, only for arms with `decision_source == tre`, and only probes
  that overlap [first grid second, load end + 90 s] (an older `safescale.json` holds every
  probe of the campaign; dropped probes are counted in a warning). `t0` = probe start, `t_hide`
  = hide, `t1` = resolution (`null` = still probing at the end), `outcome` = commit / rollback
  / probing, `reason` = rollback code, else `idle_commit`, else `terminal_reason`, `detail` =
  evidence summary.
- `gpus` copies every row of `gpu_intervals.csv` (a TP2 binding appears under both GPUs).
- `frag` (stranded TP slots): from the GPU layout only, the same rule for every arm. For each
  TP size k ≥ 2 in the registry: seconds where at least k GPUs are free (no awake binding) but
  no aligned free k-slot exists (`two_gpu_slots` for k = 2, else aligned blocks of k), merged
  into intervals. It does not use `target`: the `target` column means a different thing per
  decision source (TRE `signal_log` replicas_target, baseline scaler clamped, APA desiredScale),
  so a "below target" figure cannot be compared across arms. The page names the source when it
  shows a target value. Without `two_gpu_slots` the pairs (0,1), (2,3), ... are assumed, with a
  warning.
  It covers idle GPUs only. A placement blocked by GPUs that are awake (occupied) is not
  stranded time; see the SM 409 marks and the SafeScale probe outcomes for that (for example
  TRE shows 0 s in T7 because it never leaves a GPU idle).
- The data JSON is embedded with `<`, `>`, `&` written as `\u003c`, `\u003e`, `\u0026` (no string
  can end the script element or open a comment), and NaN / ±Inf become `null`.

Size: about 2–4 MB for a 20-minute trace × 4 arms (series rounded to 4 significant digits,
`null` for empty). Nothing is downsampled.

## Views

One shared time axis (seconds, shown as m:ss) for all three views; zoom state, cursor and arm
toggles are shared. The header holds a scorecard per arm (from `comparison.csv`; wakes /
sleeps and SafeScale probes counted in [0, load end]; decision counts in the tooltip, labelled
with the decision source), hot phases of the first arm, decision points and warnings.

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
   model, hidden (not routable) periods hatched, a TP2 binding drawn as one bar over both lanes, free
   time blank. Stranded TP-slot intervals are marked in a strip above the lanes, and SM 409
   refusals as a red ✕ on the lanes of the refused binding (in the strip when it has none).

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
`timeseries_1s.csv` columns, every `gpu_intervals.csv` row appears in the Gantt data; two arms
with the same layout and different targets get the same stranded seconds; probes only on TRE
arms, inside the arm's window, starting at their decision when the controller node has a
clock offset (with the warning); the embedded JSON cannot close the script element; missing
optional inputs give a page with a warning instead of an error; the page JS parses and its
DOM-free core (hash state, alignment) runs under `node` when available.
