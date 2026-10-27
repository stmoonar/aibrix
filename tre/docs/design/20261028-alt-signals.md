# Signal ablation: the v1 alternative signals in v2 (2026-10-28)

Status: implemented on `feat/alt-signals-20261028` (not deployed). Scope: the alternative
signals of the signal ablation (v1 10-23, D30-D40; plan `docs/v2-ablation-sensitivity-gap-20261028.md`
in the local workspace). Only the signal changes on an alt arm; the runner sets
`TRE_SIGNAL_SOURCE` and `TRE_DISABLE_ETA_GATE` (eta gate and waste ranking off on alt arms, D31).
The controller refuses to start with `TRE_SIGNAL_SOURCE` != `zm` and
`TRE_DISABLE_ETA_GATE=false` (plan section 8; review M3).

O1 (the breakpoint window: hold after a routable-count change, full window afterwards)
applies on every arm, alt arms included (same hold, same rules). The alt EMA keeps the
configured metrics window as its idle-gap length, also on an O1 suffix, as the TSS EMA
does (review M2). Only the onset saturation rescue is zm-only.

## Signals (`tre_common.alt_signals`, shared by the controller and the fit)

| `TRE_SIGNAL_SOURCE` | Value | Direction | v1 |
|---|---|---|---|
| `queue_len` | `max((running + waiting) / max(1, routable), 1)`, summed over the routable pods | lower is healthier | `1 / max(q, 1)` |
| `ttft_mean` | pods' TTFT histogram means pooled by first-token count (sum of sums / sum of counts), floor 1 ms | lower is healthier | `1 / max(mean TTFT, 1 ms)` (D38) |
| `kv_usage` | KV headroom `max(1 - min(1, sum of pod KV fill / routable), 0.001)` | higher is healthier | same |

`z = theta / value` (lower) or `value / theta` (higher), capped at `Z_MAX` = 10.
`decode_tps` / `prefill_tps` (plan 6.9) stay; `latency_p95` stays but is not v1's latency
signal and is not used. `kv_cache` (prefix-cache hit rate / 0.5, always 0 with prefix
caching off) is removed.

## Deviations from v1 (disclosed)

1. The shared tau-EMA (registry `ema_tau_ms`, the TSS tau) smooths the value (q, TTFT ms),
   not v1's H = 1/value, and replaces v1's fixed alpha per 5 s tick (D37 kept in spirit:
   the TSS smoothing is reused). Online and offline use the same function, so the fit sees
   exactly the controller's value.
2. No first token in a window: v1 held the last EMA forever and read Z = 1 (HEALTHY)
   before the first sample. v2 holds the last value only while the shared EMA keeps it
   (at most one metrics window, traffic flowing); an idle window or no prior sample is
   unknown. Reason: unknown never becomes healthy.
   On the `ttft_mean` arm, deep overload where no request gets its first token within a
   window reads as "no sample": the last value is held for one window, then the signal
   is UNKNOWN (no decision), not CRITICAL (review L6). The arm then does not scale on
   that model until a first token completes again (v1 held the last value).
3. Fail closed (v1 had no such check): `queue_len` / `ttft_mean` / `kv_usage` are unknown
   when a routable pod has no usable data in the window (scrape-stale, unknown queue, a
   short token span: the store leaves it out of `per_pod`), no replica is routable, or a
   pod lacks the gauge / histogram (a missing TTFT histogram is unknown, not "no first
   token": review L1, L3). `queue_len` used to divide the remaining pods' sum by every
   routable replica (q too low, z too high).
4. `queue_len` with routable 0 is unknown (was: divide by 1). Without a fleet view the
   window's pods are checked by count.
5. The sums run over the fleet view's routable pods, matched by identity (review L2): a
   hidden-but-awake pod (a SafeScale probe, still in the window while it drains) is left
   out of queue, fill and TTFT, and cannot stand in for a routable pod without data. The
   SafeScale observation uses the same rule.
5a. KV fill of a pod woken mid-window: the gateway docs carry no sleep state, so the
   controller averages every instant doc of a pod the fleet view reports awake, pre-wake
   samples (fill ~0) included. The offline columns follow the same rule (a pod serves
   when awake in its last sample of the window; all its samples are averaged); they used
   to drop asleep samples (review L4: the online side cannot, so the offline side moved).
   O1 holds a window that holds a wake on every arm, so a decision rarely reads one.
   A missing series voids only its own columns offline (as online: each signal reads its
   own series). `test_alt_signal_parity_20261028.py` feeds one set of raw per-pod samples
   (a wake, a missing TTFT histogram) through both paths: equal values.
6. 8b is fitted on its own data (v1 D40 reused 7b).
7. Fit method: alternative signals keep the balanced-accuracy argmax fit (theta on the
   healthy-quantile grid, queue_len on its distinct values; bands by BA argmax over the
   grids below), the existing `theta_verdict.SignalSpec` design. The TSS freeze uses
   `m2_unique_grid_plateau_midpoint`. Same labels, windows, EMA, bootstrap and family
   rule; different selection rule.

## Bands and basis

- Band grids (`tre_calibration.alt_signals.ALT_DELTA_{CRIT,HIGH}_GRID`, one definition,
  used for the merged fit and the bootstrap): delta_crit 0..0.99, delta_high 0..`Z_MAX` - 1
  = 9, step 0.01. The TSS grids (0..0.5 / 0..2) clamped ttft_mean delta_crit and queue_len
  delta_high in the first offline fit. The wider grids cover every band the classifier can
  use (tau_crit > 0; z <= Z_MAX). `params_set` bounds delta_high at 9.
- Basis stamp, symmetric with TSS `trs.numerator` / `theta_window_ms`:
  `alt_thresholds.<signal>.basis = {definition: ALT_SIGNAL_DEFINITIONS[signal], window_ms, ema_tau_ms}`.
  `fit_alt_thresholds` writes it (window = the fitting CSV's window, `ema_tau_ms` = the
  fit's tau, 0 = raw); `params_set` sets it
  (`<m>.alt_thresholds.<s>.basis.{definition,window_ms,ema_tau_ms}`, only the current
  definition id is accepted). The controller refuses to start an alt arm unless every
  model has theta, direction, delta_crit, delta_high and a basis equal to its definition,
  `TRE_METRICS_WINDOW_MS` and the tau it smooths with (`alt_ema_tau_ms(trs.ema_tau_ms)`,
  review L5). There are no default bands: the 0.2 / 0.25 fallback is removed from
  `classify.py` and the registry (a missing band there is an error). Change a definition
  id whenever the value changes.
- `merge_live_registry.py` merges each `alt_thresholds.<s>` as one fit (review M1):
  `{theta, direction, delta_crit, delta_high, basis}` come whole from live (when live has
  a theta) or whole from the release file. A live theta without a basis keeps none, so
  the alt arm refuses to start until a stamped fit is published.

## Calibration data

`calibration_dataset` adds `ttft_server_mean_ms`, `ttft_server_count`,
`kv_cache_usage_sum`, `kv_cache_pods` from the 1 Hz per-pod capture
(`scripts/alt_signal_columns.py`, same per-pod rules as online; void = empty); see
`docs/DATASET.md`. `rewindow_from_raw` adds the same columns when the cell has a 1 Hz
capture, and `calibration_campaign.ALT_SIGNALS` lists all five signals, so the campaign
fits ttft_mean / kv_usage too (not a manual step).

## Offline check (not published)

`calib_20261003` training stages (run1, run2/run2b, supp, p1/p1r2), step-counter numerator,
20 s / 10 s windows, hybrid label (500 / 75 ms, slowdown), tau 10 s, repo registry label
c/b and lambda (provisional). Output: `/data/nfs_shared_data/xxy/alt-fit-offline-20261028/alt2/`.
All bands interior. Rerun after the review fixes (KV pre-wake rule, per-column void,
`basis.ema_tau_ms`): `.../alt3/` (datasets rebuilt): every value below unchanged, the
`windows.csv` files byte-identical (single-pod capture: no mid-window wake, no missing
series), basis now `{definition, window_ms: 20000, ema_tau_ms: 10000}`.

| Signal | 7b theta / dc / dh | 8b | 14b |
|---|---|---|---|
| ttft_mean (ms) | 216.2 / .81 / .98 | 249.1 / .86 / 1.58 | 315.0 / .90 / 1.41 |
| kv_usage | .819 / .18 / .11 | .654 / .31 / .33 | .791 / .21 / .19 |
| queue_len | 31.3 / .41 / 2.61 | 27.0 / .21 / 2.84 | 86.3 / .39 / 3.70 |

ttft_mean's crit band sits far out (CRITICAL at TTFT > 5-10 x theta). The real fit runs
after the TSS freeze with its label registry and lambda, and goes out with the release.
