# Instant sampling: 0.5 s instant docs, per-pod completeness, per-doc freshness (2026-10-25)

Branch `fix/v2-sampling-20261025`. Status: implemented, not deployed.

## Problem

v1 wrote one serial Redis SET per pod per round. Under CPU throttling rounds were lost.
The controller divided the summed queue samples by the *expected* sample count, so a
lost sample read as a zero: Q too low, Z = TSS/θ 1.4–1.85× too high, false HIGH, wrong
shrink. v2 had the same shape: the window was "complete" when every tick appeared in
*some* pod (union), and a pod whose scrape stopped up to 30 s ago still passed.

## Design

1. **Two gateway periods** (`tre_common.rediskeys`; gateway env and controller env of
   the same names, checked at controller start by `gateway_cadence.py`):

   | setting | default | what uses it |
   |---|---|---|
   | `TRE_INSTANT_SAMPLE_INTERVAL_MS` | 500 | gateway `tre:v2:inst` docs; controller per-pod completeness grid |
   | `TRE_EVIDENCE_PERIOD_MS` | 10 000 | gateway `tre:v2:hist` docs and v1 legacy keys; controller phase-aligned grid, O1 grid / suffixes, SafeScale `evidence_step_ms`, saturation "latest grid", `scrape_current_pods` grid, N2 warning refresh term, gateway-clock lag bound |

   The evidence period must be a multiple of the instant period; the metrics window a
   multiple of the evidence period. Neither is the upstream `RequestTraceWriteInterval`
   (request traces, unchanged at 10 s).
2. **One MULTI/EXEC per round** for all pods (`writeTREPodMetricsToRedis`). A reader sees
   a round for every pod or for none. A histogram boundary whose tick slipped by one
   instant period is caught up (both doc kinds stamped at the boundary); a later slip
   loses the boundary (the pods are then incomplete for that window).
3. **Per-doc freshness**: an instant doc counts only if
   `timestamp − scraped_ms ≤ scaling.sample_max_scrape_age_ms` (1000; gateway clock on
   both sides). The real scrape interval is `AIBRIX_POD_METRIC_REFRESH_INTERVAL_MS=50`
   (10 workers, non-blocking queue; `scraped_ms` is stamped before the fetch).
4. **Per-pod completeness**: a pod is complete for `(start, end]` when its own valid docs
   cover ≥ `scaling.sample_min_ratio` (0.9) of the instant ticks, and its newest histogram
   doc is at or after the window's last evidence boundary. Every pod is held to the
   whole window (joined / left mid-window = incomplete). An incomplete pod is left out
   of the window (`sample_incomplete_pods`), like a scrape-stale pod (I3). While such a
   pod serves (SM fleet view), the model gets `signal_full_window=False`,
   `signal_hold_reason="samples_incomplete"`: no scale-down / release evidence; a
   receiver still acts (I4, unchanged). A pod that left is not serving: no hold. It is
   not frozen-scrape rescue evidence (that stays `scrape_stale_pods` only).
5. **Mean over valid samples**: a pod's queue gauges are the mean of its valid docs (no
   expected-sample divisor). The offline `rewindow_from_raw.aggregate_window` follows.
6. **Sampler**: `window_freshness` only waits for the end tick (the gateway-wide "round
   written" signal); interior completeness is per pod.
7. **Count rules become spans** (same span at any evidence period; deprecated count
   aliases are read as N × 10 s with a warning; both names set = refused):

   | old | new | evaluated as |
   |---|---|---|
   | `scaling.min_evidence_grids: 2` | `min_evidence_ms: 20000` | complete post-breakpoint grids × grid ≥ span |
   | `scaling.breakpoint_hold_max_windows: 6` | `breakpoint_hold_max_ms: 60000` | first held window end − one grid → current end |
   | `scaling.saturation_consecutive_ticks: 2` | `saturation_confirm_ms: 20000` | full on every window from one grid before the first full end to the current end |
   | saturation `await_timeout_ms = 3 × grid` | 30 000 fixed | unchanged rule |
   | `TRE_PAPER_STALE_MAX_WINDOWS=3` | `TRE_PAPER_STALE_MAX_MS=30000` | current window end − last good window end |
   | `TRE_METRICS_STALE_HOLD_WINDOWS=2` | `TRE_METRICS_STALE_HOLD_MS=20000` | current stale window end − last published end |
   | sampler relearn every 60 cycles | 600 000 ms | cycles = span / period |
   | H3 `QUEUE_SAMPLE_HISTORY = 8` | `QUEUE_SAMPLE_HISTORY_MS = 80000` | samples newer than newest − span |
   | SafeScale early commit `grids × step` | `early_commit_min_evidence_ms` (= `min_evidence_ms`) | newest window end − post-hide start ≥ span |

   `gateway_clock` `resume_after` is a count of clock checks (every
   `safescale.evidence_poll_s`), not of windows: unchanged.

## Evidence period 2 s

Not the default. All period uses are on `TRE_EVIDENCE_PERIOD_MS`; the count rules are
spans; the read offset defaults to `min(2 s, period / 2)`. Before 2 s goes live: θ
re-calibration on the new estimator (see the report of the branch), the calibration
capture's dump metadata (`round_interval_ms`, phases, flush wait still use the 10 s
`SCRAPE_INTERVAL_MS`), and the controller's per-refresh read cost.
