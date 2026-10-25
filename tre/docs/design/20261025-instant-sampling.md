# Instant sampling: 0.5 s instant docs, coverage flags, span-normalised tokens (2026-10-25)

Branch `fix/v2-sampling-20261025`. Status: implemented, not deployed. Revised after
review round 1 (H1, H1b, H2, M1–M5, L4–L6, S1–S5).

## Problem

v1 wrote one serial Redis SET per pod per round. Under CPU throttling rounds were lost.
The controller divided the summed queue samples by the *expected* sample count, so a
lost sample read as a zero: Q too low, Z = TSS/θ 1.4–1.85× too high, false HIGH, wrong
shrink. v2 had the same shape: the window was "complete" when every tick appeared in
*some* pod (union), and a pod whose scrape stopped up to 30 s ago still passed.

## Design

1. **Two gateway periods** (`tre_common.rediskeys`; gateway env and controller env of
   the same names):

   | setting | default | what uses it |
   |---|---|---|
   | `TRE_INSTANT_SAMPLE_INTERVAL_MS` | 500 | gateway `tre:v2:inst` docs; controller per-pod coverage grid |
   | `TRE_EVIDENCE_PERIOD_MS` | 10 000 | gateway `tre:v2:hist` docs and v1 legacy keys; controller phase-aligned grid, O1 grid / suffixes, SafeScale `evidence_step_ms`, saturation "latest grid", `scrape_current_pods` grid, N2 warning refresh term, gateway-clock lag bound |

   The evidence period must be a multiple of the instant period; the metrics window a
   multiple of the evidence period. Neither is the upstream `RequestTraceWriteInterval`
   (request traces, unchanged at 10 s).
2. **Declared cadence** (M4, L6). The gateway writes the hash `tre:v2:gateway:cadence`
   (`instant_ms`, `evidence_ms`, `written_ms`, `version` = VCS revision of the binary)
   inside the MULTI of every round. The metrics store reads it with every snapshot and
   compares it with the controller's two periods. Missing or different: the snapshot is
   stale at once (decision loops skip it) and an error is logged on each change
   (`TRE_GATEWAY_INTERVAL_CHECK=fail`, default; `warn` = log only; `off` = no check).
   Never a crash. The former inference from doc spacing (startup only, fatal, wrong on
   the mixed stamps of an upgrade) is removed.
3. **One MULTI/EXEC per round** for all pods (`writeTREPodMetricsToRedis`). A reader sees
   a round for every pod or for none.
   - Instant docs: every round, stamped `roundT` = the instant boundary nearest the tick
     (unchanged rounding: survives a small backward wall-clock step).
   - Histogram docs of evidence boundary `B = floor(roundT / evidence) × evidence`: on
     the first round with `B > lastHistT`, stamped `B`. No catch-up special case: a late
     round still writes `B`, and the doc's `scraped_ms` says when the counters were
     observed. The boundary in progress when the writer (re)starts is written too.
   - `lastHistT` moves only after `EXEC` succeeded: a failed `EXEC` retries `B` on the
     next round.
4. **Pod set** (S1, S5). A pod with a doc in the window is in the window, except an
   **I3 pod** (newest `scraped_ms` of its window docs before the window start: frozen
   metrics), which is left out entirely and listed in `scrape_stale_pods`. A pod with no
   doc is not in the window. Queue and tokens always come from this one pod set; the O1
   suffixes use the same set. A pod is never removed for low coverage or an odd span.
5. **Queue** (M2). A pod's queue gauges are the mean over all its instant docs in the
   window. A repeated `scraped_ms` is a zero-order hold of the last observation, a fair
   time-weighted estimate; no doc is filtered (no age threshold; `sample_max_scrape_age_ms`
   is removed). No expected-sample divisor, so a lost doc no longer pulls Q down.
   `rewindow_from_raw.aggregate_window` uses the same mean.
6. **Tokens** (H1, H1b, M3, S3). `tokens = (end sum − baseline sum) / Δscraped_ms × W`.
   The baseline is the newest histogram doc at or before the window start (else the
   first inside the window); the end doc is the newest in the window; `Δscraped_ms` is
   the span of their `scraped_ms`. A lost start doc (40 s span, formerly 1.33×) and no
   baseline in the lookback (20 s span, formerly 0.67×) give the true rate. Request count
   is scaled the same way; latency histograms are unchanged.
7. **Flags** (`sample_incomplete_pods`, per window and per O1 suffix):
   - coverage = distinct `scraped_ms` in the window (one pod's series) / (W / instant
     period) below `scaling.sample_min_ratio` (0.9);
   - token span outside [0.67, 1.5] × W;
   - missing or invalid `scraped_ms` (no span: the unscaled delta is kept; fail closed
     into the flag, never into UNKNOWN).
8. **`sample_quality_hold`** (H2, M1). The tick sets this context key when a pod that
   serves (SM fleet view) is flagged, I3-stale, or has no data in the window (M1:
   `serving − (per_pod − flagged)`; the window's suffix flags count when the decision
   window is a suffix). Values: `scrape_stale` / `pod_missing` / `samples_incomplete`.
   It blocks only the donor gate (planner: no shrink, release or donor for the model).
   It does not touch `signal_full_window`: the old I4 overload set it to False, which
   also capped a receiver at one step (`partial_window_max_step`). Receivers, the TSS
   and the O1 window are unaffected. `signal_hold_reason` keeps its O1 meaning.
9. **Rescue paths see all docs** (H2). `_latest_instant` (saturation rescue input) is
   the newest doc; `scrape_stale_inflight` is read for every I3 or flagged pod. In the
   H2 case (all pods scrape slowly) the TSS rescue, the saturation rescue and, for I3
   pods, the frozen-scrape rescue all work.
10. **SafeScale evidence store** (M5): `read_instant=False` - latency histograms only,
    raw deltas, no flags; the I3 rule still leaves frozen pods out.
11. **Sampler**: `window_freshness` only waits for the end tick (the gateway-wide "round
    written" signal). A snapshot the store marks stale (cadence) is not retried.
12. **Count rules become spans** (same span at any evidence period; deprecated count
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

    `gateway_clock` `resume_after` is a count of clock checks, not of windows:
    unchanged. The clock-lag fallback bound stays 2 × evidence period (L1 not changed:
    the main path is the `written_ms` offset, and a tighter bound would suspend O1 on a
    3 s CFS stall).

## Known limitations

- **L2**: the gateway never `SREM`s a pod from `tre:v2:pods:<model>`; the set expires
  only by its 2 h TTL. Deleted pods stay listed (they have no docs, so they add no data;
  readers pay one empty read each).
- The real scrape interval is `AIBRIX_POD_METRIC_REFRESH_INTERVAL_MS=50` (10 workers,
  non-blocking queue; `scraped_ms` stamped before the fetch), so coverage at a 0.5 s
  instant period is normally 1.0.

## Evidence period 2 s

Not the default yet. All period uses are on `TRE_EVIDENCE_PERIOD_MS`; the count rules
are spans; the read offset defaults to `min(2 s, period / 2)`. The calibration capture
takes its round grid from `EVIDENCE_PERIOD_MS` (`GATEWAY_ROUND_MS`); round phases come
from the histogram docs. Before 2 s goes live: θ re-calibration on the new estimator,
the remaining 10 s assumptions of the calibration tooling, and the per-refresh read cost.
