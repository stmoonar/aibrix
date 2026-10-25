# Instant sampling: 0.5 s instant docs, coverage flags, span-normalised tokens (2026-10-25)

Branch `fix/v2-sampling-20261025`. Status: implemented, not deployed. Revised after
review round 1 (H1, H1b, H2, M1–M5, L4–L6, S1–S5) and round 2 (evidence period 2 s by
default, fixed-timer fast loop, cheaper snapshot build) and round 3 (deploy order,
counter resets, retention, freshness, see the last section).

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
   | `TRE_EVIDENCE_PERIOD_MS` | 2 000 (10 000 before round 2) | gateway `tre:v2:hist` docs and v1 legacy keys; controller phase-aligned grid, O1 grid / suffixes, SafeScale `evidence_step_ms`, saturation "latest grid", `scrape_current_pods` grid, N2 warning refresh term, gateway-clock lag bound |

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
    written" signal). A snapshot the store marks stale (cadence) is not retried. Phase
    offset `min(2 s, period / 2)` = 1 s at 2 s: a published snapshot is about 1-2 s old.
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

## Evidence period 2 s (round 2, default)

- **Default** 2 000 ms in the code (`rediskeys.EVIDENCE_PERIOD_MS`, Go
  `defaultTREEvidencePeriod`), both overlays and the guard tests. Read offset
  `TRE_METRICS_PHASE_OFFSET_MS` 1 000. The window stays 30 s; v1 legacy keys stay on 10 s.
- **Fast loop on a fixed timer.** `rescue_task` runs every `TRE_RESCUE_INTERVAL_SECONDS`
  (5 s) and takes the newest published snapshot; a publish no longer wakes it
  (`SnapshotBox.wait_newer` and its trigger are removed). A snapshot is at most about
  2 s old when the loop reads it (one evidence period plus the read offset). Reading the
  same `window_end_ms` twice is safe: the EMA is time-based (`TssEma`, tau on the window
  end), band dwell is gone, and the O1 / saturation / stale rules are spans on window
  ends, so a repeated or skipped window does not advance them.
- **Cheap build.** One snapshot = two pipelined round trips (`_Batch`): (1) the pod sets
  and the cadence hash, (2) per pod the window's inst and hist docs plus the hist
  baseline as one doc (`ZREVRANGEBYSCORE … LIMIT 0 1` at or before the window start). An
  O1 suffix is built only when the model has a breakpoint inside the window (callback
  `suffix_breakpoint`, the O1 `SignalState`), and only the one suffix O1 reads
  (`ceil(bp / period) × period`). A breakpoint first seen after a read is used from the
  next snapshot (one snapshot, about 2 s, late).
- **Bench** (76, local Redis, 20 pods, 3 models, 50 builds incl. `tss_terms`, load avg
  about 6): 2 s period p50 35.6 / p99 50.6 ms without a breakpoint, 50.5 / 80.6 ms with
  one; 10 s period 26.7 / 41.0 and 40.2 / 58.9 ms.
- **Calibration tooling.**
  - `calibration_capture`: the round grid is the gateway's declared evidence period
    (`gateway_round_ms`, from `tre:v2:gateway:cadence`; 10 s for an older gateway). The
    clock-domain bounds, round phases (`mod period`) and the read-offset cap
    (`period − 500`) follow it. The flush wait stays a 12 s upper bound (it returns at
    the next doc, 0.5 s with instant docs).
  - θ fit: windows end on `FIT_GRID_MS` = 10 s (step 10 s), as before; these windows
    are a subset of the 2 s controller windows. The fit's queue signal stays the 10 s
    live-grid subsample of the 1 Hz sidecar (`LIVE_GRID_MS` = 10 s, fixed, no longer
    tied to the gateway period). So a fit samples the same window starts and the same
    queue samples as before. The live controller averages all 0.5 s instant docs, which
    is closer to `--instant-grid raw`; changing the fit's queue signal is part of the
    θ re-calibration, not of this change.
  - `r3_grid` live sampler lookback: 2 × the declared instant period (20 s fallback).
- **Percentile mode audit.** The controller reads `TRE_PERCENTILE_MODE` once
  (`config.py`) and passes it to the main and the SafeScale evidence `MetricsStore`
  (suffix windows use the same store) and to `SafeScaleConfig` (`safescale_direct`). The
  pooled p95 paths (`safescale_evidence`, `safescale_task`) use the store's `p95_rule`;
  their `"bucket_upper"` fallback applies only to a store without a rule (test fakes).
  No controller path hardcodes a mode. The offline tools take the mode from their own
  input, not from the env: the SLO label spec (`slo_labels`, `b_prime`,
  `preserve_mu`), `calibration_dataset` overrides, `r3_grid` / `rewindow_from_raw`
  `--percentile-mode`; each defaults to `bucket_upper`, the controller default. A run
  with `TRE_PERCENTILE_MODE=interpolated` must pass the same mode to these tools.
- **Before it goes live:** θ re-calibration on the new estimator (θ, λ, τ, w_p, c/b
  together), and a real-cluster smoke of the 2 s gateway write cost.

## Review round 3

- **Deploy order (H1).** New gateway + old controller: the old `sum / 3` queue reads
  60 instant docs as 20x Q, every model CRITICAL - a scale-up storm. New controller + old
  gateway: no cadence hash, every snapshot stale, no action. While images are swapped,
  in either direction, the run mode is `observe observe`
  (`deploy/RELEASE-20261025-v2-sampling.md`).
- **One pod set for queue and tokens (M2).** A token counter that went down (engine
  restart) restarts the delta at the first histogram doc after the reset; its
  `scraped_ms` starts the span (span rules apply); flagged. No usable delta (no
  histogram doc in the window, or none after the reset): the pod is out of queue and
  tokens, flagged. A pod with histogram docs and no instant doc has coverage 0: flagged.
- **Retention (M3).** Instant docs 5 min, histogram docs 30 min (gateway). About 2.7 MB
  per pod (600 x ~0.6 KB + 900 x ~2.6 KB), 53 MB for 20 pods (was 88 MB with 30 min of
  instant docs). The calibration capture spools a cell's instant docs every 60 s
  (`InstantSpool`) and a backfill merges the earlier dump's instant rows instead of
  requiring a superset; `head_within_retention` needs the spool's coverage for cells
  longer than the instant retention.
- **Freshness (M4).** A window is written when any pod has its end-boundary histogram
  doc (written in one MULTI, retried until EXEC succeeds); a round a few hundred ms late
  never gets an instant doc at the end (nearest-boundary rounding) and no longer leaves
  the window stale. Without histogram ticks (v1) the end instant tick decides.
- **Fresh cadence hash (M5).** `written_ms >= window_end - evidence period` (both gateway
  clock), else `gateway_cadence_stale` (snapshot stale, error logged on change).
- **L1** `restrict_to_serving` re-derives only the pod aggregates and keeps the store's
  flags (full window and O1 suffixes). **L2** `runTRERound()` is the loop's and the
  tests' commit decision. **L3** a histogram doc that cannot be built (schema error,
  NaN) does not move `lastHistT`. **L4** a scrape's values and its `scraped_ms` are
  stored and read under one per-pod lock, stamped at the store; an older overlapping
  fetch is dropped. **L5** `min_evidence_requests` and the low-evidence step count the
  observed requests (`request_count_observed`), not the span-scaled ones. **L6** the
  r3 live sampler looks back 2 x the declared evidence period and records no sample
  when nothing is found (`read_latest_instant` -> None). **L7** default read offset
  `min(2 s, period / 2)` everywhere.
- **Not converted yet:** the tick-counted rules (saturation confirm, breakpoint hold,
  SafeScale tail statistics) - see the evidence-not-ticks redesign.
