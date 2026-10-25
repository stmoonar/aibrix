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

    Round 4 replaced the "evaluated as" column with observation-time rules and changed
    the alias conversion: see "Evidence, not ticks (round 4)" below.

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
- **Freshness (M4; replaced in round 4 by `last_round_ms >= end`).** A window is written when any pod has its end-boundary histogram
  doc (written in one MULTI, retried until EXEC succeeds); a round a few hundred ms late
  never gets an instant doc at the end (nearest-boundary rounding) and no longer leaves
  the window stale. Without histogram ticks (v1) the end instant tick decides.
- **Fresh cadence hash (M5; round 4 moved it to the sampler: a late `last_round_ms` is
  retried and held like any late round).** `written_ms >= window_end - evidence period` (both gateway
  clock), else `gateway_cadence_stale` (snapshot stale, error logged on change).
- **L1** `restrict_to_serving` re-derives only the pod aggregates and keeps the store's
  flags (full window and O1 suffixes). **L2** `runTRERound()` is the loop's and the
  tests' commit decision. **L3** a histogram doc that cannot be built (schema error,
  NaN) does not move `lastHistT`. **L4** a scrape's values and its `scraped_ms` are
  stored and read under one per-pod lock, stamped at the store; an older overlapping
  fetch is dropped. **L5** `min_evidence_requests` and the low-evidence step count the
  observed requests (`request_count_observed`), not the span-scaled ones. **L6** the
  r3 live sampler records no sample when nothing is found (`read_latest_instant` ->
  None); round 4 removed its lookback (each pod's newest doc, with `age_ms`). **L7** default read offset
  `min(2 s, period / 2)` everywhere.
- **Not converted yet:** the tick-counted rules (saturation confirm, breakpoint hold,
  SafeScale tail statistics) - done in round 4, next section.

## Evidence, not ticks (round 4)

Design note `docs/evidence-not-ticks-20261025.md` (local workspace), approved 10-25,
SafeScale tail "one span, one value" included. Three review rounds found the same bug
class: a rule written as "N ticks / windows" changes meaning when the period changes,
and a mechanical N → N × 10 s change only moves the constant.

**Principle.** Time in a rule is observation time (gateway `scraped_ms`), never grid
time (doc boundaries). Overlapping 30 s windows are not independent evidence; only
requests are independent events. The evidence period is a storage detail: `store/`, the
sampler (`loops/metrics_task.py`), `gateway_cadence.py` and `store/evidence_grid.py` (the
SafeScale Redis evidence path, which must pick histogram docs) may read it; the
decision layer may not. Guard test `controller/tests/test_no_period_in_decision_layer.py`
forbids `grid_ms|evidence_period|period_ms|EVIDENCE_PERIOD_MS|INSTANT_SAMPLE_INTERVAL_MS|
SCRAPE_INTERVAL_MS|streak|consecutive|n_windows|grids` in `signals/`, `planning/`,
`loops/tick.py`, `rescue_task.py`, `fairness_task.py` (comments included).

**Mechanism.**

- `tre_common.metrics_schema.Observed(t_first_ms, t_last_ms, n_obs, requests, flags)` on
  every pod window, model window and suffix window, filled by the store: `t_first` = the
  `scraped_ms` of the token baseline doc, `t_last` = the newest instant `scraped_ms`,
  `n_obs` = distinct instant `scraped_ms`, `requests` unscaled. `merge` is
  order-independent (min / max / min / sum / union). `observed_of()` falls back to the
  window bounds for synthetic, v1 and offline windows.
- `tre_common.persistence`: `Persistence(span_ms).observe(t_obs, holds)` (a re-read of
  the same or an older observation changes nothing; false resets; fires when
  `t_last_true - t_first_true + 1 s >= span`), `span_reaches`, `fresh(now, t_last,
  max_age)`. `SPAN_TOLERANCE_MS` = 1 s: the fetch jitter, well below any period.
- The gateway writes `last_round_ms` (the round's instant stamp) into
  `tre:v2:gateway:cadence` in the round's MULTI (`pkg/cache/cache_tre_redis.go`).

**Rule inventory** (kind: (a) evidence amount, (b) persistence, (c) freshness, (d)
statistic gate).

| rule | kind | before | now | where |
|---|---|---|---|---|
| sampler: window written | c | the doc stamped exactly `end` exists (a round 250 ms late = stale) | `last_round_ms >= end`; a hash without it = `gateway_cadence_no_round` (stale) | `loops/metrics_task.py` `window_freshness`, `gateway_cadence.py` `cadence_problem` |
| sampler: stale hold | c | from the controller time the first stale window was given up (one period after its read) | from the first read of the first stale window | `metrics_task.py` `_mark_stale` / `_stale_for_ms` |
| sampler: phase relearn | - | every N sampled windows | every `relearn_ms` of controller time | `metrics_task.py` `run_once` |
| O1 evidence | a | complete post-breakpoint grids × grid >= 20 s | `Observed` span of the suffix (smallest suffix start >= breakpoint) >= `min_evidence_ms`, and requests >= `min_evidence_requests`; reasons `no_suffix` / `evidence_span` / `evidence_tokens` / `evidence_requests` | `signals/trs.py` `effective_window` |
| O1 hold limit | b | held windows, first one counted as one grid | `Persistence(breakpoint_hold_max_ms)` on the window's newest observation | `trs.py` `note_hold`, `tick.py` |
| saturation confirm | b | `+grid` for the first full window, `fresh_after = end - grid` | `Persistence(saturation_confirm_ms)` on the sample's observation time | `signals/saturation.py` `observe` |
| SafeScale retry after a rollback | a | any window ending after the rollback | observations spanning `early_commit_min_evidence_ms` (20 s) after it (`no_new_evidence`), then Z / routable as before | `planning/safescale.py` `rollback_retry_holds` |
| SafeScale early commit | a | post-hide grids of the newest window | newest observation − hide confirmation >= the span | `safescale.py` `_try_early_commit` |
| SafeScale commit-gate tail | d | min Z / max KV / any latency over the last `max(2, ceil(hq × N))` snapshot observations | one value per quantity over one span: the newest snapshot's tail suffix (start `floor(end - hq × W)`), raw TSS / Z, time-weighted KV of the remaining pods, its p95s | `safescale.py` `_summarize_tail`, `loops/safescale_task.py` `_tail_terms`, store `suffix_tail` |
| SafeScale immediate rollback | d | every snapshot whose window follows the hide | only windows that do not overlap a window already judged | `safescale.py` `_instant_violation` |
| H3 queue samples | a | `fresh_after = end - grid` | the sample's observation time after the breakpoint | `tick.py` `_queue_sample` |
| I3 current pods | c | scraped within one evidence period | `fresh(end, scraped, 4 × instant period)` | `store/metrics_store.py` |
| gateway clock (O1 same-clock check) | c | `written_ms` offset, else a doc-stamp lag bounded by 2 × period | `written_ms` offset only; the probe key is the pod with the newest doc; no measurement = no verdict; liveness is the sampler's | `gateway_clock.py` |
| r3 live sampler | c | lookback 2 × declared period | each pod's newest doc at or before now, with `age_ms` | `metrics_store.py` `read_latest_instant`, `deploy/scripts/r3_grid.py` |
| rescue `need` count | - | carried, unused | removed | `tick.py` |

**Deprecated count aliases** (`common/tre_common/registry.py` `_scaling_span_ms`): read
as `(N - 1) × 10 s` for the persistence rules (the first window starts the span; it is
not one grid of evidence) and `N × 10 s` for `min_evidence_grids`. So the live
`breakpoint_hold_max_windows: 6` is 50 s (was read as 60 s in round 2) and
`saturation_consecutive_ticks: 2` is 10 s (was 20 s). The new defaults are
`breakpoint_hold_max_ms: 50000`, `saturation_confirm_ms: 10000`: today's behaviour at
10 s. `breakpoint_hold_max_windows: 1` is refused (a 0 s hold limit).

**Behaviour change at 10 s too (SafeScale tail).** The commit gate's Z and KV used to
be the min / max over 2-3 overlapping snapshot observations; now they are one value on
one span. At 10 s this is usually less strict than before (one dip in one of the overlapping
windows no longer fails the gate), at 2 s it no longer gets five times the chances to
dip. The latency gate of the live controller (direct evidence) is unchanged.

**Tests.**

- `common/tests/test_persistence.py`: a verdict does not change when observations are
  thinned (seeded random timelines); the fire time moves by at most the sampling gap
  (0.5-10 s periods); a re-read is not new evidence; `Observed.merge` does not depend on
  the pod order.
- `controller/tests/test_period_invariance_20261025.py` with the renderer
  `controller/tests/period_sim.py`: three closed-loop scenarios (onset spike with a
  gateway stall; HIGH shrink with two committing probes; rollback, held by the same
  evidence, then a retry that commits) through the real store, sampler, planner loops
  (`loops/replay.py` now keeps `signal_state` / `paper_state_cache` across calls) and
  SafeScale, at P = 2 / 5 / 10 s. The probe verdicts are identical; each key event's
  delay from its cause is within 10 s + 5 s (one period of the coarser run plus one
  rescue tick) of the 10 s run's; O1 warms >= 20 s after a routable change, saturation
  fires on >= 10 s of full observations, at every period; no stale snapshot without a
  stall. Delays (s) seen:

  | event (cause) | 2 s | 5 s | 10 s |
  |---|---|---|---|
  | saturation rescue (onset) | 15 | 15 | 15 |
  | first CRITICAL scale-up (onset) | 25 | 30 | 35 |
  | O1 warm (last routable change) | 29.75 | 29.75 | 39.75 |
  | first stale snapshot (stall start) | 22 | 25 | 30 |
  | shrink probes: start / verdict (start) | 35, 70 / 21, 40 | 35, 75 / 17, 37 | 35, 75 / 17, 37 |
  | retry: verdicts (start) | 21, 31, 41 | 17, 27, 38 | 17, 27, 37 |

  The test found the stale-hold start: from the give-up time the 10 s run was 40 s
  (16 s more than at 2 s); from the first read it is 30 s.
- `controller/tests/test_instant_sampling_20261025.py`: a round 300 ms late gives no
  stale window; `test_o1_breakpoint_window_20261001.py`: leftover docs of a model that is
  down do not suspend O1.

**What still depends on the period, by design.** Data age: a window is read one
period plus the offset after its end. A suffix starts at a histogram doc, so up to one
period of post-breakpoint data is not used (`ceil`). The sampler retries a missing
window until the next boundary. Each is at most one period. In the renderer (and in
offline constructions without direct evidence) a probe's deadline runs on the snapshot
clock (`now_ms = snapshot.ts_ms`), so at 10 s a probe is judged about one period of
wall time sooner after its hide than at 2 s; the live controller's direct evidence
runs on the controller clock from the hide confirmation.

## Review round 5

An independent review of round 4 (`dd004d3c`). What changed, superseding the round-4
text above where they differ:

- **Window written = the histogram boundary (L2).** The cadence hash carries
  `last_hist_ms` (`pkg/cache/cache_tre_redis.go`): the newest histogram boundary whose
  docs were ALL built, written in the round's MULTI only then (`histOK`). The former
  `last_round_ms` moved every instant round even when the boundary's histogram docs
  failed. Freshness: a window ending at `E` is complete once `last_hist_ms >= E`; a
  hash without it is `gateway_cadence_no_hist` (stale). Trade-off: a pod whose doc can
  never be built keeps every window stale (fail-closed).
- **Persistence (L3).** A false verdict always resets, also at an observation time
  already seen; only repeated true verdicts at an old time are skipped.
- **`Observed` (L5)** is `(t_first_ms, t_last_ms)` only; counts and flags live on the
  window (`request_count_observed`, `sample_incomplete_pods`).
- **Saturation sample freshness (M2).** The store marks each pod window
  `latest_current = fresh(end, latest_instant_ms, 4 x instant period)` (the
  `scrape_current_pods` rule, bound to the producer cadence); `saturation_sample` skips
  pods that are not current, and an added pod without a current sample is not full.
- **SafeScale commit-gate tail (H1).** The tail is the post-hide span of the newest
  snapshot: the store builds the suffix from the first doc boundary at or after
  `SafeScaleStateMachine.tail_start_ms` (= `_post_hide_start`: the first gateway
  boundary after the newest doc at the confirmed hide; the planned start without an
  evidence source) - the whole window once the hide precedes it. One value per
  quantity; plus `tail_observed_ms` (observed span) and `tail_requests` (requests of the
  remaining pods). Evidence floor: no traffic, or `min_commit_samples` requests, or a
  whole-window tail (the planner's own evidence unit: a sliding window gains no
  requests by waiting - holding a low-rate model to the floor would only push every
  probe to the ceiling). Below it the gate extends (`tail_low_evidence`), up to the W
  ceiling, where it judges what there is and records `tail_low_evidence`. The N2
  startup warning (tail observations wholly post-hide) is gone with the `hq x W` span.
- **Early-commit floor (M1).** `newest observation - _post_hide_start`: both on the
  gateway clock; before, the gateway `scraped_ms` minus the controller-clock
  confirmation (off by the node skew).
- **Immediate rollback (L1).** On every new snapshot, one value over the post-hide span
  `[S, E]` (the tail p95s, max of per-pod and pooled), once it holds
  `min_commit_samples` requests - the direct path's rule. The round-4 non-overlap
  filter is gone.
- **Guard (L4).** `loops/safescale_task.py` is in the guarded decision layer.
  `planning/safescale.py` keeps an `EvidenceGrid` object and asks it for doc
  boundaries only (`evidence_start`, `next_after` for "the next snapshot", `floor` of
  the Redis-path window base), never for the period as a span of evidence.
- **Tests (M3).** `period_sim` completes requests in whole numbers (Poisson, seeded by
  pod and time), can shift every gateway stamp (`gateway_offset_ms`) and can run the
  direct evidence path (the real `DirectEvidenceCollector` on sim pods, early commit).
  New scenarios: `hold` (a TPOT threshold the probe breaks; d stays HIGH; O1 off, so
  only the rollback-retry floor holds the retry) and `direct` (gateway clock 4 s
  ahead). New per-run ground-truth test: each rule's delay from its cause is within
  `[span - 1 s, span + P + tick + 1 s]` (saturation 10 s, stale 20 s, retry floor 20 s,
  early commit 20 s + the 4 s offset). `controller/tests/test_safescale_tail_20261026.py`:
  the tail request floor extends instead of failing; the immediate rollback judges the
  post-hide span of an overlapping, pre-hide-starting window once 20 requests are in it.
  Mutations run on a throwaway copy: Persistence made count-based, the retry floor
  reverted, the early-commit floor on the controller clock, the immediate-rollback floor
  removed, the tail floor removed - each fails at least one test.

  Delays (s) seen (round 5):

  | rule (cause) | 2 s | 5 s | 10 s |
  |---|---|---|---|
  | saturation rescue (onset) | 15 | 15 | 15 |
  | first stale snapshot (stall start) | 22 | 25 | 30 |
  | retry floor: first hold other than `no_new_evidence` (rollback tick) | 19 | 23 | 23 |
  | direct early commit (hide; gateway 4 s ahead) | 26.75, 27.75 | 26.75, 26.75 | 26.75, 26.75 |

**Still period-dependent (each at most one period):** the tail and the early-commit
floor start at the first doc boundary after the hide (0..P after it); the Redis-path
window base is `floor(confirmation)` on the doc grid, and its extension / wait steps are
"the next doc boundary" (wait for a new snapshot); data age (a window is read one
period plus the offset after its end).
