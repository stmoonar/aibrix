# Release 2026-09-29: SafeScale commit evidence (controller notes)

Branch `feat/safescale-evidence-20260929` (on top of `fix/continuable-contract-20260929`
-> `feat/floor-probe-window-20260929` -> `feat/drain-policy-v1-semantics-20260929`).
Controller only; the SM, the gateway plugin and the UI are unchanged (the SM and UI
images of this release parse the new registry section with the shared parser and do not
use it; the deployed images do not parse it at all). Nothing here has been applied. Image: to be built
(`<YYYYMMDD>-<sha>`).

**Default evidence source: `direct` (plan B+D, see "Direct evidence" below).** The
controller scrapes the remaining pods' vLLM `/metrics` itself; rows 1, 2 and 5 of the
table (the gateway-doc evidence and its clock checks) describe the `redis` path, which
is now only `safescale.evidence_source: redis` (the rollback switch). The `direct` path
never falls back to it: the Redis evidence never decides a commit in `direct` mode
(2026-09-30, see "Commit exits").

## What changes

| # | Before | Now |
|---|---|---|
| 1 | an observation per 2 s tick (one 10 s snapshot counted ~5x); every observation could roll back at once, pre-hide windows included | one observation per snapshot (`window_end_ms`); the immediate rollback judges a snapshot only when `window_start_ms >= S` (its whole window follows the hide); donor health / preemption / abort still every tick |
| 2 | commit latency = hq tail of the 30 s snapshots (~56 % pre-hide at W = 20 s) | evidence = docs stamped `[S, E]` (`S` = first gateway boundary after the hide, `E` = newest snapshot; read without histogram lookback, so a pod's delta starts at its first doc >= `S`), remaining pods only (probe pods + pods asleep excluded); `n_judged` (TTFT count of the pods that have a p95 under `TRE_MIN_LATENCY_SAMPLES`) `< min_commit_samples` -> deadline one period past the newest evidence, up to `W_max`; still short at `W_max`: idle (n = 0, nothing in flight) -> commit; traffic but n = 0 -> rollback `insufficient_evidence:stalled`; otherwise the few samples are judged (`latency_gate = evaluated_low_samples`, max of per-pod and pooled p95 without a minimum; above the threshold -> rollback) - the latency gate is never skipped |
| 3 | `W_max = 2 x gateway.route_timeout_s` (300 s) | `W_max = safescale.window_ceiling_s` (60 s), extensions included; W counts from the boundary before the confirmed hide (= the probe start unless the hide was confirmed late) |
| 4 | thresholds = env 500 / 75 ms | registry `safescale.slo_mode`: `labels` (default; TPOT 75, TTFT = max(floor, k(c + bL)), L = mean prompt length of the judged window) or `fixed` (`models[].slo`); env = optional override |
| 5 | - | first evidence doc of every remaining pod stamped in `[S, N + 20 s]` (one missed gateway tick tolerated), else rollback `evidence_clock_skew` / `first_doc_outside` (ERROR log `safescale_evidence_clock_skew`); no gateway-stamp anchor (newest-doc read failed, Redis error, legacy v1 metric keys) -> rollback `evidence_clock_skew` / `anchor_unverified` at the next tick; no evidence by `W_max` (E <= S, or no doc of any remaining pod in `[S, E]`) -> rollback `evidence_empty` (ERROR log) - idle commits need docs too; clock offsets -> ERROR `safescale_clock_skew_alert` |
| 6 | - | audit fields (below), events `safescale_evidence:` / `safescale_rollback_reason:`, summary script |
| 7 | evidence = gateway docs on the 10 s grid: probe length 18-32 s by gateway phase (+10 s per extension), immediate rollback only 35-45 s after the hide | `safescale.evidence_source: direct` (default): baseline scrape of the remaining pods `baseline_delay_ms` (1 s) after the SM's hide confirmation, one poll per 2 s tick; immediate rollback `slo_violation_direct` as soon as `min_commit_samples` are judged; deadline = confirmation + W (controller clock), +2 s per extension, ceiling confirmation + 60 s; KV gate from the scrape; a direct commit only on evidence of every remaining pod covering [its planned baseline, the deadline], every one of them read in the deciding poll, checked against a fresh cluster view; any gap extends by one poll and rolls back (`evidence_incomplete:<gap>`) when it cannot heal (late baseline, restart, joined pod: at the deadline) or at the ceiling; the Redis evidence is never read for a commit |

## Direct evidence (default, 2026-09-29 B+D)

- **Baseline.** `safescale.baseline_delay_ms` (1000) after the SM confirms the hide
  (ActionQueue `on_hide_done`; the gateway applies the hide through its pod watch
  meanwhile - a baseline taken while requests still reach the hidden pods under-counts
  the remaining pods' load), the controller scrapes
  `GET http://<pod IP>:<safescale.metrics_port>/metrics` of every remaining pod of the
  model (awake and not hidden in a FRESH SM view, probe pods excluded; pod IPs from the
  SM `/v2/state` `fleet.observed`; only samples labelled with the probe's `model_name`,
  so a pod IP reused by another model's pod reads as `model_mismatch`), concurrently,
  in a dedicated thread pool (the event loop, the fast loop and the queue never wait on
  it). `scrape_timeout_s` (1 s) counts from the moment a pool thread starts the read, not
  from the submission: a read queued behind others is not a pod timeout; the queue wait
  is bounded by one more `scrape_timeout_s` (`pool_saturated`, a failure like a timeout),
  and a pool more than 75 % busy (queued + running reads, stuck ones included) logs a
  WARNING `safescale_scrape_pool_busy` (at most once a minute). The baseline keeps the
  cumulative TTFT and inter-token-latency histograms (bucket steps only) and the
  prompt-token sum / count per pod in the probe record (`direct_evidence`): a restarted
  controller continues from it. The deadline still counts from the confirmation.
  Awake but hidden pods (not the probe's) are not remaining pods.
- **Late baselines (evidence gaps).** Each pod's `baseline lag` = its baseline scrape -
  the hide confirmation is recorded (`direct_baseline_lag_ms`). A pod is `late` for the
  rest of the probe (`direct_late_pods`, pod -> `cause`, `lag_ms`, `ts_ms`) when its
  requests of some part of [confirmation + delay, now] are not in its delta: `pending_baseline` (it
  did not answer the baseline scrape; its baseline is its first later successful
  scrape), `probe_baseline` (the baseline itself was scraped later than
  confirmation + `baseline_delay_ms` + `scrape_timeout_s` = 2 s by default: a retried
  baseline after every pod failed, a controller restarted between the hide and the
  baseline, no fresh cluster view / pod IPs at the planned time), `counter_reset` /
  `family_vanished` / `process_restarted` (the pod restarted after its baseline - a
  counter went backwards, a family is gone, or `process_start_time_seconds` changed,
  which also catches a restart whose counters already passed the baseline's: its
  baseline is reset to zero - the restart follows the hide, so everything it counted
  since is post-hide - and it stays polled and judged; the requests before the restart
  are lost), `joined_after_baseline` (a pod the fresh cluster view lists as a remaining
  pod but the evidence does not know - a replica added, a sleeper that woke up: it
  served outside the evidence; it is polled from the next tick on), `asleep_before_baseline`
  (pending, then asleep). A late pod's data still drives the immediate rollback; the
  probe can no longer commit: at the deadline it rolls back
  (`rollback_reason.code = evidence_incomplete:late_baseline`, `detail` = the late pods).
  Waiting cannot fill such a hole, so the deadline is not extended for it.
- **Poll.** Every `evidence_poll_s` (2 s) the baseline pods are scraped again and
  differenced: per pod TTFT / TPOT p95 (`TRE_PERCENTILE_MODE`, per-pod minimum
  `TRE_MIN_LATENCY_SAMPLES`), model p95 = max over the pods AND the pooled p95 (the
  pods' delta histograms merged first, the minimum applied to the merged count: an
  overloaded pod that completes fewer requests than the per-pod minimum still weighs in;
  `evidence_pooled_ttft_p95_ms` / `_tpot_`), `n` = TTFT count, `n_judged` = TTFT count of
  the pods with a p95 (every request once the pooled p95 exists), `L` = prompt tokens /
  requests (labels thresholds). The Redis evidence reader applies the same pooled rule. Requests completed before the baseline are never in it (requests already
  running on a remaining pod at the hide that finish later are, as on the Redis path).
- **Immediate rollback (D).** Any poll with `n_judged >= min_commit_samples` and a p95
  above the threshold rolls back at once (`rollback_reason.code = slo_violation_direct`,
  with the p95s, thresholds, `n`, window), whatever pods are missing. The window is
  cumulative since the baseline: a violation present from the hide is caught by the
  first or second poll at 7b load (4-10 req/s per pod, 3 remaining pods) - 2-4 s after
  the baseline (3-5 s after the confirmation with the 1 s `baseline_delay_ms`); one that starts later is diluted by the earlier requests and is caught
  once the cumulative p95 crosses the threshold, or by the gate at the deadline.
- **Deadline.** `hide confirmation + W` on the controller clock (not the 10 s grid), so a
  W = 20 s probe decides at the first tick at or after confirmation + 20 s: about 20-24 s
  (the loop period is 2 s plus the tick's own work, the scrape waiting up to
  queue wait + `scrape_timeout_s`; a stale metrics snapshot pauses the evaluation). Short evidence
  extends the deadline by one poll period, up to confirmation + `W_max` (60 s). At the
  ceiling with fewer than `min_commit_samples` judged requests: idle (n = 0, no traffic
  in the snapshot tail and no request running / waiting on the pods) -> commit; traffic
  but n = 0 -> rollback `insufficient_evidence:stalled`; otherwise every sample there is
  is judged - max(per-pod p95, pooled p95) with no minimum-samples rule - and a p95 above
  the threshold fails the latency gate (rollback), else it passes
  (`latency_gate = evaluated_low_samples`, `low_sample_ttft_p95_ms` / `_tpot_`,
  `latency_samples`; a commit is marked `low_sample_commit` and counted apart in the
  summary as `low_sample_commits`). The deciding poll's reads must have been SENT at
  or after the deadline (`evidence_coverage_end_ms` >= deadline): a poll that straddles
  the deadline waits for the next one (`evidence_pending`, deadline unchanged).
- **Gates.** Latency and KV-cache come from the scrape. KV keeps the gate's existing
  aggregation - mean over the remaining pods, max over the hq tail - applied to the polls:
  per poll the mean `kv_cache_usage_perc` of the pods that answered, max over the last
  `hq` share of the polls (`kv_direct_tail_max`; the latest poll's mean is `kv_direct`);
  the Redis tail value is recorded as `kv_redis_tail_max` and used only when no pod
  reports the gauge. Z is unchanged: the tail minimum of the Redis snapshots (as v1),
  `z_source = redis_snapshot_tail`; with the earlier deadline that tail is mostly
  snapshots overlapping the hide (recorded in `tail_pre_hide_fraction_mean` / `_max`).
- **Failures.** A pod whose scrape fails (timeout, HTTP error, no IP, `pool_saturated`,
  `model_mismatch`) is left out of that poll and recorded (`direct_excluded_pods`,
  `direct_unanswered_pods`); at the intermediate ticks its latest delta (cumulative)
  still counts for two poll periods plus one scrape timeout. Beyond that - or while its
  baseline is pending - it is missing (`direct_missing_pods`). The deadline commit is
  stricter: every live remaining pod must have been read successfully in the deciding
  poll itself (a timeout is often the overload itself, and the seconds after its last
  successful read are not covered), that poll must be this tick's, and it must carry a
  fresh cluster view. Otherwise the deadline is extended by one poll (`extend_reason`
  `pods_missing` / `pods_unanswered` / `no_poll_this_tick` / `no_fresh_view` /
  `no_direct_window` / `direct_window_stale`), and at the ceiling the probe rolls back
  (`evidence_incomplete:<the same reason>`). Every remaining pod failing, even for many
  polls in a row, is the same: extend, then roll back - there is no switch to another
  evidence source. A pod still waiting for its baseline at the deadline cannot heal (it
  is late once it answers): rollback `evidence_incomplete:pending_baseline` at once. No
  baseline by the deadline -> rollback `evidence_incomplete:no_baseline`
  (a later baseline would be late anyway); no remaining pod left in the evidence ->
  `evidence_incomplete:no_live_pods`. A pod reporting its engine asleep at the baseline
  is left out (while the view agrees). A p95 in the `+Inf` bucket is reported as the
  largest finite bucket bound (records stay strict JSON).
- **Why never the Redis evidence here.** The gateway writes a doc for every ready pod
  every period from its cached scrape, with a fresh stamp even when the scrape behind
  it is stale (`pkg/cache/cache_tre_redis.go`), so a doc's stamp does not prove its
  counters are fresh; and the old fallback judged the Redis window as soon as it
  switched, not at the deadline (its end could precede the deadline). A probe record
  of the previous build that had fallen back (`direct_evidence.fallback`) is rolled back
  at the next tick (`evidence_incomplete:legacy_redis_fallback`).
- **Network.** The controller must reach every model pod IP on `metrics_port` (8000, the
  serving port; the reissue sidecar forwards `GET /metrics` to vLLM), without a proxy
  (`HTTP(S)_PROXY` is ignored for these reads; bodies above 8 MB are refused). Pod-to-pod traffic
  is open in this cluster (no NetworkPolicy); a NetworkPolicy added later must allow
  controller -> model pods on that port. No new RBAC (IPs come from the SM).
- **Clocks.** Baseline, polls and deadline use only the controller clock: the direct path
  has no cross-node clock dependency, so the "gateway and controller on the same node"
  restriction is lifted (the `redis` mode keeps the gateway-stamp anchor and its clock
  checks below).
- **Load.** One `/metrics` read per remaining pod per 2 s while a probe runs (~60 KB,
  ~20 ms measured); nothing when no probe is open. A stale metrics snapshot pauses the
  probe's evaluation (as before); the baseline is still taken.

Evidence anchor. `S = N + 10 s`, `N` = the newest gateway histogram doc stamp of the
model when the SM confirmed the hide (ActionQueue `on_hide_done`): a doc stamped `S`
is written after the confirmation, so the evidence is post-hide whatever the node clocks
say (they differ by up to 160 s here: node9; the gateway stamps docs with its own clock,
the snapshots read the same stamp grid). Residual: a doc whose scrape started just
before the confirmation and was written just after it (sub-second). Without `N` (no
doc of the model yet) `S = ceil(hide_ts / 10 s)`.

`hide_ts` = Redis `TIME` of the metrics Redis at the confirmation (the reference the
SM's `service_manager.clock_skew` check uses), else the controller clock
(`hide_anchor_source`). It anchors nothing in the evidence; the offsets
`gateway_offset_ms` (hide_ts - N, expected 0..10 s) and `controller_offset_ms`
(controller clock - hide_ts) are recorded and, beyond `evidence_clock_tolerance_s`,
logged as ERROR `safescale_clock_skew_alert` (alert only: a skewed Redis node must not
disable SafeScale when the evidence is sound). A gateway running ahead of the
controller makes `S` unreachable for the snapshots: the probe rolls back
`evidence_empty` at `W_max`, with an ERROR log.

The confirmation is the SM's (pod annotations written), not the gateway applying it;
the gateway picks the annotation up through its pod watch. Requests still routed to a
hidden pod meanwhile are outside the evidence anyway (probe pods are excluded).

## Commit exits (2026-09-30)

A probe commits only through the formal gate (`formal_commit_gate_passed`: latency ok,
Z >= `tau_low` or no Z without traffic, KV <= `kv_cache_max`; idle skips Z / KV as v1).
The latency part reaches "ok" only through these exits; everything else extends, waits
or rolls back.

Direct mode (`evidence_source: direct`), all exits require: hide confirmed; the deciding
poll is this tick's and read every live remaining pod; no late pod (late baseline,
restart, joined pod, asleep before its baseline); no missing pod; a fresh cluster view
whose remaining pods are all in the evidence; the evidence of every pod covers
[its baseline, the deadline] (`evidence_coverage_start_ms` <= planned baseline +
one scrape timeout, `evidence_coverage_end_ms` >= deadline); no judged violation.

| Exit | Extra precondition | `latency_gate` |
|---|---|---|
| D1 enough samples | `n_judged >= max(1, min_commit_samples)` with a p95, max(per pod, pooled) p95 <= thresholds (`min_commit_samples: 0` never makes the gate vacuous) | `evaluated` |
| D2 low samples at the ceiling | deadline = ceiling, n >= 1 but D1 not met (too few judged requests, or no p95 under the per-pod / pooled minimum), max(per pod, pooled) p95 without a minimum <= thresholds | `evaluated_low_samples` (+ `low_sample_commit`) |
| D3 idle at the ceiling | deadline = ceiling, n = 0, no traffic in the snapshot tail, no request running / waiting on any remaining pod | `skipped` / `idle` |

Redis mode (`evidence_source: redis`), all exits require: hide confirmed with a gateway
stamp anchor (no `newest_doc_error`), first docs within the clock tolerance, a fresh
cluster view with at least one remaining pod, each of them with docs up to the evidence
end `E`, `E` >= deadline, no judged violation. Exits R1 / R2 / R3 = D1 / D2 / D3 on the
gateway docs (R3: no request in flight in the snapshot tail; D2 / R2 also need at least
one doc of some remaining pod in [S, E]). When a doc of `E` lands late, or the
snapshot boundary has not reached the deadline, the probe waits (`evidence_pending`),
at most one gateway period past the ceiling, then rolls back `evidence_incomplete:<gap>`.

No evidence source wired at all (offline replays / direct constructions only; the
controller always wires one): the tail snapshots are judged as before 2026-09-29. In
direct mode a probe whose hide is never confirmed rolls back `hide_unconfirmed` at the
ceiling, with or without an evidence source.

Known limitation of the Redis mode: it cannot tell a doc the gateway re-wrote from its
cached (stale) scrape from a fresh one. Use it only as the rollback switch.

## Registry (structural section, restart-to-apply)

New top-level section `safescale:` (all keys optional; values = built-in defaults):

```yaml
safescale:
  slo_mode: labels            # labels | fixed
  window_ceiling_s: 60
  min_commit_samples: 20
  evidence_clock_tolerance_s: 20
  evidence_source: direct     # direct | redis
  evidence_poll_s: 2
  scrape_timeout_s: 1         # must be below evidence_poll_s
  metrics_port: 8000          # the model pods' serving port
  baseline_delay_ms: 1000     # baseline scrape after the hide confirmation (>= 0, < window_ceiling_s)
```

Apply with `deploy/scripts/merge_live_registry.py` (the release adds the section, the
live tunables are kept) -> `kubectl replace` the ConfigMap -> restart the controller.
Not needed for the defaults: a controller of this release reading a registry without
the section uses the same values. Invalid values in the section refuse the start of every
component of this release that loads the registry (controller, SM, UI) - validate the
merged file first (`merge_live_registry.py` does). Unknown keys in the section are ignored
with a WARNING (forward compatibility: a key added later must not stop an SM / UI / older
controller of this release). `safescale.evidence_source: redis` switches the whole
release back to the gateway-doc evidence (restart-to-apply, no image change).
`gateway.route_timeout_s` no longer affects the probe window.

## Controller env (`overlays/tre-v2/controller.yaml`)

| Env | Before | Now |
|---|---|---|
| `SAFE_SCALE_TTFT_P95_SLO_MS` | `500` | removed (optional override, unset) |
| `SAFE_SCALE_TPOT_P95_SLO_MS` | `75` | removed (optional override, unset) |

Everything else unchanged (`SAFE_SCALE_WINDOW_FLOOR_MS` 20000, `SAFE_SCALE_E2E_MULTIPLIER`
2, legacy `SAFE_SCALE_MIN_WINDOW_MS` 60000). Setting either removed env again pins that
threshold in both modes (`threshold_source: env_override` in the audit).

## Audit fields

In the decision details, the probe record (`terminal_details` and `window_terms`) and
the events: `evidence_start_ms`, `evidence_end_ms`, `latency_samples`,
`latency_samples_judged`, `latency_gate` (`evaluated` / `evaluated_low_samples` /
`skipped` + `latency_skip_reason` `idle` / `insufficient_evidence`), `extensions`, `clamped` (W or the
evidence cut by `W_max`; `window_clamped` = W only), `window_base_ms`,
`evidence_anchor` (`gateway_doc` / fallback), `gateway_offset_ms`,
`controller_offset_ms`, `clock_skew_alert`, `threshold_mode`, `ttft_threshold_ms`,
`tpot_threshold_ms`, `mean_prompt_tokens`, `rollback_reason` (`{"code": ...}`: 
`slo_violation`, `formal_commit_gate_failed` + `gates`, `donor_health`, `preempted`,
`hide_failed`, `hide_unconfirmed`, `evidence_empty`, `evidence_unavailable`,
`evidence_clock_skew` + `check`), `probe_wall_clock_ms`, `hide_ts_ms` /
`hide_anchor_source`, and `tail_pre_hide_fraction` (share of the latency evidence before
`S`: must be 0), `evidence_coverage_start_ms` / `evidence_coverage_end_ms` (the actual
coverage of every required pod: latest baseline / first doc, earliest poll send / last
doc), `required_pods` (Redis mode), `evidence_gap`, `wait_reason`, `low_sample_*`,
`low_sample_commit`, rollback codes `evidence_incomplete:<gap>`,
`insufficient_evidence:stalled` / `:p95_unavailable`. Direct path (2026-09-29 B+D): `evidence_source_used` (`direct` /
`redis`; `redis_fallback` only in records of the previous build), `direct_view_pods`,
`direct_in_flight`, `evidence_pooled_ttft_p95_ms` / `_tpot_`,
`hide_confirm_ms`, `baseline_delay_ms`, `direct_baseline_planned_ms` (confirmation +
delay), `direct_baseline_ts_ms` (the actual first baseline scrape), `direct_baseline_lag_ms`
(per pod), `direct_late_after_ms`, `direct_late_pods`, `direct_unanswered_pods`,
`direct_baseline_pods`, `direct_pending_pods`,
`direct_scrape_ts_ms` (every poll), `direct_pods` (per pod `n`, `ttft_p95_ms`,
`tpot_p95_ms`, `ts_ms`), `direct_excluded_pods` (pod -> reason), `direct_missing_pods`,
`direct_window_start_ms` / `_end_ms`, `kv_source` (`direct` / `redis_snapshot_tail`),
`kv_direct`, `kv_direct_tail_max`, `kv_ts_ms`,
`kv_redis_tail_max`, `z_source` (`redis_snapshot_tail`), `z_ts_ms`; rollback code
`slo_violation_direct`; `latency_source = direct`. The probe record carries
`direct_evidence` (baseline bucket steps and process start, drops, late pods, poll timestamps). `tail_pre_hide_fraction_mean` / `_max` keep describing the Z / KV
tail. New decision reasons: `evidence_extended`, `evidence_pending` (probing), `hide_unconfirmed`,
`evidence_empty`, `evidence_unavailable`, `evidence_clock_skew` (rollbacks).

Run summary: `python3 -m scripts.analysis.safescale_summary <run_dir>/safescale.json`
(resolved probes only: rollback rate, rollback-reason distribution - structured code,
else the terminal reason cut at ':' - formal-gate failures, latency-gate outcomes,
extensions, max evidence pre-hide share, and `by_evidence_source`: decided / commits /
rollbacks / rate / reasons per `evidence_source_used`). The controller GCs resolved probe
records after one hour, so for longer runs the controller log is the complete record.

## Expected behaviour change

Direct path (default): a W = 20 s probe decides about 20-24 s after the hide
confirmation (+2 s per extension, at most 60 s); a violation present from the hide rolls
back 3-5 s after the confirmation at 7b load (baseline 1 s after it). A probe whose
evidence has a gap (late baseline, restarted or joined pod, a pod unanswered at the
deadline) never commits: it rolls back at its deadline (late) or at the ceiling. The controller logs the effective settings
once at start (`safescale_config`): a misspelt key is only warned about. The bullets below describe the `redis` path (and a probe that fell back).

- Short probes (W = 20 s) now usually need extensions: the first evidence read covers
  only one gateway period (hide at B + 2..5 s -> S = B + 10 s, deadline B + 20 s). At
  light load a probe runs up to 60 s before it commits or rolls back.
- A gateway tick missed at `S` costs one period of evidence; two missed ticks roll the
  probe back (`first_doc_outside`).
- The evidence reads (one per new snapshot from the deadline on) and the hide anchor
  (Redis TIME + one ZREVRANGEBYSCORE per pod of the model) are synchronous Redis calls
  on the controller's event loop, like the probe-record writes.
- No immediate latency rollback within ~30 s of the hide (no snapshot is fully post-hide
  before that); the latency gate at the deadline takes over.
- Labels thresholds are prompt-length dependent: long-prompt models get a TTFT threshold
  above 500 ms (e.g. dsqwen-7b at L = 4000: 5 x (36.4 + 0.0527 x 4000) = 1236 ms).

## Rollback

Restore the controller Deployment object of the pre-deploy backup (image AND env), as in
`RELEASE-20260929-floor-probe-window.md` "Rollback (controller)". Compatibility:

- older controller images ignore the `safescale:` registry section; the ConfigMap can
  stay as is;
- with the env removed, older images (every image since be1e0076, incl. the deployed
  `tre-v2-controller:20260928-206a87f9`) fall back to their built-in 500 / 75 ms, i.e.
  the same thresholds the overlay used to pin, so an image-only rollback behaves as
  before; restoring the backup's Deployment brings the explicit env back anyway;
- `safescale.evidence_source: redis` + controller restart restores the gateway-doc
  evidence of 687cbd9c without an image change; the deployed images
  (`20260928-206a87f9`) ignore the whole `safescale:` section, the new keys included (they
  do not parse it). A 687cbd9c build would refuse the new keys (its parser rejected
  unknown keys); it was never built - do not build it;
- probe records written by this release carry extra keys (`hide_anchor`,
  `start_wall_ms`, `extensions`, audit fields); an older controller restoring an
  unresolved probe ignores them (and judges it the old way). Records written by an
  older controller restore here without a hide anchor: such a probe extends to `W_max`
  and rolls back `hide_unconfirmed` (fail-closed, one-time at the upgrade).
