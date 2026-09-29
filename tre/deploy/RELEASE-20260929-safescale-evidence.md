# Release 2026-09-29: SafeScale commit evidence (controller notes)

Branch `feat/safescale-evidence-20260929` (on top of `fix/continuable-contract-20260929`
-> `feat/floor-probe-window-20260929` -> `feat/drain-policy-v1-semantics-20260929`).
Controller only; the SM, the gateway plugin and the UI are unchanged (they parse the new
registry section and ignore it). Nothing here has been applied. Image: to be built
(`<YYYYMMDD>-<sha>`).

**Default evidence source: `direct` (plan B+D, see "Direct evidence" below).** The
controller scrapes the remaining pods' vLLM `/metrics` itself; rows 1, 2 and 5 of the
table (the gateway-doc evidence and its clock checks) describe the `redis` path, which
is now `safescale.evidence_source: redis` and the per-probe fallback of `direct`.

## What changes

| # | Before | Now |
|---|---|---|
| 1 | an observation per 2 s tick (one 10 s snapshot counted ~5x); every observation could roll back at once, pre-hide windows included | one observation per snapshot (`window_end_ms`); the immediate rollback judges a snapshot only when `window_start_ms >= S` (its whole window follows the hide); donor health / preemption / abort still every tick |
| 2 | commit latency = hq tail of the 30 s snapshots (~56 % pre-hide at W = 20 s) | evidence = docs stamped `[S, E]` (`S` = first gateway boundary after the hide, `E` = newest snapshot; read without histogram lookback, so a pod's delta starts at its first doc >= `S`), remaining pods only (probe pods + pods asleep excluded); `n_judged` (TTFT count of the pods that have a p95 under `TRE_MIN_LATENCY_SAMPLES`) `< min_commit_samples` -> deadline one period past the newest evidence, up to `W_max`; still short at `W_max`: idle (n = 0, nothing in flight) -> commit, traffic -> latency skipped (Z / KV judged) |
| 3 | `W_max = 2 x gateway.route_timeout_s` (300 s) | `W_max = safescale.window_ceiling_s` (60 s), extensions included; W counts from the boundary before the confirmed hide (= the probe start unless the hide was confirmed late) |
| 4 | thresholds = env 500 / 75 ms | registry `safescale.slo_mode`: `labels` (default; TPOT 75, TTFT = max(floor, k(c + bL)), L = mean prompt length of the judged window) or `fixed` (`models[].slo`); env = optional override |
| 5 | - | first evidence doc of every remaining pod stamped in `[S, N + 20 s]` (one missed gateway tick tolerated), else rollback `evidence_clock_skew` / `first_doc_outside` (ERROR log `safescale_evidence_clock_skew`); no gateway-stamp anchor (newest-doc read failed, Redis error, legacy v1 metric keys) -> rollback `evidence_clock_skew` / `anchor_unverified` at the next tick; no evidence by `W_max` (E <= S, or no doc of any remaining pod in `[S, E]`) -> rollback `evidence_empty` (ERROR log) - idle commits need docs too; clock offsets -> ERROR `safescale_clock_skew_alert` |
| 6 | - | audit fields (below), events `safescale_evidence:` / `safescale_rollback_reason:`, summary script |
| 7 | evidence = gateway docs on the 10 s grid: probe length 18-32 s by gateway phase (+10 s per extension), immediate rollback only 35-45 s after the hide | `safescale.evidence_source: direct` (default): baseline scrape of the remaining pods at the SM's hide confirmation, one poll per 2 s tick; immediate rollback `slo_violation_direct` as soon as `min_commit_samples` are judged; deadline = confirmation + W (controller clock), +2 s per extension, ceiling confirmation + 60 s; KV gate from the scrape; fallback to the `redis` path when every remaining pod fails |

## Direct evidence (default, 2026-09-29 B+D)

- **Baseline.** When the SM confirms the hide (ActionQueue `on_hide_done`), the
  controller scrapes `GET http://<pod IP>:<safescale.metrics_port>/metrics` of every
  remaining pod of the model (awake in the SM view, probe pods excluded; pod IPs from
  the SM `/v2/state` `fleet.observed`), concurrently, `scrape_timeout_s` (1 s) each, in
  a dedicated thread pool awaited with `asyncio.wait_for` (the event loop, the fast loop
  and the queue never wait on it). It keeps the cumulative TTFT and inter-token-latency
  histograms (bucket steps only) and the prompt-token sum / count per pod in the probe
  record (`direct_evidence`): a restarted controller continues from it. A baseline not
  taken at the confirmation (restart, or a probe restored from an older controller) is
  taken at the next tick - later is still post-hide.
- **Poll.** Every `evidence_poll_s` (2 s) the baseline pods are scraped again and
  differenced: per pod TTFT / TPOT p95 (`TRE_PERCENTILE_MODE`, per-pod minimum
  `TRE_MIN_LATENCY_SAMPLES`), model p95 = max over the pods, `n` = TTFT count,
  `n_judged` = TTFT count of the pods with a p95, `L` = prompt tokens / requests (labels
  thresholds). Requests completed before the baseline are never in it (requests already
  running on a remaining pod at the hide that finish later are, as on the Redis path).
- **Immediate rollback (D).** Any poll with `n_judged >= min_commit_samples` and a p95
  above the threshold rolls back at once (`rollback_reason.code = slo_violation_direct`,
  with the p95s, thresholds, `n`, window). At 7b load (4-10 req/s per pod, 3 remaining
  pods) that is the first or second poll: 2-4 s after the violation starts.
- **Deadline.** `hide confirmation + W` on the controller clock (not the 10 s grid), so a
  W = 20 s probe commits 20-22 s after the confirmation (the next 2 s tick). Short
  evidence extends the deadline by one poll period, up to confirmation + `W_max` (60 s).
  At the ceiling the rules are unchanged: idle -> commit, traffic -> latency skipped
  (Z / KV judged).
- **Gates.** Latency and KV-cache come from the scrape (KV = mean `kv_cache_usage_perc`
  of the remaining pods in the latest poll - the KV gate's existing aggregation, a mean
  over the remaining pods; the Redis tail value is recorded as `kv_redis_tail_max` and
  used only when no pod reports the gauge). Z is unchanged: the tail minimum of the Redis
  snapshots (as v1), `z_source = redis_snapshot_tail`.
- **Failures.** A pod whose scrape fails (timeout, HTTP error, no IP) is left out of that
  poll and recorded (`direct_excluded_pods`); a pod whose counters went backwards
  (restart) is dropped for good (`counter_reset`); a pod reporting its engine asleep is
  left out. A poll (or baseline) without any usable pod switches the probe to the Redis
  evidence path for good (`evidence_source_used = redis_fallback`,
  `direct_fallback.reason` `baseline_failed` / `all_pods_failed` / `no_baseline`): the
  687cbd9c logic below, clock assertions included. Redis evidence unreadable too ->
  rollback `evidence_unavailable` (fail-closed).
- **Network.** The controller must reach every model pod IP on `metrics_port` (8000, the
  serving port; the reissue sidecar forwards `GET /metrics` to vLLM). Pod-to-pod traffic
  is open in this cluster (no NetworkPolicy); a NetworkPolicy added later must allow
  controller -> model pods on that port. No new RBAC (IPs come from the SM).
- **Clocks.** Baseline, polls and deadline use only the controller clock: the direct path
  has no cross-node clock dependency, so the "gateway and controller on the same node"
  restriction is lifted - as long as a probe does not fall back to `redis` (the fallback
  keeps the gateway-stamp anchor and its clock checks below).
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
`latency_samples_judged`, `latency_gate` (`evaluated` / `skipped` + `latency_skip_reason`
`idle` / `insufficient_samples` / `p95_unavailable`), `extensions`, `clamped` (W or the
evidence cut by `W_max`; `window_clamped` = W only), `window_base_ms`,
`evidence_anchor` (`gateway_doc` / fallback), `gateway_offset_ms`,
`controller_offset_ms`, `clock_skew_alert`, `threshold_mode`, `ttft_threshold_ms`,
`tpot_threshold_ms`, `mean_prompt_tokens`, `rollback_reason` (`{"code": ...}`: 
`slo_violation`, `formal_commit_gate_failed` + `gates`, `donor_health`, `preempted`,
`hide_failed`, `hide_unconfirmed`, `evidence_empty`, `evidence_unavailable`,
`evidence_clock_skew` + `check`), `probe_wall_clock_ms`, `hide_ts_ms` /
`hide_anchor_source`, and `tail_pre_hide_fraction` (share of the latency evidence before
`S`: must be 0). Direct path (2026-09-29 B+D): `evidence_source_used` (`direct` /
`redis_fallback` / `redis`), `direct_fallback` (`reason`, `ts_ms`, `detail`),
`hide_confirm_ms`, `direct_baseline_ts_ms`, `direct_baseline_pods`, `direct_scrape_ts_ms`
(every poll), `direct_pods` (per pod `n`, `ttft_p95_ms`, `tpot_p95_ms`, `ts_ms`),
`direct_excluded_pods` (pod -> reason), `direct_window_start_ms` / `_end_ms`,
`kv_source` (`direct` / `redis_snapshot_tail`), `kv_direct`, `kv_ts_ms`,
`kv_redis_tail_max`, `z_source` (`redis_snapshot_tail`), `z_ts_ms`; rollback code
`slo_violation_direct`; `latency_source = direct`. The probe record carries
`direct_evidence` (baseline bucket steps, drops, poll timestamps, fallback). `tail_pre_hide_fraction_mean` / `_max` keep describing the Z / KV
tail. New decision reasons: `evidence_extended` (probing), `hide_unconfirmed`,
`evidence_empty`, `evidence_unavailable`, `evidence_clock_skew` (rollbacks).

Run summary: `python3 -m scripts.analysis.safescale_summary <run_dir>/safescale.json`
(resolved probes only: rollback rate, rollback-reason distribution - structured code,
else the terminal reason cut at ':' - formal-gate failures, latency-gate outcomes,
extensions, max evidence pre-hide share, and `by_evidence_source`: decided / commits /
rollbacks / rate / reasons per `evidence_source_used`). The controller GCs resolved probe
records after one hour, so for longer runs the controller log is the complete record.

## Expected behaviour change

Direct path (default): a W = 20 s probe decides 20-22 s after the hide confirmation
(+2 s per extension, at most 60 s); a violation with enough samples rolls back 2-4 s after
it starts. The bullets below describe the `redis` path (and a probe that fell back).

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
