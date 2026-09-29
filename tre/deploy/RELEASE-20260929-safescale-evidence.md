# Release 2026-09-29: SafeScale commit evidence (controller notes)

Branch `feat/safescale-evidence-20260929` (on top of `fix/continuable-contract-20260929`
-> `feat/floor-probe-window-20260929` -> `feat/drain-policy-v1-semantics-20260929`).
Controller only; the SM, the gateway plugin and the UI are unchanged (they parse the new
registry section and ignore it). Nothing here has been applied. Image: to be built
(`<YYYYMMDD>-<sha>`).

## What changes

| # | Before | Now |
|---|---|---|
| 1 | an observation per 2 s tick (one 10 s snapshot counted ~5x); every observation could roll back at once, pre-hide windows included | one observation per snapshot (`window_end_ms`); the immediate rollback judges a snapshot only when `window_start_ms >= S` (its whole window follows the hide); donor health / preemption / abort still every tick |
| 2 | commit latency = hq tail of the 30 s snapshots (~56 % pre-hide at W = 20 s) | evidence = docs stamped `[S, E]` (`S` = first gateway boundary after the hide, `E` = newest snapshot; read without histogram lookback, so a pod's delta starts at its first doc >= `S`), remaining pods only (probe pods + pods asleep excluded); `n_judged` (TTFT count of the pods that have a p95 under `TRE_MIN_LATENCY_SAMPLES`) `< min_commit_samples` -> deadline one period past the newest evidence, up to `W_max`; still short at `W_max`: idle (n = 0, nothing in flight) -> commit, traffic -> latency skipped (Z / KV judged) |
| 3 | `W_max = 2 x gateway.route_timeout_s` (300 s) | `W_max = safescale.window_ceiling_s` (60 s), extensions included; W counts from the boundary before the confirmed hide (= the probe start unless the hide was confirmed late) |
| 4 | thresholds = env 500 / 75 ms | registry `safescale.slo_mode`: `labels` (default; TPOT 75, TTFT = max(floor, k(c + bL)), L = mean prompt length of the judged window) or `fixed` (`models[].slo`); env = optional override |
| 5 | - | first evidence doc of every remaining pod stamped in `[S, N + 20 s]` (one missed gateway tick tolerated), else rollback `evidence_clock_skew` / `first_doc_outside` (ERROR log `safescale_evidence_clock_skew`); no gateway-stamp anchor (newest-doc read failed, Redis error, legacy v1 metric keys) -> rollback `evidence_clock_skew` / `anchor_unverified` at the next tick; no evidence by `W_max` (E <= S, or no doc of any remaining pod in `[S, E]`) -> rollback `evidence_empty` (ERROR log) - idle commits need docs too; clock offsets -> ERROR `safescale_clock_skew_alert` |
| 6 | - | audit fields (below), events `safescale_evidence:` / `safescale_rollback_reason:`, summary script |

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
```

Apply with `deploy/scripts/merge_live_registry.py` (the release adds the section, the
live tunables are kept) -> `kubectl replace` the ConfigMap -> restart the controller.
Not needed for the defaults: a controller of this release reading a registry without
the section uses the same values. Invalid values or unknown keys in the section refuse
the start of every component of this release that loads the registry (controller, SM,
UI) - validate the merged file first (`merge_live_registry.py` does).
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
`S`: must be 0). `tail_pre_hide_fraction_mean` / `_max` keep describing the Z / KV
tail. New decision reasons: `evidence_extended` (probing), `hide_unconfirmed`,
`evidence_empty`, `evidence_unavailable`, `evidence_clock_skew` (rollbacks).

Run summary: `python3 -m scripts.analysis.safescale_summary <run_dir>/safescale.json`
(resolved probes only: rollback rate, rollback-reason distribution - structured code,
else the terminal reason cut at ':' - formal-gate failures, latency-gate outcomes,
extensions, max evidence pre-hide share). The controller GCs resolved probe
records after one hour, so for longer runs the controller log is the complete record.

## Expected behaviour change

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
- probe records written by this release carry extra keys (`hide_anchor`,
  `start_wall_ms`, `extensions`, audit fields); an older controller restoring an
  unresolved probe ignores them (and judges it the old way). Records written by an
  older controller restore here without a hide anchor: such a probe extends to `W_max`
  and rolls back `hide_unconfirmed` (fail-closed, one-time at the upgrade).
