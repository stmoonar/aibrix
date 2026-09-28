# Observe mode semantics (2026-09-28)

User decision (2026-09-28): **observe = compute and record only, no scaling side
effects.** APA-arm experiments are driven by AIBrix's own autoscaler calling the
service-manager (SM) HTTP API, not by the TRE controller, so the SM HTTP write
API stays available in observe.

## Keys

| Key | Values | Written by | Read by |
|---|---|---|---|
| `tre:v2:controller:mode` | `active` / `observe` | console `POST /api/ops/controller/mode`, `campaign_queue.py` | controller (`ObserveModeGate`) |
| `tre:v2:sm:actuation` | `active` / `observe` | same writers, **same MULTI** as the controller mode (`tre_common.run_mode.write_run_mode`) | SM supervisor + startup admission (`SmActuation`) |
| `tre:v2:sm:maintenance` | JSON `{operation_id, kind, owner, since_ms}` | SM fleet repair (for its whole run) | SM (`ClusterSafetyGate`), console (`/v2/supervisor`) |
| `tre:v2:sm:actuation:suppressed` | LIST of JSON `{ts_ms, action, detail}` (newest first, capped 200) | SM (actions not taken in observe) | operators / console |

## Fail-closed resolution

* **Controller**: the mode key is read with a 1 s cache; a fresh (uncached) read
  is made right before every capacity-changing SM call. Key absent or not a
  mode → `observe`. Redis error → last successfully read mode; never read →
  `observe`. (Before 2026-09-28 an absent key and a Redis error meant `active`.)
* **SM**: `tre:v2:sm:actuation` if set; else derived from
  `tre:v2:controller:mode`; neither set → `observe`. Redis error → last known;
  never known → `observe`.

The console always writes both keys together, so they only differ when an
operator sets one by hand (e.g. `sm:actuation=active` with the controller in
observe for a supervised bring-up).

## Controller in observe

* Decisions are computed and published (decision snapshot, signal log) as in
  active mode.
* The ActionQueue drops every re-plannable action (scale, transfer, hide,
  defrag) — `observe_skipped`.
* No SafeScale probe is started or preempted (B8).
* Every open SafeScale probe is rolled back (level-triggered on each SafeScale
  tick, so also at a restart in observe): held one-shot actions of the probe are
  dropped and the **only** action taken is the unhide of its probe pods (the
  controller undoing its own hide). The probe resolves as a rollback
  `observe_entered`. A queued / recovered commit is never carried out: it runs
  as the unhide of its donor pods (never an unconfirmed pod, never one a fresh
  view shows asleep).
* A transfer or commit already running re-checks the mode (fresh read) before
  each capacity-changing step: donor sleep, receiver wake, each follow-up
  upscale. A donor that already slept stays asleep; its receiver is **not**
  woken. Recorded as a failed receiver result `observe_entered: …`, a JSON log
  event (`observe_entered_mid_transfer` / `observe_entered_mid_commit`) and the
  queue counters `observe_transfer_stopped_total` / `observe_commit_stopped_total`.
  A commit stopped after its donor slept resolves as a commit (the donor did
  sleep) with the reason suffix `observe_entered: follow-up upscales not issued`.
* A hide is re-checked right before its SM call (closes B8's ~1 s cache window
  between a probe start and the switch): not sent, counter
  `observe_hide_skipped_total`; the probe is then rolled back as above.

## SM supervisor in observe

Only logs and records (`tre:v2:sm:actuation:suppressed`, JSON log
`sm_supervisor_action_suppressed`, `/v2/supervisor` → `actuation.suppressed`)
what it would have done — the same action with the same detail at most once per
5 min:

| Supervisor action | observe |
|---|---|
| B7 recreate missing Deployments (`repair_missing_deployments`) | recorded only |
| drift → full fleet repair (`start_fleet_repair`) | recorded only |
| resume a stale fleet repair (`recover_stale_fleet_repairs`) | recorded only |
| reap Deployments of absent bindings (`reap_rejected_deployments`) | recorded only |
| sleep journal recovery (`recover_sleep_journal`) | runs |
| desired-state seeding (`ensure_desired_seeded`) | runs |
| startup convergence of admitted Pods (`converge_startups`) | runs |
| release orphan `starting` leases (`reap_orphan_starting_leases`) | runs |

Startup convergence runs because it only brings Pods that are not serving yet
to their recorded desired power (and, for an admission made before the switch,
wakes back the residents that admission suspended).

### Startup admission

A Pod's init gate asks `POST /v2/startup/admit`. Two cases:

* **Owned** — an SM operation creates the Pod itself (cold start of
  `PUT /v2/models/<m>/target`, defrag migration, fleet repair: operation phase
  `starting_binding`). Pre-authorized, unchanged in observe.
* **Not owned** — nobody asked for this Pod (k8s restarted it, a Deployment was
  applied / scaled by hand). In **observe** it must not sleep awake residents
  on its GPUs: the admission is refused with a retriable 409 (the gate polls
  again) and recorded as `startup_admission_sleep` with the residents it would
  have slept. Without an awake overlapping resident it sleeps nothing and is
  admitted. In active mode the admission sleeps the residents as before.

## SM HTTP API in observe

Not gated: `PUT /v2/models/<m>/target`, `PUT /v2/bindings/<id>/power`,
`PUT /v2/models/<m>/routable`, `POST /v2/defrag`, `POST /v2/fleet/repair`, …
work in both modes (APA arm, operators). Every operation records its caller as
`request.actor`: the `X-TRE-Actor` header (controller: `tre-controller`,
console: `tre-console`), else `ua=<User-Agent>;addr=<remote address>`.

## Fleet repair lock

The fleet repair no longer requires or writes the controller mode (the
supervisor used to force the controller into observe and never gave it back).
It holds `tre:v2:sm:maintenance` for its whole run (taken over from a dead SM),
together with the SM writer lock, so any other SM write gets a retriable 409
meanwhile. Deleting the key aborts the repair at its next safety check
(`MaintenanceLockLost`, HTTP 409). A key left behind by a dead SM expires
after its TTL (see "Maintenance lock mechanics" below).

## Fresh deploy

With both keys absent the controller and the SM supervisor are in observe. The
supported bring-up (`deploy/scripts/deploy_models.sh --staggered`) still
converges: it already requires the controller mode `observe` explicitly, starts
one binding at a time with every overlapping resident asleep (the not-owned
admission sleeps nothing), sleeps each new Pod itself, and wakes the baseline
through the SM HTTP API. It does not rely on B7 / fleet repair. What does not
happen by itself in observe: recreating a deleted Deployment, a drift repair,
or admitting a restarted Pod next to an awake resident — switch to active
(console) or act through the SM API. Set the mode explicitly after a deploy:
`active` for TRE-arm experiments, `observe` for APA-arm experiments and
calibration (the campaign runners set both keys per arm).

## Maintenance lock mechanics

`tre:v2:sm:maintenance` keeps its JSON payload `{operation_id, kind, owner,
since_ms}` (`since_ms` = acquire time; renewals do not change it). All three
operations are Lua scripts on the key (`tre_sm/state/safety.py`), ownership =
the stored `operation_id`:

- **Acquire**: `SET ... PX <ttl>` when the key is free. A held key is taken
  over only when its holder is one of the stale operations the new repair
  recovers (`recovered_from`, i.e. the repairs of a dead SM), when it has no
  TTL (left by a pre-TTL SM or set by hand) or is not a JSON object. Any other
  live holder refuses the acquire (`MaintenanceLockBusy`; the repair operation
  fails and nothing is touched).
- **Renew**: compare-and-`PEXPIRE`, by a background thread every
  `TRE_SM_MAINTENANCE_RENEW_S` (default 15 s) and at every safety check
  (`assert_maintenance_held`, each pressure-wait poll). TTL
  `TRE_SM_MAINTENANCE_TTL_S` (default 60 s; renew must be at most TTL/2). A
  renewal that finds the key gone or owned by another operation marks the lock
  lost; the repair aborts at its next safety check (`MaintenanceLockLost`). A
  transient Redis error is retried; the TTL bounds it.
- **Release**: compare-and-`DEL` (never deletes another operation's lock).

A dead SM's lock disappears by itself within the TTL; the controller sees the
key (presence, `since_ms`) only while a live repair renews it. Operators still
abort a repair with `DEL tre:v2:sm:maintenance`. Real-Redis tests:
`service-manager/tests/test_maintenance_lock.py` (`make check-redis`).
