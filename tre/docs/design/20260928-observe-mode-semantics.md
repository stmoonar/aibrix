# Observe mode semantics (2026-09-28)

User decision (2026-09-28): **observe = compute and record only, no scaling side
effects.** APA-arm experiments are driven by AIBrix's own autoscaler calling the
service-manager (SM) HTTP API, not by the TRE controller, so the SM HTTP write
API stays available in observe.

User decision (2026-09-28, second): the controller mode and the SM actuation are
**two independent switches**. Controller observe = the TRE controller computes
and records only. SM actuation = whether the SM supervisor may perform the
self-heal actions that restore declared state (B7 recreate, drift fleet repair,
resume repair, reap rejected, unrequested startup admission). Both experiment
arms run the SM actuation `active`, so the self-heal is the same for TRE and
APA.

## Experiment matrix

| Phase | `tre:v2:controller:mode` | `tre:v2:sm:actuation` | Other |
|---|---|---|---|
| TRE arm | `active` | `active` | no APA CRs |
| APA arm | `observe` | `active` | AIBrix APA PodAutoscaler CRs applied |
| Calibration | `observe` | `observe` | `calibration_campaign.py` refuses otherwise |
| Maintenance (staggered bring-up, between runs, after a failure) | `observe` | `observe` | `staggered_model_fleet.py` refuses otherwise |

Every deployment step sets both keys explicitly (a missing key is observe for
its reader, but that is a fail-closed fallback, not a configuration):

* `deploy/scripts/set_run_mode.sh <controller> <sm>` (one `MSET`, read back;
  `-` leaves a key unchanged; `status` prints both and warns on missing keys);
* `deploy/scripts/deploy_models.sh --run-mode CONTROLLER:SM` (plain apply: sets
  it after the apply; `--staggered --execute`: sets `observe:observe` before the
  bring-up and `--run-mode` after it);
* `deploy/scripts/toggle_tre_apa.sh tre|apa` (TRE: `active:active` once TRE is
  the only decision source; APA: `observe:active` first, before TRE scaling is
  switched off; `--keep-run-mode` leaves both alone);
* `deploy/scripts/campaign_queue.py` (per arm as in the table; `observe:observe`
  between runs; both values are read back from Redis and recorded in
  `freeze.json` (`controller_mode`, `sm_actuation`) and `command.json` /
  `run.json` (`run_mode`, `run_mode_expected`));
* the console (below).

## Keys

| Key | Values | Written by | Read by |
|---|---|---|---|
| `tre:v2:controller:mode` | `active` / `observe` | console `POST /api/ops/controller/mode` (this key only) or `POST /api/ops/run-mode`, `set_run_mode.sh`, `toggle_tre_apa.sh`, `campaign_queue.py` | controller (`ObserveModeGate`) |
| `tre:v2:sm:actuation` | `active` / `observe`, **independent** of the controller mode | console `POST /api/ops/sm/actuation` (this key only) or `POST /api/ops/run-mode`, same scripts (`tre_common.run_mode.write_run_mode`: either key or both in one MULTI) | SM supervisor + startup admission (`SmActuation`) |
| `tre:v2:sm:maintenance` | JSON `{operation_id, kind, owner, since_ms}` | SM fleet repair (for its whole run) | SM (`ClusterSafetyGate`), console (`/v2/supervisor`) |
| `tre:v2:sm:actuation:suppressed` | LIST of JSON `{ts_ms, action, detail}` (newest first, capped 200) | SM (actions not taken in observe) | operators / console |

## Fail-closed resolution

* **Controller**: the mode key is read with a 1 s cache; a fresh (uncached) read
  is made right before every capacity-changing SM call. Key absent or not a
  mode → `observe`. Redis error → last successfully read mode; never read →
  `observe`. (Before 2026-09-28 an absent key and a Redis error meant `active`.)
* **SM**: `tre:v2:sm:actuation` only. Absent or not a mode → `observe` (it is
  **no longer derived** from `tre:v2:controller:mode`). Redis error → last
  known; never known → `observe`. `/v2/supervisor` → `actuation.source` is
  `sm`, `default` (absent), `last_known` or `fail_closed`.

## Console

* `GET /api/ops/run-mode` (same payload from `GET /api/ops/controller/mode` and
  `GET /api/ops/sm/actuation`): `{"controller", "mode" (= controller),
  "sm_actuation", "raw": {"controller", "sm_actuation"} (None = absent),
  "warnings": [...]}`; a missing key gives e.g.
  `"tre:v2:sm:actuation missing -> SM treats as observe"`.
* `POST /api/ops/controller/mode {"mode"}` sets **only** the controller mode;
  `POST /api/ops/sm/actuation {"mode"}` sets **only** the SM actuation;
  `POST /api/ops/run-mode {"controller"?, "sm_actuation"?}` sets both (one
  MULTI) or either one. Answers carry the payload above plus `written`.
* The snapshot (`/api/snapshot`, SSE) carries `run_mode` (the same payload,
  sampled every 1 s); the top bar shows both switches (`(unset)` for a missing
  key) and an alert strip with the warnings. Ops & Control has one toggle per
  switch plus "set both" presets (TRE arm, APA arm, calibration/maintenance).

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

(`tre:v2:sm:actuation` = `observe`, whatever the controller mode.) Only logs
and records (`tre:v2:sm:actuation:suppressed`, JSON log
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
(`MaintenanceLockLost`, HTTP 409). A key left behind by a dead SM stays visible
(`/v2/supervisor` → `maintenance`) until the next repair takes it over; it
blocks nothing.

## Fresh deploy

With both keys absent the controller and the SM supervisor are in observe (and
the console warns). The supported bring-up
(`deploy/scripts/deploy_models.sh --staggered --execute --confirm-reset-fleet
[--run-mode CONTROLLER:SM]`) sets both keys to `observe` first and requires
both explicitly `observe` (`staggered_model_fleet.py` checks before every
binding: an SM self-heal would race the one-at-a-time bring-up). It starts one
binding at a time with every overlapping resident asleep (the not-owned
admission sleeps nothing), sleeps each new Pod itself, and wakes the baseline
through the SM HTTP API. It does not rely on B7 / fleet repair. What does not
happen by itself with the SM actuation in observe: recreating a deleted
Deployment, a drift repair, or admitting a restarted Pod next to an awake
resident — set the SM actuation to active (console) or act through the SM API.
After a deploy set both keys for the next phase (experiment matrix above); the
campaign runners set them per arm.
