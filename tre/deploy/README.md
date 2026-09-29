# TRE deploy: registry and service-manager configuration

## The registry is the single source of configuration

`deploy/registry.yaml` declares the cluster (nodes, GPUs, `max_bound_per_gpu`),
the models, the gateway (`gateway.route_timeout_s`) and the service-manager
behaviour (`service_manager:`). Every key of the `service_manager:` section is
optional; the shipped file spells out the built-in defaults with comments.

### Changing the live registry

The live registry is the `tre-v2-registry` ConfigMap (mounted at `/etc/tre`
by the controller and the service-manager).

- Change it **only through the console**: `PUT /api/params`.
- **Never** `kubectl apply` `overlays/tre-v2/params.yaml`: it is a bootstrap
  copy for a fresh install and would overwrite the live values (for example
  calibrated `theta_m`).
- The service-manager reads the registry (`placement:`, `service_manager:`,
  `gateway:`, the models) once, at start. **A placement change (or any other
  registry change) needs a service-manager restart**, not only a controller
  restart. The console's restart-to-apply (`POST /api/ops/controller/restart`,
  button "重启 controller + SM") restarts both Deployments;
  `GET /api/ops/controller/rollout` reports both (`controller`,
  `service_manager`, combined `state`). By hand:
  `kubectl -n tre-v2 rollout restart deploy/tre-v2-controller deploy/tre-v2-service-manager`.
- The console's ServiceAccount (`tre-v2-ui`) may `get` / `patch` exactly the
  Deployments `tre-v2-controller` and `tre-v2-service-manager` (Role
  `tre-v2-ui-params` in `overlays/tre-v2/rbac.yaml`, namespace `tre-v2` only);
  apply `rbac.yaml` before a console that restarts the SM.

### Checked at start

- The service-manager refuses to start when `service_manager:` / `gateway:` is
  invalid. One check is the worst-case duration of one sleeping call, for any
  number of targets (they drain and commit in parallel):
  `writer_lock_wait_s + sleep.ack_timeout_s + sleep.hard_cap_s +
  commit-lock wait + 2 x sleep.sleep_call_timeout_s + 8 x sleep.probe_timeout_s +
  sleep.physical_confirm_timeout_s + sleep.poll_interval_s + sleep.io_margin_s`
  (330.5 s with the shipped values; it counts the rollback re-probe of a failed
  commit and the last confirmation round's overshoot) must be below
  `service_manager.api_call_timeout_s`. Another: `sleep.reservation_ttl_s` must
  exceed `commit-lock wait + sleep.poll_interval_s + sleep.probe_timeout_s +
  sleep.io_margin_s` (the longest gap between two reservation renewals).
- The controller uses `service_manager.api_call_timeout_s` as its timeout for
  slow service-manager calls, unless `TRE_SM_SLOW_TIMEOUT_SECONDS` overrides it.
  It refuses to start when that timeout does not exceed the same worst case.

### Route timeout

`gateway.route_timeout_s` is the request timeout of every model route:

- `make manifests` writes it into each model HTTPRoute;
- the service-manager drain hard cap (`service_manager.sleep.hard_cap_s`)
  defaults to it and may not exceed it;
- the ext_proc route timeout in `overlays/tre-v2/gateway-extproc.yaml` must
  carry the same value (a guard test in `deploy/tests` enforces it).

### Gateway plugin contract

The service-manager and the gateway plugin share Redis keys keyed by pod
**name** only (`tre:v2:gw:seen:<pod>`, `tre:v2:gw:inflight:<pod>`). TRE model
pods must therefore have unique names across namespaces. The generated
Deployments put model, node and GPUs in the name, which guarantees this.

### Service-manager rollout

The service-manager Deployment uses `strategy: Recreate`, so only one writer
runs at a time. `terminationGracePeriodSeconds` must exceed the SIGTERM wait
`ServiceManagerConfig.shutdown_timeout_s()` (commit-lock wait + a parallel
commit + one poll round, computed from the same values as the call timeout); a
guard test enforces it. On SIGTERM the service-manager:

1. stops accepting new sleeps;
2. rolls back every drain that has not reached `/sleep`;
3. lets a sleep that is already past `/sleep` finish.

On start (and on every supervisor pass) it resolves any sleep journal entries
that a dead instance left behind. A pod whose `/sleep` may still be running is
re-opened for routing only after it read awake twice, more than
`sleep.sleep_call_timeout_s` apart.

### Run mode: controller mode and SM actuation

`tre:v2:controller:mode` and `tre:v2:sm:actuation` (`active` | `observe`) are
**independent** switches. Controller observe = the TRE controller computes and
records only (open SafeScale probes are only unhidden). SM actuation observe =
the service-manager supervisor only records what it would have done
(`tre:v2:sm:actuation:suppressed`) instead of recreating / reaping Deployments,
starting or resuming a fleet repair or sleeping residents for an unrequested
pod. The SM HTTP write API works in both modes (APA arm, operators). A missing
key is observe for its reader (the SM no longer derives its switch from the
controller mode), so **every deploy sets both explicitly**:

| Phase | controller mode | SM actuation |
|---|---|---|
| TRE arm | `active` | `active` |
| APA arm (+ APA CRs) | `observe` | `active` |
| calibration / maintenance | `observe` | `observe` |

- Scripts: `deploy/scripts/set_run_mode.sh <controller> <sm>` (or `status`);
  `deploy_models.sh --run-mode CONTROLLER:SM` (the staggered bring-up sets
  `observe:observe` first); `toggle_tre_apa.sh tre|apa` sets the arm's values
  (`--keep-run-mode` to skip); `campaign_queue.py` sets them per arm and records
  the read-back values in each run's evidence.
- Console: `POST /api/ops/controller/mode {"mode"}` (controller only),
  `POST /api/ops/sm/actuation {"mode"}` (SM only),
  `POST /api/ops/run-mode {"controller", "sm_actuation"}` (both, one MULTI);
  `GET /api/ops/run-mode` returns both plus `warnings` for missing keys.
- Redis directly:
  `kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli MSET tre:v2:controller:mode observe tre:v2:sm:actuation active`.
- Before a controller / SM rollout nothing needs to change (both read the keys
  at every use); after a fresh install or a Redis reset, set both.

`calibration_campaign.py` and `staggered_model_fleet.py` refuse to run unless
both keys are explicitly `observe`. Details:
`tre/docs/design/20260928-observe-mode-semantics.md`.

With the SM actuation in observe, a pod nobody requested (k8s restarted it, a
Deployment applied or scaled by hand) is not admitted while an awake resident
shares its GPUs (409, the gate retries; recorded as `startup_admission_sleep`).

The console shows the planner events `capacity_blocked:<model>` and
`defrag_disabled:<model>` (a CRITICAL model that got no capacity / whose TP
defrag the placement policy disables) as an alert strip under the top bar
(snapshot field `planner_alerts`: kind, model, occurrences in the last 15 min,
last time, whether the latest decision still carries it).

### Startup admission

A model pod's init container (`tre-startup-gate`, generated by
`make manifests`) posts to `POST /v2/startup/admit` every 2 s with a 15 s HTTP
timeout until it gets a 200. An admission may first drain and sleep the awake
residents on the pod's GPUs, which can take minutes, so the service-manager
runs it as a background job per (pod, UID):

- the first call starts the job and answers 200 if it finishes within 5 s,
  otherwise 202 `{"status": "in_progress"}`; later calls answer 202 while it
  runs, then its result (200) or its error (409 / 503 / 400) once;
- nothing is slept for a pod that cannot start now: pressure, the desired
  lifecycle, sleep reservations and starting / waking GPU leases on its GPUs
  are checked before (and again under the writer lock);
- the admission commit queues for the writer lock (`writer_lock_wait_s`);
- if the admission still fails after it put residents to sleep, those that
  are still desired awake are woken again.

The gate script needs no change for this (it already retries anything that is
not a 200). During a service-manager shutdown the answer is a retriable 503.

Pods the service-manager creates itself - a defrag destination, a cold start
(`PUT /v2/models/<m>/target` growing past the existing bindings), a fleet
repair - start while that writer still holds the writer lock (until the pod is
ready). The writer first takes the binding's `starting` GPU lease and records
its operation phase `starting_binding` (binding id, then the pod UID); the
gate of exactly that pod is then admitted without the writer lock (pressure,
desired lifecycle, reservations / other leases and "overlapping residents
asleep" are still checked). Any other pod keeps waiting. If such a start
fails, its Deployment is deleted again (its desired record is rolled back to
`absent`); the supervisor reaps any model Deployment whose binding is desired
`absent` and that has no Running pod.

`PUT /v2/models/<m>/routable` refuses (409) to reopen routing on a pod with a
sleep journal entry (a sleep in progress, `sleep_unconfirmed`, a failed
rollback) or whose `/is_sleeping` is not a clear "awake".

### Controller action queue

- A SafeScale commit is one queued action: the hidden donor pods sleep first,
  then each receiver is brought up to an absolute, grow-only target
  (`PUT /v2/models/<m>/target` with `at_least: true`). The target is resolved
  once at the first dispatch from the service-manager's current awake count
  (+ the planned delta, capped at `max_awake_replicas`) and then frozen: a
  retry re-sends the same target, so an SM call that succeeded but timed out
  on the controller is never applied twice.
- Before every (re)try the commit is re-checked on the models' latest signal
  state: a donor that is CRITICAL / LOW again abandons the commit (its hidden
  pods are unhidden instead); a receiver that is HEALTHY / HIGH / IDLE loses
  its upscale. This re-check (the planner's whole-model state) is a different
  criterion from the SafeScale commit gate (the probe window's tail of the
  donor's serving pods); an abandon on the first dispatch is counted as
  `commit_abandoned_after_gate_total` and logged
  (`safescale_commit_abandoned_after_gate`).
- Donor pods the service-manager reports `unconfirmed` (sleep sent, never
  confirmed) are never unhidden by the controller; its crash recovery
  resolves them.
- A commit whose donor sleep fails for good unhides the donor's hidden pods
  that are still awake.
- A SafeScale probe is `committing` from the moment its commit / rollback is
  queued until the queue finished it; only then is it resolved. After a
  controller restart a `committing` probe is re-submitted (the commit's
  follow-up upscales are left to the planner).
- A rescue scale-up of a model whose commit is waiting out a retry backoff
  preempts that retry instead of waiting behind it; it shrinks by the donor
  pods a fresh cluster view shows awake and hidden.
- Consumers for which a stale cluster view matters (retry / commit
  revalidation, preemption, failed-commit unhides) only use it while it is
  younger than 2.5 refresh periods (`TRE_FAIRNESS_INTERVAL_SECONDS`).
- A defrag runs alone: it conflicts with every other queued action. A rescue
  scale-up planned meanwhile waits for it; the decision snapshot then carries
  the event `rescue_waits_for_defrag:<models>`.
- The service-manager, the controller and the gateway plugin of this change
  must be rolled out together.
- An SM refusal `409 floor_violation` (a hide / sleep that would take a model below
  its `min_replicas` routable replicas) is not retried; the refused model is held
  out of every scale-down plan for `TRE_FLOOR_VIOLATION_COOLDOWN_TICKS` fast-loop
  ticks (default 6 x `TRE_RESCUE_INTERVAL_SECONDS` = 30 s; `0` = off). Event
  `floor_violation_hold:<model>`, counter `floor_violation_total`.

### SafeScale probe window (controller env)

`W = min(max(SAFE_SCALE_E2E_MULTIPLIER x p95_e2e, SAFE_SCALE_WINDOW_FLOOR_MS), W_max)`,
`W_max` = registry `safescale.window_ceiling_s` (default 60 s; since 2026-09-29, it
replaced `2 x gateway.route_timeout_s`). `window_terms` of every probe record carry
`W`, `W1`, `W_floor`, `W_max`, `clamped`, `dominant` (`e2e` / `floor` / `ceiling`).

Commit evidence (2026-09-29, `planning/safescale_evidence.py`, release note
`RELEASE-20260929-safescale-evidence.md`):

- one observation per metrics snapshot (keyed by `window_end_ms`); the 2 s loop still
  runs the donor-health guard and preemption / abort on every tick;
- the immediate SLO rollback judges a snapshot only when its whole window follows the
  hide (`window_start_ms >= hide`);
- at the deadline the latency check reads the docs stamped `[S, E]`: `S` = newest
  gateway doc stamp of the model when the SM confirmed the hide + one period (gateway
  clock only; `ceil(Redis TIME)` without a doc), `E` = newest snapshot, remaining pods
  only (probe pods and pods the fleet state reports asleep excluded), no histogram
  lookback. Fewer than `safescale.min_commit_samples` requests on pods with a p95: the
  deadline moves one gateway period past the newest evidence, up to `W_max` (W counts
  from the confirmed hide); still short there: no traffic -> commit, traffic -> latency
  skipped (Z / KV judged);
- thresholds: registry `safescale.slo_mode` (`labels` = the calibration label rule,
  `fixed` = `models[].slo`); env `SAFE_SCALE_TTFT_P95_SLO_MS` / `SAFE_SCALE_TPOT_P95_SLO_MS`
  are optional overrides (unset);
- clock / continuity check (fail-closed, `evidence_clock_skew`): each remaining pod's
  first evidence doc within `[S, N + safescale.evidence_clock_tolerance_s]`, and the
  newest-doc read at the hide must have worked; offsets of Redis `TIME`, the gateway
  stamps and the controller clock are alerted (`safescale_clock_skew_alert`), not
  acted on.

Z and the KV-cache fill still come from the tail of the snapshots, which may partly
precede the hide: `tail_pre_hide_fraction_mean` / `_max` record how much. The
latency evidence's own pre-hide share is `tail_pre_hide_fraction` (0 by construction,
a regression assertion). Summary of a run: `python3 -m scripts.analysis.safescale_summary
<run_dir>/safescale.json` (rollback rate, rollback reasons, latency-gate outcomes).

Env names and rollback: controller images from 2026-09-29 read the floor from
`SAFE_SCALE_WINDOW_FLOOR_MS`; older images read `SAFE_SCALE_MIN_WINDOW_MS` and refuse
to start when it is below 60000 (their N2 startup guard). The overlay sets both (new
name 20000, legacy name 60000), so an image-only rollback still starts. Precedence in
the new image: the new name wins; only the legacy name set -> it is used, with a
warning; neither -> 20000. Rule for every controller rollback: **restore the
controller Deployment object of the backup (image AND env together), never
`kubectl set image` alone** - env written for a newer image can stop an older one.

### Tests

`make check` is hermetic: the Lua scripts (sleep reservations, the fair writer
lock) run against Python models of them. `make check-redis` runs the same tests
against a real throwaway Redis container (`REDIS_TEST_IMAGE`, default
`redis:7.2-alpine`) and removes it afterwards.
