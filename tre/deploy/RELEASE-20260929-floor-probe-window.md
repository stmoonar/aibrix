# Release 2026-09-29: SafeScale probe window + replica floor (controller notes)

Branch `feat/floor-probe-window-20260929`. Controller side only; the service-manager /
registry side of the replica floor (`service_manager.replica_floor`) ships in the same
release. Nothing here has been applied. Images: to be built (`<YYYYMMDD>-<sha>`).

## Controller env changes (`overlays/tre-v2/controller.yaml`)

| Env | Value | Read by |
|---|---|---|
| `SAFE_SCALE_WINDOW_FLOOR_MS` | `20000` (new) | images from 2026-09-29 |
| `SAFE_SCALE_E2E_MULTIPLIER` | `2` (new) | images from 2026-09-29 |
| `SAFE_SCALE_MIN_WINDOW_MS` | `60000` (kept, rollback compat) | older images only |
| `TRE_FLOOR_VIOLATION_COOLDOWN_TICKS` | `6` (new; `0` = off) | images from 2026-09-29 |
| `SAFE_SCALE_MAX_WINDOW_MS`, `SAFE_SCALE_CW2_FALLBACK_MS` | removed | older images fall back to their defaults (120000 / 60000) |

Registry: no new key. `gateway.route_timeout_s` (existing) now also caps the probe
window at 2x its value (restart-to-apply, like every registry read of the controller).
Superseded in the same release train by `RELEASE-20260929-safescale-evidence.md`:
`W_max` = registry `safescale.window_ceiling_s` (60 s), and the SafeScale threshold env
`SAFE_SCALE_TTFT_P95_SLO_MS` / `SAFE_SCALE_TPOT_P95_SLO_MS` left the overlay.

Behaviour: see `README.md` "SafeScale probe window (controller env)" and "Controller
action queue" (floor-violation hold). New decision events:
`safescale_tail_pre_hide:<model>:...`, `floor_violation_hold:<model>`; the
`safescale_probe_window:` event gains `:max=..:clamped=..`.

## Why the legacy env stays at 60000

Controller images before 2026-09-29 validate `SAFE_SCALE_MIN_WINDOW_MS` at startup
(`min*(1-hq) >= metrics window + refresh + read offset`, i.e. >= 56 s with today's
values) and raise `ValueError` otherwise. The 914dd51f overlay set it to 20000: rolling
back only the image would have crash-looped the controller. Checked: the config of
`tre-v2-controller:20260928-206a87f9` (source 206a87f9) refuses the 914dd51f env and
starts on this overlay's env (with its old 60-120 s band).

Trade-off: two env names for one knob until the older images are retired (then drop
`SAFE_SCALE_MIN_WINDOW_MS` from the overlay and the guard test). The new image ignores
the legacy value when the new name is set, so there is no ambiguity in what runs.

## Rollback (controller)

Restore image AND env together - the whole Deployment object from the backup:

```bash
B=/data/nfs_shared_data/xxy/backups/<pre-deploy backup>
# controller observe first (console or set_run_mode.sh observe <sm>), then:
kubectl -n tre-v2 get deploy tre-v2-controller -o yaml > /tmp/controller-before-rollback.yaml
python3 - "$B/tre-v2-ns.yaml" > /tmp/controller-rollback.yaml <<'PY'
import sys, yaml
items = yaml.safe_load(open(sys.argv[1]))["items"]
[dep] = [i for i in items if i["kind"] == "Deployment" and i["metadata"]["name"] == "tre-v2-controller"]
for key in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
    dep["metadata"].pop(key, None)
dep.pop("status", None)
yaml.safe_dump(dep, sys.stdout, sort_keys=False)
PY
kubectl replace -f /tmp/controller-rollback.yaml
kubectl -n tre-v2 rollout status deploy/tre-v2-controller --timeout=300s
```

Never `kubectl set image deploy/tre-v2-controller ...` alone. If an image-only rollback
already happened, the overlay's `SAFE_SCALE_MIN_WINDOW_MS=60000` keeps it starting; a
`CrashLoopBackOff` with `SAFE_SCALE_MIN_WINDOW_MS minus the commit-gate tail span must
be >=` in the log means the env is newer than the image. SM, controller and
gateway-plugins images are still switched together (plugins -> SM -> controller); roll
back in the reverse order.

## Service-manager notes (replica floor review fixes)

* New optional registry keys (structural `service_manager` section; merge with
  `deploy/scripts/merge_live_registry.py`, then restart the SM):
  `replica_floor.log_interval_s` (60: floor WARNINGs at most once per outcome / path /
  model per interval; counters always count) and `startup_admission.gate_seen_s` (30) /
  `startup_admission.drift_grace_s` (600). An older SM image ignores all three, so an
  SM rollback needs no registry change.
* Startup admission never waits on the floor: the SM wakes another replica first (own
  writer phase, within `max_awake_replicas`); otherwise the start is exempt and counted
  as `floor_exempt:startup:<model>`. After the start converges, the suspended resident
  stays asleep and the make-up replica keeps serving (a swap: same awake count).
* A Pod whose startup gate is polling is reported as `startup_admission_pending`
  (informational; `GET /v2/supervisor` lists it under `informational`) and does not
  trigger a fleet repair for up to `drift_grace_s`; longer waits are drift again.
* A defrag move (make-before-break) can exceed `max_awake_replicas` by one replica for
  the duration of the move. The startup make-up never exceeds it: at the cap there is
  no make-up and the start is exempt (`makeup_failed: max_awake_replicas`).
* Floor counters (`GET /v2/sleep` -> `floor.counts`) are read from the persistent sleep
  stats (Redis), so they match `stats` and survive an SM restart.
* HTTP callers may only name the sleep paths `scale_down`, `urgent`,
  `safescale_commit` and `apa`; `repair`, `startup`, `defrag` and `default` are 400.
