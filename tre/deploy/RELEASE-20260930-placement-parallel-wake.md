# Release notes: placement + parallel wake (branch feat/placement-parallel-wake-20260930)

Notes only. Nothing is built or deployed from this branch yet; the release procedure
(backup, `rollback.sh`, user confirmation) follows the usual recipe
(`RELEASE-20260930-calib-capture.md`).

## Compatibility rules (must hold at every step)

1. **The service-manager, controller and UI images are swapped in the same release**
   (order as usual: gateway plugin -> SM (Recreate) -> controller -> UI).
   - A new SM answers a failed wake with 409 `error: wake_failed` (it was 400). An
     old controller treats any 409 as retriable and would re-send the wake. The
     new controller does not re-send it: it cools the GPU down (placement.wake_cooldown)
     and re-plans elsewhere.
   - A new controller sends hinted model targets (`hints`, `avoid_gpus`). An old SM
     ignores those fields and wakes by its own ranking, which is safe but loses the
     structured refusals.
2. **Write the new registry keys only after all three images run this release.**
   The components released before it (images `20260930-f8ccb0ca`) refuse unknown
   `placement:` keys at load: a pod restarted on an old image would crash on
   `placement.placement_penalty` / `placement.wake_cooldown`. `registry.yaml` therefore
   ships them commented out; `deploy/tests/test_registry_compat_20260930.py` guards
   that. The new `service_manager:` keys (`test_hooks`, `operations.max_records`,
   `wake.recovery_unknown_attempts`, `wake.transport_recheck_s`,
   `startup_admission.placeholder_max_s`) are ignored by older parsers but are
   commented out as well, to keep the defaults visible.
3. **Wakes in flight at the swap.** The SM Recreate stops the old SM mid-wake at worst.
   The old SM has no wake journal, so there is nothing to recover. The new SM
   rebuilds the leases at bootstrap (awake, starting, and journaled waking).
4. **gpu-truth.** A wake no longer waits for a fresh sample. The agent should serve
   refresh requests (`refresh_seq`); agents without them still work, but after a local
   power change their samples are trusted only once published again.

## New observable surface

- `GET /v2/wake`: wake counters and journal.
- `/v2/state` gains `gpus[]` and `nodes{}`.
- Structured 409 bodies.
- JSON log events: `wake_start`, `wake_done`, `wake_failed`, `gpu_truth_fallback`,
  `startup_placeholder`, `startup_placeholder_released`, `container_restart_placeholder`,
  `wake_recovery_gave_up`, `wake_handed_to_recovery`.
- Controller decision events: `gpu_cooldown`, `wake_refused`, `placement_retry`.

## Test hooks (acceptance only)

Set `service_manager.test_hooks: true` (SM restart), then set
`tre:v2:sm:fault:refuse_wake:<node>/<gpu>` (409 `gpu_busy`) or
`tre:v2:sm:fault:fail_wake:<node>/<gpu>` (the wake fails after /wake_up, which
exercises the compensating sleep) with a TTL. Turn the hooks off again afterwards.
