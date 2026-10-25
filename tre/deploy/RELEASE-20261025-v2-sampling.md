# Release 2026-10-25: v2 instant sampling, 2 s evidence period - DRAFT

Plan only: nothing here has been built or applied. Confirm with the owner before any
step that touches the cluster (a parallel session may be using it). Branch
`fix/v2-sampling-20261025`; design `tre/docs/design/20261025-instant-sampling.md`.

Images that change: gateway-plugins (TRE doc writer), controller. The SM image is
swapped in the same order as always (plugins -> SM -> controller) even when its code is
unchanged, so the three stay one release. Overlay change: `tre-v2-redis` memory request
256Mi / limit 2Gi (was 128Mi / 512Mi).

## Why the order matters (review round 3, H1)

Mixed versions are not safe while the controller acts:

* **New gateway, old controller:** the old controller averages the instant docs over the
  expected sample count of its 10 s grid (`sum / 3`). With 0.5 s instant docs that is 60
  samples divided by 3: Q is 20x too large, Z 20x too small, every model reads CRITICAL
  and the controller starts a scale-up storm.
* **New controller, old gateway:** safe. The old gateway writes no
  `tre:v2:gateway:cadence` hash, so every snapshot is stale
  (`gateway_cadence_missing`) and the decision loops do nothing
  (`controller/tests/test_gateway_cadence.py`, case `missing`).
* **Round 4: new controller, round-3 gateway:** also safe, also stale. The round-4
  controller needs `last_round_ms` in the cadence hash (the window-written rule); a
  gateway built before round 4 writes the hash without it, so every snapshot is stale
  (`gateway_cadence_no_round`, case `no_round`). The gateway image of this release
  must be running before the controller can act.

Rule: while the images are being swapped, in either direction, the run mode is
`observe observe`.

## Redis restart (the limit change restarts the pod)

`tre-v2-redis` has no volume: a pod restart wipes the SM leases and bindings, the
operation journal and every stream. Save an RDB first and load it back afterwards. The
procedure exists in `tre_v1/deploy`: `00_stop_v2.sh` (read-only RDB dump with
`redis-cli --rdb` through `kubectl exec`, checked for the `REDIS` magic) and
`rollback.sh` `restore_v2_redis` (a local `redis:7.2-alpine` loads the RDB on
127.0.0.1:16379 and `MIGRATE ... COPY REPLACE` copies every key into the service IP).
Take the dump only after the writers (controller, SM) are scaled to 0, so nothing is
written after it. Keys with a TTL can expire in between; that is expected.

## Steps

0. Backup to `<backups>/v2-sampling-20261025/`: the three Deployments, the
   `tre-v2-redis` Deployment, the registry ConfigMap, and an RDB dump. Write
   `rollback.sh` (step 6). Exclusive window marker on, parallel sessions told.
1. `tre/deploy/scripts/set_run_mode.sh observe observe`.
2. Redis: scale controller and SM to 0; RDB dump (as `00_stop_v2.sh`); apply the new
   `redis.yaml`; wait Ready; load the RDB (as `restore_v2_redis`); compare `DBSIZE` with
   the dump.
3. Swap images: gateway-plugins -> SM (Recreate) -> controller (tags in the overlays and
   in `deploy/tests/test_kustomize_overlays.py`). The overlays set
   `TRE_INSTANT_SAMPLE_INTERVAL_MS=500` and `TRE_EVIDENCE_PERIOD_MS=2000` on both the
   gateway and the controller; they must be equal.
4. Verify:
   - `HGETALL tre:v2:gateway:cadence`: `instant_ms 500`, `evidence_ms 2000`, `version` =
     the plugin image's sha, `written_ms` within a few seconds of now (gateway clock),
     `last_round_ms` present and moving every 0.5 s;
   - controller log: `gateway cadence ok`, no `GATEWAY CADENCE` error, snapshots
     published every 2 s (`metrics_sampler`), no `metrics_window_stale`;
   - `/v2/...` decision snapshot: `stale: false`, `sample_quality_hold` null on a quiet
     model.
5. Restore the run mode (`active active` for the TRE arm).
6. Rollback: the same procedure in reverse order - `observe observe`, controller -> SM
   -> plugins back to the previous images, verify, restore the run mode. The redis limit
   may stay (it does not depend on the images); reverting it repeats step 2.

## Behaviour changes to expect (round 4, "Evidence, not ticks")

Design `tre/docs/design/20261025-instant-sampling.md`, section "Evidence, not ticks
(round 4)".

- **Registry aliases.** The live registry still uses the deprecated count keys. They
  are now read as `(N - 1) × 10 s`: `breakpoint_hold_max_windows: 6` = 50 s (round 2
  read 60 s), `saturation_consecutive_ticks: 2` = 10 s (round 2 read 20 s), which is
  what they did on the 10 s grid. `min_evidence_grids: 2` stays 20 s. The controller
  logs a deprecation warning per key; replacing them with `breakpoint_hold_max_ms` /
  `saturation_confirm_ms` / `min_evidence_ms` is optional (a registry change, released
  by the usual registry rules).
- **SafeScale commit gate (also at 10 s).** Z and KV are one value over the probe's
  last `hq × W` (raw TSS on that span, time-weighted KV of the remaining pods), not the
  min / max over the last 2-3 overlapping snapshots. Expect fewer `z_below_tau_low` /
  `kv_cache` rollbacks caused by one dip; the audit field `tail_observation_count` is
  now always 1 and `tail_start_ms` / `tail_end_ms` / `tail_span_ms` show the span.
- **Rollback retry.** A probe rolled back for capacity is retried only after 20 s of
  new observations (`safescale_rollback_hold:<m>:no_new_evidence`), then only on a
  higher Z or another routable count (as before). Before, any newer window freed it -
  2 s after the rollback at the 2 s period.
- **Stale hold.** The 20 s hold now counts from the first read of the first missing
  window (about one period sooner than before).
- **O1 gateway-clock check.** The doc-stamp fallback is gone: with no `written_ms` to
  measure, the check gives no verdict instead of suspending O1 (leftover docs of a
  model that is down no longer suspend it).

## Not covered here

- theta, lambda, tau, w_p and c/b must be re-calibrated on the new estimator before a
  data-producing run (registry rule: atomic update with the controller).
- Memory: 20 pods x (600 instant docs x ~0.6 KB + 900 histogram docs x ~2.6 KB) is about
  53 MB of sampling docs; the 2Gi limit is for the capped streams (`tre:v2:bl:*`,
  `tre:v2:controller:signal_log`) and the operation journal.
