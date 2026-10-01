# C1: deficit-sized rescue scale-up, no scale-up cooldown (2026-10-01)

User decision (2026-10-01): **scale up by the deficit in one step and drop the
scale-up cooldown; scale-down stays with SafeScale** (scale-up aggressive,
scale-down cautious). Branch `feat/c1-deficit-scaleup-20261001`; review round
1 folded in (section "Review round").

## Before

* Fast loop (rescue, every `TRE_RESCUE_INTERVAL_SECONDS`, 5 s): a CRITICAL
  receiver got `ceil(scale_step_ratio * n)` replicas per decision, i.e. **+1 for
  n <= 10** (`planner._scale_step`, `scale_step_ratio = 0.1`). Immediate donors
  (IDLE / HIGH) gave one step each.
* Review F4 cooldown (`loops/tick._action_cooldowns`, `TRE_ACTION_COOLDOWN`,
  default on): after a successful scale-up of a model, its next action waits
  until a metrics window *starts* after that scale-up completed, about
  W (30 s) + one window step + one tick, ~45-60 s in the recorded runs.
* So 1 -> 4 replicas took three decide-wake-settle rounds (verify run
  `C-crit-20261001-141922`: dsqwen-7b 1 -> 3 in ~60 s, 4 not reached in the
  episode).

## C1

### Rescue target (fast loop, CRITICAL receiver)

```
desired = min( max(n + 1, ceil(n * tau_crit / Z)),                  # bring Z back to tau_crit
               max(n + 1, floor(ratio * n), n + step_pods),         # ratio = scaling.rescue_max_step_ratio (2.0)
                                                                    # step_pods = scaling.rescue_max_step_pods (0)
               scaling cap (max_awake_replicas),
               capacity the plan finds )
```

`n` = routable replicas, `Z` = the classified (EMA) decision-window signal,
`tau_crit` = the model's band. `Z <= 0` / missing or `n = 0` -> `n + 1`. The
formula assumes Z proportional to n at fixed load (Z is a per-replica speed
over a per-replica queue). `planner.rescue_desired`. `step_pods` gives the
HPA default scale-up policy shape `max(100 %, +4 pods)` when set to 4; at 0 the
cap is the ratio alone.

The whole `desired - n` is planned in one tick through the existing receiver
paths, each able to take several replicas: sleeping-binding wakes (joint
assignment, one hinted `put_model_target`, which the SM wakes in parallel),
cold create into free GPUs, free TP slot pairs (several per tick; a defrag plan
stays one migration), and the same-GPU donor -> receiver relay.

### Donors stay cautious

The donor side is unchanged: an immediate IDLE / HIGH donor gives **one step
per tick** (`ceil(0.1 * n_d)`), a relay never more than the receiver still
needs; middle-zone donors only behind SafeScale probes. A rescue whose deficit
exceeds what the donors give this tick takes the rest from the next ticks (the
target bookkeeping below plans only the remainder). The opt-in
`scaling.donor_surplus_release` (default false) lets an immediate donor give its
surplus in one tick: IDLE down to its floor, HIGH down to
`ceil(n_d * tau_high / Z_d)` (its projected Z exactly tau_high, never less than
one step). With it on, a HIGH donor can land on the HIGH/HEALTHY edge and turn
LOW on a small rise (ping-pong); with it off that cannot happen faster than one
step per tick, as before C1.

### Idempotence and in-flight protection (no cooldown needed)

Every rescue scale-up carries a `RescuePlan(target, desired, base, covered)`.
The ActionQueue keeps the last one per model (`rescue_targets()`): issued,
parts outstanding (each part tied to its own record, so a part of a superseded
target never completes its successor), replicas really added (`covered`; a
partial hinted wake counts what the SM picked; pods a SafeScale preemption gives
back count without a dispatch), `done_ms`. The planner tick turns a record whose
effect the decision signal does not reflect yet into a `RescueBasis(base,
covered)`. "Reflected" = the model's window starts at or after
`done_ms + k * trs.ema_tau_ms` (`scaling.rescue_settle_ema_k`, default 2: the
EMA'd Z lags the raw window by about its time constant; 0 = the F4 rule; a
model on the legacy fixed-alpha EMA has no extension).

* Until then the desired is computed from `base` (the replicas the signal
  describes) and only `desired - covered` is planned. An unrefreshed window
  yields the same desired, so repeated ticks add nothing
  (`rescue_target_hold:<m>:desired=..:covered=..`).
* A deeper CRITICAL in a newer window (load still rising) raises the target by
  the difference only. During that period the target stays bounded by the cap
  **of the base**, `max(base + 1, floor(ratio * base), base + step_pods)`: with
  n = 1 and ratio 2 the first round reaches 2 and the next one starts only once
  the signal reflects it.
* A failed or partial target counts only what was added, so the rest is
  re-planned on the next tick.
* While an action of the model is queued or running it is not planned again
  (`inflight_models`, unchanged); a raise would wait behind it on the model
  resource anyway, so this costs at most one tick.
* Once reflected, the record is ignored and the target is recomputed from the
  current routable count.
* SafeScale preemption: one preemption per receiver per tick (tick and queue);
  the pods it gives back are deducted across all of the receiver's scale-up
  parts, and a target covered entirely by them is still recorded (done now).
* Restart: the queue persists `_last_done` and the rescue records per model
  (Redis hash `tre:v2:controller:scale_memory`, controller state store) and
  reloads them at start; a target still running at the restart is taken as done
  at load time. A restored target issued more than
  `TRE_SCALE_MEMORY_MAX_AGE_SECONDS` (50 s, ~ W + settle) before the start is
  dropped, and so is one the first tick finds contradicted by the fleet (fewer
  routable replicas than it had counted before its scale-up), so a restart
  never holds a model on stale memory. Best effort: a Redis error only loses
  the memory (the previous behaviour) and is logged; the controller's state
  Redis client has a socket / connect timeout (`TRE_REDIS_SOCKET_TIMEOUT_SECONDS`,
  2 s), so a stalled Redis cannot block dispatch. The metrics-read client has its
  own (`TRE_REDIS_METRICS_SOCKET_TIMEOUT_SECONDS`, 10 s: measured on the live
  tre-v2 Redis, read-only, per call p99 0.35 ms / max 73 ms, a whole tick's reads
  p99 162 ms / max 241 ms; default max(10 s, 5 x p99)). The SM fleet view carries no
  wake timestamps, so it could not serve as the source.

Decision log: `rescue_target:<m>:n=..:z=..:desired=..:covered=..:planned=..`
and the `rescue` object on each scale-up action of the decision snapshot.

### Cooldown

`scaling.scale_up_cooldown_enabled` (default false) re-enables the F4 hold of
a CRITICAL receiver's scale-up. Unchanged: the F4 hold of scale-downs (after a
scale-up or scale-down), the floor-violation hold, LOW receivers (the slow
loop) and the `TRE_ACTION_COOLDOWN` switch itself.

### Unchanged

* Slow loop (fairness, LOW receivers): one step per receiver per tick, the
  same donors and SafeScale piggyback.
* Every scale-down: idle proactive shrink, HIGH proactive SafeScale probe,
  probe / commit / rollback, middle-zone probes, same-slot TP shrink, and (by
  default) the immediate donors' one step per tick.

## Relation to the paper (tre_paper section 4)

* Fast-loop rescue: the paper does not bound the receiver's step; C1 sizes it
  from the signal the paper already uses (Z against tau_crit) and keeps the
  scaling cap. `rescue_max_step_ratio` / `rescue_max_step_pods` bound a single
  decision against a Z far below tau_crit (noise, cold window); they are
  configuration bounds, not paper quantities.
* Bounded pairwise transfer: at most one transfer per pair per tick,
  re-evaluated on the next tick's Z. This bounds the **donor side** too - a
  donor gives one step per tick - and C1 keeps it by default
  (`donor_surplus_release` off). The slow loop (rebalancing) is untouched; the
  implementation applies its one step per LOW receiver, so two LOW receivers in
  one tick each get one pair (pre-existing behaviour, not changed by C1).
* The paper has no cooldown. C1's default (no scale-up cooldown) is closer to
  it; the window-reflection rule survives only as the rescue target
  bookkeeping, which never blocks a larger target.

## Configuration

Registry top-level `scaling:` (controller only, read at start,
restart-to-apply; images before 2026-10-01 ignore the section):

| Key | Default | Meaning |
|---|---|---|
| `rescue_max_step_ratio` | `2.0` | rescue target at most `max(n+1, floor(ratio*n), n+step_pods)`; `0` = legacy one step per window |
| `rescue_max_step_pods` | `0` (shipped registry: `4`, user decision 2026-10-01) | `step_pods` above (4 = HPA's `max(100%, +4 pods)`) |
| `scale_up_cooldown_enabled` | `false` | F4 hold of a CRITICAL receiver's scale-up |
| `donor_surplus_release` | `false` | immediate donors give their surplus instead of one step |
| `rescue_settle_ema_k` | `2.0` | a target counts as reflected `k * trs.ema_tau_ms` after the window start passes it |

The scaling cap of both experiment arms is `models[].max_awake_replicas` (4 for
every model; TRE planner, SM and the APA `maxReplicas`). `models[].max_replicas`
(8 for 7b / 8b, 4 for 14b) is the GPU layout - how many bindings `make manifests`
renders - not a replica ceiling; it is left unchanged.

Rollback without a new image: `rescue_max_step_ratio: 0` and
`scale_up_cooldown_enabled: true` give the previous behaviour.

## Counterfactual on recorded runs

The evidence holds the per-tick decision log, not the raw windows, so the
ticks were walked at decision level (`loops/replay.py` needs metrics
snapshots): legacy (+1, F4 cooldown) vs C1 on a counterfactual replica count,
Z rescaled with Z ~ n, wake 2.5 s (sm_ops: reserve + wake_up + commit), receiver
capacity assumed available, settle extension 20 s (k = 2, tau 10 s).

* `C-crit-20261001-141922` dsqwen-7b (n=1, Z 0.012..0.05 << tau_crit 0.56;
  the first 30 s are warmup-suppressed): legacy 1->2->3->4 at +32/+68/+103 s;
  C1 (ratio 2) 1->2 at +32 s, 2->4 at +88 s; ratio 3: 1->3 at +32 s, 4 at
  +88 s; ratio 4 or `rescue_max_step_pods: 4`: 1->4 at +32 s.
* `A-smoke-tre-20261001-135629`: every CRITICAL episode starts at n=1 and is
  resolved by its first +1, so C1 issues the same targets.

## Risks

* Overshoot: a cold or noisy window with Z far below tau_crit asks for up to
  the cap at once; the surplus is reclaimed only by the (slow, guarded)
  scale-down path. Bounded by the ratio / pods caps and the scaling cap.
* With one-step donors a large deficit is filled from donors over several
  ticks (free and sleeping capacity are taken at once).
* The Z ~ n assumption is rough at small n (prefill-heavy bursts, KV limits).
* `rescue_settle_ema_k` lengthens the hold before the next round (20 s at the
  default); it never blocks a larger target computed from the base.
