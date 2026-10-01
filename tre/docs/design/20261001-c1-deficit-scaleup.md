# C1: deficit-sized rescue scale-up, no scale-up cooldown (2026-10-01)

User decision (2026-10-01): **scale up by the deficit in one step and drop the
scale-up cooldown; scale-down stays with SafeScale.** Branch
`feat/c1-deficit-scaleup-20261001`.

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
desired = min( max(n + 1, ceil(n * tau_crit / Z)),      # bring Z back to tau_crit
               max(n + 1, floor(ratio * n)),              # ratio = scaling.rescue_max_step_ratio (2.0)
               scaling cap (max_awake_replicas),
               capacity the plan finds )
```

`n` = routable replicas, `Z` = the classified (EMA) decision-window signal,
`tau_crit` = the model's band. `Z <= 0` / missing or `n = 0` -> `n + 1`. The
formula assumes Z proportional to n at fixed load (Z is a per-replica speed
over a per-replica queue). `planner.rescue_desired`.

The whole `desired - n` is planned in one tick through the existing paths, each
already able to take several replicas: sleeping-binding wakes (joint
assignment, one hinted `put_model_target`, which the SM wakes in parallel),
cold create into free GPUs, free TP slot pairs (now several per tick; a defrag
plan stays one migration), and the same-GPU donor -> receiver relay. An
immediate donor gives its surplus instead of one step: IDLE down to its floor,
HIGH down to `ceil(n_d * tau_high / Z_d)` (its projected Z stays >= tau_high),
never less than one step (`planner._donor_give`). Middle-zone donors stay
behind SafeScale probes with one step.

### Idempotence and in-flight protection (no cooldown needed)

Every rescue scale-up carries a `RescuePlan(target, desired, base, covered)`.
The ActionQueue keeps the last one per model (`rescue_targets()`): issued,
parts outstanding, replicas really added (`covered`; a partial hinted wake
counts what the SM picked), `done_ms`. The planner tick turns a record whose
effect the model's window does not reflect yet (still running, or
`window_start < done_ms`, the F4 rule) into a `RescueBasis(base, covered)`:

* the desired is computed from `base` (the replicas the window's Z describes),
  and only `desired - covered` is planned. An unrefreshed window yields the same
  desired, so repeated ticks add nothing (`rescue_target_hold:<m>:desired=..:covered=..`);
* a deeper CRITICAL in a newer window (load still rising) raises the target by
  the difference only;
* a failed or partial target counts only what was added, so the rest is
  re-planned on the next tick;
* while an action of the model is queued or running it is not planned again
  (`inflight_models`, unchanged); a raise would wait behind it on the model
  resource anyway, so this costs at most one tick;
* once the window starts after `done_ms` the record is ignored and the target
  is recomputed from the current routable count.

Decision log: `rescue_target:<m>:n=..:z=..:desired=..:covered=..:planned=..`.

### Cooldown

`scaling.scale_up_cooldown_enabled` (default false) re-enables the F4 hold of
a CRITICAL receiver's scale-up. Unchanged: the F4 hold of scale-downs (after a
scale-up or scale-down), the floor-violation hold, LOW receivers (the slow
loop) and the `TRE_ACTION_COOLDOWN` switch itself.

### Unchanged

* Slow loop (fairness, LOW receivers): one step per receiver per tick, the
  same donors and SafeScale piggyback.
* Every scale-down: idle proactive shrink, HIGH proactive SafeScale probe,
  probe / commit / rollback, middle-zone probes, same-slot TP shrink.

## Relation to the paper (tre_paper section 4)

* Fast-loop rescue: the paper does not bound the rescue step; C1 sizes it from
  the signal the paper already uses (Z against tau_crit) and keeps the scaling
  cap. `rescue_max_step_ratio` bounds a single decision against a Z far below
  tau_crit (noise, cold window); it is a configuration bound, not a paper
  quantity.
* Bounded pairwise transfer (slow-loop rebalancing, at most one transfer per
  tick, re-evaluated on the next tick's Z): untouched. The implementation
  applies "one step" per LOW receiver; with two LOW receivers in one tick each
  gets one pair (pre-existing behaviour, not changed by C1).
* The paper has no cooldown. C1's default (no scale-up cooldown) is closer to
  it; the window-reflection rule survives only as the rescue target
  bookkeeping, which never blocks a larger target.

## Configuration

Registry top-level `scaling:` (controller only, read at start,
restart-to-apply; images before 2026-10-01 ignore the section):

| Key | Default | Meaning |
|---|---|---|
| `rescue_max_step_ratio` | `2.0` | rescue target at most `max(n+1, floor(ratio*n))`; `0` = legacy one step per window |
| `scale_up_cooldown_enabled` | `false` | F4 hold of a CRITICAL receiver's scale-up |

Rollback without a new image: `rescue_max_step_ratio: 0` and
`scale_up_cooldown_enabled: true` give the previous behaviour.

## Counterfactual on recorded runs

The evidence holds the per-tick decision log, not the raw windows, so the
ticks were walked at decision level (`loops/replay.py` needs metrics
snapshots): legacy (+1, cooldown) vs C1 on a counterfactual replica count,
Z rescaled with Z ~ n, wake 2.5 s (sm_ops: reserve + wake_up + commit), capacity
assumed available.

* `C-crit-20261001-141922` dsqwen-7b (n=1, Z 0.012..0.05 << tau_crit 0.56;
  the first 30 s are warmup-suppressed): legacy 1->2->3->4 at +32/+68/+103 s;
  C1 1->2 at +32 s, 2->4 at +68 s (ratio 2 caps the first step at 2n = 2).
  With ratio 4 the first decision would go 1->4 at +32 s.
* `A-smoke-tre-20261001-135629`: every CRITICAL episode starts at n=1 and is
  resolved by its first +1, so C1 issues the same targets.

## Risks

* Overshoot: a cold or noisy window with Z far below tau_crit asks for up to
  `ratio * n` at once; the surplus is reclaimed only by the (slow, guarded)
  scale-down path. Bounded by `rescue_max_step_ratio` and the scaling cap.
* Donors give more per tick (IDLE to floor, HIGH to its tau_high level), so a
  donor that turns busy right after has fewer replicas to fall back on.
* The Z ~ n assumption is rough at small n (prefill-heavy bursts, KV limits).
* The bookkeeping lives in the controller process: a restart forgets it, the
  next tick then plans from the current routable count (at most one extra
  target while the window still predates the last wake).
