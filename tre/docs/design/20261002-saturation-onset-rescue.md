# Onset saturation rescue (2026-10-02)

Branch `feat/saturation-onset-rescue-20261002` (on `fix/tick-safescale-none-20261002`
@ 61bd5019). Registry `scaling.saturation_*`. Code: `tre_controller/signals/saturation.py`
(rule and per-model state), `loops/tick.py` (`_apply_saturation_rescue`,
`_note_saturation_steps`), `planning/planner.py` (`saturation_need`). Offline check:
`controller/tools/saturation_onset_replay.py`.

## Problem

The TSS numerator is the token total of the requests that **completed** in the window
(the gateway only records `request_prompt_tokens` / `request_generation_tokens`
histograms at completion). At a load onset nothing has completed yet:

1. the numerator is 0, Z is undefined, and the model is classified HEALTHY (request-rate
   idle rule) and then dropped from the plan as incomplete - the planner does not see it;
2. once the first requests complete, the traffic onset is an O1 breakpoint, and the O1
   evidence gate holds the receiver for two complete grids (`no_complete_grid`,
   `evidence_grids`), about 20 s.

Recorded case (`A-smoke-tre-20261002-180116`, dsllama-8b, 12 rps from t = 0, one
replica): running 42 / 99 / 161 at 5 / 10 / 15 s, 55 waiting and KV cache 0.997 at 25 s
- the engine is full - while the first scale-up was planned at 45 s (tick 49 s).

## v1 and v2

| | numerator 0, queue > 0 | consequence |
|---|---|---|
| v1 | Z = 0 -> CRITICAL immediately | fast at onset, but a single long request that has not completed yet also reads CRITICAL |
| v2 (before this change) | Z undefined -> no receiver (idle rule) + O1 evidence gate | no false CRITICAL; a real flash crowd waits 40-50 s |
| v2 + this change | Z undefined -> no receiver **unless the engines are full** on 2 consecutive windows | flash crowd rescued after 2 windows; a lone long request stays healthy |

## Rule

Per model and per metrics window (10 s gateway grid; one count per distinct window end -
the rescue and fairness loops read every window more than once and never count it twice):

* **Eligible** only while the TSS cannot decide:
  * `numerator_zero`: the window's TSS numerator `Y_m` is 0 (tokens present, i.e. not a
    scrape gap), or
  * `o1_hold`: the receiver gate is not warm (`signal_warm` false: the O1 evidence gate,
    or the ADR-0013 onset guard when O1 is off / suspended).

  TSS signal source (`zm`) only. When the TSS is warm, the TSS / Z / C1 rules decide alone.
* **Engine full**, from each routable pod's **newest gateway instant sample** in the
  window (not the 30 s average; samples older than one grid before the window end are
  ignored; hidden probe pods and sleeping pods excluded):
  `sum(num_requests_waiting) > 0` **or** `mean(kv_cache_usage_perc) >= saturation_kv_threshold`.
* **Confirmed** on `saturation_consecutive_ticks` (2) consecutive eligible windows. The
  model is then a CRITICAL receiver with `saturation_rescue = True`; the warmup / dwell
  suppression of the planner does not apply to it (its own confirmation replaces them).
* **Step size - bounded doubling**: target = `min(max(n + 1, floor(factor * n)),
  max_awake_replicas)`, n = routable replicas (1 -> 2 -> 4). Not C1's
  `ceil(n * tau_crit / Z)`: a Z near 0 sends that to the cap. The O1 low-evidence step
  cap (`breakpoint_partial_max_step`) does not apply to this path (see below). A target
  an earlier rescue has not reached yet counts as covered (C1 rescue bookkeeping).
* **After each step** the count restarts; windows are counted again only once the
  routable count moved off its value at the decision, starting with the window after the
  one on which the change was first seen (that window's sample may predate the change).
  A step whose change never shows (SM refusal, observe mode) releases the wait after
  three grids (30 s).
* Not full (only running requests, KV below the threshold: a single long or stuck
  request) keeps the TSS verdict - no receiver.

Capacity order is the existing fast-loop order: sleeping-replica wakes, then free GPUs
(`critical_idle_capacity` / TP slots), then IDLE / HIGH donors released immediately
(`critical_donor_immediate`), then middle-zone SafeScale donors.

## Parameters (registry `scaling:`)

| key | default | meaning |
|---|---|---|
| `saturation_rescue` | `true` | on / off |
| `saturation_kv_threshold` | `0.9` | mean KV-cache fill that counts as full (0.01-1) |
| `saturation_consecutive_ticks` | `2` | consecutive eligible full windows (>= 1) |
| `saturation_max_step_factor` | `2` | target factor per step (>= 1; the target is at least n + 1) |

Validation follows the `breakpoint_*` keys (`parse_scaling_config`: integers via
`_scaling_count`, booleans strict, numbers finite and in range). Added to
`deploy/registry.yaml` and `deploy/overlays/tre-v2/params.yaml`. Restart-to-apply like
the rest of `scaling:`; images without this change ignore the keys (logged as unknown).

## Interaction with O1, C1, SafeScale and donors

* **O1**: unchanged. The saturation path is only eligible while O1 holds (or the
  numerator is 0); once O1 lets the window through, the warm TSS decides. Every rescue
  step changes the routable count, which is an O1 breakpoint: the next step can only come
  from the saturation path (engine still full on fresh windows) or, once the TSS is warm
  again, from C1.
* **C1**: a saturation step is tagged with the same `RescuePlan` (base = n, desired =
  target), so `rescue_bases` keeps C1 from re-asking for a target not yet reflected, and a
  later C1 decision builds on it.
* **`breakpoint_partial_max_step` (checked 2026-10-02)**: it is in effect. The registry
  value is parsed into `ScalingRegistryConfig.breakpoint_partial_max_step`, `registry.scaling()`
  is the `config` object in `loops/tick.py::_scaling_options`, and with
  `breakpoint_window: true` it reaches `PlanConfig.partial_window_max_step`;
  `signal_evidence_requests` is set in the tick context on every partial window and read
  by `critical_need`. The cap applies only when the post-breakpoint window holds fewer
  than `breakpoint_lowevidence_requests` (10) completed requests. In A-smoke-tre the
  first decision (window 45 s, n = 1, Z = 0.298, desired 3) was taken on a 20 s suffix
  at ~3000 decode tokens/s, i.e. far more than 10 completed requests, so the cap did not
  apply and 1 -> 3 (planned 2) is the intended C1 result. A tick-level test now covers
  the wiring end to end (`test_partial_max_step_is_wired_from_the_registry_through_the_tick`).
  The two mechanisms are disjoint: the cap bounds a TSS decision on a thin partial window;
  the saturation path is not a TSS decision, has its own bound (doubling) and re-confirms
  saturation on fresh windows after every step, so the cap is not applied to it.
* **SafeScale**: unchanged. A saturation receiver that has an active probe preempts it
  (`receiver_need_upscale`), exactly like a TSS CRITICAL receiver. A probe's hide is an O1
  breakpoint: during the following hold the saturation path watches the remaining pods;
  if they are full, that is a real overload and the rescue (and the probe rollback) is
  the intended reaction. The replay found no such case (below).
* **Donors**: unchanged - only IDLE / HIGH models are immediate donors; a model the
  saturation path declares CRITICAL is a receiver and therefore not a donor in that tick.

### No scale-down protection period

No fixed "scale up only" period (Knative-panic style) is added after a saturation step.
A hot switch takes 2-3 s, so a step that turns out too large is cheap to undo. The
existing state gates already prevent flapping: (1) every step changes the routable
count, an O1 breakpoint, and O1 keeps the model out of every scale-down - HIGH / IDLE
donor and middle-zone donor - until a whole window lies after it
(`donor_suppressed_breakpoint_window`), so no decision is taken on a window that still
describes the old replica count; (2) donors are only taken from IDLE / HIGH models, and a
model that was saturated a moment ago reads neither once its window is clean; (3)
SafeScale shrinks only after its own observation window without SLO violations. A timed
protection would only add controller state.

## Observability

* Event `saturation_rescue:<model>:n=<n>:target=<t>:waiting=<w>:kv=<kv>:reason=<numerator_zero|o1_hold>:ticks=<k>:planned_max=<p>`
  in the rescue tick's events (`ctrl_ticks`, decision snapshot), followed by the usual
  `rescue_target:` event; `saturation_pending:<model>:<k>/<K>:reason=...:waiting=...:kv=...`
  for a full window not yet confirmed.
* Decision snapshot `model_states.<model>`: `saturation_rescue`, `saturation_reason`,
  `saturation_ticks`, `saturation_waiting`, `saturation_kv` (present when the feature is on).
* Startup log `saturation_rescue_config`. The signal log shows the receiver as tier `crit`.
* `PodWindowMetrics.latest_waiting / latest_running / latest_gpu_cache / latest_instant_ms`
  (store): the newest instant sample per pod (not part of equality).

## Offline replay

`controller/tools/saturation_onset_replay.py` (read-only) runs the controller's
`SaturationTracker` and `saturation_target` over recorded runs: eligibility from the
recorded tick of each window, engine gauges from the 5 s sampler (`pod_gauges.jsonl`,
nearest sample to the window end), routable pods from `layout.jsonl`. Exact up to the
first simulated step; later steps are indicative (the recorded gauges belong to a
different replica count; a simulated step is assumed to land 3 s after the tick and
capacity to be found; after the TSS warms again the C1 hand-off is estimated from the
recorded Z rescaled to the simulated n). Times are seconds after load start; "tick" is
when the decision is taken (the wake lands ~2-3 s later).

| run | model / load | recorded first scale-up | new rule: first step | new rule: reaches 4 (indicative) | false trigger |
|---|---|---|---|---|---|
| A-smoke-tre-20261002-180116 | 8b, 12 rps from 0 | window 45, tick 49.4: 1 -> 3 (C1); 3 routable at 52 s | window 35, tick 39.4: 1 -> 2 (`o1_hold`, waiting 122, KV 0.99) | tick 69.4: 2 -> 4 (backlog of the first pod: waiting 96, mean KV 0.51) | none: 7b 0-70 s (4 rps) 3 eligible windows, 0 full (max KV 0.06); 7b whole run 15 eligible / 0 full; 14b 3 / 0; 8b after 80 s (SafeScale probe holds) 17 / 0 (max KV 0.80) |
| C-crit-20261002-175810 | 7b, 200 concurrent, 1500-word prompts | window 35, tick 38.8: +1 (donor); 2 at 63 s, 4 at 102 s | window 15, tick 18.8: 1 -> 2 (`o1_hold`, waiting 172) | tick 48.8 via C1 hand-off (Z ~ 0.017) | - (genuine overload) |
| E-14b-crit-20261001-143058 | 14b, 400 concurrent decode-heavy (pre-O1 build, onset guard) | window 66, tick 74.4: +1 (wake did not land; stayed at 1) | window 16, tick 24.4: 1 -> 2 (`numerator_zero`, waiting 104, KV 0.72) | tick 54.4: 2 -> 4 | prefill-heavy phase (242-402 s): TSS warm throughout, path not eligible |

Reading: the first rescue step comes 10 s (A), 20 s (C) and 50 s (E) earlier than the
recorded first scale-up. The low-load and long-request segments are eligible (numerator
zero / O1 hold) but never "full", so they never trigger.

## Risks

* **Backlog on the first replica**: vLLM's waiting queue is per pod; after 1 -> 2 the
  first pod still drains its backlog, so `sum(waiting) > 0` can confirm the next step
  although the new replica has room (A-smoke: 2 -> 4 decided at 69 s while the recorded
  run settled at 2-3 replicas and read HIGH at 95 s). Bounded by doubling and
  `max_awake_replicas`; the excess is given back by the normal scale-down paths once the
  window is clean. If this proves costly, a per-pod rule (a pod is full) is the natural
  refinement.
* **Gauge freshness**: the rule reads the gateway's 10 s instant docs; a stale doc is
  ignored (no sample = not full), so a scrape gap delays, never triggers.
* **SafeScale probe holds**: a probe hide puts the donor under the O1 gate; if its
  remaining pods are full the probe is rolled back (intended). Replay: no such case.
* **Signal-source ablations**: the path is TSS-only (`signal_source: zm`); alternative
  signal arms are unaffected.

## Tests

`controller/tests/test_saturation_rescue_20261002.py`: numerator 0 + waiting on 2
windows -> CRITICAL (1 window -> nothing, re-reads do not count); running-only / KV < 0.9
-> stays healthy; O1 hold + KV >= 0.9 -> trigger; warm TSS -> never; bounded doubling
1 -> 2, two fresh windows after the change, 2 -> 4, capped by `max_awake_replicas`; a
step that never lands; free capacity before donors, immediate HIGH donor without it; no
incomplete-drop; sample selection (hidden / stale pods); store latest-sample fields;
config parsing and validation; the `breakpoint_partial_max_step` wiring.
