# Onset saturation rescue (2026-10-02)

Branch `feat/saturation-onset-rescue-20261002` (on `fix/tick-safescale-none-20261002`
@ 61bd5019; review fixes in a follow-up commit). Registry `scaling.saturation_*`. Code:
`tre_controller/signals/saturation.py` (rule and per-model state), `loops/tick.py`
(`_apply_saturation_rescue`, `_note_saturation_steps`), `planning/planner.py`
(`saturation_need`, receiver order). Offline check: `controller/tools/saturation_onset_replay.py`.

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
    scrape gap);
  * `o1_hold`: the receiver gate is not warm (`signal_warm` false: the O1 evidence gate,
    or the ADR-0013 onset guard when O1 is off / suspended) **and** the breakpoint is the
    traffic onset or the routable change of the rescue's own last step. Once a window of
    the current traffic period was warm, an O1 hold caused by anything else - a C1
    scale-up, an immediate donor release, a SafeScale hide or unhide - is not eligible;
    the path re-opens only after an idle window (no token: the predicate that clears the
    O1 onset). An O1 resume after a suspension records an onset without an idle window
    and does not re-open it.

  TSS signal source (`zm`) only. A warm TSS decides alone (TSS / Z / C1 rules).
* **Engine full**, from each routable pod's **newest gateway instant sample** in the
  window (not the 30 s average; samples older than one grid before the window end are
  ignored; hidden probe pods and sleeping pods excluded):
  `sum(num_requests_waiting) > 0`, **or** `mean(kv_cache_usage_perc) >= saturation_kv_threshold`
  with at least 2 requests running (one very long prompt - a 32k-context prefill - can
  fill the KV cache alone).
* **Confirmed** on `saturation_consecutive_ticks` (2) consecutive eligible full windows.
  The model is then a CRITICAL receiver with `saturation_rescue = True`; the planner's
  warmup / dwell suppression does not apply to it (its own confirmation replaces them).
* **Step size - bounded doubling**: target = `min(max(n + 1, floor(factor * n)),
  max_awake_replicas)`, n = routable replicas (1 -> 2 -> 4). Not C1's
  `ceil(n * tau_crit / Z)`: a Z near 0 sends that to the cap. The O1 low-evidence step
  cap (`breakpoint_partial_max_step`) does not apply to this path (see below). A target an
  earlier rescue has not reached yet counts as covered (C1 rescue bookkeeping).
* **After each step** the count restarts. Windows count again only once the routable
  count rose above its value at the decision (`saturation_step_landed`), from the window
  after the one the change was seen on (its sample may predate it). **The pods the step
  added must be full themselves** (their own newest sample: waiting > 0, or KV above the
  threshold with >= 2 running): vLLM's waiting queue is per pod, and a backlog left on the
  old pod is no reason for the next step. *Every* added pod must be full - deliberately
  conservative: one added replica with room means capacity exists. The pods before the
  step are the model's awake, not hidden bindings of the fleet view at the decision (not
  only those with a fresh sample). A step may land in parts over several windows (target
  4: 2 -> 3, then 3 -> 4); every rise up to its target is the step's own
  (`saturation_step_landed` each time), only a fall or a rise above the target is external. A step whose change never shows (SM refusal,
  observe mode) releases the wait after three grids (`saturation_step_unconfirmed`) and the
  next step is a first step again. A routable change the rescue did not cause (donor,
  probe, C1) restarts the count (`saturation_reset_external`).
* Not full (only running requests, or one request filling the KV cache) keeps the TSS
  verdict - no receiver.
* **Several receivers in one tick**: TSS-confirmed CRITICAL receivers are planned first
  (their order unchanged); saturation receivers after them, the largest backlog per
  replica (`waiting / n`) first.

Capacity order is the existing fast-loop order: sleeping-replica wakes, then free GPUs
(`critical_idle_capacity` / TP slots), then IDLE / HIGH donors released immediately
(`critical_donor_immediate`), then middle-zone SafeScale donors.

## Parameters (registry `scaling:`)

| key | default | meaning |
|---|---|---|
| `saturation_rescue` | `true` | on / off |
| `saturation_kv_threshold` | `0.9` | mean KV-cache fill that counts as full (0.01-1; with >= 2 running) |
| `saturation_consecutive_ticks` | `2` | consecutive eligible full windows (>= 1) |
| `saturation_max_step_factor` | `2` | target factor per step, 1-4 (the target is at least n + 1) |

Validation follows the `breakpoint_*` keys (`parse_scaling_config`: integers via
`_scaling_count`, booleans strict, numbers finite and in range). Added to
`deploy/registry.yaml` and `deploy/overlays/tre-v2/params.yaml`. The tracker re-reads
them from the registry every tick, like `PlanConfig`. Images without this change ignore
the keys (logged as unknown). A per-model KV threshold is not implemented.

## Interaction with O1, C1, SafeScale and donors

* **O1**: unchanged. The saturation path only acts while O1 holds a model at its traffic
  onset (or inside the rescue's own step chain) or the numerator is 0; once O1 lets a
  window through, the warm TSS decides, and later O1 holds from other breakpoints are left
  to the TSS.
* **C1**: a saturation step is tagged with the same `RescuePlan` (base = n, desired =
  target), so `rescue_bases` keeps C1 from re-asking for a target not yet reflected, and a
  later C1 decision builds on it. After a C1 scale-up the saturation path stays out (the
  TSS was warm), so a backlog on the old pods cannot add a second, saturation-driven step.
* **`breakpoint_partial_max_step` (checked 2026-10-02)**: it is in effect. The registry
  value is parsed into `ScalingRegistryConfig.breakpoint_partial_max_step`,
  `registry.scaling()` is the `config` object in `loops/tick.py::_scaling_options`, and
  with `breakpoint_window: true` it reaches `PlanConfig.partial_window_max_step`;
  `signal_evidence_requests` is set in the tick context on every partial window and read
  by `critical_need`. The cap applies only when the post-breakpoint window holds fewer
  than `breakpoint_lowevidence_requests` (10) completed requests. In A-smoke-tre the
  first decision (window 45 s, n = 1, Z = 0.298, desired 3) was taken on a 20 s suffix at
  ~3000 decode tokens/s, i.e. far more than 10 completed requests, so the cap did not
  apply and 1 -> 3 (planned 2) is the intended C1 result. A tick-level test covers the
  wiring end to end (`test_partial_max_step_is_wired_from_the_registry_through_the_tick`).
  The two mechanisms are disjoint: the cap bounds a TSS decision on a thin partial window;
  the saturation path is not a TSS decision, has its own bound (doubling, added pods
  full) and re-confirms on fresh windows after every step, so the cap is not applied to it.
* **SafeScale**: unchanged. A saturation receiver that has an active probe preempts it
  (`receiver_need_upscale`) like a TSS CRITICAL receiver, and the pods the preemption
  gives back are deducted from the step (tick restore deduction). A probe's hide (or
  unhide) on a model whose TSS was warm is a foreign breakpoint: the hold that follows is
  not eligible, so busy donors' probes are not rolled back by this path.
* **Donors**: unchanged - only IDLE / HIGH models are immediate donors; a model the
  saturation path declares CRITICAL is a receiver and therefore not a donor in that tick.
  A donor release on a warm model is a foreign breakpoint (no re-scaling by this path).

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
  for a full window not yet confirmed; `saturation_step_landed:<model>:<n>-><m>`,
  `saturation_step_unconfirmed:<model>:n=<n>:waited_ms=<ms>` (refused / observe mode),
  `saturation_reset_external:<model>:<a>-><b>` (a routable change the rescue did not cause).
  Each event is reported once, on the first read of its window (the re-reads of the
  rescue / fairness loops do not repeat it).
* Decision snapshot `model_states.<model>` (present when the feature is on):
  `saturation_rescue`, `saturation_reason`, `saturation_ticks`, `saturation_waiting`,
  `saturation_kv`, `saturation_running`, `saturation_pods`, `saturation_sample_ms`,
  `saturation_awaiting_step`, `saturation_count_after`, `saturation_pod_samples`
  (per routable pod: waiting / KV / running of its newest sample).
* Startup log `saturation_rescue_config`. The signal log shows the receiver as tier `crit`.
* `PodWindowMetrics.latest_waiting / latest_running / latest_gpu_cache / latest_instant_ms`
  (store): the newest instant sample per pod (not part of equality).

## Offline replay

`controller/tools/saturation_onset_replay.py` (read-only) runs the controller's
`SaturationTracker` and `saturation_target` over recorded runs: eligibility from the
recorded tick of each window, engine gauges from the 5 s sampler (`pod_gauges.jsonl`),
routable pods from `layout.jsonl`. Limits:

* the gauge used for a window is the sampler's sample nearest to the window end within
  +-3 s - it can lie up to 3 s after the window end (a small look-ahead the controller,
  which reads the gateway doc stamped at the window end, does not have);
* exact up to the first simulated step only; afterwards the recorded gauges belong to a
  different replica count, a step is assumed to land 3 s after the tick and capacity to
  be found, the pods a simulated step adds are not in the recording (so the "added pods
  full" check never confirms a second saturation step there), and once the TSS is warm
  again the C1 hand-off is estimated from the recorded Z rescaled to the simulated n;
* the idle windows are approximated by the recorded windows with numerator 0 and Q at qmin;
* `E-14b-crit-20261001-143058` was recorded with a build before O1 (onset warmup guard),
  so its "hold" windows are the onset guard's.

Times are seconds after load start; "tick" is when the decision is taken (the wake lands
~2-3 s later).

| run | model / load | recorded first scale-up | new rule: first step | reaches 4 (indicative) | false trigger |
|---|---|---|---|---|---|
| A-smoke-tre-20261002-180116 | 8b, 12 rps from 0 | window 45, tick 49.4: 1 -> 3 (C1); peak 3 | window 35, tick 39.4: 1 -> 2 (`o1_hold`, waiting 122, KV 0.99) | no second saturation step (the added pod is not full; the old pod drains its backlog); not during the onset | none: 7b 0-70 s (4 rps) 3 eligible windows, 0 full (max KV 0.06); 7b / 14b whole run 3 eligible each, 0 full; 8b after 80 s (C1 / SafeScale breakpoints) 0 eligible |
| C-crit-20261002-175810 | 7b, 200 concurrent, 1500-word prompts | window 35, tick 38.8: +1 (donor); 2 at 63 s, 4 at 102 s | window 15, tick 18.8: 1 -> 2 (`o1_hold`, waiting 172) | tick 48.8 via C1 hand-off (Z ~ 0.017) | - (genuine overload) |
| E-14b-crit-20261001-143058 | 14b, 400 concurrent decode-heavy (pre-O1 build) | window 66, tick 74.4: +1 (wake did not land; stayed at 1) | window 16, tick 24.4: 1 -> 2 (`numerator_zero`, waiting 104, KV 0.72) | tick 74.4 via C1 hand-off | prefill-heavy phase (242-402 s): 0 eligible windows (TSS warm throughout) |

Reading: the first rescue step comes 10 s (A), 20 s (C) and 50 s (E) earlier than the
recorded first scale-up. Low-load and long-request windows are eligible but never "full";
after a C1 / donor / SafeScale breakpoint on a warm model nothing is eligible.

## Risks

* **Backlog on the first replica**: addressed by the added-pods rule; a model whose new
  replica fills up immediately (a real flash crowd) still doubles again.
* **Gauge freshness**: the rule reads the gateway's 10 s instant docs; a stale doc is
  ignored (no sample = not full), so a scrape gap delays, never triggers.
* **Restart mid-load**: the tracker is in-process, and after a controller restart the
  traffic is a new onset for O1 too: if the engines are full on the first two windows the
  rescue may take one step (they are full); the first warm window closes the path again.
* **Signal-source ablations**: the path is TSS-only (`signal_source: zm`); alternative
  signal arms are unaffected.

## Tests

`controller/tests/test_saturation_rescue_20261002.py`: numerator 0 + waiting on 2
windows -> CRITICAL (1 window -> nothing, re-reads do not count); running-only, or one
request filling the KV cache -> stays healthy; O1 hold at the onset + KV >= 0.9 ->
trigger; warm TSS -> never; O1 holds after a C1 scale-up, a donor release and a SafeScale
hide on a warm model -> never; a new onset re-opens; bounded doubling 1 -> 2 -> 4 with two
fresh windows after the change; a step landing in parts; step pods from the fleet view
(a stale-sampled old pod is not "added"); events once per window; O1 resume after a
suspension does not re-open; old-pod backlog with an idle added pod -> no second step;
capped by `max_awake_replicas`; external routable change restarts the count; observe mode
-> `saturation_step_unconfirmed`, then counting restarts; probe preemption covers the step
(restore deduction) and its unhide lands it; free capacity before donors, immediate HIGH
donor without it; TSS CRITICAL before saturation receivers, then by waiting / n;
no incomplete-drop; sample selection (hidden / stale pods); store latest-sample fields;
decision-snapshot export; config parsing, validation (factor <= 4) and per-tick
re-read; the `breakpoint_partial_max_step` wiring.
