# tre_baselines

The baseline autoscalers of TRE v2 (scaling parts only) as one small service, the
**baseline shell**. Each arm is an adaptation and is named so in decision records and
docs: **Chiron-global** (`TRE_BL_POLICY=chiron`; only the global instance loop scales, B
is static = engine max_num_seqs; the virtual Alg.1 B is a sensitivity row), **TokenScale-colocated** (`tokenscale`; the PD-disaggregated velocity policy
on colocated P+D replicas; the paper's cells, no V_P), **PreServe** (`preserve`; the paper's mLSTM
Tier-1 with v1's paper_past weights, Tier-2 on `max_tokens` plus the paper's Tab. 2 load errors, v1's
faithful composition; `PreServe-oracle` only as the oracle sensitivity). Every tick (5 s = the TRE fast
loop, `TRE_BL_TICK_S`) it builds a snapshot (service
manager `/v2/state`, each awake pod's `/metrics`, the gateway request-event stream
`tre:v2:bl:req:<model>`, the replay marker `tre:v2:bl:replay_t0`), asks the selected
policy for a desired replica count per model, clamps it to the registry bounds and sends
`PUT /v2/models/{m}/target` to the service manager. All timestamps are Redis server time.

Rules every arm shares (2026-10-05 decision, `docs/baselines-methodology-20261005.md` of the
local workspace):

- **Unknown is not idle.** A scale-down needs complete evidence
  (`snapshot.evidence_gaps`): every serving pod scraped, running/waiting present (a missing
  gauge is `None`, never 0), no stale scrape, and for the event-driven arms (TokenScale,
  PreServe) a gap-free event history covering every request in flight. After the shell
  starts or the event stream is trimmed/recreated, the requests already running (the
  cohort, seeded from the engine running+waiting) block scale-down until the engine is
  covered by tracked requests again - state, no timer. Otherwise the decision holds at
  `awake` with reason `incomplete` (policy reason and target kept in `inputs`); the shell
  applies the same gate as a backstop. Scale-up uses whatever evidence there is.
- **One sleep path for every arm.** Scale-downs go to the SM on the abort path
  (`TRE_BL_ABORT_SLEEP_PATH`, default `urgent`, must be one of the registry's
  `service_manager.sleep.no_drain_paths`): hide, gateway ack, `/sleep mode=abort`,
  sidecar continuation. The shell never drains and never sends `drain_budget_s`;
  aborts/continuations are a reported cost of each arm.
- **Donor before dependent.** No scale-up is sent while a scale-down call is in flight
  (action `wait_donor`); the SM answers a scale-down after the sleep is committed and the
  GPU released.
- **Execution right is re-checked right before every SM call** (owner lock still ours,
  controller still `observe`); otherwise the call is dropped (`sm_result.error =
  "dropped"`).
- **No silent SLO fallback.** An actuating shell refuses to start without the registry's
  live idle-TTFT fit (`slo.ttft_idle_c_ms` / `ttft_idle_b_ms_per_token`).

```
tre_baselines/   snapshot.py (policy contract)  sources.py  loop.py  sm_client.py  main.py
                 policies/{chiron,tokenscale,preserve,static}.py   trace_oracle.py
                 policies/preserve_{tier1,anticipator,predictor,load_errors,history}.py
                 tools/   arm.py  chiron_theta.py  tokenscale_buckets.py  velocity_ladder.py  freeze_params.py
                          preserve_history.py  tokenscale_profile.py  preserve_mu.py (old procedure)
weights/         preserve-mlstm-20261024-paper/{code,conv,pooled}-paper.npz (v1 paper_past, REPORT.md)
requirements-baselines.txt   numpy==1.26.4 (mLSTM forward pass; the image sets *_NUM_THREADS=1)
examples/        one parameter file per policy (chiron.yaml, tokenscale.yaml, preserve.yaml)
tests/           unit tests + test_bl_e2e_real_redis.py (fake environment, real Redis)
```

## Run the shell locally in dry-run

Dry-run is the default: decisions are logged, the service manager is never called.

```bash
cd tre
export PYTHONPATH=common:deploy:baselines
export TRE_SM_URL=http://<sm-host>:8000 TRE_REDIS_URL=redis://<redis-host>:6379/0
export TRE_REGISTRY_PATH=deploy/registry.yaml
export TRE_BL_POLICY=chiron TRE_BL_POLICY_CONFIG=baselines/examples/chiron.yaml
export TRE_BL_LOG_DIR=/tmp/bl-logs TRE_BL_HTTP_PORT=8080
python3 -m tre_baselines.main          # one JSONL line per model per tick in $TRE_BL_LOG_DIR
curl -s localhost:8080/healthz ; curl -s localhost:8080/livez ; curl -s localhost:8080/metrics
```

`TRE_BL_DRY_RUN=false` actuates, but only in ticks where the shell holds the owner lock
`tre:v2:bl:owner` **and** the TRE controller is in observe mode: the shell reads
`tre:v2:controller:mode` every tick (missing = observe) and otherwise logs
`guard_controller_active` instead of calling the SM (metric `tre_bl_controller_guard`);
the SM worker checks both again right before each call. A refused scale-up is not a
model-wide pause (the SM's `retry_after_s` scopes the refused GPUs): the shell records the
refused `node/gpu`, the blocking binding and the state the call was decided on, logs
`wait_refusal` (metric `tre_bl_refusal_waits_total`) and sends nothing while that state is
unchanged; the next scale-up goes on the first tick where the blocking binding is no
longer awake, a GPU of a sleeping binding of the model became free, or the store version /
wakeable GPU set of `/v2/state` changed (`retry_evidence` on the line). Only a cause
`/v2/state` cannot show (writer lock, failed wake, text-only 409) is also retried after the
SM's `retry_after_s` (none: next tick). A failed scale-down or an unanswered call backs off
(`max(retry_after_s, tick)`, doubling, capped at `TRE_BL_BACKOFF_MAX_S` = 10 s; action
`backoff`; a refused scale-down is retried at once when the SM state changes). A scale-up
is sent one replica at a time
(`awake+1`, `awake+2`, ... each grow-only) and stops at the first 409, because the SM
refuses a whole multi-replica target when one wake is blocked; granted-then-refused is
`sm_result.partial_fill` (metric `tre_bl_partial_fills_total`) and is asked again next
tick. An SM that fills partially itself (200 `unfilled`, or a 409 listing wakes done) is
counted the same way. Remaining
timers and what they guard: the owner-lock TTL (mutual exclusion after a crash), the SM
call timeout, the SM's `retry_after_s` for refusals whose cause `/v2/state` does not show
and the backoff cap (liveness), `/livez` stall (k8s probe); the policy windows (TokenScale `window_s`, PreServe
`window_s` and its once-per-window scale-down) are the papers' mechanisms. Every decision line also goes
to the Redis stream `tre:v2:bl:decisions` (`TRE_BL_DECISION_STREAM`, MAXLEN ~ 100000).
`/healthz` (readiness) turns 503 after repeated failed ticks; `/livez` (liveness) only
checks that the loop runs, so an SM outage never restarts the pod. The full list of
environment variables is in the docstring of `tre_baselines/config.py`. In the cluster it
is the deployment `tre-v2-baseline-scaler` (`deploy/baselines/tre/`), off (0 replicas) by
default.

## Arm tool

```bash
python3 -m tre_baselines.tools.arm enable --policy chiron|tokenscale|preserve [--dry-run-shell] [--models a,b] [--execute]
python3 -m tre_baselines.tools.arm disable --collect-dir <dir> [--execute]   # or --skip-collect
python3 -m tre_baselines.tools.arm mark-replay --trace <path> --seed <int> [--execute]
```

Without `--execute` it only prints the `kubectl` commands. With it, `enable` first checks
that the controller is in `observe` mode (`tre:v2:controller:mode`) and that no APA
`PodAutoscaler` targets the managed models, then sets `TRE_BL_POLICY` /
`TRE_BL_DRY_RUN`, scales the deployment to 1 and waits for `/healthz` 200 and the owner
lock; `disable` first copies the decision logs out of the pod (`kubectl cp
<ns>/<pod>:/var/log/tre-baselines <collect-dir>/<pod>`, `--collect-dir` required with
`--execute` unless `--skip-collect`; a failed copy aborts before scaling), then scales to 0
and waits for the lock to go; `mark-replay` writes
`tre:v2:bl:replay_t0` = `{t0_ms (Redis TIME), trace_path, seed}`. Options `--namespace`,
`--deployment`, `--redis-url` (env `TRE_REDIS_URL`; else `kubectl exec` into
`--redis-deploy`). Nothing about the cluster is hard-coded.

## Parameters

Per-policy parameters live in `examples/*.yaml` (copy, fill in, and ship as the
`tre-v2-baseline-<policy>` ConfigMap, see `deploy/baselines/tre/policy-configmaps.yaml`);
each policy's docstring lists every key, marking what is from the paper and what is our
choice (`# not in paper`). TokenScale velocities, Chiron theta and PreServe mu have no
usable default and must be measured (`tools/`); the policies refuse to start without them:

- Chiron-global: `batch_mode: static` with `static_b` = engine max_num_seqs (256) for the main
  runs (Chiron's ablation with static batch sizes; the engine batch size cannot change at run time,
  so a virtual Alg.1 B only tracks load noise), `batch_mode: alg1` as a sensitivity row
  (`deploy/baselines/tre/sensitivity/chiron-alg1.yaml`); `busy_def: effective` (busy = ceil(sum(running+waiting) / mean B),
  the packed busy count; the paper's IBP assumes packing routing, ours spreads) for the main
  runs, `nonidle` / `at_cap` only as sensitivity runs; part of the same adaptation: B is capped
  (alg1) at min(max_num_seqs, floor(num_gpu_blocks x block_size / (in + out))) of the trace shape
  (`kv_request_tokens`, engine cache info), recorded as `b_max` / `b_max_src` per decision; theta = theta_trace from `tools/chiron_theta --method peak_mean
  --interval-s 5` on the replayed trace (CPU only), 1/3 as a sensitivity row.
- **2026-10-26 (v1 port, docs/campaign-v2-20261026-plan.md)**: TokenScale cells use the paper's boundaries
  (input < 256 / < 1024 / else, output < 100 / < 350 / >= 350) with each cell's calibration point = the mean
  (in, out) of the campaign's replayed requests in it: `tools/tokenscale_buckets --trace-root <set> --traces ...
  --out <report>`. V_b (peak of full steps, v1 D53 queue-growth criterion, two full steps in a row stop) and
  PreServe mu (normalized-latency SLO = 3 x isolated NL, every request in a step's window, v1 D58) are measured
  direct to the pods with `tools/velocity_ladder {ladder|isolated|mu|summarize}` (module docstring). This
  replaces the knee / single-shape procedure below for the main runs.
- **Freeze step.** `python3 -m tre_baselines.tools.freeze_params tokenscale --report <buckets report> --profile
  <ladder profile> --write` and `... preserve --mu <mu.yaml> --write` render the whole policy block of
  `deploy/baselines/tre/policy-configmaps.yaml` from the calibration outputs (input sha256 in the header);
  until then the blocks say `calibration: pending` and the policies refuse to start (`--pending` renders that).
- TokenScale: no V_P (v1 D51); lambda' per cell = true input + the cell's represented output after the
  simulated 15 % misbucketing, drawn from `hash(seed, model, client request id)` (`x-tre-request-id`, carried by
  the gateway's bl events as `client_id`; a sidecar reissue keeps it, so a reissue lands in the same cell).
- PreServe: `window_s` 30 (v1 D18); mu = variant B of `velocity_ladder mu`: cells whose P share lies in the
  traces' pooled 30 s window share **[p5, p95]** (deviation from v1's [min, max], recorded in the output:
  v1's range was narrow because its traces were; `--p-share-range minmax` restores it). Tier-1 needs the
  history before t0 next to the trace (below). The old knee / closed-loop procedure (`tokenscale_profile`,
  `preserve_mu`) and its sensitivity rows (TokenScale-aggressive / window10, PreServe window600) are retired.

## PreServe history before t0

The mLSTM's first forecast needs the 25 windows before the replay (paper section 5.3; v1 #8). The policy
reads `history_path`, default `<trace dir>/pre_t0.json` if present, else `<trace dir>/preserve_history.json`
(format: `policies/preserve_history.py`; a file for another trace / seed, window or fewer than `seq_len`
windows refuses the policy):

- synthetic traces: the generator's `pre_t0.json` (`tre-pre-t0-v1`, rule in the spec's `pre_t0`);
- `Real_*`: `python3 -m tre_baselines.tools.preserve_history real --counts <v1 history/Real_*_slice_*_tok70.json>
  --trace <trace dir>/traces_tre.effective.json --out <dir>/preserve_history.json` (the Azure windows right
  before the slice x the trace's rate / the slice's rate x the trace's mean lengths);
- `preserve_history check --trace T [--history H]` loads it as the policy does and prints the last history
  window next to the trace's first one.

The campaign runner copies the history file next to its copy of the trace, keeping `<trace>/seed<k>/`.

## Trace volume (PreServe Tier-1)

PreServe reads the replayed trace from `params.trace_path` inside the pod. The deployment
mounts a volume named `traces` at `/etc/tre-baselines-traces` (read-only); it is an
`emptyDir` in the shipped manifest, so nothing environment-specific is baked in. The image
also carries `tre/replayer/traces_*` under `/app/tre/replayer/` (and `tre_replayer` itself,
which segment traces need). To serve other traces, patch the volume in your own overlay,
e.g.

```yaml
# kustomization.yaml of an overlay on deploy/baselines/tre
patches:
  - target: {kind: Deployment, name: tre-v2-baseline-scaler}
    patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata: {name: tre-v2-baseline-scaler, namespace: tre-v2}
      spec:
        template:
          spec:
            volumes:            # strategic merge: matched by name
              - name: traces
                emptyDir: null  # drop the default source
                persistentVolumeClaim: {claimName: <your-trace-pvc>, readOnly: true}
                # or configMap: {name: <cm-holding-trace.json>}
                # or hostPath: {path: <dir-on-every-node>, type: Directory}
```

and set `trace_path: /etc/tre-baselines-traces/<case>/trace.json` in the PreServe policy
file. The replay marker must name the same trace (last `trace_match_parts` components).

## Tests

```bash
cd tre && make check          # includes baselines/tests (the e2e file is skipped)
cd tre && make check-redis    # also test_bl_e2e_real_redis.py against a throwaway Redis (needs docker)
```

## Known gaps before on-cluster runs

- **Client header.** Clients do not send `x-tre-bl-in-tokens` yet; the gateway falls back
  to a character-count estimate (poor for Chinese text). Keep the header in the plan:
  until clients send it, TokenScale sits in `degraded_estimate_frac` (it refuses to trust
  a window where more than `max_estimate_frac` of the counts are estimates).
- **Replay t0.** `arm mark-replay` stamps `t0_ms` when it runs; the replayer's own warm-up
  can make that t0 lead the first real send. The replayer should write the marker at its
  first send (not done: `tre/replayer` is outside this line); until then PreServe Tier-1
  windows may start early by the warm-up.
- **PreServe map** (2026-10-26, v1 parity). A request enters the look-ahead map at its `arr`
  event, which the gateway writes once the target pod is chosen (v1 `routed`): `P` = `in_tokens`,
  `D_hat` = `max_tokens` (or its load-error prediction), the head starting at the `arr` time.
  So requests queued inside vLLM (waiting, or in prefill) count too, and non-streaming requests
  are in the map. v1 discloses this as a strengthening against the paper, which books a request
  at prefill done; v2 booked at `ft` (first token) until this date. `ft` no longer changes the
  map. A reissue (sidecar retry / continuation) is a new gateway request: a new entry with its
  own `in_tokens` / `max_tokens` (Tier-1 does not count it). A request leaves on its `done`
  (completion or abort), when its pod leaves the snapshot, or when the engine reports nothing in
  flight at a scrape at least 1 s after its `arr` (v1's margin: a request routed just before
  the scrape may not have reached the engine yet). A request without `done` after
  `max(D_hat, max_tokens) x done_tpot_bound_s + done_margin_s` (0.2 s/token + 180 s, v1 D65:
  above the 150 s route timeout) is expired as a lost done (`expired_routed`), and Tier-2 holds
  for one window (gap `done_lost`), as v1.
- **PreServe slept pods** (2026-10-26). A pod that leaves the snapshot drops its map and its
  buffered events, and any event for it stamped at or before the last tick that saw it absent
  is discarded (also when read later, and also an `arr` the gateway routed to it until the
  hide receipt), so a woken pod starts with an empty map. A `done` of such a request counts as
  `done_pod_gone` (not `unknown_req`). Before this fix, events buffered while the pod slept were
  applied at its wake (smoke 2026-10-26: 51 stale requests, map U0 0.23 vs engine KV 0.011).
  Cost: an `arr` routed to a just-woken pod before the first tick that lists it is discarded too
  (at most one tick of requests on that pod, an under-count); a pod never listed before still
  has its early events buffered.
- **PreServe deviations from v1** (2026-10-26). (a) v1 `_apply_pending` applied the events
  buffered while a pod slept at its wake (the same bug as above); v2 drops them. (b) A
  `reissue=retried` request whose original pod stays listed is in the map twice (original and
  retry) until the original's `done`; v1 ended the original at its interruption. (c) v1's
  `arr` de-duplication and its done-before-routed tombstones are not needed: the v2 gateway
  writes `arr` once per request, and `arr` and `done` come in order from one gateway.
- **T1 schema.** The service manager's structured 409 body and the new `/v2/state` fields
  are provisional (the T1 line is not merged); the shell parses both plain and structured
  refusals (and honours `retry_after_s` when present).
- **Frozen parameters.** Chiron theta_trace is frozen in `deploy/baselines/tre/policy-configmaps.yaml`;
  TokenScale and PreServe are `calibration: pending` until the 2026-10-26 ladder (freeze step above).
  PreServe's `trace_path` must name the replayed trace in the trace volume (75/76 cluster: overlay
  `deploy/baselines/tre-cluster-75-76`, hostPath).
- **Stale owner.** The pre-call check leaves the time between the check and the SM
  handling the request; closing it needs the SM to reject a stale owner generation.
- The E1 trace format and where the `*.effective.json` (per-request schedule) lands are
  not verified; PreServe Tier-1 needs the trace of the replay and its seed.
