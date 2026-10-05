# tre_baselines

The baseline autoscalers of TRE v2 (scaling parts only) as one small service, the
**baseline shell**. Each arm is an adaptation and is named so in decision records and
docs: **Chiron-global** (`TRE_BL_POLICY=chiron`; only the global instance loop scales, B
is virtual), **TokenScale-colocated** (`tokenscale`; the PD-disaggregated velocity policy
on colocated P+D replicas), **PreServe-oracle** (`preserve`; Tier-1 is the replayed trace
plus noise, not mLSTM; `max_tokens` as the length prediction). Every tick it builds a snapshot (service
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
                 tools/   arm.py  chiron_theta.py  tokenscale_buckets.py  tokenscale_profile.py  preserve_mu.py
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
the SM worker checks both again right before each call. After an SM refusal a model backs
off (`max(retry_after_s, tick)`, doubling, capped at `TRE_BL_BACKOFF_MAX_S` = 10 s; action
`backoff`); a refusal is retried at once when the SM state version changes. Remaining
timers and what they guard: the owner-lock TTL (mutual exclusion after a crash), the SM
call timeout and the backoff cap (liveness: retries whose cause `/v2/state` does not
show), `/livez` stall (k8s probe); the policy windows (TokenScale `window_s`, PreServe
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

- Chiron-global: `busy_def: nonidle` (paper IBP) for the main runs, `at_cap` only as a
  sensitivity run; theta = theta_trace from `tools/chiron_theta --method peak_mean
  --interval-s 5` on the replayed trace (CPU only), 1/3 as a sensitivity row.
- TokenScale-colocated: `tools/tokenscale_buckets` on the trace (edges, centers,
  `median_in`), then `tools/tokenscale_profile --sender http --gateway-url
  <chat endpoint URL> --sm-url <SM> --i-have-user-approval` per model with exactly one
  awake replica (closed loop 1..32, 60 s steps, 15 s warm-up; ~1 h per model).
- PreServe-oracle: mu from `tools/preserve_mu` over the calibration capture (needs the
  registry with the fitted c/b); `window_s` 600 (sensitivity 60), `noise_sigma` 0.0772
  (sensitivity 0.30).

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
- **mu window.** `tools/preserve_mu.py` profiles over 30 s windows while Tier-1 plans over
  600 s windows; a 30 s maximum is biased high relative to what a replica sustains over
  10 min (fewer replicas planned). Use a 600 s-equivalent window if the capture allows,
  else disclose.
- **PreServe map.** Requests enter the look-ahead map at `ft` (prefill done); non-streaming
  requests have no `ft` until their output is complete, so they are not in the map (the
  experiments stream). Requests still in prefill are not in the map either (paper).
- **ft vs TTFT.** The gateway's `ft` event (first byte through Envoy) is used as "prefill
  done"; its alignment with vLLM's TTFT is not verified.
- **T1 schema.** The service manager's structured 409 body and the new `/v2/state` fields
  are provisional (the T1 line is not merged); the shell parses both plain and structured
  refusals (and honours `retry_after_s` when present).
- **Campaign.** `campaign_queue.py` does not have the three baseline arms yet (it should
  call the arm tool: `enable`, `mark-replay` at replay start, `disable --collect-dir`).
- **Unmeasured parameters.** TokenScale velocities V_b / V_P, Chiron theta_trace and
  PreServe mu have not been measured on the current engine; the example files hold
  placeholders (the policies refuse to start on them). Freeze all of them in one config
  commit before any comparison.
- **Stale owner.** The pre-call check leaves the time between the check and the SM
  handling the request; closing it needs the SM to reject a stale owner generation.
- The E1 trace format and where the `*.effective.json` (per-request schedule) lands are
  not verified; PreServe Tier-1 needs the trace of the replay and its seed.
