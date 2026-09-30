# tre_baselines

The baseline autoscalers of TRE v2 (Chiron, TokenScale, PreServe; scaling parts only) as
one small service, the **baseline shell**. Every tick it builds a snapshot (service
manager `/v2/state`, each awake pod's `/metrics`, the gateway request-event stream
`tre:v2:bl:req:<model>`, the replay marker `tre:v2:bl:replay_t0`), asks the selected
policy for a desired replica count per model, clamps it to the registry bounds and sends
`PUT /v2/models/{m}/target` to the service manager. All timestamps are Redis server time.

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
curl -s localhost:8080/healthz ; curl -s localhost:8080/metrics
```

`TRE_BL_DRY_RUN=false` actuates (needs the owner lock `tre:v2:bl:owner`, and the TRE
controller must be in observe mode). The full list of environment variables is in the
docstring of `tre_baselines/config.py`. In the cluster it is the deployment
`tre-v2-baseline-scaler` (`deploy/baselines/tre/`), off (0 replicas) by default.

## Arm tool

```bash
python3 -m tre_baselines.tools.arm enable --policy chiron|tokenscale|preserve [--dry-run-shell] [--models a,b] [--execute]
python3 -m tre_baselines.tools.arm disable [--execute]
python3 -m tre_baselines.tools.arm mark-replay --trace <path> --seed <int> [--execute]
```

Without `--execute` it only prints the `kubectl` commands. With it, `enable` first checks
that the controller is in `observe` mode (`tre:v2:controller:mode`) and that no APA
`PodAutoscaler` targets the managed models, then sets `TRE_BL_POLICY` /
`TRE_BL_DRY_RUN`, scales the deployment to 1 and waits for `/healthz` 200 and the owner
lock; `disable` scales to 0 and waits for the lock to go; `mark-replay` writes
`tre:v2:bl:replay_t0` = `{t0_ms (Redis TIME), trace_path, seed}`. Options `--namespace`,
`--deployment`, `--redis-url` (env `TRE_REDIS_URL`; else `kubectl exec` into
`--redis-deploy`). Nothing about the cluster is hard-coded.

## Parameters

Per-policy parameters live in `examples/*.yaml` (copy, fill in, and ship as the
`tre-v2-baseline-<policy>` ConfigMap, see `deploy/baselines/tre/policy-configmaps.yaml`);
each policy's docstring lists every key, marking what is from the paper and what is our
choice (`# not in paper`). TokenScale velocities, Chiron theta and PreServe mu have no
usable default and must be profiled (`tools/`); the policies refuse to start without them.

## Tests

```bash
cd tre && make check          # includes baselines/tests (the e2e file is skipped)
cd tre && make check-redis    # also test_bl_e2e_real_redis.py against a throwaway Redis (needs docker)
```

## Known gaps before on-cluster runs

- Clients do not send the `x-tre-bl-in-tokens` header yet; the gateway falls back to a
  character-count estimate (poor for Chinese text), and TokenScale degrades when too many
  counts are estimates.
- `campaign_queue.py` does not have the three baseline arms yet (it should call the arm
  tool: `enable`, `mark-replay` at replay start, `disable`).
- The service manager's structured 409 body and the new `/v2/state` fields are provisional
  (the T1 line is not merged); the shell parses both plain and structured refusals.
- The decision log is on an `emptyDir`: it is gone with the pod (the last line per model is
  also kept in Redis for 1 h). Copy it out before `disable`, or patch in a volume.
- The "controller is in observe mode" guard exists only in the arm tool; the shell itself
  does not check it.
- The E1 trace format and where the `*.effective.json` (per-request schedule) lands are not
  verified; PreServe Tier-1 needs the trace of the replay and its seed.
- TokenScale velocities V_b / V_P, Chiron theta and PreServe mu (and the shell's tick /
  window settings) have not been measured on the current engine; the example files hold
  placeholders. `tools/tokenscale_profile.py` has no HTTP sender yet.
