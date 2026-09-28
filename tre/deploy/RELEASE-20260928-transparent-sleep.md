# Release 2026-09-28: transparent sleep (TRE v2)

Branch `tre/v2-transparent-sleep-release` (worktree `aibrix-wt/release`), based on
`tre/reissue-sidecar-v2` @ 3ec45753 (integration 47a00dee = service-manager + gateway
transparent-sleep work, plus the retry / continuation sidecar). Plan:
`docs/plan-20260927-v2-transparent-sleep-portability.md` (D1-D10).

Nothing in this document has been applied. Deploy only after confirming with the user
that no other session is using the cluster.

## 1. What changes

| Component | Before (live) | This release |
|---|---|---|
| model pods | `vllm/vllm-openai:0.10.1-sleep`, vLLM on :8000 | `vllm-openai-tre:0.30.0-ts-02ad6c9e`, reissue sidecar on :8000, vLLM on 127.0.0.1:8001, `--sleep-reject-new --abort-return-token-ids`, default cumem sleep backend |
| service-manager | `tre-v2-service-manager:20260924-4e9ab85c`, registry baked in the image | `tre-v2-service-manager:20260927-849bba24`, `strategy: Recreate`, registry from ConfigMap `/etc/tre`, sleep primitive hide -> gateway ack -> drain -> /sleep (D1-D4) |
| controller | `tre-v2-controller:20260924-2caa0514` | `tre-v2-controller:20260927-849bba24` |
| gateway-plugins | `aibrix/gateway-plugins:20260924-43aa0c31-nozmq2` | `aibrix/gateway-plugins:20260927-849bba24-nozmq2` (route-gen acks, in-flight counts, `x-tre-exclude-pod`, requests without `routing-strategy` go through ext_proc, vLLM metric-rename fallback) |
| gateway Service | none | ClusterIP `tre-gateway` in `envoy-gateway-system` (in-cluster clients: the sidecar) |
| ui | `tre-v2-ui:20260924-4e9ab85c` | unchanged (tre/ui untouched) |
| registry ConfigMap | `cluster` + `models` only | adds `vllm`, `gateway`, `reissue`, `service_manager`; new per-model keys |

## 2. Images

| Image | ID | 76 (node10) | 75 (node9) |
|---|---|---|---|
| `vllm-openai-tre:0.30.0-ts-02ad6c9e` | `0cbc7eecaef1` | yes | yes |
| `tre-v2-service-manager:20260927-849bba24` | `d6e49b8006c3` | yes | - |
| `tre-v2-controller:20260927-849bba24` | `09250d5dbfa6` | yes | - |
| `aibrix/gateway-plugins:20260927-849bba24-nozmq2` | `a081871d4615` | yes | - |
| `tre-v2-ui:20260924-4e9ab85c` (unchanged) | `07e60f502378` | yes | - |

- The control-plane Deployments (service-manager, controller, gateway-plugins, ui, redis)
  are pinned to node10 (`nodeSelector`), so their images exist only on 76. Model pods
  run on both nodes.
- vLLM image: fork `stmoonar/vllm` branch `tre/transparent-sleep` @ 02ad6c9eca (label
  `org.opencontainers.image.revision`), a pure-Python patch `ced6857afa..02ad6c9eca --
  vllm/` over `vllm/vllm-openai:0.30.0`. It includes the wait-mode sleep fix.
  `vllm-openai-tre:0.30.0-ts-a2659293` (`c75b2f6feaa3`) is also present on both nodes
  and **must not be used** (waiting requests were held until wake).
- CUDA 13 image on host driver 550 via the image's cuda-compat 580: `cuInit` returns 0
  and the driver API reports 13000 inside the image on both nodes (container toolkit
  1.17.8 on 75, 1.18.2 on 76).
- Rollback images are still present on 76: `tre-v2-service-manager:20260924-4e9ab85c`
  (`8eb1a92f39e5`), `tre-v2-controller:20260924-2caa0514` (`518bef0902a7`),
  `aibrix/gateway-plugins:20260924-43aa0c31-nozmq2` (`9889fe739c81`); and on both nodes
  `vllm/vllm-openai:0.10.1-sleep` (`6a3a5efad777`). Do not `docker image prune` on 76.

## 3. vLLM 0.30 migration notes (registry)

- `vllm.env` (new, registry-wide) merges over the built-in `VLLM_SERVER_DEV_MODE=1`
  (0.30 mounts `/sleep`, `/wake_up`, `/is_sleeping` only in dev mode); `models[].vllm_env`
  merges over it. Shipped: `VLLM_SERVER_DEV_MODE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `HF_HUB_OFFLINE=1`. `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` was
  shipped first and removed on 2026-09-28 (deploy test): with the cumem backend every
  sleep after the first wake failed (`CUDART error: invalid argument` in
  `CuMemAllocator.sleep`), see section 4.
  `VLLM_USE_MODELSCOPE=True` (hard-coded before) is dropped: the 0.30 image was
  GPU-validated with `HF_HUB_OFFLINE=1` and local weight paths.
- `models[].max_model_len`: dsllama-8b 32768 (at util 0.85 0.30 leaves ~15.8 GiB of KV
  cache, below the full 131072 context, and refuses to start); dsqwen-14b 12288 (moved
  out of `vllm_extra_args`); dsqwen-7b null (131072).
- `models[].sleep_mode_backend: null` = vLLM default cumem. `pinned_weights` is the
  documented option (sleep ~0.07 s, wake ~0.4 s slower, level 1 only).
- `--swap-space` removed (gone in 0.30).
- `models[].vllm_features: [sleep_reject_new, abort_return_token_ids]` renders the two
  fork flags (only while `reissue.enabled`).
- Every flag the generator emits was found in `vllm serve --help=all` of the image;
  each rendered command was parsed with the image's own parser +
  `validate_parsed_serve_args` + `create_engine_config` (TP1 models; the TP2 config
  check needs visible GPUs and was only parsed). The launcher
  `python3 -m vllm.entrypoints.openai.api_server` still works in 0.30 (deprecation
  warning only).
- vLLM 0.30 no longer exports `vllm:gpu_cache_usage_perc` or
  `vllm:time_per_output_token_seconds`. The gateway-plugins build of this release reads
  `vllm:kv_cache_usage_perc` / `vllm:inter_token_latency_seconds` instead when the old
  name is absent (least-gpu-cache routing and the Redis TPOT histogram behind the
  SafeScale SLO check depend on them). The APA baseline manifests
  (`deploy/baselines/apa/*`) still target `gpu_cache_usage_perc`; with 0.30 pods they
  must target `kv_cache_usage_perc` (the aibrix-system controller-manager binary maps
  both names) before any APA run.
- The calibrated theta predates the new engine: recalibrate after deployment (D9, P5).

`--sleep-reject-new` behaviour the sidecar relies on:
- a request reaching a sleeping / paused engine gets `503` + `Retry-After: 1`,
  body `{"error": {"type": "EngineSleeping"}}`;
- a streaming request rejected after the response started gets HTTP 200 and a first SSE
  event `{"error": {"type": "EngineSleeping", "code": 503}}` then `[DONE]`. The sidecar
  treats an EngineSleeping error as the first event, before anything reached the client,
  as retryable (`tre_reissue/sidecar.py`, `is_sleeping_error` in the stream loop ->
  `_retry(..., "engine_sleeping")`; test
  `test_engine_sleeping_error_inside_the_stream_is_retried`);
- multi-API-server setups can see ~0.5 s of extra retryable 503s after a wake (TRE runs
  one API server per pod; the sidecar retries them anyway).

## 4. Memory sanity (computed, no GPU run)

Layout (registry, `max_bound_per_gpu: 3`): every one of the 8 GPUs holds three bindings:
one dsqwen-7b, one dsllama-8b, one rank of a dsqwen-14b TP2 slot.

GPU (A100 40 GiB = 40960 MiB):
- awake binding at util 0.85: 0.85 x 40 GiB = 34.0 GiB (measured awake 33.7-34.6 GiB);
- sleeping residual with cumem: 1.1-2.1 GiB per process (measured 7b/8b); the 14b rank
  is not measured on 0.30 (assume <= 2.6 GiB with NCCL buffers);
- worst case per GPU: 34.0 + 2.1 + 2.6 = 38.7 GiB <= 40 GiB (headroom ~1.3 GiB); nominal
  (1.1 GiB residuals) 36.2 GiB. The 0.30 startup check (free memory >= util x total when
  the engine starts) passes with two sleeping neighbours: 40 - 2.1 - 2.6 - ~0.5 (own
  context) = 34.8 GiB >= 34.0 GiB, margin ~0.8 GiB in the worst case.
- the service-manager wake gate (`wake.max_used_fraction 0.2` = 8 GiB) sees <= 4.7 GiB
  of sleeping residents.

Host RAM per node (4 x 7b, 4 x 8b, 2 x 14b TP2 bindings). With cumem the pinned host
copy stays after the first sleep (torch's CachingHostAllocator keeps it), also while
awake, so every binding counts:
- with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` (NOT shipped: breaks every
  cumem sleep after the first wake, reproduced 3/3 on 0.30.0-ts-02ad6c9e): ~1.0 x weights +
  process baseline: 7b 17.6 GiB (measured) x 4 + 8b ~18 GiB x 4 + 14b ~34 GiB x 2
  = ~210 GiB (weights only: 15 x 4 + 16 x 4 + 28 x 2 = 180 GB);
- without it (shipped; power-of-two rounding, ~1.9 x; 7b measured 30.1 GiB): ~360 GiB;
- nodes: 1007 GiB total, ~660 GiB available now on each node (buff/cache is
  reclaimable; ~273 GiB `shared` already in use). Fits with either setting.

## 5. Before deploying

1. Confirm with the user that the cluster is free (no campaign / parallel session). On
   2026-09-28 00:00 the GPUs of 75 showed ~37 GiB used each outside the cluster pods
   (a validation run on 75?): all 8 GPUs must be free of foreign processes.
2. Controller mode AND SM actuation (independent switches since 2026-09-28) both
   `observe`: `bash deploy/scripts/set_run_mode.sh status`; set them with
   `bash deploy/scripts/set_run_mode.sh observe observe` or the console
   (`POST /api/ops/run-mode {"controller":"observe","sm_actuation":"observe"}`).
   After the deploy set both for the next phase (TRE arm `active active`, APA arm
   `observe active`).

### Backup

```bash
B=/data/nfs_shared_data/xxy/backups/tre-v2-state-$(date +%Y%m%d-%H%M)
mkdir -p $B
kubectl -n tre-v2 get deploy,svc,cm,sa,role,rolebinding -o yaml > $B/tre-v2-ns.yaml
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/live-registry.yaml
kubectl -n default get deploy,svc -l tre.aibrix.io/managed=true -o yaml > $B/default-models.yaml
kubectl -n default get svc -l model.aibrix.ai/name -o yaml > $B/default-model-svcs.yaml
kubectl -n tre-v2 get httproute -o yaml > $B/httproutes.yaml
kubectl get envoyextensionpolicy,envoypatchpolicy,clienttrafficpolicy -A -o yaml > $B/envoy-policies.yaml
kubectl get referencegrant -n default -o yaml > $B/referencegrants.yaml
kubectl get clusterrole/tre-gateway-plugins-role clusterrolebinding/tre-gateway-plugins-rolebinding \
    clusterrole/tre-v2-readonly clusterrolebinding/tre-v2-readonly -o yaml > $B/cluster-rbac.yaml
# Redis (desired state, journals, leases): RDB snapshot
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SAVE
kubectl -n tre-v2 exec deploy/tre-v2-redis -- cat /data/dump.rdb > $B/tre-v2-redis.rdb
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --scan --pattern 'tre:v2:*' | sort > $B/redis-keys.txt
```

## 6. Apply order

All commands on 76 in the release worktree:
`cd /data/nfs_shared_data/xxy/aibrix-wt/release/tre`.

1. **Registry ConfigMap** (structural keys have no console path; `PUT /api/params`
   edits only per-model tunables, and `overlays/tre-v2/params.yaml` must never be
   applied). Merge the release registry with the live one (live tunables win key by key,
   release adds images / engine args / new sections), review, write:
   ```bash
   kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > /tmp/live.yaml
   PYTHONPATH=common:deploy python3 deploy/scripts/merge_live_registry.py \
       --live /tmp/live.yaml --release deploy/registry.yaml --out /tmp/merged.yaml
   diff /tmp/live.yaml /tmp/merged.yaml | less
   kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=/tmp/merged.yaml \
       --dry-run=client -o yaml | kubectl replace -f -
   ```
   Dry run on 2026-09-28: "no live tunable differs from the release file" (live trs /
   slo / alt_thresholds / replica keys equal the release file); the merged file equals
   `deploy/registry.yaml`. The running old service-manager reads its registry from its
   image and the old controller only at start, so this step changes nothing live yet.
2. **Gateway**: plugin (new image, `ROUTING_ALGORITHM=least-gpu-cache`,
   `TRE_GW_COORDINATION=true`), ext_proc routes (no-header requests via ext_proc), the
   stable Service (rendered through kustomize for its name / namespace params):
   ```bash
   kubectl apply -f deploy/overlays/tre-v2/gateway-plugins.yaml
   kubectl -n tre-v2 rollout status deploy/tre-gateway-plugins
   kubectl apply -f deploy/overlays/tre-v2/gateway-extproc.yaml
   kubectl kustomize deploy/overlays/tre-v2 | python3 -c 'import sys,yaml; print(yaml.safe_dump([d for d in yaml.safe_load_all(sys.stdin) if d and d["kind"]=="Service" and d["metadata"]["name"]=="tre-gateway"][0]))' > /tmp/tre-gateway-svc.yaml
   kubectl apply -f /tmp/tre-gateway-svc.yaml
   ```
   Check: `redis-cli ZRANGE tre:v2:gw:instances 0 -1 WITHSCORES` shows the new plugin pod
   with an advancing score; `kubectl -n envoy-gateway-system get endpoints tre-gateway`
   lists the Envoy proxy pod(s).
3. **service-manager** (`strategy: Recreate`, one writer; mounts the ConfigMap):
   ```bash
   kubectl apply -f deploy/overlays/tre-v2/service-manager.yaml
   kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager
   ```
   Check logs for registry validation and the clock-skew line; `GET /v2/audit`,
   `GET /v2/supervisor`. The new service-manager also drives the old 0.10.1 pods (it
   probes `/version` and sends a plain `/sleep` below 0.18).
4. **controller** (stays in observe mode):
   ```bash
   kubectl apply -f deploy/overlays/tre-v2/controller.yaml
   kubectl -n tre-v2 rollout status deploy/tre-v2-controller
   ```
5. **Model pods** (vLLM 0.30 + sidecar). The generated Deployments use the default
   RollingUpdate: applying them over the running pods would start each new pod next to
   the old one on the same GPU, its startup gate would wait for memory the old pod never
   frees, and the rollout would never finish. Replace them instead:
   ```bash
   kubectl -n default delete deploy -l tre.aibrix.io/managed=true --wait=false
   kubectl apply -k deploy/models    # sidecar ConfigMap, Services, HTTPRoutes, 20 Deployments
   ```
   The new pods wait in their `tre-startup-gate` init container; the service-manager
   admits them as the old pods release their GPUs (pressure / gpu-truth checks) and
   brings each binding to its desired lifecycle (resident or asleep); the desired state
   in Redis is kept (binding ids are unchanged) and seeded from the registry where
   missing (D7). If the supervisor sees the missing Deployments first, its fleet repair
   creates the same objects from the same registry (the runtime-create path renders
   identical Deployments, tested), so `kubectl apply` is then a no-op.
   Watch: `kubectl -n default get deploy -l tre.aibrix.io/managed=true`,
   `GET /v2/fleet/state`, `GET /v2/audit` (expect no issue once 20 are ready). If gates
   stall > 10 min: SM logs, `GET /v2/operations`, then `POST /v2/fleet/repair`
   (controller must be in observe).
6. Keep the controller in observe until the tests below pass.

SM API from 76: `kubectl -n tre-v2 port-forward svc/tre-v2-service-manager 18000:8000`.

## 7. Rollback

To the manifests of tag `pre-transparent-sleep-20260927` (= main 1f307163) and the old
images:

```bash
git -C /data/nfs_shared_data/xxy/aibrix worktree add /tmp/tre-rollback pre-transparent-sleep-20260927
R=/tmp/tre-rollback/tre/deploy
# 1. controller observe (console), then model pods back to 0.10.1 without sidecar
kubectl -n default delete deploy -l tre.aibrix.io/managed=true --wait=false
kubectl apply -k $R/models
kubectl -n default delete cm tre-reissue-sidecar --ignore-not-found
# 2. control plane (old SM reads the registry baked into its image)
kubectl apply -f $R/overlays/tre-v2/service-manager.yaml
kubectl apply -f $R/overlays/tre-v2/controller.yaml
kubectl apply -f $R/overlays/tre-v2/gateway-plugins.yaml
kubectl apply -f $R/overlays/tre-v2/gateway-extproc.yaml
kubectl -n envoy-gateway-system delete svc tre-gateway --ignore-not-found
# 3. registry ConfigMap back to the backup (keeps the live tunables of that moment)
kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=$B/live-registry.yaml \
    --dry-run=client -o yaml | kubectl replace -f -
kubectl -n tre-v2 rollout restart deploy/tre-v2-controller
```
- The old service-manager has no `strategy: Recreate` in its manifest: applying it over
  the new one switches the strategy back; delete the new pod first if two writers must
  never overlap (`kubectl -n tre-v2 scale deploy/tre-v2-service-manager --replicas=0`
  before the apply).
- Redis keys added by this release (`tre:v2:gw:*`, sleep journal / reservations) are
  ignored by the old code; restore `$B/tre-v2-redis.rdb` only if the desired state got
  corrupted (stop redis, copy the file to `/data/dump.rdb`, start).

## 8. Deployment test plan (in this order; stop at the first failure)

Every step: `GET /v2/audit` empty afterwards, no pod restarts, sidecar metric
`tre_reissue_total{kind=...}` recorded, `vllm:num_requests_paused` sane.

1. **Empty-load sleep / wake**: each model, one binding, `PUT /v2/bindings/<id>/power
   {"awake": false}` then `{"awake": true}`. Expect hide -> ack -> drain(0) -> /sleep,
   `/is_sleeping` true, GPU used ~1.1-2.1 GiB (also the 14b ranks), wake ~2 s, the first
   request after wake answers correctly. Check `tre.aibrix.io/route-gen` bumped and
   `tre:v2:gw:seen:<pod>` acked.
2. **Loaded sleep / wake**: 8 concurrent streams on one model, sleep one awake binding
   (drain path, no abort): all streams complete, `x-tre-continued` absent, requests
   that hit the sleeping pod retried by the sidecar (`kind=retry`).
3. **Cold start**: `PUT /v2/models/<m>/target` beyond the awake bindings, and one Deployment
   deleted + re-created: gate admission, serial per GPU, no OOM, 0.30 startup memory
   check passes next to two sleeping neighbours (worst case in section 4).
4. **Defrag**: `POST /v2/defrag {"tp_size": 2}` when a TP2 slot is fragmented; relocation
   completes, audit clean.
5. **Failure injection**: kill a vLLM process mid-sleep; delete the gateway-plugins pod
   during a hide (ack timeout -> rollback, pod routable again); `/sleep` without the
   hidden header -> 409 from the sidecar; `/sleep?level=2` -> 400 and the engine keeps
   serving.
6. **SafeScale commit**: controller active on one model with a probe window; a
   hidden-probe commit sleeps via `safescale_commit` drain budget; TPOT p95 is present
   in the controller metrics (Redis TPOT histogram filled from
   `vllm:inter_token_latency_seconds`).
7. **Controller restart with a committing probe**: restart the controller while a
   SafeScale commit is in progress; the frozen absolute target is re-sent, never applied
   twice.
8. **SM SIGTERM mid-drain**: `kubectl delete pod` of the service-manager during a long
   drain: the drain rolls back (pod routable again), a sleep past `/sleep` finishes, the
   new instance resolves the journal (awake read twice before reopening).
9. **P1 acceptance**: 32 concurrent streams (max_tokens 1024) through the gateway while
   50 random SM sleep / wake operations run -> 0 `finish_reason=abort`, 0 HTTP errors,
   0 client timeouts; gateway ack latency p99 recorded; audit clean.
10. **P3 acceptance**: 32 streams in flight, forced abort sleep
    (`PUT /v2/bindings/<id>/power {"awake": false, "drain_budget_s": 0}`) -> 100% complete
    responses (token-id seam, no duplicate / missing tokens vs a serial reference),
    `x-tre-continued` recorded in the replayer log, non-continuable requests (n>1,
    logprobs) drained instead of aborted.

Afterwards: recalibrate theta on the new engine (D9 / P5) before any campaign; switch
the APA baseline to `kv_cache_usage_perc` before any APA run.

## 9. Commits

- `27923798` gateway: read renamed vLLM metrics (kv_cache_usage_perc, inter_token_latency_seconds)
- `849bba24` deploy: migrate model pods to the vLLM 0.30 fork image (plan D9) (images built here)
- `0859cff6` deploy: tre-v2 images of the transparent-sleep release (849bba24)
- this document + `deploy/scripts/merge_live_registry.py`
