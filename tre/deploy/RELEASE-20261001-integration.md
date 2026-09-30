# Release 2026-10-01: integration (calibration chat sender + sidecar keep-alive + placement / parallel wake) - DRAFT

Plan only: nothing here has been built or applied. Confirm with the owner before any
step that touches the cluster (a parallel session may be using it; check that
`TRE_EXCLUSIVE_WINDOW` is not held by someone else).

Branch `integ/tre-v2-20261001` (worktree `aibrix-wt/integ-20261001`), from main
19781f50 (live control plane = `20260930-f8ccb0ca`):

1. `--no-ff` merge of `feat/calib-chat-sender-20260930` (dcb8d5f3; contains
   `feat/prompt-zh-en-mix-20260930` 487b9d5b): zh/en 1:1 prompt corpus, calibration
   routing `least-gpu-cache`, provenance, calibration sender on `/v1/chat/completions`
   + `ignore_eos`, preflight CLI `python3 -m scripts.calib_preflight`. Host-side tooling
   only: no image content.
2. `--no-ff` merge of `fix/sidecar-keepalive-20260930` (621fbd34): sidecar keep-alive
   (loopback pool 2 s, one fresh-connection re-send within 1 s, server keep-alive 75 s),
   vLLM `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`, Envoy -> pod idle timeout 60 s
   (EnvoyPatchPolicy ORIGINAL_DST clusters + BackendTrafficPolicy), registry
   `gateway.upstream_idle_timeout_s`, regenerated model manifests. Addendum at the end of
   `RELEASE-20260930-integration.md`.
3. `--no-ff` merge of `feat/placement-parallel-wake-20260930` (c13988d8): one GPU
   ranking for controller and SM, three-phase parallel wake with a wake journal,
   structured 409 + GPU cooldown, startup / restart placeholders, fault hooks
   (`service_manager.test_hooks`, default off). Notes:
   `RELEASE-20260930-placement-parallel-wake.md` (its compatibility rules apply here).
4. Integration fixes: a restart / bootstrap placeholder conflict keeps the binding a
   suspect (SM); outside-window reconnect skips counted in their own metric
   `tre_reissue_local_reconnect_skipped_total{reason="outside_window"}` (sidecar; the
   re-send counter `tre_reissue_local_reconnect_total` keeps `result=ok|fail` only).

Verified on the branch (2026-10-01): `make check` 2956 passed / 4 skipped;
`make check-redis` 12 passed; Go (`pkg/plugins/gateway cache metrics types utils`,
`-tags nozmq`) all ok; the four parallel-wake / wake-failure / review-fix / GPU-cooldown
test files 3 x green; `make manifests` leaves no diff. The f8ccb0ca parser (controller,
SM `check_service_manager_config`, UI `apply_and_validate`) loads the repo
`registry.yaml`, the `params.yaml` copy and the live registry merged with this release
without an error: an old pod restarted mid-release still starts.

## 0. Calibration gate (unchanged)

The live theta predates `vllm-openai-tre:0.30.0-ts-8dc0f2a7` (8b tokenization -18 %).
This release does not change it: `active active` only for the acceptance items of
section 8 that need it, then back. No TRE-arm experiment until the recalibration on
this release is applied (console `PUT /api/params` + restart of controller and SM).

## Conventions

`REPO=/data/nfs_shared_data/xxy/aibrix`, `WT=$REPO-wt/integ-20261001`. `SHA` = the
integration commit the images are built from, fixed **once, before any build and before
the tag-bump commit** (which moves HEAD): `SHA=$(git -C $WT rev-parse --short=8 HEAD)`,
then written down as a literal. `TAG=$(date +%Y%m%d)-$SHA` (`<YYYYMMDD>-<git short
sha>`). If the owner merges the branch into main first (a `--no-ff` merge whose tree
equals the integration HEAD), build from that main commit instead and use its sha - the
tag sha must always be the commit checked out in the clean clone. Node names come from
the registry, never retyped:

```bash
NODES=$(python3 -c 'import yaml,sys; print(" ".join(n["name"] for n in yaml.safe_load(open(sys.argv[1]))["cluster"]["nodes"]))' $WT/tre/deploy/registry.yaml)
set -- $NODES; N1=$1; N2=$2        # wave order below: the node NOT holding most awake pods first
```

Ssh aliases of the two nodes: `<node-a-ssh>` / `<node-b-ssh>` (see `~/.ssh/config`).

## 1. Images

| Component | Image | Build | Why |
|---|---|---|---|
| gateway plugin | `aibrix/gateway-plugins:$TAG-nozmq2` | 76, **clean clone** | Go code identical to the live f8ccb0ca build (only `ENV_VARS.md` changed); rebuilt so the four control-plane images share one tag and one provenance. Skipping it is allowed (keep `20260930-f8ccb0ca-nozmq2`, then leave its overlay line and guard assert unchanged). |
| service-manager | `tre-v2-service-manager:$TAG` | 76, `tre/` context | parallel wake, journal, placeholders, placement, structured 409, integration fix |
| controller | `tre-v2-controller:$TAG` | 76, `tre/` context | placement policy, GPU cooldown, hinted wakes |
| UI | `tre-v2-ui:$TAG` | 76, `tre/` context | must parse the new `placement:` / `service_manager:` / `reissue:` keys (an old UI rejects them on `PUT /api/params`) |
| model pods | `vllm-openai-tre:0.30.0-ts-8dc0f2a7` | **no rebuild** | no vLLM code change: `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` is read at start by the existing image (`vllm.envs`, default 5); the sidecar code comes from ConfigMap `tre-reissue-sidecar`. The pods must be **recreated** (new env + new sidecar script). Image ID `a96754e185b1` on both nodes (checked 2026-10-01). |

Check the model image on both nodes first:

```bash
for h in <node-a-ssh> <node-b-ssh>; do ssh root@$h "docker image inspect vllm-openai-tre:0.30.0-ts-8dc0f2a7 --format '{{.Id}}'"; done
```

### 1.1 Clean clone (plugin must come from it)

`build_gateway_plugins_nozmq.sh` refuses uncommitted `pkg/ cmd/ go.mod go.sum`, and in a
worktree (`.git` is a file) Go stamps no `vcs.revision`: only the tag would tell the
source. So:

```bash
SRC=/tmp/aibrix-clean-$SHA
git clone --no-hardlinks $REPO $SRC && git -C $SRC checkout --detach $SHA
git -C $SRC status --porcelain                        # must print nothing
nohup $SRC/tre/deploy/scripts/build_gateway_plugins_nozmq.sh aibrix/gateway-plugins:$TAG-nozmq2 \
    > /tmp/build-gwp-$TAG.log 2>&1 &
cd $SRC/tre
for c in service-manager controller ui; do
  nohup docker build -f $c/Dockerfile -t tre-v2-$c:$TAG . > /tmp/build-$c-$TAG.log 2>&1 &
done
# poll `docker images | grep $TAG` (4 images); never wait on the build in the ssh session
docker create --name gwp-check aibrix/gateway-plugins:$TAG-nozmq2 && docker cp gwp-check:/gateway-plugins /tmp/gwp-$TAG && docker rm gwp-check
docker run --rm -v /tmp:/t golang:1.22 go version -m /t/gwp-$TAG | grep -E 'vcs.revision|vcs.modified|-tags'
# expect: vcs.revision=$SHA..., vcs.modified=false, -tags=nozmq
```

### 1.2 Both nodes

The control-plane pods are pinned to one node by the overlays, but a node change or a
rollback must not depend on it: copy all four to the other node and compare IDs.

```bash
for i in aibrix/gateway-plugins:$TAG-nozmq2 tre-v2-service-manager:$TAG tre-v2-controller:$TAG tre-v2-ui:$TAG; do
  docker save $i | ssh root@<other-node-ssh> docker load
  echo "$i $(docker image inspect $i --format '{{.Id}}') $(ssh root@<other-node-ssh> docker image inspect $i --format '{{.Id}}')"
done
```

Do **not** `docker image prune` on 76.

### 1.3 Tag-bump commit (two places each)

One commit on `integ/tre-v2-20261001` that changes only tags:

- `tre/deploy/overlays/tre-v2/gateway-plugins.yaml`, `service-manager.yaml`,
  `controller.yaml`, `ui.yaml`: `20260930-f8ccb0ca` -> `$TAG`;
- guard test `tre/deploy/tests/test_kustomize_overlays.py`: the four `_image(...) ==`
  asserts;
- `make check` green, commit. Then `kubectl diff -f` of the four overlays must show the
  image line only (no env / RBAC / strategy change in this release).

## 2. Backup and rollback.sh (before any apply)

```bash
B=/data/nfs_shared_data/xxy/backups/pre-integ-20261001-$(date +%H%M)
mkdir -p $B
kubectl -n tre-v2 get deploy,ds,svc,cm,sa,role,rolebinding -o yaml > $B/tre-v2-ns.yaml
kubectl -n tre-v2 get deploy tre-gateway-plugins tre-v2-service-manager tre-v2-controller tre-v2-ui -o yaml > $B/control-plane-deploys.yaml
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/live-registry.yaml
kubectl -n default get deploy,svc -l tre.aibrix.io/managed=true -o yaml > $B/default-models.yaml
kubectl -n default get cm tre-reissue-sidecar -o yaml > $B/tre-reissue-sidecar-cm.yaml
kubectl -n tre-v2 get httproute,referencegrant -o yaml > $B/routes.yaml
kubectl -n tre-v2 get envoypatchpolicy,backendtrafficpolicy,envoyextensionpolicy,clienttrafficpolicy -o yaml > $B/tre-v2-envoy-policies.yaml
kubectl -n envoy-gateway-system get svc -o yaml > $B/envoy-gateway-system-svcs.yaml   # NodePort 31094
bash $WT/tre/deploy/scripts/set_run_mode.sh status > $B/run-mode.txt 2>&1
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SAVE
kubectl -n tre-v2 exec deploy/tre-v2-redis -- cat /data/dump.rdb > $B/tre-v2-redis.rdb
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --scan --pattern 'tre:*' | sort > $B/redis-keys.txt
for k in tre:v2:sm:state tre:v2:sm:version tre:v2:sm:desired tre:v2:sm:desired_version tre:v2:sm:observed \
         tre:v2:sm:gpu_leases tre:v2:sm:sleep_reservations tre:v2:sm:sleep_ops tre:v2:controller:mode tre:v2:sm:actuation; do
  echo "== $k $(kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli TYPE $k)"
  kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --no-raw DUMP "$k"
done > $B/redis-key-dumps.txt      # human-readable record; the rdb above is the restore source
cp /data/nfs_shared_data/xxy/backups/pre-integ-20260930-1214/awake_ctl.py $B/   # moves / restores the awake set
git -C $REPO rev-parse main > $B/main-sha.txt; echo $SHA > $B/release-sha.txt
```

`$B/rollback.sh` (reviewed before section 3; `DRY_RUN=1` makes every write a
`--dry-run=server` replace / `--dry-run=client` create and prints the awake moves):

1. `set_run_mode.sh observe observe`.
2. Registry ConfigMap back to `$B/live-registry.yaml`
   (`kubectl create configmap tre-v2-registry --from-file=registry.yaml=$B/live-registry.yaml
   --dry-run=client -o yaml | kubectl replace -f -`) first, so each restored component
   starts once on the old registry (f8ccb0ca refuses the new `placement:` / `reissue:`
   keys; the live registry of this release carries neither unless section 8 wrote
   `test_hooks`, which old parsers ignore).
3. Whole Deployment objects from `$B/control-plane-deploys.yaml` (image + env +
   strategy), metadata / status stripped, `kubectl replace -f`, `rollout status`; order
   controller -> SM (stays Recreate) -> plugin -> UI. Never `kubectl set image` alone.
   The old SM rebuilds the GPU leases at bootstrap (the never-expiring `waking` /
   `starting` leases of the new SM are replaced); the new Redis keys
   (`tre:v2:sm:wake_ops`, restart ledger, fault keys) are ignored by it. A new controller
   with an old SM (or the reverse) may only run in `observe`: the old controller retries
   a 409 `wake_failed`, the old SM ignores `hints` / `avoid_gpus`.
4. Envoy policies: `kubectl replace -f` the objects of `$B/tre-v2-envoy-policies.yaml`
   (tre-v2 namespace only). Restoring the 1 h Envoy default brings the 502/503
   keep-alive race back but is otherwise safe.
5. Model pods (only if a model-side problem): sidecar ConfigMap from
   `$B/tre-reissue-sidecar-cm.yaml`, then the same two-wave delete-then-create as
   section 7 with the Deployments from `$B/default-models.yaml` (per
   `tre.aibrix.io/node`, metadata stripped), awake set moved off each node first.
6. Redis rdb only if the desired state is corrupted.
7. Restore the run mode recorded in `$B/run-mode.txt`.

## 3. Run mode

```bash
cd $WT/tre && bash deploy/scripts/set_run_mode.sh observe observe && bash deploy/scripts/set_run_mode.sh status
```

Checkpoint: both read back `observe`. No wake / sleep is planned by TRE from here to
the end of section 7 except the explicit `awake_ctl.py` moves.

## 4. Control plane: plugin -> SM -> controller -> UI (same batch)

```bash
cd $WT/tre
kubectl apply -f deploy/overlays/tre-v2/gateway-plugins.yaml && kubectl -n tre-v2 rollout status deploy/tre-gateway-plugins
kubectl apply -f deploy/overlays/tre-v2/service-manager.yaml && kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager   # Recreate
kubectl apply -f deploy/overlays/tre-v2/controller.yaml     && kubectl -n tre-v2 rollout status deploy/tre-v2-controller
kubectl apply -f deploy/overlays/tre-v2/ui.yaml             && kubectl -n tre-v2 rollout status deploy/tre-v2-ui
```

Checkpoints (stop and roll back at the first failure):

- plugin: `redis-cli ZRANGE tre:v2:gw:instances 0 -1 WITHSCORES` lists only the new pod,
  score advancing; no panic in its log.
- SM: registry validation OK; bootstrap log shows the lease rebuild (awake + starting +
  journaled waking = none); `GET /v2/wake` answers (empty journal); `GET /v2/state` has
  `gpus[]` / `nodes{}`; supervisor running. A `container_restart_conflict` at bootstrap
  is an alert to read, not a stop (the binding is kept a suspect).
- controller: starts, no unknown-key WARNING, `safescale_config` unchanged from 09-30.
- UI: `GET /api/params` renders; console run mode `observe/observe`.
- `GET /v2/audit` **once** (never polled): healthy.

Rollback point R1: section 2 steps 1, 3, 7 (registry untouched so far).

## 5. Registry ConfigMap (structural merge; never `params.yaml`)

```bash
cd $WT/tre
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > /tmp/live.yaml
PYTHONPATH=common:deploy python3 deploy/scripts/merge_live_registry.py \
    --live /tmp/live.yaml --release deploy/registry.yaml --out /tmp/merged.yaml
diff /tmp/live.yaml /tmp/merged.yaml
kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=/tmp/merged.yaml \
    --dry-run=client -o yaml | kubectl replace -f -
kubectl -n tre-v2 rollout restart deploy/tre-v2-service-manager deploy/tre-v2-controller
kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager && kubectl -n tre-v2 rollout status deploy/tre-v2-controller
```

Dry run 2026-10-01 against the live ConfigMap: "no live tunable differs from the release
file"; the semantic diff is exactly two leaves:

- `vllm.env.VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '75'` (read by the SM's runtime creates; the
  manifests of section 7 already carry it);
- `gateway.upstream_idle_timeout_s: 60`.

All other new keys (`placement.placement_penalty`, `placement.wake_cooldown`,
`reissue.upstream_keepalive_s|local_reconnect_*|server_keepalive_s`,
`service_manager.test_hooks|operations|wake.recovery_unknown_attempts|wake.transport_recheck_s|startup_admission.placeholder_max_s`)
stay commented out: the built-in defaults apply, and a component rolled back to
f8ccb0ca still starts. Write any of them into the live registry only in a later change,
after this release has run for a while (and never before all three images are on it).
Tunables are never set here: console `PUT /api/params` + restart of controller and SM.

Checkpoint: `diff` shows only the two leaves (plus comments); both pods restart once and
log registry OK. Rollback point R2: section 2 step 2.

## 6. Gateway resources (tre-v2 only; never aibrix-system)

```bash
cd $WT/tre/deploy
kubectl diff -f overlays/tre-v2/gateway-extproc.yaml           # expect: 4 x typed_extension_protocol_options idle_timeout 60s in EnvoyPatchPolicy tre-original-dst
kubectl diff -f gateway-hardening/backendtrafficpolicy-tre-v2.yaml   # expect: 3 x timeout.http.connectionIdleTimeout 60s
kubectl apply -f overlays/tre-v2/gateway-extproc.yaml
kubectl apply -f gateway-hardening/backendtrafficpolicy-tre-v2.yaml
```

Both files are namespaced `tre-v2`; `gateway-hardening/backendtrafficpolicy-aibrix-system.yaml`
is **not** applied (ADR-0008). The read-only `kubectl diff` of 2026-10-01 showed exactly
the expected changes.

Checkpoints: `kubectl -n tre-v2 get envoypatchpolicy tre-original-dst -o
jsonpath='{.status}'` Accepted / Programmed; the tre-v2 Envoy config dump
(`kubectl -n envoy-gateway-system port-forward <tre-v2 envoy pod> 19000`, then
`GET /config_dump`) shows `idle_timeout: 60s` on the four `.../rule/original-dst`
clusters; `kubectl -n envoy-gateway-system get svc | grep
31094` still lists the tre-v2 Envoy Service (if Envoy Gateway re-created it with a
random NodePort, patch it back to 31094: known gotcha); one request per model through
31094 returns 200. Rollback point R3: section 2 step 4.

## 7. Model pods: delete then create, two waves by node

Never roll in place (the new pod waits for GPU memory the old one never frees). Stay in
`observe observe`. The sidecar ConfigMap embeds `sidecar.py` and goes first (running
sidecars keep the loaded script; only new pods use the new one):

```bash
cd $WT/tre/deploy/models
kubectl apply -f tre-reissue-sidecar.yaml
for N in $N1 $N2; do
  python3 $B/awake_ctl.py off "$N"          # move the awake set to the other node first (wake there, then sleep here)
  kubectl -n default delete deploy -l tre.aibrix.io/managed=true,tre.aibrix.io/node=$N --wait=true
  kubectl apply $(for f in *-$N-gpu-*.yaml; do printf -- '-f %s ' "$f"; done)
  # wait until the node's 10 pods are 2/2 Running, 0 restarts (startup gate admits them
  # serially per GPU; ~5.5 min per node on 2026-09-30); GET /v2/fleet/state converged;
  # then the next wave. Template: deploy-test-20260930-integ/wave.sh.
done
kubectl apply -k .      # Services / HTTPRoutes / ReferenceGrant; the Deployments must be a no-op
```

Checkpoints per wave: 10/10 Ready, 0 restarts; every engine container has
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`, every sidecar the six new `TRE_REISSUE_*KEEPALIVE*` /
`*RECONNECT*` / `TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S` env vars; the sidecar `/metrics` shows
`tre_reissue_local_reconnect_skipped_total` (new script loaded); SM logs
`startup_placeholder` for each admitted pod and no `startup_placeholder_overdue`; KV
numbers equal to 09-30 (8b 16.76 GiB / 137,248 tokens). A transient `power_mismatch`
in an audit taken during a startup sleep commit is expected (clears within ~1 min).
Gates stalled > 10 min: SM log, `GET /v2/operations`, then `POST /v2/fleet/repair`.

Then:

1. **20/20 Ready** (`kubectl -n default get pods -l tre.aibrix.io/managed=true`).
2. **gpu-truth fresh on both nodes**: for each node `redis-cli TTL tre:gpu_truth:<node>`
   > 0 and `seq` advancing across two reads 10 s apart (compare within one node only;
   node clocks differ by ~160 s). A Running agent with a frozen `seq` = NVML died inside
   the container: `kubectl -n tre-v2 delete pod <gpu-truth pod> --force --grace-period=0`.
3. **Baseline awake layout restored** with `awake_ctl.py restore` (wake on empty GPUs
   first, then sleep the non-target replicas, then the remaining wakes / sleeps):
   7b `<node9>/0`, 8b `<node9>/1`, 14b `<node10>/0,1` (the layout of
   `verify-precalib-20261001/env.sh` `BASELINE_AWAKE`).
4. `GET /v2/audit` **once**: healthy.

Rollback point R4: section 2 step 5 (models) and/or R1-R3.

## 8. Hand-over to the acceptance run

Acceptance: `76:/data/nfs_shared_data/xxy/verify-precalib-20261001/` (`PLAN.md`,
`scripts/`, run order P0-P9, ~1 h 40 min core). Before it:

- F6 uses `CALIB_PREFLIGHT_CMD="python3 -m scripts.calib_preflight ..."` run from
  `$WT/tre/deploy` (or main after the merge).
- F4(c) `--inject` needs the SM fault hooks: add `test_hooks: true` under
  `service_manager:` in the live registry (fetch live -> edit that one key ->
  `kubectl replace`, the same way as section 5), restart SM **and** controller, then
  `F4_INJECT_CMD='kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SETEX
  tre:v2:sm:fault:refuse_wake:<node>/<gpu> 600 1'`, `F4_CLEAR_CMD` = the matching
  `DEL`. **After the acceptance run set it back to false (or remove the key) and restart
  both again**; confirm with `redis-cli --scan --pattern 'tre:v2:sm:fault:*'` empty.
- F5 checks `tre_reissue_local_reconnect_total{result="fail"}`; outside-window skips
  are now in `tre_reissue_local_reconnect_skipped_total` (not a failure).
- Final state after acceptance: `observe active` (the resting mode), baseline layout,
  one audit.

## 9. Known issues to watch

- **Fresh / wiped Redis**: the startup gate 400s because the SM has no desired record:
  `POST /v2/fleet/repair`, reset the desired state (`POST /v2/fleet/seed`), restart the SM.
- **SM scans pods only at start / reload**: a model Deployment created after the SM
  started is invisible to scale calls (200, nothing done) until the next reload - keep
  the order control plane -> models, and restart the SM if the waves had to be redone.
- **NodePort 31094** drifts if Envoy Gateway re-creates its Service: patch it back.
- **gpu-truth NVML** can die silently (pod Running, keys stale): force-delete the pod.
- **Placement conflicts**: the controller now cools a GPU down after a structured 409
  (`gpu_cooldown` / `placement_retry` decision events) instead of re-sending the wake.
- The placement branch's known gap (`avoid_gpus` is a snapshot) and the partial-rollback
  rules of `RELEASE-20260930-placement-parallel-wake.md` apply.

## 10. Time estimate

| Step | Wall clock | Cluster impact |
|---|---|---|
| builds (parallel) + save/load to the other node + tag bump + `make check` | 20-25 min | none |
| backup + rollback.sh dry run | 5-10 min | none |
| run mode + control plane | ~5 min | observe only |
| registry merge + restart | ~3 min | observe only |
| gateway resources | ~3 min | idle-timeout change on live traffic |
| model waves (2 x ~6 min + awake moves) | ~15 min | serving capacity halves per wave |
| gpu-truth, baseline restore, audit | ~5 min | |
| **total** | **~60 min** (cluster-touching part ~30-35 min) | |

After acceptance: record image IDs and the deployed sha in `HANDOFF.md`. Merging
`integ/tre-v2-20261001` into main and pushing are separate, owner-approved steps.
