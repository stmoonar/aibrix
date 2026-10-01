# Release 2026-10-01: integration (calibration chat sender + sidecar keep-alive + placement / parallel wake + unified client) - DRAFT

Plan only: nothing here has been built or applied. Confirm with the owner before any
step that touches the cluster (a parallel session may be using it). The cluster-touching
part runs inside an exclusive window (section W, opened before section 2 and closed at
the end of section 8).

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
5. `--no-ff` merge of `feat/unified-client-20260930` (09efa281, based on dcb8d5f3; T5):
   one sending core (SSE parser, pooled async transport, request profiles), a
   process-pool open loop (`procpool`: N processes x asyncio, pre-sharded absolute-time
   sending) under `openloop.py`, `loadgen_v1` reduced to a shell over profile `e1_v1`
   (v1's request: chat, stream, no `ignore_eos`, `max_tokens` from the trace; the old v1
   sender is removed), and `campaign_queue` manifests with `client_profile` (default
   `e1_v1`; `replay` = the pre-2026-10-01 campaign request, completions + `ignore_eos`).
   Host-side tooling only: no image, registry, overlay or manifest change. Client
   numbers of the old senders (late sends under bursts, TTFT ~125 ms high) are not
   comparable with the new ones.

Verified on the branch (2026-10-01, after the T5 merge): `make check` 3008 passed / 4 skipped;
`make check-redis` 12 passed; Go (`pkg/plugins/gateway cache metrics types utils`,
`-tags nozmq`) all 19 packages ok; the four parallel-wake / wake-failure / review-fix / GPU-cooldown
test files green (3 x before T5, 1 x after); the T5 multi-process / equivalence tests
(`replayer/tests/test_unified_client*.py`, `deploy/tests/test_openloop*.py`,
`deploy/tests/test_probe_label_parity.py`, `loadgen_v1/tests`) green; `make manifests`
leaves no diff. The f8ccb0ca parser (controller,
SM `check_service_manager_config`, UI `apply_and_validate`) loads the repo
`registry.yaml`, the `params.yaml` copy and the live registry merged with this release
without an error: an old pod restarted mid-release still starts.

Release tooling (`tre/deploy/scripts/release/`, no built-in paths / dates / node names;
all read-only unless stated, `DRY_RUN=1` where they write):

| Script | Use |
|---|---|
| `release_wave.sh <node>` | one model wave: manifest self-check (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`), awake set off the node, delete, wait for the pods to go, apply, wait Ready, wave checkpoint. Needs `WT`, `B`. |
| `release_checks.py` | `manifests` (self-check), `wave-checkpoint` (no `starting` lease / admission annotation / running operation / wake journal entry), `gpu-truth` (TTL + `seq` advancing per node) |
| `awake_ctl.py` | `show`, `off <node>`, `restore <sm-state.json>`, `restore-ids <id>...`, `expect <sm-state.json> <id>...` (SM API) |
| `redis_state.py` | `dump` / `restore` of the durable Redis keys (DUMP / RESTORE REPLACE), `delete-keys` |
| `registry_set.py` | set / remove one structural registry key, validated by the release's parser before it is written |
| `strip_obj.py` | strip server fields from backed-up objects (`--all` for a whole List) |
| `rollback.sh` | generic rollback, copied into `$B` and run from there (section 2) |

Self-tested 2026-10-01 against the live cluster in read-only / server-dry-run mode:
`release_wave.sh` (`DRY_RUN=1`, node10), `rollback.sh` (`DRY_RUN=1`, incl.
`RESTORE_REDIS=1`), `gpu-truth`, `wave-checkpoint`, `awake_ctl.py expect`,
`registry_set.py` (set / unset round trip equal; an unknown `placement:` key rejected),
`redis_state.py dump` (37 durable keys). `manifests` fails, as intended, on main's
manifests (no keep-alive env).

## 0. Calibration gate (unchanged)

The live theta predates `vllm-openai-tre:0.30.0-ts-8dc0f2a7` (8b tokenization -18 %).
This release does not change it: `active active` only for the acceptance items of
section 8 that need it, then back. No TRE-arm experiment until the recalibration on
this release is applied (console `PUT /api/params` + restart of controller and SM).

## Conventions

```bash
REPO=/data/nfs_shared_data/xxy/aibrix
WT=$REPO-wt/integ-20261001            # or the clean clone of section 1.1 (same commit + tag bump)
VP=/data/nfs_shared_data/xxy/verify-precalib-20261001    # acceptance run (section 8)
SHA=<fixed once, see below>; TAG=<YYYYMMDD>-$SHA
S=$WT/tre/deploy/scripts/release
NODES=$(python3 -c 'import yaml,sys; print(" ".join(n["name"] for n in yaml.safe_load(open(sys.argv[1]))["cluster"]["nodes"]))' $WT/tre/deploy/registry.yaml)
. $VP/env.sh                          # BASELINE_AWAKE (7b node9/0, 8b node9/1, 14b node10/0,1), EXPECT_CP_TAG, ...
export EXPECT_CP_TAG=$TAG             # acceptance G6 then checks the four control-plane tags
```

`SHA` = the commit the images are built from, fixed **once, before any build and before
the tag-bump commit** (which moves HEAD): `SHA=$(git -C $WT rev-parse --short=8 HEAD)`,
written down as a literal. `TAG=$(date +%Y%m%d)-$SHA`. If the owner merges the branch
into main first (a `--no-ff` merge whose tree equals the integration HEAD), build from
that main commit and use its sha: the tag sha is always the commit checked out in the
clean clone. Temporary files carry the tag: `/tmp/<what>-$TAG.*`. Ssh aliases of the
two nodes: `<node9-ssh>` / `<node10-ssh>` (`~/.ssh/config`).

**Wave order: node10 first, then node9** (derived, not typed):

```bash
WAVE1=$(for b in $BASELINE_AWAKE; do echo "$b"; done | awk -F/ '$3 ~ /,/ {print $2}')   # node of the baseline TP-2 (14b) binding
WAVE2=$(for n in $NODES; do [ "$n" != "$WAVE1" ] && echo "$n"; done)
echo "wave 1: $WAVE1   wave 2: $WAVE2"     # expect node10, then node9
```

Why: in the baseline node10 holds only the 14b (GPUs 0,1) awake, so wave 1 needs one
move (wake 14b on node9/2,3 - free in the baseline - then sleep node10/0,1); wave 2
then moves the whole awake set onto wave 1's freshly created pods, a live check of the
new manifests before node9 is touched. The 2026-09-30 release used the same order.

## W. Exclusive window (open before section 2, close at the end of section 8)

```bash
WIN=/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW
[ -e $WIN ] && { cat $WIN; echo "window held - stop and ask the owner"; }
[ -e $WIN ] || echo "validation $(date -Iseconds) $(date -Iseconds -d '+100 min')" > $WIN
cat $WIN      # one line: validation <start ISO> <expected end ISO>
```

Parallel sessions do not start tests / load while it exists. If the deployment overruns,
rewrite the line with a new end time. Remove it (`rm -f $WIN`) at the end of section 8 -
or after `rollback.sh` finished, if the release is rolled back. Builds and the tag-bump
`make check` (section 1) run **before** the window is opened.

## 1. Images

| Component | Image | Build | Why |
|---|---|---|---|
| gateway plugin | `aibrix/gateway-plugins:$TAG-nozmq2` | 76, **clean clone** | Go code identical to the live f8ccb0ca build (only `ENV_VARS.md` changed); **rebuilt anyway** so the four control-plane images share one tag and one provenance (`EXPECT_CP_TAG=$TAG` of the acceptance run requires it) |
| service-manager | `tre-v2-service-manager:$TAG` | 76, `tre/` context | parallel wake, journal, placeholders, placement, structured 409, integration fix |
| controller | `tre-v2-controller:$TAG` | 76, `tre/` context | placement policy, GPU cooldown, hinted wakes |
| UI | `tre-v2-ui:$TAG` | 76, `tre/` context | must parse the new `placement:` / `service_manager:` / `reissue:` keys (an old UI rejects them on `PUT /api/params`) |
| model pods | `vllm-openai-tre:0.30.0-ts-8dc0f2a7` | **no rebuild** | no vLLM code change: `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` is read at start by the existing image (`vllm.envs`, default 5); the sidecar code comes from ConfigMap `tre-reissue-sidecar`. The pods must be **recreated** (new env + new sidecar script). Image ID `a96754e185b1` on both nodes (checked 2026-10-01). |

Check the model image on both nodes first:

```bash
for h in <node9-ssh> <node10-ssh>; do ssh root@$h "docker image inspect vllm-openai-tre:0.30.0-ts-8dc0f2a7 --format '{{.Id}}'"; done
```

### 1.1 Clean clone (the plugin must come from it)

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
docker create --name gwp-check-$TAG aibrix/gateway-plugins:$TAG-nozmq2 && docker cp gwp-check-$TAG:/gateway-plugins /tmp/gwp-$TAG && docker rm gwp-check-$TAG
docker run --rm -v /tmp:/t golang:1.22 go version -m /t/gwp-$TAG | grep -E 'vcs.revision|vcs.modified|-tags'
# expect: vcs.revision=$SHA..., vcs.modified=false, -tags=nozmq
```

### 1.2 Both nodes

The control-plane pods are pinned to one node by the overlays, but a node change or a
rollback must not depend on it: copy all four to the other node and compare IDs.

```bash
for i in aibrix/gateway-plugins:$TAG-nozmq2 tre-v2-service-manager:$TAG tre-v2-controller:$TAG tre-v2-ui:$TAG; do
  docker save $i | ssh root@<node9-ssh> docker load
  echo "$i $(docker image inspect $i --format '{{.Id}}') $(ssh root@<node9-ssh> docker image inspect $i --format '{{.Id}}')"
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

## 2. Backup and rollback.sh (window open; before any apply)

```bash
B=/data/nfs_shared_data/xxy/backups/pre-integ-$(date +%Y%m%d-%H%M)
mkdir -p $B && cp $S/rollback.sh $S/strip_obj.py $S/awake_ctl.py $S/redis_state.py $B/
SM=http://$(kubectl -n tre-v2 get svc tre-v2-service-manager -o jsonpath='{.spec.clusterIP}'):8000
kubectl -n tre-v2 get deploy,ds,svc,cm,sa,role,rolebinding -o yaml > $B/tre-v2-ns.yaml
kubectl -n tre-v2 get deploy tre-gateway-plugins tre-v2-service-manager tre-v2-controller tre-v2-ui -o yaml > $B/control-plane-deploys.yaml
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/live-registry.yaml
kubectl -n default get deploy,svc -l tre.aibrix.io/managed=true -o yaml > $B/default-models.yaml
kubectl -n default get cm tre-reissue-sidecar -o yaml > $B/tre-reissue-sidecar-cm.yaml
kubectl -n tre-v2 get httproute,referencegrant -o yaml > $B/routes.yaml
kubectl -n tre-v2 get envoypatchpolicy,backendtrafficpolicy,envoyextensionpolicy,clienttrafficpolicy -o yaml > $B/tre-v2-envoy-policies.yaml
kubectl -n envoy-gateway-system get svc -o yaml > $B/envoy-gateway-system-svcs.yaml   # NodePort 31094
bash $WT/tre/deploy/scripts/set_run_mode.sh status > $B/run-mode.txt 2>&1
curl -sf $SM/v2/state > $B/sm-state.json                      # awake set before the release (awake_ctl restore)
curl -sf $SM/v2/fleet/state > $B/sm-fleet-state.json
python3 $B/awake_ctl.py expect $B/sm-state.json $BASELINE_AWAKE   # must print "matches"; else stop and ask the owner
# Redis: durable keys (the restore source) + an RDB copy (forensics / second source)
python3 $B/redis_state.py dump $B/redis-state.json            # keys under tre: without a TTL
R="kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli"
L0=$($R LASTSAVE); $R BGSAVE; while [ "$($R LASTSAVE)" = "$L0" ]; do sleep 1; done
kubectl -n tre-v2 exec deploy/tre-v2-redis -- cat /data/dump.rdb > $B/tre-v2-redis.rdb
$R --scan --pattern 'tre:*' | sort > $B/redis-keys.txt
git -C $REPO rev-parse main > $B/main-sha.txt; echo $SHA > $B/release-sha.txt
DRY_RUN=1 bash $B/rollback.sh > $B/rollback-dryrun.log 2>&1; tail -3 $B/rollback-dryrun.log
```

**Redis facts (checked 2026-10-01):** Deployment `tre-v2-redis`, one replica, **no
volume** (`/data` is the container's own layer), `save 3600 1 300 100 60 10000`,
`appendonly no`, `enable-debug-command no`. So an RDB copied into the pod does not come
back through a restart (the restarted container starts on an empty `/data`), and
`DEBUG RELOAD` is unavailable: the state is restored key by key with
`redis_state.py restore` (DUMP / RESTORE REPLACE), which `rollback.sh` does with
`RESTORE_REDIS=1` while controller and SM are scaled to 0. To use the RDB instead:
`docker run -d --rm --name redis-rdb-$TAG -p 127.0.0.1:16399:6379 -v <dir with the
rdb as dump.rdb>:/data redis:7.2-alpine`, then `REDIS_URL=redis://127.0.0.1:16399/0
python3 $B/redis_state.py dump /tmp/redis-from-rdb-$TAG.json` and restore that file.

`$B/rollback.sh` (`tre/deploy/scripts/release/rollback.sh`, reads only `$B`; review it
and its dry-run log before section 3). What it does, in order:

1. Both switches `observe` (Redis `tre:v2:controller:mode`, `tre:v2:sm:actuation`).
2. Registry ConfigMap back to `$B/live-registry.yaml` first, so each restored component
   starts once on the old registry (f8ccb0ca refuses the new `placement:` / `reissue:`
   keys; the live registry of this release carries none of them - `test_hooks` of
   section 8 is ignored by old parsers).
3. Whole Deployment objects from `$B/control-plane-deploys.yaml` (image + env +
   strategy), metadata / status stripped (`strip_obj.py`), `kubectl replace -f`,
   `rollout status`; controller -> SM (Recreate) -> plugin -> UI. Never `kubectl set
   image` alone. The old SM rebuilds the GPU leases at bootstrap (the never-expiring
   `waking` / `starting` leases of the new SM are replaced).
   **Once the old SM runs**, it deletes `tre:v2:sm:wake_ops`, `tre:v2:sm:wake_stats`,
   `tre:v2:sm:restart_seen` and SCANs away `tre:v2:sm:fault:*` (names =
   `tre_common.rediskeys` `SM_WAKE_OPS_KEY`, `SM_WAKE_STATS_KEY`, `SM_RESTART_SEEN_KEY`,
   `SM_FAULT_KEY_PREFIX`). Why: the old SM neither reads nor updates them, so they only
   go stale; a later re-upgrade would recover wakes settled long ago (wake journal),
   compare container restart counts with outdated values (spurious restart placeholders)
   and honour a leftover fault key as soon as `test_hooks` is on. Deleting them only
   after the old SM is up guarantees the new SM (Recreate) no longer writes them.
   A new controller with an old SM (or the reverse) may only run in `observe`: the old
   controller retries a 409 `wake_failed`, the old SM ignores `hints` / `avoid_gpus`.
4. (`RESTORE_REDIS=1`, only if the desired state is corrupted) controller + SM scaled to
   0, `redis_state.py restore $B/redis-state.json`, scaled back to 1.
5. Envoy policies of `tre-v2` from `$B/tre-v2-envoy-policies.yaml`, stripped with
   `strip_obj.py --all` (resourceVersion / uid / managedFields / last-applied / status
   dropped; the script refuses an object outside `tre-v2`), `kubectl replace -f`.
   Restoring Envoy's 1 h default brings the keep-alive race back but is otherwise safe.
6. Model pods (skip with `SKIP_MODELS=1`): sidecar ConfigMap from the backup, then per
   node (default order: `cluster.nodes` of the backed-up registry; pass
   `NODES="$WAVE1 $WAVE2"`) awake set off, delete, `kubectl wait --for=delete`, create
   the stripped backed-up Deployments, wait Ready.
7. Awake set back to `$B/sm-state.json` (`awake_ctl.py restore`).
8. Then by hand: `GET /v2/audit` once, run mode from `$B/run-mode.txt`
   (`set_run_mode.sh <controller> <sm>`), `rm -f $WIN`.

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
- all four images = `$TAG` (`kubectl -n tre-v2 get deploy -o wide`).
- `GET /v2/audit` **once** (never polled): healthy.

Rollback point R1: `rollback.sh` with `SKIP_MODELS=1 SKIP_ENVOY=1` (registry untouched
so far, so its step 2 is a no-op).

## 5. Registry ConfigMap (structural merge; never `params.yaml`)

```bash
cd $WT/tre
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > /tmp/live-registry-$TAG.yaml
PYTHONPATH=common:deploy python3 deploy/scripts/merge_live_registry.py \
    --live /tmp/live-registry-$TAG.yaml --release deploy/registry.yaml --out /tmp/merged-registry-$TAG.yaml
diff /tmp/live-registry-$TAG.yaml /tmp/merged-registry-$TAG.yaml
kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=/tmp/merged-registry-$TAG.yaml \
    --dry-run=client -o yaml | kubectl replace -f -
kubectl -n tre-v2 rollout restart deploy/tre-v2-service-manager deploy/tre-v2-controller deploy/tre-v2-ui
for d in tre-v2-service-manager tre-v2-controller tre-v2-ui; do kubectl -n tre-v2 rollout status deploy/$d; done
```

The UI is restarted with them: it validates `PUT /api/params` against the registry it
read at start.

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

Checkpoint: `diff` shows only the two leaves (plus comments); the three pods restart
once and log registry OK. Rollback point R2: `rollback.sh` with `SKIP_MODELS=1
SKIP_ENVOY=1`.

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
clusters; `kubectl -n envoy-gateway-system get svc | grep 31094` still lists the tre-v2
Envoy Service (if Envoy Gateway re-created it with a random NodePort, patch it back to
31094: known gotcha); one request per model through 31094 returns 200. Rollback point
R3: `rollback.sh` with `SKIP_MODELS=1` (or only step 5 by hand).

## 7. Model pods: delete then create, two waves by node

Never roll in place (the new pod waits for GPU memory the old one never frees). Stay in
`observe observe`.

**7.0 gpu-truth fresh before any wave** (the SM's wake / create gates read it):

```bash
python3 $S/release_checks.py gpu-truth --nodes "$NODES"
```

Every node `OK` (TTL > 0, `seq` advancing; each node compared with itself only - node
clocks differ by ~160 s). A Running agent with a frozen `seq` = NVML died inside the
container: `kubectl -n tre-v2 delete pod <gpu-truth pod of that node> --force
--grace-period=0`, wait for the new pod, re-run the check.

**7.1 sidecar ConfigMap** (embeds `sidecar.py`; running sidecars keep the script they
loaded, only new pods use the new one):

```bash
kubectl apply -f $WT/tre/deploy/models/tre-reissue-sidecar.yaml
```

**7.2 waves** (`release_wave.sh`, parametrised by `WT` and `B`; do not reuse the
2026-09-30 `wave.sh`, which has its paths built in):

```bash
export WT B
DRY_RUN=1 bash $S/release_wave.sh $WAVE1          # dry run first: plan + server dry-run
bash $S/release_wave.sh $WAVE1                     # ~6 min; log $B/wave-$WAVE1.log
bash $S/release_wave.sh $WAVE2                     # only after wave 1 printed "done"
kubectl apply -k $WT/tre/deploy/models             # Services / HTTPRoutes / ReferenceGrant; Deployments must be a no-op
```

Each wave: (1) `release_checks.py manifests` refuses to go on unless every Deployment
of the node has `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75` in its engine container (override /
extend with `EXPECT_ENV`); (2) `awake_ctl.py off`; (3) delete the Deployments, then
`kubectl wait --for=delete pod -l tre.aibrix.io/managed=true,tre.aibrix.io/node=$N`;
(4) apply, wait until all pods are 2/2 Running (startup gate admits them serially per
GPU; ~5.5 min per node on 2026-09-30); (5) **wave checkpoint**
(`release_checks.py wave-checkpoint --node $N`, retried up to 10 min), all must hold
before the next wave:

- `HGETALL tre:v2:sm:gpu_leases` has no lease with `"phase":"starting"` on the node;
- no pod of the node still has the annotation `tre.aibrix.io/startup-admitted-uid`
  (the SM clears it when the startup converged);
- `GET /v2/operations` has no `status: running` operation;
- the wake journal `tre:v2:sm:wake_ops` is empty.

**Stuck checkpoint** (a starting lease / admission annotation that does not clear within
10 min): read the SM log for that binding, then **delete that pod** (`kubectl -n default
delete pod <pod>`; its Deployment re-creates it and the startup gate admits it again).
Why deleting the pod is the fix: a `starting` lease never expires (TTL 0 since
2026-09-30, a cold vLLM load outlasts any fixed TTL) and `POST /v2/fleet/repair` does not
release it; it is released only when the placeholder converges or its pod is gone
(`reap_orphan_starting_leases`), or when the engine container is not running and does
not read awake (`reap_stale_startup_placeholders`). A running operation that does not
finish: `GET /v2/operations/<id>`, SM log; the supervisor recovers stale ones.

Per-wave extra checks: every sidecar has the six new `TRE_REISSUE_*KEEPALIVE*` /
`*RECONNECT*` / `TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S` env vars; the sidecar `/metrics`
shows `tre_reissue_local_reconnect_skipped_total` (new script loaded); SM logs
`startup_placeholder` for each admitted pod and no `startup_placeholder_overdue`; KV
numbers equal to 09-30 (8b 16.76 GiB / 137,248 tokens). A transient `power_mismatch`
in an audit taken during a startup sleep commit is expected (clears within ~1 min).
Gates stalled > 10 min: SM log, `GET /v2/operations`, then `POST /v2/fleet/repair`.

**7.3 after both waves:**

1. **20/20 Ready** (`kubectl -n default get pods -l tre.aibrix.io/managed=true`);
   `release_checks.py wave-checkpoint` (all nodes) passes.
2. **gpu-truth fresh** again: `python3 $S/release_checks.py gpu-truth --nodes "$NODES"`.
3. **Baseline awake layout restored** from the backup - it was checked equal to
   `BASELINE_AWAKE` in section 2 (7b node9/0, 8b node9/1, 14b node10/0,1):
   ```bash
   python3 $S/awake_ctl.py expect $B/sm-state.json $BASELINE_AWAKE   # guard: the file is the baseline
   python3 $S/awake_ctl.py restore $B/sm-state.json --dry-run
   python3 $S/awake_ctl.py restore $B/sm-state.json
   curl -sf $SM/v2/state > /tmp/sm-state-after-$TAG.json
   python3 $S/awake_ctl.py expect /tmp/sm-state-after-$TAG.json $BASELINE_AWAKE   # must match
   ```
4. `GET /v2/audit` **once**: healthy.
5. **Resting run mode** (the acceptance run P0 expects it):
   ```bash
   bash $WT/tre/deploy/scripts/set_run_mode.sh observe active
   bash $WT/tre/deploy/scripts/set_run_mode.sh status     # must read controller_mode=observe sm_actuation=active
   ```

Rollback point R4: `rollback.sh` (models included, `NODES="$WAVE1 $WAVE2"`).

## 8. Hand-over to the acceptance run

Acceptance: `$VP` (`PLAN.md`, `scripts/`, run order P0-P9, ~1 h 40 min core), started
in `observe active` with `EXPECT_CP_TAG=$TAG` exported. Before / during it:

- **Client = this checkout.** The smoke (`$VP/scripts/A_smoke.sh`, derived from
  `smoke-e1-20260930/tools/run_arm.sh`) still runs `python3 -m tre_loadgen_v1 --stage all`
  with only `tre/loadgen_v1` on `PYTHONPATH`; since T5 that is a shell that sends through
  the sibling `tre/replayer` (profile `e1_v1`, procpool). It sends with the unified
  client only if `TRE_DIR` is a checkout containing T5: `env.sh` defaults to
  `$TRE_REPO/tre` = main, which still has the old v1 sender until the branch is merged.
  So export `TRE_DIR=$WT/tre` (or merge first), and do not put another `tre_replayer` on
  `PYTHONPATH` (the shell warns and records which one it used). The 09-30
  `smoke-e1-20260930/tools/run_arm.sh` (outside the repo) takes the same override since
  2026-10-01: `TRE_DIR=$WT/tre bash run_arm.sh <arm> ...` (default unchanged: main;
  backup `run_arm.sh.bak-20261001`).
  Client-side latencies of this run are not comparable with 09-30 smoke numbers (old
  sender) - compare server-side and error counts instead.
- **Campaign request.** `campaign_queue` manifests without `client_profile` now send
  `e1_v1` for every arm (TRE / APA / E1); a manifest that must reproduce an older
  campaign request sets `"client_profile": "replay"`. The run's `command.json` and
  `run_trace_summary.json` record the profile.

- F6 uses `CALIB_PREFLIGHT_CMD="python3 -m scripts.calib_preflight ..."` run from
  `$WT/tre/deploy` (or main after the merge).
- F5 checks `tre_reissue_local_reconnect_total{result="fail"}`; outside-window skips
  are now in `tre_reissue_local_reconnect_skipped_total` (not a failure).

**`service_manager.test_hooks` (F4(c) `--inject` only).** It is a structural key (no
console path; `PUT /api/params` cannot set it) and only the SM reads it. Turn it on:

```bash
cd $WT/tre
bash deploy/scripts/set_run_mode.sh observe observe            # restart the SM only while nothing acts
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > /tmp/live-registry-hooks-$TAG.yaml
PYTHONPATH=common:deploy python3 $S/registry_set.py --in /tmp/live-registry-hooks-$TAG.yaml \
    --out /tmp/registry-hooks-on-$TAG.yaml --set service_manager.test_hooks=true   # validated by THIS release's parser
diff /tmp/live-registry-hooks-$TAG.yaml /tmp/registry-hooks-on-$TAG.yaml      # only test_hooks: true
kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=/tmp/registry-hooks-on-$TAG.yaml \
    --dry-run=client -o yaml | kubectl replace -f -
kubectl -n tre-v2 rollout restart deploy/tre-v2-service-manager && kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager
# F4: F4_INJECT_CMD='kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SETEX tre:v2:sm:fault:refuse_wake:<node>/<gpu> 600 1'
#     F4_CLEAR_CMD='kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli DEL tre:v2:sm:fault:refuse_wake:<node>/<gpu>'
```

Only the SM is restarted (the controller does not read the key; in `observe observe`
a SM restart interrupts no action). Turn it off right after F4(c), the same way with
`--unset service_manager.test_hooks` (removing the key = default false), `replace`,
SM restart, then `redis-cli --scan --pattern 'tre:v2:sm:fault:*'` must print nothing
(`python3 $B/redis_state.py delete-keys --scan 'tre:v2:sm:fault:*'` otherwise) and the
run mode goes back to what the acceptance step needs. Should it be left on by mistake,
the next `merge_live_registry.py` drops it anyway (the merge keeps only the console
tunables of the live registry; `test_hooks` is not one), but do not rely on that.

End of the deployment (acceptance may keep its own window): after section 7.3 passed,
`rm -f $WIN` unless the acceptance run follows immediately under the same window (then
rewrite its end time and remove it after P9).

## 9. Known issues to watch

- **Fresh / wiped Redis**: the startup gate 400s because the SM has no desired record:
  `POST /v2/fleet/repair`, reset the desired state (`POST /v2/fleet/seed`), restart the
  SM.
- **SM scans pods only at start / reload**: a model Deployment created after the SM
  started is invisible to scale calls (200, nothing done) until the next reload - keep
  the order control plane -> models, and restart the SM if the waves had to be redone.
- **NodePort 31094** drifts if Envoy Gateway re-creates its Service: patch it back.
- **gpu-truth NVML** can die silently (pod Running, keys stale): force-delete the pod
  (section 7.0).
- **Starting leases never expire**: a stuck startup is cleared by deleting its pod,
  not by fleet repair (section 7.2).
- **Placement conflicts**: the controller now cools a GPU down after a structured 409
  (`gpu_cooldown` / `placement_retry` decision events) instead of re-sending the wake.
- The placement branch's known gap (`avoid_gpus` is a snapshot) and the partial-rollback
  rules of `RELEASE-20260930-placement-parallel-wake.md` apply.

## 10. Time estimate

| Step | Wall clock | Cluster impact |
|---|---|---|
| builds (parallel) + save/load to the other node + tag bump + `make check` | 20-25 min | none (before the window) |
| window + backup + rollback.sh dry run | 5-10 min | none |
| run mode + control plane | ~5 min | observe only |
| registry merge + restart (SM, controller, UI) | ~3 min | observe only |
| gateway resources | ~3 min | idle-timeout change on live traffic |
| gpu-truth + model waves (2 x ~6 min + awake moves + checkpoints) | ~20 min | serving capacity halves per wave |
| gpu-truth, baseline restore, audit, `observe active` | ~5 min | |
| **total** | **~65 min** (cluster-touching part ~35-40 min) | |

After acceptance: record image IDs and the deployed sha in `HANDOFF.md`. Merging
`integ/tre-v2-20261001` into main and pushing are separate, owner-approved steps.
