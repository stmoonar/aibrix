# Release 2026-09-30: integration (serve args + drain / floor / SafeScale evidence + tokenizer fix) - DRAFT

Plan only: nothing here has been built or applied. Confirm with the owner before any
step that touches the cluster (a parallel session may be using it).

Branch `integ/tre-v2-20260930` (worktree `aibrix-wt/integ-20260930`), from main
b9cf38d9:

1. `--no-ff` merge of `feat/serve-args-v1-align-20260929` (faa979f4; the live model pods
   already run its manifests: `max_model_len: null`, 14b `--enable-chunked-prefill
   --max-num-batched-tokens 2048`);
2. `--no-ff` merge of `feat/safescale-evidence-20260929` (bd2ee23c; contains
   drain-policy -> floor-probe-window -> continuable-contract -> safescale-evidence);
3. vLLM image `vllm-openai-tre:0.30.0-ts-2be2d647` -> `vllm-openai-tre:0.30.0-ts-8dc0f2a7`
   (tokenizer byte-level BPE fix, fork commit 8dc0f2a7) for all three models.

Component notes, compatibility and per-feature rollback details:
`RELEASE-20260929-floor-probe-window.md`, `RELEASE-20260929-safescale-evidence.md`,
`README.md` (SafeScale probe window, action queue, sleep paths).

## 0. Calibration gate (theta) - read first

A new vLLM image invalidates the calibrated theta (registry comment, plan D9; AGENTS:
after an image change theta, lambda, tau and w_p are recalibrated and updated atomically
together with the controller). The live theta was fitted on 2026-09-24, before the 0.30
migration; this release adds the tokenizer fix (8b's byte-level BPE tokenization changes,
which moves the TSS numerator) on top of the 14b chunked prefill that is already live.
Therefore:

- `active active` is used **only** for the smoke / acceptance steps 6.5-6.7 below;
  afterwards controller and SM go back to `observe observe`.
- **No TRE-arm experiment runs on this release until theta / lambda / tau / w_p are
  recalibrated on `vllm-openai-tre:0.30.0-ts-8dc0f2a7`** (then applied through console
  `PUT /api/params` + restart of controller and SM together), or the owner records an
  explicit waiver in `HANDOFF.md`.
- Evidence for the 8b decision: step 6.1 also compares token counts of the old and the
  new image's tokenizer on the calibration prompts.

## Conventions

Throughout: `REPO=/data/nfs_shared_data/xxy/aibrix`, `WT=$REPO-wt/integ-20260930`
(sibling of the repo: `/data/nfs_shared_data/xxy/aibrix-wt/integ-20260930`), `SHA` = the
integration commit the images are built from, **fixed once before any build and before
the tag-bump commit of 1.3** (`SHA=$(git -C $WT rev-parse --short=8 HEAD)` evaluated
now, then written down as a literal; the tag-bump commit moves HEAD),
`TAG=$(date +%Y%m%d)-$SHA`. Node names below come from the registry
(`cluster.nodes[].name`); read them, do not retype them:

```bash
NODES=$(PYTHONPATH=$WT/tre/common python3 -c 'import yaml,sys; print(" ".join(n["name"] for n in yaml.safe_load(open(sys.argv[1]))["cluster"]["nodes"]))' $WT/tre/deploy/registry.yaml)
```

## 1. Images

| Component | Image | Built where | Notes |
|---|---|---|---|
| model pods | `vllm-openai-tre:0.30.0-ts-8dc0f2a7` | already present | ID `a96754e185b1` on both nodes; no build |
| gateway plugin | `aibrix/gateway-plugins:$TAG-nozmq2` | 76, **clean clone** (below) | Go code changed (continuability contract, `tre_transparent_sleep.go`) |
| service-manager | `tre-v2-service-manager:$TAG` | 76, `tre/` context | replica floor, no-drain sleep paths, startup admission |
| controller | `tre-v2-controller:$TAG` | 76, `tre/` context | SafeScale direct evidence, window W, floor client |
| UI | `tre-v2-ui:$TAG` | 76, `tre/` context | no UI code change; rebuilt so the console validates `PUT /api/params` with the same `tre_common` registry parser as SM / controller |

Check the vLLM image on both nodes before anything else:

```bash
for h in <node9-ssh-host> <node10-ssh-host>; do ssh root@$h "docker image inspect vllm-openai-tre:0.30.0-ts-8dc0f2a7 --format '{{.Id}}'"; done
# both: sha256:a96754e185b1...
```

The control-plane pods (plugin, SM, controller, UI) are pinned by the overlays to one
node (`kubernetes.io/hostname` nodeSelector, currently node10): build there; no
`docker save | docker load` needed unless that selector changes.

### 1.1 Gateway plugin: clean clone, not the worktree

Why: (a) `build_gateway_plugins_nozmq.sh` refuses a tree with uncommitted changes under
`pkg/ cmd/ go.mod go.sum`; (b) in a git worktree (`.git` is a file) Go silently stamps
**no** `vcs.revision` / `vcs.modified` into the binary, so `go version -m` could not
prove which commit the image contains - only a primary checkout gets the stamp. Build
from a fresh clone checked out at the integration commit:

```bash
SRC=/tmp/aibrix-clean-$SHA
git clone --no-hardlinks $REPO $SRC
git -C $SRC checkout --detach $SHA
git -C $SRC status --porcelain            # must print nothing
nohup $SRC/tre/deploy/scripts/build_gateway_plugins_nozmq.sh aibrix/gateway-plugins:$TAG-nozmq2 \
    > /tmp/build-gwp-$TAG.log 2>&1 &
# poll: docker images | grep $TAG ; then verify the stamp:
docker create --name gwp-check aibrix/gateway-plugins:$TAG-nozmq2 && docker cp gwp-check:/gateway-plugins /tmp/gwp-$TAG && docker rm gwp-check
go version -m /tmp/gwp-$TAG | grep -E 'vcs.revision|vcs.modified|-tags'   # revision = $SHA..., modified=false, nozmq
```

### 1.2 SM / controller / UI

Build context must be `tre/`. The worktree is fine for these (Python; provenance is the
tag), but building from the same clean clone keeps one source for all four images:

```bash
cd $SRC/tre
for c in service-manager controller ui; do
  nohup docker build -f $c/Dockerfile -t tre-v2-$c:$TAG . > /tmp/build-$c-$TAG.log 2>&1 &
done
# poll: docker images | grep $TAG   (4 images incl. the plugin)
```

### 1.3 Tag bump commit (two places each)

On `integ/tre-v2-20260930` in `$WT`, one commit that changes only tags:

- `tre/deploy/overlays/tre-v2/gateway-plugins.yaml`, `service-manager.yaml`,
  `controller.yaml`, `ui.yaml`: `20260928-206a87f9` -> `$TAG`;
- guard test `tre/deploy/tests/test_kustomize_overlays.py` (the four `_image(...) ==`
  asserts);
- `make check` green, commit. The images keep the code sha in their tag; the tag-bump
  commit changes no image content.

Never build 687cbd9c (an intermediate SafeScale commit whose parser rejects the new
`safescale:` keys).

## 2. Backup and rollback.sh (before any apply)

```bash
B=/data/nfs_shared_data/xxy/backups/tre-v2-state-$(date +%Y%m%d-%H%M)-pre-integ
mkdir -p $B
kubectl -n tre-v2 get deploy,svc,cm,sa,role,rolebinding -o yaml > $B/tre-v2-ns.yaml
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/live-registry.yaml
kubectl -n default get deploy,svc -l tre.aibrix.io/managed=true -o yaml > $B/default-models.yaml
kubectl -n default get cm tre-reissue-sidecar -o yaml > $B/tre-reissue-sidecar-cm.yaml
kubectl -n default get svc -l model.aibrix.ai/name -o yaml > $B/default-model-svcs.yaml
kubectl -n tre-v2 get httproute -o yaml > $B/httproutes.yaml
kubectl get envoyextensionpolicy,envoypatchpolicy,clienttrafficpolicy -A -o yaml > $B/envoy-policies.yaml
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SAVE
kubectl -n tre-v2 exec deploy/tre-v2-redis -- cat /data/dump.rdb > $B/tre-v2-redis.rdb
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --scan --pattern 'tre:v2:*' | sort > $B/redis-keys.txt
git -C $REPO rev-parse main > $B/main-sha.txt
```

`$B/rollback.sh` must exist and be reviewed before step 3. Requirements:

1. Controller and SM to observe first (`set_run_mode.sh observe observe`).
2. Registry ConfigMap first (item 4), so every restored component starts once, on the
   old registry. Then images in reverse order: controller -> SM -> gateway plugin
   (-> UI).
3. **controller and SM: restore the whole Deployment object from `$B/tre-v2-ns.yaml`
   (image AND env AND strategy), never `kubectl set image` alone** - strip
   `resourceVersion`, `uid`, `creationTimestamp`, `generation`, `managedFields`,
   `status`, then `kubectl replace -f`; `rollout status` each (snippet:
   `RELEASE-20260929-floor-probe-window.md` "Rollback (controller)"; same for
   `tre-v2-service-manager`, which stays `strategy: Recreate`). Same whole-object
   restore for `tre-gateway-plugins` and `tre-v2-ui`.
4. Registry ConfigMap back to `$B/live-registry.yaml` (`kubectl create configmap
   tre-v2-registry --from-file=registry.yaml=$B/live-registry.yaml --dry-run=client -o
   yaml | kubectl replace -f -`) - required for a full rollback: it brings `vllm_image`
   back to ts-2be2d647 for the SM's runtime creates. (The old images ignore the new
   `safescale:` / `replica_floor` / `startup_admission` / `sleep.no_drain_paths` keys,
   so an image-only rollback also starts; only `sleep.budgets_s` paths are strict, and
   this release adds none.)
5. Model pods back to ts-2be2d647: sidecar ConfigMap from `$B/tre-reissue-sidecar-cm.yaml`
   (strip `resourceVersion`, `uid`, `creationTimestamp`, `managedFields` before
   `kubectl replace -f`), then the same two-wave delete-then-create as section 5 with the Deployments from
   `$B/default-models.yaml` (split per `tre.aibrix.io/node`, metadata stripped).
6. Redis: restore `$B/tre-v2-redis.rdb` only if the desired state is corrupted.
7. Never build or deploy 687cbd9c.
8. Partial rollback: an old controller with the new SM treats the new SM's `409
   floor_violation` as retriable (`RETRIABLE_STATUSES = {409, 503}`, up to 6 tries) and
   has no floor hold. That combination may only run in `observe`; otherwise roll the SM
   back with it. (New controller + old SM: no floor on the SM side; also observe only.)

## 3. Registry ConfigMap

Structural changes only (no console path); `overlays/tre-v2/params.yaml` is **never**
applied. Dry run 2026-09-30 against the live ConfigMap: "no live tunable differs from the
release file"; merged file == `deploy/registry.yaml`. Structural diff live -> release
(17 leaves):

- `models[dsqwen-7b|dsllama-8b|dsqwen-14b].vllm_image`: ts-2be2d647 -> ts-8dc0f2a7
- new `service_manager.replica_floor` {enforce: true, log_interval_s: 60}
- new `service_manager.startup_admission` {gate_seen_s: 30, drift_grace_s: 600}
- new `service_manager.sleep.no_drain_paths`: [safescale_commit, urgent, apa]
- new top-level `safescale` {slo_mode: labels, window_ceiling_s: 60,
  min_commit_samples: 20, evidence_clock_tolerance_s: 20, evidence_source: direct,
  evidence_poll_s: 2, scrape_timeout_s: 1, baseline_delay_ms: 1000, metrics_port: 8000}
- serve args (`vllm_extra_args`, `max_model_len: null`): already live, no change.

Redo the dry run right before applying (tunables may have moved):

```bash
cd $WT/tre
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > /tmp/live.yaml
PYTHONPATH=common:deploy python3 deploy/scripts/merge_live_registry.py \
    --live /tmp/live.yaml --release deploy/registry.yaml --out /tmp/merged.yaml
diff /tmp/live.yaml /tmp/merged.yaml     # expect only the leaves above (+ comments)
kubectl -n tre-v2 create configmap tre-v2-registry --from-file=registry.yaml=/tmp/merged.yaml \
    --dry-run=client -o yaml | kubectl replace -f -
```

The running (old) images read the registry at start only and ignore the new keys, so
this step changes nothing live until the restarts below. Tunables changed later go
through console `PUT /api/params` + restart of **both** controller and SM.

## 4. Control plane (plugin -> SM -> controller -> UI)

Precondition: `bash deploy/scripts/set_run_mode.sh observe observe`. All three
(plugin, SM, controller) are switched together, in this order:

```bash
cd $WT/tre
kubectl apply -f deploy/overlays/tre-v2/gateway-plugins.yaml && kubectl -n tre-v2 rollout status deploy/tre-gateway-plugins
kubectl apply -f deploy/overlays/tre-v2/service-manager.yaml && kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager   # Recreate: one writer
kubectl apply -f deploy/overlays/tre-v2/controller.yaml     && kubectl -n tre-v2 rollout status deploy/tre-v2-controller
kubectl apply -f deploy/overlays/tre-v2/ui.yaml             && kubectl -n tre-v2 rollout status deploy/tre-v2-ui
```

Checks after each: pod Running, no restart; SM log shows registry validation OK and the
floor / no-drain settings; controller log `safescale_config` shows
`evidence_source=direct` and no unknown-key WARNING; `redis-cli ZRANGE
tre:v2:gw:instances 0 -1 WITHSCORES` shows the new plugin pod with an advancing score.
The controller env of the overlay (`SAFE_SCALE_WINDOW_FLOOR_MS` 20000,
`SAFE_SCALE_E2E_MULTIPLIER` 2, legacy `SAFE_SCALE_MIN_WINDOW_MS` 60000,
`TRE_FLOOR_VIOLATION_COOLDOWN_TICKS` 6; `SAFE_SCALE_TTFT/TPOT_P95_SLO_MS` removed) goes
with the image; see the 09-29 release notes. `GET /v2/audit` one-shot, never polled.

**Behaviour change to confirm with the owner before step 4:** with the env overrides
removed and `safescale.slo_mode: labels` (the default), the SafeScale TTFT threshold is
`max(slo.ttft_floor_ms, k * (c + b * L))` of the judged window instead of the fixed
500 ms of v1; with `max_model_len` now 131072, long prompts get a clearly looser TTFT
gate (e.g. 7b at L = 4000: 1236 ms). This is intended (the calibration label rule), but
for v1-aligned runs set `safescale.slo_mode: fixed` (`models[].slo`, 500 / 75 ms) via
the registry merge + controller restart.

## 5. Model pods: delete then create, two waves by node

Never roll in place (a new pod waits for GPU memory the old pod never frees). Stay in
`observe observe`. The sidecar ConfigMap embeds `sidecar.py` and must go first:

```bash
cd $WT/tre/deploy/models
kubectl apply -f tre-reissue-sidecar.yaml            # ConfigMap (+ anything else in that file)
set -- $NODES; N1=$1; N2=$2
for N in $N1 $N2; do                                  # wave 1, then wave 2
  kubectl -n default delete deploy -l tre.aibrix.io/managed=true,tre.aibrix.io/node=$N --wait=true
  kubectl apply $(for f in *-$N-gpu-*.yaml; do printf -- '-f %s ' "$f"; done)
  # wait: all Deployments of $N Ready (startup gate admits them serially per GPU),
  # GET /v2/fleet/state converged, sleeping residents asleep, before the next wave
  kubectl -n default get deploy -l tre.aibrix.io/managed=true,tre.aibrix.io/node=$N
done
kubectl apply -k .                                    # Services / HTTPRoutes / ReferenceGrant; Deployments must be no-op
```

If the supervisor's fleet repair recreates a missing Deployment first, it renders the
same object from the same registry (so the apply is a no-op). Gates stalled > 10 min:
SM logs, `GET /v2/operations`, then `POST /v2/fleet/repair` (controller in observe).
Every pod: `image: vllm-openai-tre:0.30.0-ts-8dc0f2a7`, sidecar on :8000.

**8b KV headroom (watch in both waves and on any SM cold create):** measured 2026-09-29 on
the current 0.30 fork image (record: `/data/nfs_shared_data/xxy/deploy-test-20260929-serve-args/RUNLOG.md`,
`kv-summary.txt`): all 10 8b pods got the same KV, 16.76 GiB / 137,248 tokens, including
the extra case of a GPU whose two sleeping neighbours (7b + 14b, 2646 MiB together) were
present. So KV capacity does **not** depend on sleeping neighbours: vLLM budgets
util x total device memory, and the startup check only requires free memory >= that
budget (log: `Free memory on device (36.23/39.38 GiB) on startup. Desired GPU memory
utilization is (0.85, 33.47 GiB)`). The 15.79 GiB refusal of 2026-09-26 came from another
image (the pinned validation image) and does not apply to the current one.
Real risks: (1) the margin over the 16 GiB needed for max_model_len 131072 is only
~0.76 GiB (6,176 tokens), so a full-length 131072 request fits once at a time;
(2) if a co-resident on the same GPU is **awake**, the startup check fails (free memory
below the budget). If an 8b pod does not come up: stop the wave, controller stays
observe, read the pod log and tell the two cases apart:
- "not enough free memory" (neighbour awake): put the neighbour to sleep first or change
  the start order, then retry that binding alone;
- "KV cache too small for max_model_len": pin 8b `max_model_len` (e.g. the previous
  32768) - see P3-A below.

**P3-A (pinning 8b `max_model_len`).** Order matters so live equals the repo: set
`max_model_len` in `registry.yaml` (and the `overlays/tre-v2/params.yaml` bootstrap copy),
run `make manifests`, **commit to the integ branch first**; only then run
`merge_live_registry.py` + `kubectl replace` + SM restart and recreate only the 8b
Deployments. Changing 8b's context length changes the theta calibration basis (request
length mix / KV pressure), so decide the value **before** recalibrating 8b, not after.
Record the deviation from v1 in `HANDOFF.md`.

## 6. Smoke and acceptance (in order, stop at the first failure)

Keep `observe observe` through 6.1-6.3; switch to the TRE arm (`active active`) only for
6.5-6.7 and only with the owner's go-ahead.

1. **Tokenizer self-check** (gate; CPU, before section 5; run the loop below on **each**
   node, i.e. on 76 and again over ssh on 75, against that node's local image):
   `forks/vllm/tools/check_tokenizer_consistency.py` from fork commit 8dc0f2a7
   (worktree `forks/vllm-wt-tokenizer`), one run per model, each must **exit 0**:
   ```bash
   T=/data/nfs_shared_data/xxy/forks/vllm-wt-tokenizer/tools
   for W in $(python3 -c 'import yaml,sys; print(" ".join(m["weights_path"] for m in yaml.safe_load(open(sys.argv[1]))["models"]))' $WT/tre/deploy/registry.yaml); do
     docker run --rm --entrypoint python3 -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 \
       -v $T:/tools:ro -v $W:$W:ro vllm-openai-tre:0.30.0-ts-8dc0f2a7 \
       /tools/check_tokenizer_consistency.py $W; echo "$W exit=$?"
   done
   ```
   The BOS line is a warning only (known, see backlog).
   Calibration-gate evidence (section 0): tokenize a sample of the calibration prompts
   (the prompt texts the calibration / loadgen runs send) with `get_tokenizer(<weights>)`
   inside the old image `ts-2be2d647` and inside `ts-8dc0f2a7`, per model; report the
   per-prompt token-count difference (expected: 8b differs, 7b / 14b identical). Any
   difference -> that model's theta must be recalibrated before a TRE-arm run.
   **P3-B commands.** Sample source: the calibration prompt files
   `<run>/<model>/prompts/<cell>/*.prompts.jsonl` (one JSON per line, field `prompt`;
   written by `tre_replayer.engine.prompt_store.materialize_prompts`, see
   `tre/deploy/scripts/openloop.py`), e.g.
   `/data/nfs_shared_data/xxy/calibration_supp_20260923/<model>/prompts/*/*.prompts.jsonl`.
   Take the first 200 lines and count tokens in both images, then diff:
   ```bash
   W=<weights_path of the model>; F=<one .prompts.jsonl of that model>
   for IMG in ts-2be2d647 ts-8dc0f2a7; do
     docker run --rm --entrypoint python3 -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1        -v $W:$W:ro -v $F:/p.jsonl:ro vllm-openai-tre:0.30.0-$IMG -c "
   import json,sys
   from vllm.tokenizers import get_tokenizer
   t=get_tokenizer('$W')
   ps=[json.loads(l)['prompt'] for l,_ in zip(open('/p.jsonl'),range(200))]
   print(json.dumps([len(t.encode(p,add_special_tokens=False)) for p in ps]))" > /tmp/tok-$IMG.json
   done
   python3 -c "import json;a,b=[json.load(open('/tmp/tok-ts-%s.json'%x)) for x in ('2be2d647','8dc0f2a7')];d=[y-x for x,y in zip(a,b)];print('n',len(d),'differ',sum(1 for x in d if x),'max|d|',max(map(abs,d)))"
   ```
   (If the import path `vllm.tokenizers` differs in the image, use the same
   `get_tokenizer` import as `check_tokenizer_consistency.py`.)
2. **Fleet**: 20/20 Ready, no restarts; `GET /v2/audit` (one call) empty; each model
   answers `/v1/completions` and `/v1/chat/completions` through the gateway NodePort.
3. **Serve args**: pod args show no `--max-model-len` (each model serves its maximum);
   14b has `--enable-chunked-prefill --max-num-batched-tokens 2048`; `/v1/models`
   `max_model_len` = 131072.
4. **Continuation (reissue) on a no-drain path**: a long streaming completion on a pod,
   then sleep that pod via an `urgent` / `apa` path: the client stream completes without
   truncation, sidecar `tre_reissue_total` increments, the sleep outcome `aborted`
   counts only non-continuable requests. Repeat once with a non-continuable request
   (e.g. `n=2`): it is cut off and counted, never waited for, never rolled back.
5. **Replica floor**: with a model at `min_replicas` routable replicas, a SafeScale hide
   / `scale_down` / urgent sleep of it is refused `409 floor_violation`; APA
   `/scale_service` is clamped; `GET /v2/sleep` -> `floor.counts` increments; the
   controller logs `floor_violation_hold:<model>` and re-plans.
6. **SafeScale direct**: one **commit** (steady light load on a model above its floor:
   probe decides ~20-24 s after the hide confirmation, `evidence_source_used=direct`,
   `latency_gate=evaluated`, commit, pod asleep) and one **rollback** (load that violates
   the SLO on the remaining pods: `rollback_reason.code=slo_violation_direct` within a
   few seconds, pod routable again). Summarise with
   `python3 -m scripts.analysis.safescale_summary <run_dir>/safescale.json`;
   `evidence_pre_hide_fraction_max` must be 0 (the summary covers direct probes since
   this release). Require at least one `evidence_source_used=direct` probe whose
   remaining pods are on **each** node (proves the controller reaches pod
   IP:`metrics_port` across nodes; there is no NetworkPolicy today).
7. **CRIT path**: drive one model's Z_m past `tau_crit`: fast-loop urgent wake within
   seconds, donors released on the no-drain `urgent` path without breaking their floor,
   no stuck reservation, `GET /v2/audit` empty afterwards.

Any failure: controller back to observe, then `$B/rollback.sh` as far as needed.
After 6.7 (pass or fail): `set_run_mode.sh observe observe` again (section 0).

## 7. After acceptance

Record image IDs and the deployed sha in `HANDOFF.md`. The release stays in
`observe observe` until the calibration gate of section 0 is closed (recalibration on
ts-8dc0f2a7 applied atomically, or the owner's waiver recorded). Merging
`integ/tre-v2-20260930` into main and pushing are separate, owner-approved steps.

Known limitations (deferred, see the local backlog): SafeScale P2-2 / P3-1..P3-6 of the
implementation review; a SafeScale commit does not re-check the floor when a remaining
pod drops out during the probe (the probed pod is already hidden, so the commit's
floor guard sees nothing to remove) - abnormal path only.
