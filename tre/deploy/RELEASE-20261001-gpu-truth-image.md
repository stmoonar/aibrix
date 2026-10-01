# Release 2026-10-01: gpu-truth agent on its own image - DRAFT

Plan only: the image is built and loaded on both nodes, nothing is applied. Confirm with
the owner before any step that touches the cluster (a parallel session may be using it).
Only the `tre-v2/tre-v2-gpu-truth` DaemonSet changes (and, at the end, its old ConfigMap
is deleted); no model pod, no registry ConfigMap, no `aibrix-system` object.

## What is in it

Branch `feat/gpu-truth-image-20261001` (worktree `aibrix-wt/gpu-truth-image-20261001`)
from `integ/tre-v2-20261001b` bdea7a72:

1. e14151b1 `tre/gpu-truth/Dockerfile` + `requirements.txt`: the agent
   (`deploy/scripts/gpu_truth_agent.py`) baked into `python:3.11.11-slim-bookworm` with
   `redis==6.4.0` (the version it ran with on the old image), user 65532,
   `NVIDIA_DRIVER_CAPABILITIES=utility`. No CUDA userspace: `nvidia-smi` and
   `libnvidia-ml.so` come from the host driver through the NVIDIA container runtime
   (docker `default-runtime: nvidia` on both nodes), as they did for the borrowed
   `vllm/vllm-openai:0.10.1-sleep`, which cannot be rebuilt from source.
2. 8ab65e8d: `registry.yaml` (+ the `params.yaml` bootstrap copy) gains a top-level
   `gpu_truth.image`, read only by `gen_gpu_truth_manifest.py`; node affinity from
   `cluster.nodes`; no agent ConfigMap any more (one copy of the script, in the image);
   non-root, read-only root fs, no privilege escalation, `RollingUpdate maxUnavailable 1`.
3. 8d8b2565 liveness:
   - the agent touches `/run/gpu-truth/heartbeat` (1 Mi memory `emptyDir`) after every
     sample that was collected AND written to Redis;
   - `livenessProbe` exec `python3 /app/gpu_truth_agent.py --check-heartbeat
     /run/gpu-truth/heartbeat --heartbeat-max-age-s 60` (60 s = 6 samples),
     `initialDelaySeconds 120`, `periodSeconds 30`, `timeoutSeconds 10`,
     `failureThreshold 3`: a restart needs the heartbeat stale for >= 2 min;
   - the agent exits 3 after 6 failed `nvidia-smi` samples in a row (~1 min), so the
     kubelet restarts the container; Redis faults do not count (they only let the
     heartbeat go stale, so a long Redis outage also restarts the agent, harmlessly).
4. Tag-bump commit (this plan): image `tre-v2-gpu-truth:20261001-8d8b2565`.

Agent command line, Redis URL and keys are unchanged (`--interval-s 10 --refresh-poll-s
0.25 --ttl-s 120`; `tre:gpu_truth:<node>`, refresh counter).

Image `tre-v2-gpu-truth:20261001-8d8b2565`, ID
`sha256:85b981a6c406748a6e702dbe6ff1831059a5fc9b2c1f7ed86501a4c8f057e0ef`, 133 MB, built
from a clean clone at 8d8b2565 on 76, `docker save | docker load` to 75 (same ID).

Verified 2026-10-01 on 76 and 75, Kubernetes style (no `--gpus`; only
`NVIDIA_VISIBLE_DEVICES=all`, so the default nvidia runtime injects, as for the pod),
`--read-only`, `no-new-privileges`, tmpfs heartbeat, uid 65532, against a throwaway
Redis container on a private docker network (not the TRE Redis): the agent published
seq 4 within 7 s with the node's 4 GPUs (registry UUIDs), the heartbeat was fresh and
the probe returned 0 (and 1 with a 1 ms max age); with no GPU visible the agent exited
3 after 6 failed samples. Read-only GPU queries, no GPU memory used.
`make check` 3138 passed / 4 skipped; `kubectl kustomize deploy/overlays/tre-v2`
renders; the new `registry.yaml` loads with the registry parsers of ba5b558f (running)
and f8ccb0ca (top-level `gpu_truth:` is ignored).

Independent of the models: it can be deployed with every model pod down (or up), and
independently of `RELEASE-20261001b.md`. While a node's pod is swapped, its last sample
stays in Redis until the 120 s TTL; a wake / cold-start refresh in that gap times out
and the gate fails closed (safe; with models down there are none).

## NVML silent failure: what liveness does and does not cover

Historically (2026-09-21) a gpu-truth pod stayed Running while `nvidia-smi` failed inside
it (exit 255, NVML gone); the key went stale and only a **force delete of the pod**
recovered it. Liveness now makes the kubelet restart the container in that state (agent
self-exit after 6 failed samples, or the stale-heartbeat probe). **Whether a container
restart, without a new pod, recovers NVML has not been verified.** If after a liveness
restart the pod's `restartCount` keeps growing and `tre:gpu_truth:<node>` stays stale
(section 3 checks), fall back to the old fix (section 5).

## Commands

Run in ONE shell on the control-plane node (76), in order; each block is meant to be
pasted as is. Stop at the first unexpected output.

### 0. Variables and preconditions

```bash
cd /data/nfs_shared_data/xxy/aibrix-wt/gpu-truth-image-20261001/tre   # or the merged checkout
S=deploy/scripts/release
IMG=tre-v2-gpu-truth:20261001-8d8b2565
ID=sha256:85b981a6c406748a6e702dbe6ff1831059a5fc9b2c1f7ed86501a4c8f057e0ef
SEL=app.kubernetes.io/name=tre-v2-gpu-truth
HOSTS="192.168.223.76 192.168.223.75"          # ssh names of the GPU nodes
N1=nscc-ds-4a100-node9                          # worker first
N2=nscc-ds-4a100-node10                         # control-plane second
NODES="$N1 $N2"
grep -q "image: $IMG" deploy/overlays/tre-v2/gpu-truth.yaml && echo manifest-ok
for h in $HOSTS; do echo "$h $(ssh root@$h docker image inspect -f '{{.Id}}' $IMG)"; done   # both = $ID
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o wide
kubectl -n tre-v2 get pods -l $SEL -o wide
python3 $S/release_checks.py gpu-truth --nodes "$NODES"     # baseline; a FAIL = NVML already dead there
```

### 1. Backup and rollback script

```bash
B=/data/nfs_shared_data/xxy/backups/pre-gpu-truth-image-$(date +%Y%m%d-%H%M); mkdir -p $B
echo $B > /data/nfs_shared_data/xxy/backups/pre-gpu-truth-image.path    # later shells: B=$(cat ...path)
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o yaml > $B/ds-gpu-truth.yaml
kubectl -n tre-v2 get cm tre-v2-gpu-truth-agent -o yaml > $B/cm-gpu-truth-agent.yaml
cp $S/strip_obj.py $B/
python3 $B/strip_obj.py - - - $B/ds-gpu-truth.yaml > $B/ds-gpu-truth.clean.yaml
python3 $B/strip_obj.py - - - $B/cm-gpu-truth-agent.yaml > $B/cm-gpu-truth-agent.clean.yaml
cat > $B/rollback-gpu-truth.sh <<'EOF'
#!/usr/bin/env bash
# Restore the gpu-truth ConfigMap + DaemonSet backed up next to this script
# (old image + mounted agent script). The DaemonSet then rolls both nodes itself.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
if kubectl -n tre-v2 get cm tre-v2-gpu-truth-agent >/dev/null 2>&1; then
  kubectl replace -f "$B/cm-gpu-truth-agent.clean.yaml"
else
  kubectl create -f "$B/cm-gpu-truth-agent.clean.yaml"
fi
kubectl replace -f "$B/ds-gpu-truth.clean.yaml"
kubectl -n tre-v2 rollout status ds/tre-v2-gpu-truth --timeout=300s
EOF
chmod +x $B/rollback-gpu-truth.sh
kubectl replace --dry-run=server -f $B/ds-gpu-truth.clean.yaml      # restore path valid
kubectl replace --dry-run=server -f $B/cm-gpu-truth-agent.clean.yaml
ls -l $B
```

### 2. Switch to OnDelete and apply the new template (no pod restarts)

```bash
kubectl -n tre-v2 patch ds tre-v2-gpu-truth --type merge \
  -p '{"spec":{"updateStrategy":{"type":"OnDelete","rollingUpdate":null}}}'
python3 -c 'import yaml; d=yaml.safe_load(open("deploy/overlays/tre-v2/gpu-truth.yaml")); d["spec"]["updateStrategy"]={"type":"OnDelete"}; print(yaml.safe_dump(d, sort_keys=False))' > /tmp/gpu-truth-ondelete.yaml
kubectl diff -f /tmp/gpu-truth-ondelete.yaml    # image, command, env, securityContext, livenessProbe, volumes
kubectl apply -f /tmp/gpu-truth-ondelete.yaml
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o jsonpath='{.spec.updateStrategy.type} {.spec.template.spec.containers[0].image} {.spec.template.spec.volumes}{"\n"}'
kubectl -n tre-v2 get pods -l $SEL -o wide       # still the old pods, 0 new restarts
```

### 3. Swap node 1, check, then node 2

Node 1:

```bash
N=$N1
P=$(kubectl -n tre-v2 get pods -l $SEL --field-selector spec.nodeName=$N -o jsonpath='{.items[*].metadata.name}'); echo old=$P
kubectl -n tre-v2 delete pod $P --wait=true
for i in $(seq 60); do P2=$(kubectl -n tre-v2 get pods -l $SEL --field-selector spec.nodeName=$N -o jsonpath='{.items[*].metadata.name}'); [ -n "$P2" ] && [ "$P2" != "$P" ] && break; sleep 2; done; echo new=$P2
kubectl -n tre-v2 wait pod/$P2 --for=condition=Ready --timeout=120s
kubectl -n tre-v2 get pod $P2 -o jsonpath='{.spec.containers[0].image} restarts={.status.containerStatuses[0].restartCount}{"\n"}'
kubectl -n tre-v2 logs $P2 --tail=20                      # no nvidia-smi / publish errors
python3 $S/release_checks.py gpu-truth --nodes "$N" --gap-s 15     # OK: TTL > 0 and seq advances
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --raw GET tre:gpu_truth:$N   # 4 GPUs, registry UUIDs
kubectl -n tre-v2 exec $P2 -- python3 /app/gpu_truth_agent.py --check-heartbeat /run/gpu-truth/heartbeat; echo probe-rc=$?   # 0
```

`seq` restarts from 1 with the new process (only `refresh_seq` is compared by the
service-manager, against its own counter). STOP and run section 5 if the pod is not
Ready, the log shows errors, the check FAILs or the probe returns non-zero; node 2 still
runs the old pod.

Node 2: the same block with `N=$N2`:

```bash
N=$N2
P=$(kubectl -n tre-v2 get pods -l $SEL --field-selector spec.nodeName=$N -o jsonpath='{.items[*].metadata.name}'); echo old=$P
kubectl -n tre-v2 delete pod $P --wait=true
for i in $(seq 60); do P2=$(kubectl -n tre-v2 get pods -l $SEL --field-selector spec.nodeName=$N -o jsonpath='{.items[*].metadata.name}'); [ -n "$P2" ] && [ "$P2" != "$P" ] && break; sleep 2; done; echo new=$P2
kubectl -n tre-v2 wait pod/$P2 --for=condition=Ready --timeout=120s
kubectl -n tre-v2 get pod $P2 -o jsonpath='{.spec.containers[0].image} restarts={.status.containerStatuses[0].restartCount}{"\n"}'
kubectl -n tre-v2 logs $P2 --tail=20
python3 $S/release_checks.py gpu-truth --nodes "$N" --gap-s 15
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --raw GET tre:gpu_truth:$N
kubectl -n tre-v2 exec $P2 -- python3 /app/gpu_truth_agent.py --check-heartbeat /run/gpu-truth/heartbeat; echo probe-rc=$?
```

### 4. Restore RollingUpdate (template unchanged: no restarts) and verify

```bash
kubectl apply -f deploy/overlays/tre-v2/gpu-truth.yaml
kubectl -n tre-v2 rollout status ds/tre-v2-gpu-truth --timeout=60s
kubectl diff -f deploy/overlays/tre-v2/gpu-truth.yaml && echo no-diff
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o jsonpath='{.spec.updateStrategy} updated={.status.updatedNumberScheduled}/{.status.desiredNumberScheduled}{"\n"}'
python3 $S/release_checks.py gpu-truth --nodes "$NODES" --gap-s 30   # OK on every node
sleep 180; kubectl -n tre-v2 get pods -l $SEL -o wide                  # after the first probes: 0 restarts
kubectl -n tre-v2 get events --field-selector reason=Unhealthy | grep gpu-truth || echo no-liveness-failures
```

With models up later, a service-manager wake / cold-start gate gets fresh samples (no
gpu-truth refresh timeout in the SM log). Repeat `release_checks.py gpu-truth` after ~1 h.

### 5. Rollback, or NVML not recovered by a restart

Whole release (old image + ConfigMap; both nodes roll):

```bash
B=$(cat /data/nfs_shared_data/xxy/backups/pre-gpu-truth-image.path)
$B/rollback-gpu-truth.sh
python3 deploy/scripts/release/release_checks.py gpu-truth --nodes "nscc-ds-4a100-node9 nscc-ds-4a100-node10"
```

One node stuck (`restartCount` growing and the key not fresh after liveness restarts):
force delete that pod, as before this release:

```bash
N=nscc-ds-4a100-node9      # the stuck node
kubectl -n tre-v2 get pods -l app.kubernetes.io/name=tre-v2-gpu-truth --field-selector spec.nodeName=$N -o jsonpath='{.items[0].metadata.name} restarts={.items[0].status.containerStatuses[0].restartCount}{"\n"}'
kubectl -n tre-v2 delete pod -l app.kubernetes.io/name=tre-v2-gpu-truth --field-selector spec.nodeName=$N --force --grace-period=0
python3 deploy/scripts/release/release_checks.py gpu-truth --nodes "$N" --gap-s 30
```

Record in `docs/` whether a container restart alone ever recovered NVML (unverified
today).

Code rollback: revert 8d8b2565..HEAD (liveness + tag) or 8ab65e8d..HEAD (whole switch)
on the branch; the images can stay on the nodes.

### 6. Cleanup (after a clean day)

`kubectl apply` does not prune; the old ConfigMap is only needed through the backup:

```bash
kubectl -n tre-v2 delete cm tre-v2-gpu-truth-agent
```

Merge the branch into the next integration branch; a later `merge_live_registry.py` run
carries `gpu_truth:` into the live registry, which the components ignore.

## Notes

- New agent version: rebuild from `tre/` in a clean checkout (`docker build -f
  gpu-truth/Dockerfile -t tre-v2-gpu-truth:<YYYYMMDD>-<sha> .`), load it on every node,
  bump `gpu_truth.image` in `registry.yaml` + `params.yaml` + `EXPECTED_IMAGE` in
  `deploy/tests/test_gpu_truth_daemonset.py`, rerun `python3 deploy/gen_gpu_truth_manifest.py`.
- Liveness tuning: `gen_gpu_truth_manifest.py --heartbeat-max-age-s / --max-collect-failures`
  (probe timing constants in the generator); the guard test pins the defaults.
- Offline builds: `BASE_IMAGE` is a build arg; the single `pip install` pin needs a PyPI
  index or mirror reachable from the build.
