# Release 2026-10-01: gpu-truth agent on its own image - DRAFT

Plan only: the image is built and loaded on both nodes, nothing is applied. Confirm with
the owner before any step that touches the cluster (a parallel session may be using it).
Only the `tre-v2/tre-v2-gpu-truth` DaemonSet changes; no other object, no model pod, no
`aibrix-system` object, no registry ConfigMap.

## What is in it

Branch `feat/gpu-truth-image-20261001` (worktree `aibrix-wt/gpu-truth-image-20261001`)
from `integ/tre-v2-20261001b` bdea7a72:

1. e14151b1 `tre/gpu-truth/Dockerfile` + `tre/gpu-truth/requirements.txt`: the node
   GPU-truth agent (`deploy/scripts/gpu_truth_agent.py`, unchanged) baked into
   `python:3.11.11-slim-bookworm` with `redis==6.4.0` (the version the agent ran with on
   the old image), user 65532, `NVIDIA_DRIVER_CAPABILITIES=utility`. No CUDA userspace:
   `nvidia-smi` and `libnvidia-ml.so` come from the host driver through the NVIDIA
   container runtime (docker `default-runtime: nvidia` on both nodes), exactly as they did
   for the old image. Replaces `vllm/vllm-openai:0.10.1-sleep`, which only lent python3 +
   nvidia-smi and cannot be rebuilt from source.
2. 8ab65e8d: `registry.yaml` (and the `params.yaml` bootstrap copy) gains a top-level
   `gpu_truth.image`; `gen_gpu_truth_manifest.py` reads it, takes the node affinity from
   `cluster.nodes` (no hard-coded node names) and no longer emits the agent ConfigMap
   (the script is in the image: one copy, no drift). DaemonSet: non-root, read-only root
   fs, `allowPrivilegeEscalation: false`, `RollingUpdate maxUnavailable: 1`. Agent
   command line unchanged (`--interval-s 10 --refresh-poll-s 0.25 --ttl-s 120`, same
   Redis URL), so the Redis contract (`tre:gpu_truth:<node>`, refresh counter) is the
   same.

Image: `tre-v2-gpu-truth:20261001-e14151b1`, ID
`sha256:9ed077c6c1573ef9f5d3abba432426621024291fa5716b0ca081f27eeb9c35f0`, 133 MB, built
from a clean clone at e14151b1 on 76 and `docker save | docker load`-ed to 75 (same ID).

Verified 2026-10-01 (read-only GPU queries, no GPU memory used):
- 76: `docker run --rm --gpus all --entrypoint nvidia-smi <img> -L` lists the 4 A100s;
- 76 and 75, the Kubernetes path (no `--gpus`, only `-e NVIDIA_VISIBLE_DEVICES=all`, so
  the default nvidia runtime does the injection, as for the pod): the agent's own
  `collect_nvidia_smi()` as uid 65532 returns the 4 GPUs of the node with the registry's
  UUIDs; also with `--read-only --security-opt no-new-privileges`;
- without `NVIDIA_VISIBLE_DEVICES` the container has no `nvidia-smi` (nothing leaks in).
- `make check` 3127 passed / 4 skipped; `kubectl kustomize deploy/overlays/tre-v2`
  renders (the gpu-truth part is the DaemonSet only); the new `registry.yaml` loads with
  the registry parsers of ba5b558f (running) and f8ccb0ca (they read named top-level
  sections only; `gpu_truth:` is ignored).

Independent of the models: the agent only reads `nvidia-smi` and writes Redis, so it can
be deployed with every model pod down (and also with models up). It is independent of
`RELEASE-20261001b.md` (either order). During the swap of a node's pod the last sample
stays in Redis until its 120 s TTL; a wake / cold-start refresh request in that gap
times out and the gate fails closed (safe; with models down there are none).

## 0. Variables and preconditions

```bash
cd /data/nfs_shared_data/xxy/aibrix-wt/gpu-truth-image-20261001/tre   # or main after merge
S=deploy/scripts/release
IMG=tre-v2-gpu-truth:20261001-e14151b1
ID=sha256:9ed077c6c1573ef9f5d3abba432426621024291fa5716b0ca081f27eeb9c35f0
NODES=$(python3 -c 'import yaml; print(" ".join(n["name"] for n in yaml.safe_load(open("deploy/registry.yaml"))["cluster"]["nodes"]))')
# the image is on every node (ssh hosts of the two nodes; adjust to your inventory)
for h in 192.168.223.76 192.168.223.75; do ssh root@$h docker image inspect -f '{{.Id}}' $IMG; done   # both = $ID
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o wide; kubectl -n tre-v2 get pods -l app.kubernetes.io/name=tre-v2-gpu-truth -o wide
python3 $S/release_checks.py gpu-truth --nodes "$NODES"     # baseline: OK on every node
```

If a node's baseline already FAILs (frozen seq = NVML died), note it: the new pod on that
node is the fix, not a regression.

## 1. Backup

```bash
B=/data/nfs_shared_data/xxy/backups/pre-gpu-truth-image-$(date +%Y%m%d-%H%M); mkdir -p $B
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o yaml > $B/ds-gpu-truth.yaml
kubectl -n tre-v2 get cm tre-v2-gpu-truth-agent -o yaml > $B/cm-gpu-truth-agent.yaml
cp $S/strip_obj.py $B/
python3 $B/strip_obj.py - - - $B/ds-gpu-truth.yaml > $B/ds-gpu-truth.clean.yaml
python3 $B/strip_obj.py - - - $B/cm-gpu-truth-agent.yaml > $B/cm-gpu-truth-agent.clean.yaml
cat > $B/rollback-gpu-truth.sh <<'EOF'
#!/usr/bin/env bash
# Restore the gpu-truth ConfigMap + DaemonSet backed up next to this script.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
kubectl -n tre-v2 get cm tre-v2-gpu-truth-agent >/dev/null 2>&1 \
  && kubectl replace -f "$B/cm-gpu-truth-agent.clean.yaml" \
  || kubectl create -f "$B/cm-gpu-truth-agent.clean.yaml"
kubectl replace -f "$B/ds-gpu-truth.clean.yaml"
kubectl -n tre-v2 rollout status ds/tre-v2-gpu-truth --timeout=300s
EOF
chmod +x $B/rollback-gpu-truth.sh
kubectl replace --dry-run=server -f $B/ds-gpu-truth.clean.yaml   # the restore path is valid
```

## 2. Swap, one node at a time with a check in between

The DaemonSet is first switched to `OnDelete`, so applying the new template replaces no
pod; each node's pod is then deleted by hand and checked before the next.

```bash
kubectl -n tre-v2 patch ds tre-v2-gpu-truth --type merge \
  -p '{"spec":{"updateStrategy":{"type":"OnDelete","rollingUpdate":null}}}'
python3 - <<'EOF' > /tmp/gpu-truth-ondelete.yaml
import yaml
doc = yaml.safe_load(open("deploy/overlays/tre-v2/gpu-truth.yaml"))
doc["spec"]["updateStrategy"] = {"type": "OnDelete"}
print(yaml.safe_dump(doc, sort_keys=False))
EOF
kubectl diff -f /tmp/gpu-truth-ondelete.yaml          # image, command path, env, securityContext, volumes removed
kubectl apply -f /tmp/gpu-truth-ondelete.yaml          # no pod restarts (OnDelete)
kubectl -n tre-v2 get ds tre-v2-gpu-truth -o jsonpath='{.spec.template.spec.volumes}{"\n"}'   # empty
```

For each node, worker first (node9, then the control-plane node10):

```bash
N=nscc-ds-4a100-node9      # then the other node of $NODES
pod_on() { kubectl -n tre-v2 get pods -l app.kubernetes.io/name=tre-v2-gpu-truth \
             --field-selector spec.nodeName=$1 -o jsonpath='{.items[*].metadata.name}'; }
P=$(pod_on $N); kubectl -n tre-v2 delete pod $P --wait=true
for i in $(seq 60); do P2=$(pod_on $N); [ -n "$P2" ] && [ "$P2" != "$P" ] && break; sleep 2; done
kubectl -n tre-v2 wait pod/$P2 --for=condition=Ready --timeout=120s
kubectl -n tre-v2 get pod $P2 -o jsonpath='{.spec.containers[0].image}{"\n"}'   # = $IMG
kubectl -n tre-v2 logs $P2 --tail=20
python3 $S/release_checks.py gpu-truth --nodes "$N" --gap-s 15    # OK: TTL > 0 and seq advances
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --raw GET tre:gpu_truth:$N   # 4 GPUs, registry UUIDs
```

`seq` restarts with the new agent process (it counts the publishes of one process; the
service-manager only uses `refresh_seq` against its own counter). STOP and roll back
(section 4) if the pod is not Ready, the log shows `nvidia-smi` errors, or the check
FAILs; the other node still runs the old pod.

Then restore the declared strategy (template unchanged: no pod restarts):

```bash
kubectl apply -f deploy/overlays/tre-v2/gpu-truth.yaml
kubectl -n tre-v2 rollout status ds/tre-v2-gpu-truth --timeout=60s
kubectl diff -f deploy/overlays/tre-v2/gpu-truth.yaml   # empty
```

## 3. Verify

```bash
python3 $S/release_checks.py gpu-truth --nodes "$NODES" --gap-s 30   # OK on every node
kubectl -n tre-v2 get pods -l app.kubernetes.io/name=tre-v2-gpu-truth -o wide   # one per node, 0 restarts
```

With models up, also confirm a service-manager wake/cold-start headroom gate gets a
fresh sample (the SM log shows no `gpu-truth` refresh timeout). Repeat the gpu-truth
check after ~1 h: the NVML-silent-failure mode shows as a frozen seq on a Running pod
(fix: force-delete that pod).

## 4. Rollback

```bash
$B/rollback-gpu-truth.sh      # B = the backup directory of section 1
python3 $S/release_checks.py gpu-truth --nodes "$NODES"
```

It restores the old ConfigMap and DaemonSet (borrowed vLLM image + mounted script). Code
rollback: revert 8ab65e8d (manifest / registry) on the branch; the image can stay on the
nodes.

## 5. Cleanup (after the new pods ran clean for a day)

`kubectl apply` does not prune: the old ConfigMap `tre-v2-gpu-truth-agent` stays until
deleted, and section 4 needs it only through the backup. Once satisfied:
`kubectl -n tre-v2 delete cm tre-v2-gpu-truth-agent`. Merge the branch into the next
integration branch; a later `merge_live_registry.py` run carries `gpu_truth:` into the
live registry, which the components ignore.

## Notes

- A new agent version = rebuild the image (`docker build -f gpu-truth/Dockerfile -t
  tre-v2-gpu-truth:<YYYYMMDD>-<sha> .` from `tre/` in a clean checkout), load it on every
  node, bump `gpu_truth.image` in `registry.yaml` + `params.yaml` + `EXPECTED_IMAGE` in
  `deploy/tests/test_gpu_truth_daemonset.py`, and rerun the generator.
- Offline builds: `BASE_IMAGE` is a build arg (default the public
  `python:3.11.11-slim-bookworm`); `pip install` of the single pin needs a PyPI index or
  mirror reachable from the build.
