#!/usr/bin/env bash
# rollback.sh - roll a TRE v2 release back to the state backed up in its directory.
# Copied into the backup directory $B by the release plan (deploy/RELEASE-*.md, section
# "Backup") together with strip_obj.py, awake_ctl.py, redis_state.py; it reads only $B:
#   live-registry.yaml, control-plane-deploys.yaml, tre-v2-envoy-policies.yaml,
#   tre-reissue-sidecar-cm.yaml, default-models.yaml, sm-state.json, redis-state.json.
#
# Order: both switches observe -> registry ConfigMap -> controller -> SM -> plugin -> UI
#        -> drop the release's Redis keys (once the old SM runs) -> [Redis state]
#        -> Envoy policies (tre-v2 only) -> models (per node) -> awake set.
# Usage: ./rollback.sh                    everything
#        SKIP_MODELS=1 ./rollback.sh      no sidecar ConfigMap / model Deployments
#        ONLY_MODELS=1 ./rollback.sh      sidecar ConfigMap + model Deployments + awake set only
#        SKIP_ENVOY=1 ./rollback.sh       leave the Envoy policies as they are
#        NODES="<node> <node>" ./rollback.sh   model waves for these nodes, in this order
#                                         (default: cluster.nodes of live-registry.yaml)
#        RESTORE_REDIS=1 ./rollback.sh    also restore the durable Redis keys (redis-state.json);
#                                         only if the desired state is corrupted
#        DRY_RUN=1 ./rollback.sh          replace/delete --dry-run=server, create --dry-run=client,
#                                         awake moves and Redis writes printed only
# Env: TRE_NS (tre-v2), MODEL_NS (default), REDIS_DEPLOY (tre-v2-redis). Never touches
# aibrix-system.
set -euo pipefail
B="${B:-$(cd "$(dirname "$0")" && pwd)}"
TRE_NS="${TRE_NS:-tre-v2}"
MODEL_NS="${MODEL_NS:-default}"
REDIS_DEPLOY="${REDIS_DEPLOY:-tre-v2-redis}"
DRY="${DRY_RUN:-0}"
SRV=""; CLI=""; PYDRY=""
if [ "$DRY" = 1 ]; then SRV="--dry-run=server"; CLI="--dry-run=client"; PYDRY="--dry-run"; fi
NODES="${NODES:-$(python3 -c 'import sys,yaml; print(" ".join(n["name"] for n in yaml.safe_load(open(sys.argv[1]))["cluster"]["nodes"]))' "$B/live-registry.yaml")}"
REDIS="kubectl -n $TRE_NS exec deploy/$REDIS_DEPLOY -- redis-cli"
W="$(mktemp -d "${TMPDIR:-/tmp}/tre-rollback-$(date +%Y%m%d-%H%M%S)-XXXX")"
run() { echo "+ $*"; "$@"; }
say() { if [ "$DRY" = 1 ]; then echo "[dry-run skip] $*"; else run "$@"; fi; }
podset() {
  local sel
  sel=$(kubectl -n "$TRE_NS" get deploy "$1" -o go-template='{{range $k,$v := .spec.selector.matchLabels}}{{$k}}={{$v}},{{end}}' | sed 's/,$//')
  kubectl -n "$TRE_NS" get pods -l "$sel" -o name | sort | tr '\n' ' '
}
echo "backup: $B   work: $W   nodes: $NODES   dry-run: $DRY"

echo "== 0. both switches observe"
say $REDIS MSET tre:v2:controller:mode observe tre:v2:sm:actuation observe

if [ "${ONLY_MODELS:-0}" != 1 ]; then
  echo "== 1. registry ConfigMap back to the backup (first: each restored component starts once, on it)"
  kubectl -n "$TRE_NS" create configmap tre-v2-registry --from-file=registry.yaml="$B/live-registry.yaml" \
      --dry-run=client -o yaml > "$W/registry-cm.yaml"
  run kubectl replace -f "$W/registry-cm.yaml" $SRV

  echo "== 2. control plane: whole Deployment objects (image AND env AND strategy): controller -> SM -> plugin -> UI"
  for D in tre-v2-controller tre-v2-service-manager tre-gateway-plugins tre-v2-ui; do
    python3 "$B/strip_obj.py" Deployment "$TRE_NS" "$D" "$B/control-plane-deploys.yaml" > "$W/$D.yaml"
    echo "   $D -> $(grep -m1 'image:' "$W/$D.yaml" | awk '{print $NF}')"
    BEFORE=$(podset "$D")
    run kubectl replace -f "$W/$D.yaml" $SRV
    if [ "$DRY" != 1 ]; then
      run kubectl -n "$TRE_NS" rollout status "deploy/$D" --timeout=300s
      sleep 3
      if [ "$BEFORE" = "$(podset "$D")" ]; then   # template unchanged: restart to re-read the registry
        run kubectl -n "$TRE_NS" rollout restart "deploy/$D"
        run kubectl -n "$TRE_NS" rollout status "deploy/$D" --timeout=300s
      fi
    fi
    if [ "$D" = tre-v2-service-manager ]; then
      # The old SM runs now (Recreate: the new one is gone, nothing writes these any more).
      # Keys of the release (tre_common.rediskeys SM_WAKE_OPS_KEY, SM_WAKE_STATS_KEY,
      # SM_RESTART_SEEN_KEY, SM_FAULT_KEY_PREFIX): the old SM neither reads nor updates them,
      # so they only go stale; a later re-upgrade would then recover wakes that were settled
      # long ago (wake journal), compare container restart counts against old values
      # (spurious restart placeholders) and honour a leftover fault key as soon as
      # test_hooks is on. Drop them.
      echo "   drop the release's SM keys"
      python3 "$B/redis_state.py" delete-keys tre:v2:sm:wake_ops tre:v2:sm:wake_stats tre:v2:sm:restart_seen \
          --scan 'tre:v2:sm:fault:*' $PYDRY
    fi
  done

  if [ "${RESTORE_REDIS:-0}" = 1 ]; then
    echo "== 2b. durable Redis keys from redis-state.json (controller + SM stopped meanwhile)"
    say kubectl -n "$TRE_NS" scale deploy/tre-v2-controller deploy/tre-v2-service-manager --replicas=0
    [ "$DRY" = 1 ] || kubectl -n "$TRE_NS" wait --for=delete pod -l app.kubernetes.io/name=tre-v2-service-manager --timeout=120s || true
    python3 "$B/redis_state.py" restore "$B/redis-state.json" $PYDRY
    say kubectl -n "$TRE_NS" scale deploy/tre-v2-service-manager deploy/tre-v2-controller --replicas=1
    [ "$DRY" = 1 ] || { kubectl -n "$TRE_NS" rollout status deploy/tre-v2-service-manager --timeout=300s; kubectl -n "$TRE_NS" rollout status deploy/tre-v2-controller --timeout=300s; }
  fi

  if [ "${SKIP_ENVOY:-0}" != 1 ]; then
    echo "== 3. Envoy policies of $TRE_NS back to the backup (metadata stripped)"
    python3 "$B/strip_obj.py" --all "$B/tre-v2-envoy-policies.yaml" > "$W/envoy-policies.yaml"
    python3 -c 'import sys,yaml; bad=[i["metadata"]["name"] for i in yaml.safe_load(open(sys.argv[1]))["items"] if i["metadata"].get("namespace")!=sys.argv[2]]; sys.exit("objects outside %s: %s" % (sys.argv[2], bad) if bad else 0)' "$W/envoy-policies.yaml" "$TRE_NS"
    run kubectl replace -f "$W/envoy-policies.yaml" $SRV
  fi
fi

if [ "${SKIP_MODELS:-0}" != 1 ]; then
  echo "== 4. sidecar ConfigMap back to the backup"
  python3 "$B/strip_obj.py" - - - "$B/tre-reissue-sidecar-cm.yaml" > "$W/sidecar-cm.yaml"
  run kubectl replace -f "$W/sidecar-cm.yaml" $SRV
  echo "== 5. model Deployments: delete then re-create per node (never an in-place roll)"
  for NODE in $NODES; do
    OUT="$W/models-$NODE.yaml"
    python3 "$B/strip_obj.py" --models "$NODE" - - "$B/default-models.yaml" > "$OUT"
    NAMES=$(python3 -c "import yaml,sys; print(' '.join(i['metadata']['name'] for i in yaml.safe_load(open(sys.argv[1]))['items']))" "$OUT")
    [ -n "$NAMES" ] || { echo "   ($NODE: nothing selected)"; continue; }
    N=$(echo $NAMES | wc -w)
    echo "-- wave $NODE ($N Deployments): first move the awake set off $NODE"
    python3 "$B/awake_ctl.py" off "$NODE" $PYDRY
    run kubectl -n "$MODEL_NS" delete deploy $NAMES --ignore-not-found --wait=true $SRV
    [ "$DRY" = 1 ] || kubectl -n "$MODEL_NS" wait --for=delete pod -l "tre.aibrix.io/managed=true,tre.aibrix.io/node=$NODE" --timeout=600s || true
    run kubectl create -f "$OUT" $CLI
    if [ "$DRY" != 1 ]; then
      echo "   waiting for this wave to be 2/2 Ready (up to 25 min)"
      for i in $(seq 1 150); do
        L=$(kubectl -n "$MODEL_NS" get pods -l "tre.aibrix.io/managed=true,tre.aibrix.io/node=$NODE" --no-headers)
        NR=$(echo "$L" | awk 'NF && ($2!="2/2"||$3!="Running")' | grep -c . || true)
        TOT=$(echo "$L" | grep -c . || true)
        [ "$NR" = 0 ] && [ "$TOT" -ge "$N" ] && break
        sleep 10
      done
      kubectl -n "$MODEL_NS" get pods -l "tre.aibrix.io/managed=true,tre.aibrix.io/node=$NODE" --no-headers
    fi
  done
  echo "== 6. awake set back to the pre-deploy one (sm-state.json)"
  python3 "$B/awake_ctl.py" restore "$B/sm-state.json" $PYDRY
fi
echo "== done. GET /v2/audit once; then restore the run mode recorded in $B/run-mode.txt"
echo "   (set_run_mode.sh <controller> <sm>) and remove the exclusive-window marker if this session set it."
