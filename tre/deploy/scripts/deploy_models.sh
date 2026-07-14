#!/usr/bin/env bash
# Declaratively deploy the TRE-v2 model bindings: per-(model,slot) single-replica
# Deployments (D7 UUID-bound), per-model Services, per-model HTTPRoutes, and the
# cross-namespace ReferenceGrant. Idempotent -- safe to re-run.
#
# The committed manifests under deploy/models/ already carry the fixed GPU UUIDs
# (collected by collect_gpu_uuids.py). GPU UUIDs are stable per physical card, so
# routine redeploys need no regeneration. If the hardware changes, refresh with:
#   # gather `nvidia-smi -L` from each node into files, then:
#   python3 deploy/collect_gpu_uuids.py --registry deploy/registry.yaml \
#       --node-output nscc-ds-4a100-node9=/tmp/node9.txt \
#       --node-output nscc-ds-4a100-node10=/tmp/node10.txt
#   make manifests   # regenerates deploy/models/
# WARNING (co-residency / creation-order, root-caused 2026-07-06): a plain
# `kubectl apply -k models` creates all bindings CONCURRENTLY. On a fresh cluster
# each GPU's 3 D7-bound pods would then load at gpu_memory_utilization=0.9
# (~36 GiB each) simultaneously -> >40 GiB per card -> mass OOM. The fleet must be
# brought up STAGGERED: create <=1 loading pod per GPU at a time, wait vLLM ready,
# /sleep it (drops to ~2 GiB), then create the next. Rounds = max bindings/GPU (3).
# Plain apply is CORRECT ONLY for re-applying an already-present resident fleet.
# Fresh bring-up and recovery must use --staggered, which scales the fleet to zero
# and then starts/sleeps exactly one binding at a time with physical verification.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$DEPLOY_DIR/models"

if [[ "${1:-}" == "--staggered" ]]; then
  shift
  exec python3 "$DEPLOY_DIR/scripts/staggered_model_fleet.py" \
    --models-dir "$MODELS_DIR" "$@"
fi

echo "[deploy_models] applying $MODELS_DIR ..."
kubectl apply -k "$MODELS_DIR"

echo "[deploy_models] applied. service-manager reconcile will discover the bindings."
echo "[deploy_models] verify with: curl -s http://<sm>:8000/v2/state | python3 -m json.tool"
