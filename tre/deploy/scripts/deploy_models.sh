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
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$DEPLOY_DIR/models"

echo "[deploy_models] applying $MODELS_DIR ..."
kubectl apply -k "$MODELS_DIR"

echo "[deploy_models] applied. service-manager reconcile will discover the bindings."
echo "[deploy_models] verify with: curl -s http://<sm>:8000/v2/state | python3 -m json.tool"
