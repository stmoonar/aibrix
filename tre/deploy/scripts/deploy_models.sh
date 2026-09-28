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
#
# RUN MODE (2026-09-28): the controller mode (tre:v2:controller:mode) and the SM
# actuation (tre:v2:sm:actuation) are independent switches; a missing key is observe
# for its reader, so a deploy sets both explicitly:
#   --run-mode CONTROLLER:SM   e.g. active:active (TRE arm), observe:active (APA arm),
#                              observe:observe (calibration / maintenance)
# Plain apply: sets --run-mode after the apply; without it, prints the current values
# and warns about missing keys. --staggered --execute: sets observe:observe BEFORE the
# bring-up (it scales the fleet to zero and must not race the SM self-heal), then
# --run-mode (if given) after it; otherwise both stay observe.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$DEPLOY_DIR/models"
SET_RUN_MODE="$DEPLOY_DIR/scripts/set_run_mode.sh"

STAGGERED=0
EXECUTE=0
RUN_MODE=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --staggered) STAGGERED=1 ;;
    --run-mode)
      [[ $# -ge 2 ]] || { echo "[deploy_models][ERROR] --run-mode needs CONTROLLER:SM" >&2; exit 2; }
      RUN_MODE="$2"; shift ;;
    --execute) EXECUTE=1; PASS+=("$1") ;;
    *) PASS+=("$1") ;;
  esac
  shift
done
if [[ -n "$RUN_MODE" && ! "$RUN_MODE" =~ ^(active|observe):(active|observe)$ ]]; then
  echo "[deploy_models][ERROR] --run-mode must be CONTROLLER:SM with active|observe, got '$RUN_MODE'" >&2
  exit 2
fi

apply_run_mode() {
  if [[ -n "$RUN_MODE" ]]; then
    echo "[deploy_models] run mode -> controller=${RUN_MODE%%:*} sm_actuation=${RUN_MODE##*:}"
    bash "$SET_RUN_MODE" "${RUN_MODE%%:*}" "${RUN_MODE##*:}"
  else
    echo "[deploy_models] run mode unchanged (pass --run-mode CONTROLLER:SM to set both):"
    bash "$SET_RUN_MODE" status
  fi
}

if [[ "$STAGGERED" -eq 1 ]]; then
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "[deploy_models] staggered bring-up: controller + SM actuation -> observe first"
    bash "$SET_RUN_MODE" observe observe
  fi
  python3 "$DEPLOY_DIR/scripts/staggered_model_fleet.py" \
    --models-dir "$MODELS_DIR" ${PASS[@]+"${PASS[@]}"}
  if [[ "$EXECUTE" -eq 1 ]]; then
    apply_run_mode
  fi
  exit 0
fi

echo "[deploy_models] applying $MODELS_DIR ..."
kubectl apply -k "$MODELS_DIR"

echo "[deploy_models] applied. service-manager reconcile will discover the bindings."
echo "[deploy_models] verify with: curl -s http://<sm>:8000/v2/state | python3 -m json.tool"
apply_run_mode
