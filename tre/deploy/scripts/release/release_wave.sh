#!/usr/bin/env bash
# release_wave.sh <node>: one delete-then-create wave of the model Deployments of <node>
# (never an in-place roll: a new pod waits for GPU memory the old one never frees).
#
#   1. self-check the manifests to apply (every Deployment of <node> carries EXPECT_ENV);
#   2. move the awake set off <node> (awake_ctl.py off);
#   3. delete the node's model Deployments, wait until their pods are gone;
#   4. apply the node's manifests, wait until all its pods are 2/2 Running;
#   5. wave checkpoint (no starting lease / admission annotation / running operation).
#
# Required env:
#   WT   the checkout the release is deployed from (integration worktree or the clean
#        clone the images were built from); manifests = ${MODELS_DIR:-$WT/tre/deploy/models}
#   B    the backup / run directory of this release (logs go to $B/wave-<node>.log)
# Optional env:
#   EXPECT_ENV   space separated NAME=VALUE pairs the engine container must carry
#                (default: VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75)
#   MODEL_NS (default), READY_TIMEOUT_S (1500), CHECKPOINT_WAIT_S (600)
#   DRY_RUN=1    awake moves printed, delete/apply as --dry-run=server, no waits
set -euo pipefail
N="${1:?usage: release_wave.sh <node>}"
: "${WT:?set WT to the release checkout}"
: "${B:?set B to the backup / run directory}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$WT/tre/deploy/models}"
MODEL_NS="${MODEL_NS:-default}"
EXPECT_ENV="${EXPECT_ENV:-VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1500}"
DRY="${DRY_RUN:-0}"
SRV=""; AWK_DRY=""
if [ "$DRY" = 1 ]; then SRV="--dry-run=server"; AWK_DRY="--dry-run"; fi
SEL="tre.aibrix.io/managed=true,tre.aibrix.io/node=$N"
mkdir -p "$B"
exec > >(tee -a "$B/wave-$N.log") 2>&1
ts() { date +%T; }

echo "$(ts) wave $N: manifests from $MODELS_DIR (expect: $EXPECT_ENV)"
EXP=(); for kv in $EXPECT_ENV; do EXP+=(--expect-env "$kv"); done
FILES=$(python3 "$HERE/release_checks.py" manifests --models-dir "$MODELS_DIR" --node "$N" "${EXP[@]}")
NF=$(echo "$FILES" | grep -c . || true)
NEXP=$(python3 -c 'import sys,yaml; print(sum(1 for f in sys.argv[1:] for d in yaml.safe_load_all(open(f)) if d and d.get("kind")=="Deployment"))' $FILES)
echo "$(ts) wave $N: $NF files, $NEXP Deployments"

echo "$(ts) wave $N: move the awake set off $N"
python3 "$HERE/awake_ctl.py" off "$N" $AWK_DRY

echo "$(ts) wave $N: delete Deployments ($SEL)"
kubectl -n "$MODEL_NS" delete deploy -l "$SEL" --wait=true $SRV
if [ "$DRY" != 1 ]; then
  kubectl -n "$MODEL_NS" wait --for=delete pod -l "$SEL" --timeout=600s || true
  LEFT=$(kubectl -n "$MODEL_NS" get pods -l "$SEL" --no-headers 2>/dev/null | grep -c . || true)
  [ "$LEFT" = 0 ] || { echo "$(ts) wave $N: $LEFT pods still present after 600 s - stop"; exit 1; }
fi

T0=$(date +%s)
echo "$(ts) wave $N: apply"
kubectl apply $SRV $(for f in $FILES; do printf -- '-f %s ' "$f"; done)
if [ "$DRY" = 1 ]; then echo "$(ts) wave $N: dry run done"; exit 0; fi

while :; do
  L=$(kubectl -n "$MODEL_NS" get pods -l "$SEL" --no-headers 2>/dev/null || true)
  NR=$(echo "$L" | awk 'NF && ($2!="2/2" || $3!="Running")' | grep -c . || true)
  TOT=$(echo "$L" | grep -c . || true)
  EL=$(( $(date +%s) - T0 ))
  if [ "$NR" = 0 ] && [ "$TOT" -ge "$NEXP" ]; then break; fi
  [ "$EL" -lt "$READY_TIMEOUT_S" ] || { echo "$(ts) wave $N: not Ready after ${EL}s - stop"; echo "$L"; exit 1; }
  [ $((EL % 60)) -lt 10 ] && echo "$(ts) t=${EL}s ready=$((TOT - NR))/$NEXP"
  sleep 10
done
echo "$(ts) wave $N: Ready t=$(( $(date +%s) - T0 ))s"
kubectl -n "$MODEL_NS" get pods -l "$SEL" -o wide --no-headers

echo "$(ts) wave $N: checkpoint"
python3 "$HERE/release_checks.py" wave-checkpoint --node "$N" --wait-s "${CHECKPOINT_WAIT_S:-600}"
echo "$(ts) wave $N: done"
