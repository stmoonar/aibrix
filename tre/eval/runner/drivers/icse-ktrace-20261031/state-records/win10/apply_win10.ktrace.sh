#!/usr/bin/env bash
# DRAFT (not executed by the analysis agent). Switch the live TRE v2 to TSS window 10 s, EMA tau 0 (D4-prime fit):
# registry (theta_m x0.5, theta_window_ms 10000, ema_tau_ms 0, all 3 models) + controller env
# TRE_METRICS_WINDOW_MS=10000, restart SM then controller. WRITES THE CLUSTER.
# Why not params_set.py alone: trs.ema_tau_ms is locked there, and params_set --restart would roll the
# controller (strategy Recreate) with TRE_METRICS_WINDOW_MS=20000 against theta_window_ms=10000 ->
# controller refuses to start (app.py:274) -> no controller. Env + restart go out in ONE controller patch.
set -euo pipefail
W5=/data/nfs_shared_data/xxy/analysis-tss-params-20261030/win10
B=/data/nfs_shared_data/xxy/backups/tss-win10-20261030
T=/data/nfs_shared_data/xxy/aibrix/tre
NS=tre-v2; CTL=tre-v2-controller; SMD=tre-v2-service-manager
export TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
log() { echo "[$(date +%F' '%T)] $*"; }
NEWREG=$W5/registry/registry-win10.yaml
RV=$(kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.metadata.resourceVersion}')
LIVE_ID=$(python3 -c "import json;print(json.load(open('$W5/registry/meta.json'))['registry_id_live'])")
NEWH=$(python3 -c "import json;print(json.load(open('$W5/registry/meta.json'))['params_hash_new'])")

# 0. guards
if pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null || pgrep -af '^python3 .*campaign\.py run' >/dev/null; then
  echo "ABORT: a runner is active"; exit 1; fi
bash $T/deploy/scripts/set_run_mode.sh status
cur=$(kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_METRICS_WINDOW_MS")].value}')
[ "$cur" = 20000 ] || { echo "ABORT: TRE_METRICS_WINDOW_MS is '$cur', expected 20000"; exit 1; }

# 1. backups (+ rollback.sh next to them)
mkdir -p $B
kubectl -n $NS get cm tre-v2-registry -o yaml > $B/cm-tre-v2-registry.before.yaml
kubectl -n $NS get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/registry.before.yaml
kubectl -n $NS get deploy $CTL -o yaml > $B/deploy-controller.before.yaml
kubectl -n $NS get deploy $SMD -o yaml > $B/deploy-sm.before.yaml
kubectl -n $NS set env deploy/$CTL --list > $B/controller-env.before.txt
python3 $T/deploy/scripts/release/params_set.py --status > $B/params-status.before.txt
cp $W5/registry/registry-win10.yaml $W5/registry/meta.json $W5/registry/diff.txt $W5/rollback_win10.sh $W5/cm_replace.py $B/
log "backups in $B"

# 2. registry (resourceVersion + content id locked to what registry-win10.yaml was built from)
python3 $W5/cm_replace.py $NEWREG --backup-dir $B --expect-rv "$RV" --expect-id "$LIVE_ID"

# 3. SM restart (reads the registry only at start), annotations as params_set.py restart()
NOW=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
ANN="{\"kubectl.kubernetes.io/restartedAt\":\"$NOW\",\"tre.dev/restarted-by\":\"apply_win10.sh\",\"tre.dev/restart-reason\":\"tss window 10s + ema tau 0 (win10)\",\"tre.dev/params-hash\":\"$NEWH\"}"
kubectl -n $NS patch deploy $SMD --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN}}}}"
kubectl -n $NS rollout status deploy/$SMD --timeout=300s

# 4. controller: env + restart annotation in ONE patch (one Recreate rollout)
kubectl -n $NS patch deploy $CTL --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN},\"spec\":{\"containers\":[{\"name\":\"controller\",\"env\":[{\"name\":\"TRE_METRICS_WINDOW_MS\",\"value\":\"10000\"}]}]}}}}"
kubectl -n $NS rollout status deploy/$CTL --timeout=300s || { echo "controller rollout failed -> bash $B/rollback_win10.sh"; exit 2; }

# 5. verify
python3 $T/deploy/scripts/release/params_set.py --status
sleep 20
kubectl -n $NS logs deploy/$CTL --tail=2000 | grep -c "not fitted on this controller" && { echo "BASIS MISMATCH"; exit 2; } || true
kubectl -n $NS logs deploy/$CTL --tail=400 | grep trs_calc_result | tail -1 | grep -o 'theta_m[^,]*' || true   # expect 230.31 / 233.56 / 302.84
kubectl -n $NS get pod -l app.kubernetes.io/name=tre-v2-controller -o wide
python3 $T/eval/runner/arm_registry.py id   # expect 5028a1209a5fe38d (= campaign registry_expect.registry_id)
log "win10 live. Revert after the campaign: bash $B/rollback_win10.sh"
