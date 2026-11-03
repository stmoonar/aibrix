#!/usr/bin/env bash
# 2026-10-31 apply the w10refit registry for the TRE-only campaign campaign-icse-w10refit-20261031. WRITES THE CLUSTER.
# Modelled on analysis-tss-params-20261030/win10/apply_win10.sh. Live registry af9aef4701804743 (standard, W 20 s)
#   -> refit-fixedslo-w10-20261031/registry-final/registry-w10refit-fixed.yaml (id b1eb8f0c7593a86c):
#      win10 basis (theta_window_ms 10000, ema_tau_ms 0, numerator step_counters) + theta / tau_crit / tau_high of the
#      10 s fixed-SLO refit + safescale.slo_mode fixed (probe thresholds = models[].slo 500 / 75)
#   + controller env TRE_METRICS_WINDOW_MS=10000; restart SM, then controller (env + restart in ONE patch:
#     the controller refuses to start when TRE_METRICS_WINDOW_MS != trs.theta_window_ms, app.py theta_basis_errors).
# Locks: live content id af9aef4701804743 + live content sha == the win10 base's live-registry.yaml + live
# resourceVersion read now (cm_replace --expect-rv: 409-style refusal on a concurrent edit in between).
# Revert: bash /data/nfs_shared_data/xxy/campaign-icse-w10refit-20261031/rollback_w10refit.sh
set -euo pipefail
X=/data/nfs_shared_data/xxy
C=$X/campaign-icse-w10refit-20261031
F=$X/refit-fixedslo-w10-20261031/registry-final
B=$X/backups/tss-w10refit-20261031
T=$X/aibrix/tre
NS=tre-v2; CTL=tre-v2-controller; SMD=tre-v2-service-manager
export TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
log() { echo "[$(date +%F' '%T)] $*"; }
NEWREG=$F/registry-w10refit-fixed.yaml
ID_STD=af9aef4701804743
ID_NEW=b1eb8f0c7593a86c
STD_SHA=4c40a90cb2abc1d53f4ea1533c0a9dcc3d51b81cfdbf1bc55bf53a5c42b33194   # sha256 of the standard registry text
reg_id() { python3 $T/eval/runner/arm_registry.py id "$@"; }
NEWH=$(python3 -c "import json;print(json.load(open('$F/meta.json'))['params_hash_new'])")

# 0. guards (read-only)
if pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null || pgrep -af '^python3 .*campaign\.py run' >/dev/null; then
  echo "ABORT: a runner is active"; exit 1; fi
bash $T/deploy/scripts/set_run_mode.sh status
[ "$(reg_id --registry-file $NEWREG)" = $ID_NEW ] || { echo "ABORT: $NEWREG id != $ID_NEW"; exit 1; }
[ "$(reg_id)" = $ID_STD ] || { echo "ABORT: live registry id $(reg_id) != $ID_STD"; exit 1; }
LIVE_SHA=$(kubectl -n $NS get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' | sha256sum | cut -c1-64)
[ "$LIVE_SHA" = $STD_SHA ] || { echo "ABORT: live registry text sha $LIVE_SHA != standard $STD_SHA"; exit 1; }
cur=$(kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_METRICS_WINDOW_MS")].value}')
[ "$cur" = 20000 ] || { echo "ABORT: TRE_METRICS_WINDOW_MS is '$cur', expected 20000"; exit 1; }
RV=$(kubectl -n $NS get cm tre-v2-registry -o jsonpath='{.metadata.resourceVersion}')

# 1. backups (+ the scripts and the final registry next to them)
mkdir -p $B
kubectl -n $NS get cm tre-v2-registry -o yaml > $B/cm-tre-v2-registry.before.yaml
kubectl -n $NS get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/registry.before.yaml
[ "$(sha256sum < $B/registry.before.yaml | cut -c1-64)" = $STD_SHA ] || { echo "ABORT: backup sha mismatch"; exit 1; }
kubectl -n $NS get deploy $CTL -o yaml > $B/deploy-controller.before.yaml
kubectl -n $NS get deploy $SMD -o yaml > $B/deploy-sm.before.yaml
kubectl -n $NS set env deploy/$CTL --list > $B/controller-env.before.txt
python3 $T/deploy/scripts/release/params_set.py --status > $B/params-status.before.txt
cp $NEWREG $F/meta.json $F/diff-vs-win10.txt $F/diff-vs-standard.txt $C/apply_w10refit.sh $C/rollback_w10refit.sh $C/cm_replace.py $B/
log "backups in $B (live rv $RV)"

# 2. registry (resourceVersion + content id locks)
python3 $C/cm_replace.py $NEWREG --backup-dir $B --expect-rv "$RV" --expect-id $ID_STD

# 3. SM restart (reads the registry only at start), annotations as params_set.py restart()
NOW=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
ANN="{\"kubectl.kubernetes.io/restartedAt\":\"$NOW\",\"tre.dev/restarted-by\":\"apply_w10refit.sh\",\"tre.dev/restart-reason\":\"w10 fixed-SLO refit + safescale fixed (w10refit)\",\"tre.dev/params-hash\":\"$NEWH\"}"
kubectl -n $NS patch deploy $SMD --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN}}}}"
kubectl -n $NS rollout status deploy/$SMD --timeout=300s

# 4. controller: env + restart annotation in ONE patch (one Recreate rollout)
kubectl -n $NS patch deploy $CTL --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN},\"spec\":{\"containers\":[{\"name\":\"controller\",\"env\":[{\"name\":\"TRE_METRICS_WINDOW_MS\",\"value\":\"10000\"}]}]}}}}"
kubectl -n $NS rollout status deploy/$CTL --timeout=300s || { echo "controller rollout failed -> bash $C/rollback_w10refit.sh"; exit 2; }

# 5. verify
python3 $T/deploy/scripts/release/params_set.py --status
[ "$(reg_id)" = $ID_NEW ] || { echo "live registry id $(reg_id) != $ID_NEW after apply"; exit 2; }
sleep 20
if kubectl -n $NS logs deploy/$CTL --tail=2000 | grep -q "not fitted on this controller"; then echo "BASIS MISMATCH"; exit 2; fi
SM_LINE=$(kubectl -n $NS logs deploy/$CTL | grep -m1 'safescale_config' | grep -o 'slo_mode[^,]*' || true)   # JSON-in-JSON: quotes escaped
echo "controller $SM_LINE"
echo "$SM_LINE" | grep -q fixed || { echo "controller safescale_config does not say slo_mode fixed"; exit 2; }
kubectl -n $NS logs deploy/$CTL --tail=400 | grep trs_calc_result | tail -1 | grep -o 'theta_m[^,]*' || true   # expect 652.7 / 650.9 / 591.6
kubectl -n $NS get pod -l app.kubernetes.io/name=tre-v2-controller -o wide
log "w10refit live: registry $ID_NEW, TRE_METRICS_WINDOW_MS 10000, safescale.slo_mode fixed. Revert: bash $C/rollback_w10refit.sh"
