#!/usr/bin/env bash
# DRAFT rollback of apply_win10.sh: registry back to the backed-up text (id af9aef4701804743),
# controller TRE_METRICS_WINDOW_MS back to 20000, restart SM then controller. WRITES THE CLUSTER.
# Run between batches only (no runner active). Same one-patch rule as apply: the controller must
# never start with env and registry stamp disagreeing (Recreate -> no controller).
set -euo pipefail
B=/data/nfs_shared_data/xxy/backups/tss-win10-20261030
T=/data/nfs_shared_data/xxy/aibrix/tre
NS=tre-v2; CTL=tre-v2-controller; SMD=tre-v2-service-manager
export TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
WANT_ID=af9aef4701804743
if pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null || pgrep -af '^python3 .*campaign\.py run' >/dev/null; then
  echo "ABORT: a runner is active"; exit 1; fi
[ "$(python3 $T/eval/runner/arm_registry.py id --registry-file $B/registry.before.yaml)" = $WANT_ID ] || { echo "ABORT: backup id mismatch"; exit 1; }
mkdir -p $B/rollback
python3 $B/cm_replace.py $B/registry.before.yaml --backup-dir $B/rollback
H=$(python3 -c "import sys;sys.path[:0]=['$T/common','$T/deploy'];from scripts import registry_params as rp;print(rp.params_hash(open('$B/registry.before.yaml').read()))")
NOW=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
ANN="{\"kubectl.kubernetes.io/restartedAt\":\"$NOW\",\"tre.dev/restarted-by\":\"rollback.sh\",\"tre.dev/restart-reason\":\"revert tss win10\",\"tre.dev/params-hash\":\"$H\"}"
kubectl -n $NS patch deploy $SMD --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN}}}}"
kubectl -n $NS rollout status deploy/$SMD --timeout=300s
kubectl -n $NS patch deploy $CTL --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN},\"spec\":{\"containers\":[{\"name\":\"controller\",\"env\":[{\"name\":\"TRE_METRICS_WINDOW_MS\",\"value\":\"20000\"}]}]}}}}"
kubectl -n $NS rollout status deploy/$CTL --timeout=300s
python3 $T/deploy/scripts/release/params_set.py --status
python3 $T/eval/runner/arm_registry.py id    # expect af9aef4701804743
kubectl -n $NS set env deploy/$CTL --list | grep TRE_METRICS_WINDOW_MS   # expect 20000
