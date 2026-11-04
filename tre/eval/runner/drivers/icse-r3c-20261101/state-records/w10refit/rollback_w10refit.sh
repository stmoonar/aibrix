#!/usr/bin/env bash
# 2026-10-31 rollback of apply_w10refit.sh: registry back to the standard text (id af9aef4701804743, W 20 s),
# controller TRE_METRICS_WINDOW_MS back to 20000, restart SM then controller. WRITES THE CLUSTER.
# Equivalent to backups/tss-win10-20261030/rollback_win10.sh, with the w10refit backup dir. Source text:
# backups/tss-w10refit-20261031/registry.before.yaml (written by apply_w10refit.sh), else the win10 backup
# (same text, sha 4c40a90c...; covers an apply that failed before its backup). Also safe after a partial apply.
# Run between batches only (no runner active). Same one-patch rule as apply: the controller must never start with
# env and registry stamp disagreeing (Recreate -> no controller).
set -euo pipefail
X=/data/nfs_shared_data/xxy
C=$X/campaign-icse-w10refit-20261031
B=$X/backups/tss-w10refit-20261031
T=$X/aibrix/tre
NS=tre-v2; CTL=tre-v2-controller; SMD=tre-v2-service-manager
export TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
WANT_ID=af9aef4701804743
ID_NEW=b1eb8f0c7593a86c
STD_SHA=4c40a90cb2abc1d53f4ea1533c0a9dcc3d51b81cfdbf1bc55bf53a5c42b33194
if pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null || pgrep -af '^python3 .*campaign\.py run' >/dev/null; then
  echo "ABORT: a runner is active"; exit 1; fi
SRC=$B/registry.before.yaml
[ -s "$SRC" ] || SRC=$X/backups/tss-win10-20261030/registry.before.yaml
[ "$(sha256sum < $SRC | cut -c1-64)" = $STD_SHA ] || { echo "ABORT: $SRC is not the standard registry text"; exit 1; }
[ "$(python3 $T/eval/runner/arm_registry.py id --registry-file $SRC)" = $WANT_ID ] || { echo "ABORT: backup id mismatch"; exit 1; }
LIVE=$(python3 $T/eval/runner/arm_registry.py id)
case "$LIVE" in $ID_NEW|$WANT_ID) ;; *) echo "ABORT: live registry id $LIVE is neither $ID_NEW nor $WANT_ID (foreign edit?)"; exit 1;; esac
mkdir -p $B/rollback
python3 $C/cm_replace.py $SRC --backup-dir $B/rollback
H=$(python3 -c "import sys;sys.path[:0]=['$T/common','$T/deploy'];from scripts import registry_params as rp;print(rp.params_hash(open('$SRC').read()))")
NOW=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
ANN="{\"kubectl.kubernetes.io/restartedAt\":\"$NOW\",\"tre.dev/restarted-by\":\"rollback_w10refit.sh\",\"tre.dev/restart-reason\":\"revert w10refit\",\"tre.dev/params-hash\":\"$H\"}"
kubectl -n $NS patch deploy $SMD --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN}}}}"
kubectl -n $NS rollout status deploy/$SMD --timeout=300s
kubectl -n $NS patch deploy $CTL --type strategic -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":$ANN},\"spec\":{\"containers\":[{\"name\":\"controller\",\"env\":[{\"name\":\"TRE_METRICS_WINDOW_MS\",\"value\":\"20000\"}]}]}}}}"
kubectl -n $NS rollout status deploy/$CTL --timeout=300s
python3 $T/deploy/scripts/release/params_set.py --status
python3 $T/eval/runner/arm_registry.py id    # expect af9aef4701804743
kubectl -n $NS set env deploy/$CTL --list | grep TRE_METRICS_WINDOW_MS   # expect 20000
