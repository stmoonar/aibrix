#!/usr/bin/env bash
# Reset TRE state between evaluation traces (endgame plan 6.4 / R2/R4):
#   1. scale every model back to awake=1 (serving floor),
#   2. archive + clear the Redis decision stream,
#   3. restart the controller pod to clear in-memory EMA / SafeScale state.
# Idempotent. Requires: kubectl context on the tre-v2 cluster; SM reachable.
set -euo pipefail

SM_CLUSTERIP="$(kubectl -n tre-v2 get svc tre-v2-service-manager -o jsonpath='{.spec.clusterIP}')"
SM="http://${SM_CLUSTERIP}:8000"
ARCHIVE_DIR="${1:-/root/tre-experiments/reset_archive}"
mkdir -p "$ARCHIVE_DIR"
STAMP="$(kubectl -n tre-v2 get pod -l app.kubernetes.io/name=tre-v2-controller -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null || echo unknown)"

echo "[reset] scaling all models to awake=1"
for m in $(curl -s "${SM}/v2/state" | python3 -c 'import sys,json;[print(k) for k in json.load(sys.stdin)["models"]]'); do
  curl -s -X PUT "${SM}/v2/models/${m}/target" -H 'Content-Type: application/json' -d '{"wake_replicas":1}' >/dev/null
  echo "  ${m} -> awake=1"
done

echo "[reset] archiving + clearing Redis decision snapshot"
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --scan --pattern 'tre:v2:decision:*' \
  | while read -r k; do kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli DUMP "$k" >/dev/null 2>&1 || true; done
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli --scan --pattern 'tre:v2:decision:*' > "${ARCHIVE_DIR}/decision_keys_${STAMP}.txt" 2>/dev/null || true
kubectl -n tre-v2 exec deploy/tre-v2-redis -- sh -c "redis-cli --scan --pattern 'tre:v2:decision:*' | xargs -r redis-cli DEL" >/dev/null 2>&1 || \
  kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli DEL tre:v2:decision:latest >/dev/null 2>&1 || true

echo "[reset] restarting controller to clear EMA / SafeScale state"
kubectl -n tre-v2 rollout restart deploy/tre-v2-controller
kubectl -n tre-v2 rollout status deploy/tre-v2-controller --timeout=120s

echo "[reset] done. reconcile:"
curl -s -X POST "${SM}/v2/reconcile" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  version",d["version"],"warnings",d["warnings"][:3])'
