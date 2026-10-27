#!/usr/bin/env bash
# Real-cluster smoke of the service-manager POST /v2/plans (design docs/design/20261027-sm-plans.md §8 items 1-4, 6):
#   single   /target +1 / -1 and a one-item plan (+1 / -1)
#   multi    one wake item at_least awake+2 -> both /wake_up in one hold (phases_ms)
#   models   two wake items of different models in one plan (hold ~ one wake, not the sum), then one plan that
#            sleeps both back and wakes a third model
#   donor    receiver with a donor allowance (wave 2 on the freed GPU when no GPU is free; relay `donors`)
#   tp2      TP2 receiver over two TP1 donors
# Every step writes the request and the response (<out>/<nn>-<step>.{req,resp}.json) and prints status, phases_ms,
# lock_wait_ms and per item awake_before -> awake_after, wave / source of each woken pod. At the end every model is
# put back to its starting awake count and /v2/audit is read ONCE (never polled).
#
# Default: prints the plans only (nothing is sent). --run sends them. Needs: the cluster in `observe active`
# (controller observes, SM actuates), baseline-scaler at 0 replicas, no APA CR, the exclusive-window marker
# (validation) and the user's go-ahead. Not covered here: §8 item 5 (fail_wake hook: registry
# service_manager.test_hooks) and item 7 (SM pod killed in wave 2): manual.
#
# Env: SM_URL (default: http://<ClusterIP of $SM_SVC in $TRE_NS>:$SM_PORT), TRE_NS tre-v2,
#      SM_SVC tre-v2-service-manager, SM_PORT 8000, M_TP1A dsqwen-7b, M_TP1B dsllama-8b, M_TP2 dsqwen-14b,
#      OUT (default /tmp/sm-plans-smoke-<UTC time>), CURL_MAX_S 400 (> api_call_timeout_s 360), SETTLE_S 20,
#      RUN_MODE_SCRIPT (set_run_mode.sh beside this tree; its status must read observe / active for --run).
set -euo pipefail
RUN=0; [ "${1:-}" = "--run" ] && RUN=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRE_NS="${TRE_NS:-tre-v2}"; SM_SVC="${SM_SVC:-tre-v2-service-manager}"; SM_PORT="${SM_PORT:-8000}"
A="${M_TP1A:-dsqwen-7b}"; B="${M_TP1B:-dsllama-8b}"; C="${M_TP2:-dsqwen-14b}"
OUT="${OUT:-/tmp/sm-plans-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"; CURL_MAX_S="${CURL_MAX_S:-400}"; SETTLE_S="${SETTLE_S:-20}"
RUN_MODE_SCRIPT="${RUN_MODE_SCRIPT:-$HERE/../set_run_mode.sh}"
if [ -z "${SM_URL:-}" ]; then
  if [ "$RUN" = 1 ]; then
    SM_URL="http://$(kubectl -n "$TRE_NS" get svc "$SM_SVC" -o jsonpath='{.spec.clusterIP}'):$SM_PORT"
  else
    SM_URL="http://<sm>:$SM_PORT"
  fi
fi
mkdir -p "$OUT"; N=0
log(){ echo "[$(date -Is)] $*" | tee -a "$OUT/smoke.log"; }

awake(){  # model -> awake count (POST /models_replicas: store view, awake and not hidden)
  [ "$RUN" = 1 ] || { echo "N_$1"; return; }
  curl -sf -m 30 -X POST "$SM_URL/models_replicas?models=$1" | python3 -c "import json,sys; print(json.load(sys.stdin)[sys.argv[1]])" "$1"
}

report(){  # resp file -> one summary block
  python3 - "$1" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as exc:  # noqa: BLE001
    print(f"  (no JSON response: {exc})"); sys.exit(0)
r = r.get("detail", r) if isinstance(r.get("detail"), dict) else r
ph = r.get("phases_ms") or {}
print(f"  plan {r.get('plan_id')} status={r.get('status')} lock_wait_ms={ph.get('lock_wait')} phases_ms={ph}")
for kind in ("sleeps", "wakes"):
    for it in r.get(kind) or []:
        woke = [(w.get("serve_id"), w.get("node"), w.get("gpu_ids"), f"wave{w.get('wave')}", w.get("source"),
                 w.get("donor_binding_ids")) for w in it.get("woke") or []]
        print(f"  {kind[:-1]} {it.get('model')}: {it.get('status')} {it.get('awake_before')} -> {it.get('awake_after')}"
              f" taken={it.get('taken')} unfilled={it.get('unfilled')} clamped={it.get('clamped_by_floor')}"
              f" retriable={it.get('retriable')} donors={it.get('donors')} woke={woke}"
              + (f" error={it.get('error')}" if it.get("error") else ""))
PY
}

send(){  # name method path json
  N=$((N + 1)); local f; f="$OUT/$(printf %02d $N)-$1"
  printf '%s\n' "$4" > "$f.req.json"
  log "== $1: $2 $3 $4"
  [ "$RUN" = 1 ] || return 0
  local code; code=$(curl -s -m "$CURL_MAX_S" -o "$f.resp.json" -w '%{http_code}' -X "$2" "$SM_URL$3" \
    -H 'content-type: application/json' --data-binary @"$f.req.json" || echo 000)
  log "  HTTP $code"; report "$f.resp.json" | tee -a "$OUT/smoke.log"
  sleep "$SETTLE_S"
}
plan(){ send "$1" POST /v2/plans "$2"; }

if [ "$RUN" = 1 ]; then
  st="$("$RUN_MODE_SCRIPT" status 2>/dev/null | tr '\n' ' ' || true)"
  case "$st" in *controller_mode=observe*sm_actuation=active*) ;; *) echo "run mode is '$st', need controller observe / sm active" >&2; exit 2 ;; esac
  [ -e "${TRE_EXCLUSIVE_WINDOW:-/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW}" ] || { echo "no exclusive-window marker" >&2; exit 2; }
fi
A0=$(awake "$A"); B0=$(awake "$B"); C0=$(awake "$C")
log "SM $SM_URL out $OUT start awake: $A=$A0 $B=$B0 $C=$C0 (run=$RUN)"
inc(){ [ "$RUN" = 1 ] && echo $(($1 + $2)) || echo "$1+$2"; }

# 1. single replica
send single-target-up PUT "/v2/models/$A/target" "{\"wake_replicas\": $(inc "$A0" 1)}"
send single-target-down PUT "/v2/models/$A/target" "{\"wake_replicas\": $A0, \"sleep_path\": \"scale_down\"}"
plan single-plan-up "{\"plan_id\": \"smoke-single-up\", \"wakes\": [{\"model\": \"$A\", \"at_least\": $(inc "$A0" 1), \"reason\": \"smoke\"}]}"
plan single-plan-down "{\"plan_id\": \"smoke-single-down\", \"sleeps\": [{\"model\": \"$A\", \"at_most\": $A0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}]}"
# 2. multi replica: two wakes of one model in one hold
plan multi-up "{\"plan_id\": \"smoke-multi-up\", \"wakes\": [{\"model\": \"$B\", \"at_least\": $(inc "$B0" 2), \"reason\": \"smoke\"}]}"
plan multi-down "{\"plan_id\": \"smoke-multi-down\", \"sleeps\": [{\"model\": \"$B\", \"at_most\": $B0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}]}"
# 3. multi-model: two models woken in one hold; then both slept back + a third woken in one plan
plan models-up "{\"plan_id\": \"smoke-models-up\", \"wakes\": [{\"model\": \"$A\", \"at_least\": $(inc "$A0" 1), \"reason\": \"smoke\"}, {\"model\": \"$B\", \"at_least\": $(inc "$B0" 1), \"reason\": \"smoke\"}]}"
plan models-swap "{\"plan_id\": \"smoke-models-swap\", \"sleeps\": [{\"model\": \"$A\", \"at_most\": $A0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}, {\"model\": \"$B\", \"at_most\": $B0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}], \"wakes\": [{\"model\": \"$C\", \"at_least\": $(inc "$C0" 1), \"reason\": \"smoke\"}]}"
plan models-restore "{\"plan_id\": \"smoke-models-restore\", \"sleeps\": [{\"model\": \"$C\", \"at_most\": $C0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}]}"
# 4. donor -> receiver (the relay is used when no GPU is free for the receiver; see `woke[].source` / donor ids)
plan donor "{\"plan_id\": \"smoke-donor\", \"wakes\": [{\"model\": \"$B\", \"at_least\": $(inc "$B0" 1), \"donors\": [{\"model\": \"$A\", \"max\": 1, \"sleep_path\": \"urgent\"}], \"reason\": \"smoke\"}]}"
# 6. TP2 receiver over two TP1 donors
plan tp2 "{\"plan_id\": \"smoke-tp2\", \"wakes\": [{\"model\": \"$C\", \"at_least\": $(inc "$C0" 1), \"donors\": [{\"model\": \"$A\", \"max\": 2, \"sleep_path\": \"urgent\"}], \"reason\": \"smoke\"}]}"
# restore: sleeps first, then wakes back to the start counts
plan restore-sleep "{\"plan_id\": \"smoke-restore-sleep\", \"sleeps\": [{\"model\": \"$B\", \"at_most\": $B0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}, {\"model\": \"$C\", \"at_most\": $C0, \"sleep_path\": \"scale_down\", \"reason\": \"smoke\"}]}"
plan restore-wake "{\"plan_id\": \"smoke-restore-wake\", \"wakes\": [{\"model\": \"$A\", \"at_least\": $A0, \"reason\": \"smoke\"}, {\"model\": \"$B\", \"at_least\": $B0, \"reason\": \"smoke\"}, {\"model\": \"$C\", \"at_least\": $C0, \"reason\": \"smoke\"}]}"
if [ "$RUN" = 1 ]; then
  log "end awake: $A=$(awake "$A") $B=$(awake "$B") $C=$(awake "$C") (start $A0 $B0 $C0)"
  curl -s -m 120 "$SM_URL/v2/audit" > "$OUT/audit.json" && log "audit (one call): $(head -c 400 "$OUT/audit.json")"
  log "check gpu-truth never showed two awake on one GPU during the run (tre:gpu_truth:* / gpu_truth.jsonl)"
fi
log "done: $OUT"
