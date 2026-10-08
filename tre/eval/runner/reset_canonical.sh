#!/usr/bin/env bash
# Put the cluster back in the campaign's canonical idle state (run_campaign.sh calls it after
# every arm, and with --force before a retry and when a pre-check needs it):
#   run mode observe/active; decision source NONE (0 APA CRs, baseline scaler at 0 replicas, no
#   baseline owner lock); no hidden binding; awake set == $BASELINE (runner.env); engines idle;
#   (2026-10-08) controller ablation switches false (unset counts as false).
# Usage: reset_canonical.sh [--check | --force]
#   --check  report only: exit 0 when canonical, 1 when not (read-only)
#   (none)   reset only when not canonical
#   --force  always run the full sequence
# Exit 0 = canonical afterwards, 3 = could not reach it (see the output).
# CHANGES CLUSTER STATE (except --check): run mode, APA CRs, baseline scaler, SM awake set,
# controller ablation switches (set env + rollout).
set -euo pipefail
# shellcheck source=lib_env.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_env.sh"
need_env TRE_DIR BASELINE TRE_NS SM_SVC SM_PORT APA_NS BL_NS BL_DEPLOY MODEL_NS MODEL_SELECTOR ENGINE_PORT \
  REDIS_SVC REDIS_PORT GW_NS GW_SELECTOR CONTROLLER_DEPLOY
MODE="${1:-auto}"
case "$MODE" in --check|--force|auto) ;; *) echo "usage: $0 [--check|--force]" >&2; exit 2;; esac
TRE="$TRE_DIR"
REDIS_DEPLOY="${REDIS_DEPLOY:-tre-v2-redis}"
BL_OWNER_KEY="${BL_OWNER_KEY:-tre:v2:bl:owner}"
SMIP=$(kubectl -n "$TRE_NS" get svc "$SM_SVC" -o jsonpath='{.spec.clusterIP}')
SM=http://$SMIP:$SM_PORT
export SM_URL=$SM
REDIS_HOST=$(kubectl -n "$TRE_NS" get svc "$REDIS_SVC" -o jsonpath='{.spec.clusterIP}')
REDIS_URL="redis://$REDIS_HOST:$REDIS_PORT/0"
log() { echo "[$(date +%F' '%T)] reset: $*"; }
WANT=$(echo $BASELINE | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')
sm_state() { curl -s --max-time 10 "$SM/v2/state"; }
awake_ids() { sm_state | python3 -c 'import sys,json;d=json.load(sys.stdin);print(" ".join(sorted(b["binding_id"] for b in d["bindings"] if b["awake"])))'; }
hidden_n() { sm_state | python3 -c 'import sys,json;print(sum(bool(b.get("hidden")) for b in json.load(sys.stdin)["bindings"]))'; }
run_mode() { bash "$TRE/deploy/scripts/set_run_mode.sh" status 2>/dev/null | tr '\n' ' ' | sed 's/ $//'; }
mode_set() { bash "$TRE/deploy/scripts/set_run_mode.sh" "$1" "$2" >/dev/null; log "run mode -> $(run_mode)"; }
n_apa() { kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o name 2>/dev/null | grep -c . || true; }
bl_replicas() { kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0; }
bl_owner() { kubectl -n "$TRE_NS" exec "deploy/$REDIS_DEPLOY" -- redis-cli --raw GET "$BL_OWNER_KEY" 2>/dev/null | tr -d '\r' || true; }
CTL_ENV_KEYS="TRE_ABLATION_DISABLE_SAFESCALE TRE_ABLATION_DISABLE_SLOW_LOOP"   # = run_arm_pilot.sh note 14
ctl_env_get() { kubectl -n "$TRE_NS" get deploy "$CONTROLLER_DEPLOY" -o jsonpath="{.spec.template.spec.containers[0].env[?(@.name==\"$1\")].value}"; }
ctl_not_false() { local k v; for k in $CTL_ENV_KEYS; do v=$(ctl_env_get "$k" | tr '[:upper:]' '[:lower:]'); [ "${v:-false}" = false ] || echo "$k=$v"; done; }
ARM_TOOL() { env PYTHONPATH="$TRE/common:$TRE/deploy:$TRE/baselines" python3 -m tre_baselines.tools.arm "$@" \
               --namespace "$BL_NS" --deployment "$BL_DEPLOY" --redis-url "$REDIS_URL" --gw-namespace "$GW_NS" --gw-selector "$GW_SELECTOR"; }

problems() {   # prints one line per deviation from the canonical state
  local rm got h a r o c
  rm=$(run_mode); case "$rm" in *"controller_mode=observe"*"sm_actuation=active"*) ;; *) echo "run mode: $rm";; esac
  got=$(awake_ids || echo "?"); [ "$got" = "$WANT" ] || echo "awake [$got] != canonical [$WANT]"
  h=$(hidden_n || echo "?"); [ "$h" = 0 ] || echo "hidden bindings: $h"
  a=$(n_apa); [ "$a" = 0 ] || echo "APA CRs live: $a"
  r=$(bl_replicas); [ "${r:-0}" = 0 ] || echo "baseline scaler replicas: $r"
  o=$(bl_owner); [ -z "$o" ] || echo "baseline owner lock: $o"
  c=$(ctl_not_false | tr '\n' ' '); [ -z "$c" ] || echo "controller ablation switches not false: $c"
}

P=$(problems)
if [ -z "$P" ] && [ "$MODE" != --force ]; then log "canonical: $(run_mode); awake [$WANT]"; exit 0; fi
[ -z "$P" ] || log "not canonical: $(echo "$P" | tr '\n' ';')"
[ "$MODE" = --check ] && exit 1

mode_set observe observe
# controller ablation switches back to production (false) - after observe/observe, before anything else
CF=$(ctl_not_false | sed 's/=.*/=false/' | tr '\n' ' ')
if [ -n "$CF" ]; then
  # shellcheck disable=SC2086
  if kubectl -n "$TRE_NS" set env "deploy/$CONTROLLER_DEPLOY" $CF && kubectl -n "$TRE_NS" rollout status "deploy/$CONTROLLER_DEPLOY" --timeout=180s; then
    log "controller switches -> $CF"
  else
    log "ERROR controller switch reset failed ($CF)"
  fi
fi
# baseline shell: disable through the arm tool (releases the owner lock), then make sure it is at 0
if [ "$(bl_replicas)" != 0 ] || [ -n "$(bl_owner)" ]; then
  ARM_TOOL disable --skip-collect --execute 2>&1 | tail -5 || log "WARN arm disable failed"
  [ "$(bl_replicas)" = 0 ] || kubectl -n "$BL_NS" scale "deploy/$BL_DEPLOY" --replicas=0
fi
# APA CRs off (decision source NONE); the run mode is set explicitly here
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode 2>&1 | tail -3 || log "WARN toggle tre failed"
for i in $(seq 1 60); do [ "$(hidden_n || echo 1)" = 0 ] && break; sleep 5; done
[ "$(hidden_n || echo 1)" = 0 ] || log "WARN hidden bindings remain after 300 s"
log "awake before restore: [$(awake_ids || echo ?)]"
# shellcheck disable=SC2086
python3 "$TRE/deploy/scripts/release/awake_ctl.py" restore-ids $BASELINE 2>&1 | tail -5 || log "ERROR restore-ids failed"
mode_set observe active
for i in $(seq 1 60); do   # every Ready model pod idle (running + waiting == 0), max 300 s
  BUSY=$(kubectl -n "$MODEL_NS" get pods -l "$MODEL_SELECTOR" -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}' | while read -r IP; do
    [ -n "$IP" ] || continue
    curl -s --max-time 3 "http://$IP:$ENGINE_PORT/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}'
  done | awk '{s+=$1} END {print s+0}')
  [ "$BUSY" = 0 ] && break; sleep 5
done
[ "${BUSY:-0}" = 0 ] || log "WARN engines still busy after 300 s ($BUSY)"
P=$(problems)
if [ -n "$P" ]; then log "ERROR still not canonical: $(echo "$P" | tr '\n' ';')"; exit 3; fi
log "canonical: $(run_mode); awake [$WANT]"
