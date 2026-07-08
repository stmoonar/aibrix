#!/usr/bin/env bash
# Switch experiment 3's decision source between TRE and the APA (KVCache) baseline, or
# report which one is currently in charge. Exactly ONE decision source must be active at a
# time, so every switch STOPS the old source and verifies it is gone BEFORE starting the new
# one (endgame plan / REFACTOR_PLAN experiment-3 arms; supersedes the old
# CustomTraceGenerator/toggle_tre_apa_hot_switch.sh, which used pre-v2 resource names).
#
#   tre    : APA off  -> delete APA PodAutoscaler CRs, verify none remain,
#            then TRE on -> set ENABLE_TRE_SCALING=true and restart the controller.
#   apa    : TRE off  -> set ENABLE_TRE_SCALING=false, wait for the rollout, verify it is off,
#            then APA on -> apply the APA PodAutoscaler CRs.
#   status : print the active source (checks the controller env AND the live PA CRs).
#
# Both the TRE controller (tre-v2 ns) and the patched aibrix podautoscaler controller
# (aibrix-system) route scaling through service-manager, so leaving both live would let them
# fight over the same pods -- hence the strict stop-old-then-start-new ordering here.
set -euo pipefail

TRE_NS="${TRE_NS:-tre-v2}"
CONTROLLER_DEPLOY="${CONTROLLER_DEPLOY:-tre-v2-controller}"
APA_NS="${APA_NS:-default}"
APA_DIR="${APA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../baselines/apa" && pwd)}"
APA_CRS=(dsqwen-7b-apa.yaml dsllama-8b-apa.yaml dsqwen-14b-apa.yaml)

log() { echo "[toggle] $*"; }
die() { echo "[toggle][ERROR] $*" >&2; exit 1; }

tre_scaling_enabled() {
  local v
  v="$(kubectl -n "$TRE_NS" get deploy "$CONTROLLER_DEPLOY" \
        -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="ENABLE_TRE_SCALING")].value}' 2>/dev/null || true)"
  [[ "$v" == "true" || "$v" == "True" || "$v" == "1" ]]
}

apa_cr_count() {
  kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai \
    -l tre.aibrix.io/baseline=apa -o name 2>/dev/null | grep -c . || true
}

delete_apa_crs() {
  for f in "${APA_CRS[@]}"; do
    kubectl -n "$APA_NS" delete -f "$APA_DIR/$f" --ignore-not-found --wait=true
  done
}

apply_apa_crs() {
  for f in "${APA_CRS[@]}"; do
    kubectl -n "$APA_NS" apply -f "$APA_DIR/$f"
  done
}

set_tre_scaling() {
  kubectl -n "$TRE_NS" set env "deploy/$CONTROLLER_DEPLOY" "ENABLE_TRE_SCALING=$1"
  kubectl -n "$TRE_NS" rollout status "deploy/$CONTROLLER_DEPLOY" --timeout=120s
}

cmd_tre() {
  log "switching to TRE"
  log "1/3 stopping APA baseline: deleting PodAutoscaler CRs"
  delete_apa_crs
  local n; n="$(apa_cr_count)"
  [[ "$n" -eq 0 ]] || die "APA still has $n PodAutoscaler CR(s); refusing to enable TRE (would double-drive scaling)"
  log "2/3 verified 0 APA PodAutoscaler CRs"
  log "3/3 enabling TRE scaling + restarting controller"
  set_tre_scaling true
  kubectl -n "$TRE_NS" rollout restart "deploy/$CONTROLLER_DEPLOY"
  kubectl -n "$TRE_NS" rollout status "deploy/$CONTROLLER_DEPLOY" --timeout=120s
  log "done: TRE is the active decision source"
}

cmd_apa() {
  log "switching to APA (KVCache baseline)"
  log "1/3 stopping TRE: ENABLE_TRE_SCALING=false"
  set_tre_scaling false
  if tre_scaling_enabled; then die "TRE scaling still enabled after set env; refusing to apply APA (would double-drive scaling)"; fi
  log "2/3 verified TRE scaling is off"
  log "3/3 applying APA PodAutoscaler CRs"
  apply_apa_crs
  log "done: APA is the active decision source"
}

cmd_status() {
  local tre="off" n
  tre_scaling_enabled && tre="on"
  n="$(apa_cr_count)"
  echo "TRE scaling (ENABLE_TRE_SCALING): $tre"
  echo "APA PodAutoscaler CRs live:        $n"
  if [[ "$tre" == "on" && "$n" -eq 0 ]]; then
    echo "active decision source: TRE"
  elif [[ "$tre" == "off" && "$n" -gt 0 ]]; then
    echo "active decision source: APA"
  elif [[ "$tre" == "off" && "$n" -eq 0 ]]; then
    echo "active decision source: NONE (both stopped)"
  else
    echo "active decision source: CONFLICT (both TRE and APA are live -- run 'tre' or 'apa' to fix)"
  fi
}

main() {
  case "${1:-}" in
    tre) cmd_tre ;;
    apa) cmd_apa ;;
    status) cmd_status ;;
    *) echo "usage: $0 {tre|apa|status}" >&2; exit 2 ;;
  esac
}

main "$@"
