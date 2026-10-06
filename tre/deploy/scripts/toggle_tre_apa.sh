#!/usr/bin/env bash
# Switch experiment 3's decision source between TRE and the APA (KVCache) baseline, or
# report which one is currently in charge. Exactly ONE decision source must be active at a
# time, so every switch STOPS the old source and verifies it is gone BEFORE starting the new
# one (endgame plan / REFACTOR_PLAN experiment-3 arms; supersedes the old
# CustomTraceGenerator/toggle_tre_apa_hot_switch.sh, which used pre-v2 resource names).
#
#   tre    : APA off  -> delete APA PodAutoscaler CRs, verify none remain (and no baseline
#            shell holds the owner lock), restart the controller, then controller active.
#   apa    : TRE off  -> controller observe (verified), no baseline owner lock, then APA on
#            -> apply the APA PodAutoscaler CRs.
#   status : print the active source and the run mode.
#
# Decision source (2026-10-07): the TRE controller acts only in run mode `active`; in
# `observe` it still computes and records signals and decisions (counterfactual log of
# the APA and baseline arms) but never actuates. The controller's former scaling env
# switch was removed (it stopped the whole decision pipeline, not just scaling). So the source is read from
#   TRE      = tre:v2:controller:mode == active
#   APA      = live PodAutoscaler CRs (label tre.aibrix.io/baseline=apa)
#   BASELINE = the baseline shell's owner lock tre:v2:bl:owner is held
# exactly one of them = that source, none = NONE, more than one = CONFLICT.
#
# Run mode (2026-09-28, set_run_mode.sh): the controller mode and the SM actuation are
# independent switches, and BOTH arms run the SM actuation active (symmetric self-heal):
#   tre -> controller active  + SM active   (set once TRE is the only decision source)
#   apa -> controller observe + SM active   (set FIRST: the controller stops actuating before the APA CRs go live)
# --keep-run-mode leaves both keys alone (campaign_queue.py sets them itself per arm).
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
# Inert 0-replica scale anchors; the patched podautoscaler still resolves scaleTargetRef
# to read the pod selector for KVCache scraping, so APA errors FailedGetScale without them.
APA_ANCHORS=(dsqwen-7b-apa-anchor.yaml dsllama-8b-apa-anchor.yaml dsqwen-14b-apa-anchor.yaml)

SET_RUN_MODE="${SET_RUN_MODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_run_mode.sh}"
REDIS_DEPLOY="${REDIS_DEPLOY:-tre-v2-redis}"
KUBECTL="${KUBECTL:-kubectl}"
CONTROLLER_MODE_KEY="tre:v2:controller:mode"
BL_OWNER_KEY="tre:v2:bl:owner"
KEEP_RUN_MODE=0

log() { echo "[toggle] $*"; }

# set_run_mode <controller_mode> <sm_actuation>, unless --keep-run-mode.
set_run_mode() {
  if [[ "$KEEP_RUN_MODE" -eq 1 ]]; then
    log "run mode left unchanged (--keep-run-mode)"
    return
  fi
  log "run mode: controller=$1 sm_actuation=$2"
  bash "$SET_RUN_MODE" "$1" "$2"
}
die() { echo "[toggle][ERROR] $*" >&2; exit 1; }

rcli() { "$KUBECTL" -n "$TRE_NS" exec "deploy/$REDIS_DEPLOY" -- redis-cli --raw "$@"; }

# Redis reads are fail-closed: an unreadable key aborts instead of reading as observe / no
# owner (else `apa --keep-run-mode` could apply the CRs while the controller is still active).
# Controller run mode, trimmed + lowercased like the controller's parse_mode. Empty = key
# missing = observe (controller semantics); anything but active / observe aborts.
controller_mode() {
  local v; v="$(rcli GET "$CONTROLLER_MODE_KEY" | tr -d '\r')" || die "redis unreadable (GET $CONTROLLER_MODE_KEY)"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "$v" in
    active) echo active ;;
    observe|"") echo observe ;;
    *) die "unexpected controller mode '$v' in $CONTROLLER_MODE_KEY" ;;
  esac
}

# Baseline shell owner lock holder ("" = no shell holds it).
bl_owner() {
  local v; v="$(rcli GET "$BL_OWNER_KEY" | tr -d '\r')" || die "redis unreadable (GET $BL_OWNER_KEY)"
  printf '%s' "$v"
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

apply_apa_anchors() {
  for f in "${APA_ANCHORS[@]}"; do
    kubectl -n "$APA_NS" apply -f "$APA_DIR/$f"
  done
}

delete_apa_anchors() {
  for f in "${APA_ANCHORS[@]}"; do
    kubectl -n "$APA_NS" delete -f "$APA_DIR/$f" --ignore-not-found --wait=true
  done
}

cmd_tre() {
  log "switching to TRE"
  log "1/3 stopping APA baseline: deleting PodAutoscaler CRs"
  delete_apa_crs
  local n owner; n="$(apa_cr_count)"
  [[ "$n" -eq 0 ]] || die "APA still has $n PodAutoscaler CR(s); refusing to enable TRE (would double-drive scaling)"
  owner="$(bl_owner)"
  [[ -z "$owner" ]] || die "a baseline shell holds $BL_OWNER_KEY ($owner); refusing to enable TRE (would double-drive scaling)"
  log "2/3 verified 0 APA PodAutoscaler CRs and no baseline owner lock; deleting APA scale anchors"
  delete_apa_anchors
  log "3/3 restarting the controller"
  kubectl -n "$TRE_NS" rollout restart "deploy/$CONTROLLER_DEPLOY"
  kubectl -n "$TRE_NS" rollout status "deploy/$CONTROLLER_DEPLOY" --timeout=120s
  set_run_mode active active
  local m; m="$(controller_mode)"
  if [[ "$m" == active ]]; then
    log "done: TRE is the active decision source (controller mode active)"
  else
    log "done: controller mode is $m; the controller will not actuate until run mode is active (not TRE-driven yet)"
  fi
}

cmd_apa() {
  log "switching to APA (KVCache baseline)"
  # Controller observe first: the TRE controller stops acting before anything else
  # changes (it keeps computing and logging its decisions). SM actuation active, as in
  # the TRE arm.
  set_run_mode observe active
  log "1/3 stopping TRE: controller run mode observe"
  [[ "$(controller_mode)" == observe ]] || die "TRE controller mode is still active; refusing to apply APA (would double-drive scaling)"
  local owner; owner="$(bl_owner)"
  [[ -z "$owner" ]] || die "a baseline shell holds $BL_OWNER_KEY ($owner); refusing to apply APA (would double-drive scaling)"
  log "2/3 verified the TRE controller observes and no baseline owner lock"
  log "3/3 applying APA scale anchors + PodAutoscaler CRs"
  apply_apa_anchors
  apply_apa_crs
  log "done: APA is the active decision source"
}

cmd_status() {
  local mode n owner sources=()
  mode="$(controller_mode)"
  n="$(apa_cr_count)"
  owner="$(bl_owner)"
  echo "TRE controller mode:               $mode (observe = decisions recorded, no actuation)"
  echo "APA PodAutoscaler CRs live:        $n"
  echo "baseline owner lock ($BL_OWNER_KEY): ${owner:-none}"
  [[ "$mode" == active ]] && sources+=(TRE)
  [[ "$n" -gt 0 ]] && sources+=(APA)
  [[ -n "$owner" ]] && sources+=(BASELINE)
  if [[ ${#sources[@]} -eq 0 ]]; then
    echo "active decision source: NONE (TRE observes, no APA CR, no baseline shell)"
  elif [[ ${#sources[@]} -eq 1 ]]; then
    echo "active decision source: ${sources[0]}"
  else
    echo "active decision source: CONFLICT (${sources[*]} are all live -- run 'tre' or 'apa' to fix)"
  fi
  bash "$SET_RUN_MODE" status || true
}

main() {
  if [[ "${2:-}" == "--keep-run-mode" ]]; then
    KEEP_RUN_MODE=1
  elif [[ -n "${2:-}" ]]; then
    echo "usage: $0 {tre|apa|status} [--keep-run-mode]" >&2; exit 2
  fi
  case "${1:-}" in
    tre) cmd_tre ;;
    apa) cmd_apa ;;
    status) cmd_status ;;
    *) echo "usage: $0 {tre|apa|status} [--keep-run-mode]" >&2; exit 2 ;;
  esac
}

main "$@"
