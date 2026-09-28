#!/usr/bin/env bash
# Set (or show) the two INDEPENDENT run-mode switches in the tre-v2 Redis
# (user decision 2026-09-28; tre/docs/design/20260928-observe-mode-semantics.md):
#
#   tre:v2:controller:mode  active|observe  TRE controller acts / only computes and records
#   tre:v2:sm:actuation     active|observe  SM supervisor self-heal (B7 recreate, drift fleet
#                                           repair, resume repair, reap, unrequested admission)
#
# A missing key is observe for its reader (fail-closed), so every deployment step sets
# both explicitly:
#
#   TRE-arm experiment        set_run_mode.sh active  active
#   APA-arm experiment        set_run_mode.sh observe active
#   calibration / maintenance set_run_mode.sh observe observe
#
# usage: set_run_mode.sh <controller_mode|-> <sm_actuation|->   ("-" = leave unchanged)
#        set_run_mode.sh status
# Both given -> one MSET (atomic). The values are read back and verified.
# Equivalent console API: POST /api/ops/run-mode {"controller": ..., "sm_actuation": ...}.
set -euo pipefail

TRE_NS="${TRE_NS:-tre-v2}"
REDIS_DEPLOY="${REDIS_DEPLOY:-tre-v2-redis}"
KUBECTL="${KUBECTL:-kubectl}"
CONTROLLER_KEY="tre:v2:controller:mode"
SM_KEY="tre:v2:sm:actuation"

die() { echo "[run-mode][ERROR] $*" >&2; exit 1; }
rcli() { "$KUBECTL" -n "$TRE_NS" exec "deploy/$REDIS_DEPLOY" -- redis-cli --raw "$@"; }

show() {
  local c s
  c="$(rcli GET "$CONTROLLER_KEY")"
  s="$(rcli GET "$SM_KEY")"
  echo "controller_mode=${c:-<missing: observe>}"
  echo "sm_actuation=${s:-<missing: observe>}"
  [[ -n "$c" ]] || echo "[run-mode][WARN] $CONTROLLER_KEY missing -> controller treats as observe" >&2
  [[ -n "$s" ]] || echo "[run-mode][WARN] $SM_KEY missing -> SM treats as observe" >&2
}

valid() { [[ "$1" == "active" || "$1" == "observe" || "$1" == "-" ]]; }

main() {
  if [[ "${1:-}" == "status" ]]; then show; return; fi
  [[ $# -eq 2 ]] || die "usage: $0 <controller_mode|-> <sm_actuation|-> | status"
  local controller="$1" sm="$2"
  valid "$controller" || die "controller mode must be active|observe|-, got '$controller'"
  valid "$sm" || die "sm_actuation must be active|observe|-, got '$sm'"
  if [[ "$controller" != "-" && "$sm" != "-" ]]; then
    rcli MSET "$CONTROLLER_KEY" "$controller" "$SM_KEY" "$sm" >/dev/null
  elif [[ "$controller" != "-" ]]; then
    rcli SET "$CONTROLLER_KEY" "$controller" >/dev/null
  elif [[ "$sm" != "-" ]]; then
    rcli SET "$SM_KEY" "$sm" >/dev/null
  else
    die "nothing to set"
  fi
  if [[ "$controller" != "-" ]]; then
    [[ "$(rcli GET "$CONTROLLER_KEY")" == "$controller" ]] || die "$CONTROLLER_KEY read back differs"
  fi
  if [[ "$sm" != "-" ]]; then
    [[ "$(rcli GET "$SM_KEY")" == "$sm" ]] || die "$SM_KEY read back differs"
  fi
  show
}

main "$@"
