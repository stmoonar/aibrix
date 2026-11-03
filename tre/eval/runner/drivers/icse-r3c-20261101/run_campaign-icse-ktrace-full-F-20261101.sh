#!/usr/bin/env bash
# [repo] Repo copy of the driver as it was run: only paths changed (CAMPAIGN_DATA_ROOT = data root, CAMPAIGN_WORK_DIR = log / STOP / runner-stdout dir,
# [repo] TRE_REPO = the tre/ tree; spec, guards, registry snapshot and trace sha256 list come from the repo). See tre/eval/runner/drivers/README.md.
# 2026-11-01 ICSE r3c HALF-run driver: campaign-icse-ktrace-full-F-20261101.yaml (5 r3c traces cut to [0, 400) s, seed 1;
# TRE + TokenScale on Alternating / Prefill_mixed / Simultaneous, TRE + TokenScale + PreServe on Real_code / Real_conv:
# 12 arms). WRITES THE CLUSTER only through the runner (arms, run modes, baseline-scaler); it does NOT switch the
# registry and does NOT roll anything back: it runs under the live registry and aborts unless that is ID_EXPECT
# (= the spec's registry_expect). Modelled on campaign-icse-tune-20261031/run_all8.sh (marker, STOP forward, guards).
#
# Launch:
#   cd $CAMPAIGN_DATA_ROOT/campaign-icse-r3c-20261101 && setsid nohup bash run_campaign-icse-ktrace-full-F-20261101.sh >> run_campaign-icse-ktrace-full-F-20261101.out 2>&1 < /dev/null &
# Clean stop (no race): touch $CAMPAIGN_DATA_ROOT/campaign-icse-r3c-20261101/STOP
#   forwarded within 5 s to the results root (the runner's own STOP file: it stops after the current arm).
#   Or touch $ROOT/STOP directly (the runner checks it before every arm).
# Resume: delete $C/STOP and $ROOT/STOP, launch again (done arms are skipped by the runner).
# Registry other than tune C (e.g. tune D): make it live by hand, then set ID_EXPECT below AND registry_id in the spec
# to its id (python3 $T/eval/runner/arm_registry.py id) and re-run the dry-run.
# Logs: run_campaign-icse-ktrace-full-F-20261101.log (this script), campaign-icse-ktrace-full-F-20261101.out (runner stdout); results + campaign.log + status.json in $ROOT.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # [repo] this driver directory (guards, trace sha256 lists)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
C="${CAMPAIGN_WORK_DIR:-$X/campaign-icse-r3c-20261101}"                       # [repo] logs, STOP file, runner stdout (original: the campaign dir next to the spec)
mkdir -p "$C"
T="${TRE_REPO:-$(cd "$HERE/../../../.." && pwd)}"                      # [repo] the tre/ tree of this checkout
TRACE_ROOT=$X/traces-icse-ktrace-20261031
FULL_ROOT=$X/traces-icse-ktrace-20261031
R=campaign-icse-ktrace-full-F-20261101              # round of the spec (marker)
SPEC=$T/eval/runner/campaigns/campaign-icse-ktrace-full-F-20261101.yaml; ROOT=$X/campaign-icse-ktrace-full-F-20261101-run
MAIN_SHA=4ab51261                            # runner / client / images (spec header)
ID_EXPECT=f3ac339a6e5de9be                   # tune D; keep equal to the spec registry_id
WINDOW_EXPECT=10000                          # TRE_METRICS_WINDOW_MS of the live controller (tune A / C / D: 10 s)
NS=tre-v2; CTL=tre-v2-controller
OWNER=owner=icse-r3c-run_campaign-icse-ktrace-full-F-20261101
export MARKER=$X/TRE_EXCLUSIVE_WINDOW TRE_REPO=$T TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
LOG=$C/run_campaign-icse-ktrace-full-F-20261101.log
RUNNER_PID=""; MARKER_OURS=0
log() { echo "[$(date +%FT%T%z)] $*" | tee -a "$LOG"; }
reg_id() { python3 $T/eval/runner/arm_registry.py id; }
ctl_window() { kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_METRICS_WINDOW_MS")].value}'; }
counts() { python3 -c "import json;print(json.load(open('$1/status.json'))['counts'])" 2>&1; }
stopped() { [ -e "$C/STOP" ] || [ -e "$ROOT/STOP" ]; }

stop_runner() {   # clean stop of a running campaign: STOP file, then wait for it (at most one arm, ~14 min)
  [ -n "$RUNNER_PID" ] && kill -0 "$RUNNER_PID" 2>/dev/null || return 0
  log "runner $RUNNER_PID still running: STOP file -> $ROOT, waiting for it to exit"
  touch "$ROOT/STOP"
  while kill -0 "$RUNNER_PID" 2>/dev/null; do sleep 10; done
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM
  stop_runner
  if [ "$MARKER_OURS" = 1 ] && grep -qE "round=$R( |$)" "$MARKER" 2>/dev/null; then rm -f "$MARKER"; log "marker removed"; fi
  log "run_r3c_half exit rc=$rc; live registry $(reg_id), window $(ctl_window) (nothing to roll back)"
  exit $rc
}
trap on_exit EXIT
trap 'log "signal: stopping"; exit 130' INT TERM
trap '' HUP

log "run_r3c_half start pid $$ host $(hostname) main $(git -C $T rev-parse --short HEAD) spec $(sha256sum < $SPEC | cut -c1-12)"

# ---- 0. guards (read-only, except the marker)
for p in $(pgrep -f '^bash ([^ ]*/)?run_[A-Za-z0-9_]*\.sh( |$)'); do
  [ "$p" = $$ ] && continue
  [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")" = $$ ] && continue     # our own command-substitution subshell
  kill -0 "$p" 2>/dev/null || continue
  case "$(ps -o args= -p "$p" 2>/dev/null)" in
    *run_campaign.sh*) ;;                                                 # checked below via campaign.py
    *) log "ABORT: another campaign driver is running (pid $p: $(ps -o args= -p "$p" | head -c 200))"; exit 1 ;;
  esac
done
pgrep -af '^python3 .*campaign\.py run' >/dev/null && { log "ABORT: a campaign runner is active"; exit 1; }
pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null && { log "ABORT: run_arm_pilot.sh is active"; exit 1; }
stopped && { log "ABORT: a STOP file is present ($C/STOP or $ROOT/STOP; remove it to start)"; exit 1; }
[ "$(git -C $T/.. status --porcelain | wc -l)" = 0 ] || { log "ABORT: main tree dirty"; exit 1; }
[ "$(git -C $T/.. rev-parse HEAD)" = "$(git -C $T/.. rev-parse origin/main)" ] || { log "ABORT: main != origin/main"; exit 1; }
[ "$(git -C $T/.. rev-parse --short=8 HEAD)" = $MAIN_SHA ] || { log "ABORT: main $(git -C $T/.. rev-parse --short=8 HEAD) != $MAIN_SHA"; exit 1; }
for TR in $TRACE_ROOT $FULL_ROOT; do
  [ -s "$TR/SOURCES.md" ] || { log "ABORT: $TR/SOURCES.md missing (trace set incomplete)"; exit 1; }
  [ -z "$(find $TR -name pre_t0.json -print -quit)" ] || { log "ABORT: pre_t0.json in $TR (would shadow preserve_history.json)"; exit 1; }
done
grep -q "registry_id: $ID_EXPECT" "$SPEC" || { log "ABORT: $SPEC does not expect registry $ID_EXPECT (keep spec and ID_EXPECT equal)"; exit 1; }
LIVE=$(reg_id)
[ "$LIVE" = $ID_EXPECT ] || { log "ABORT: live registry id $LIVE != $ID_EXPECT (this driver never switches the registry)"; exit 1; }
[ "$(ctl_window)" = $WINDOW_EXPECT ] || { log "ABORT: TRE_METRICS_WINDOW_MS $(ctl_window) != $WINDOW_EXPECT"; exit 1; }
if [ -e "$MARKER" ]; then
  grep -qE "round=$R( |$)" "$MARKER" || { log "ABORT: foreign exclusive-window marker: $(head -c 300 "$MARKER")"; exit 1; }
  log "marker already ours (adopted): $(head -1 "$MARKER")"
else
  S=$(date +%FT%T%:z); E=$(date -d '+3 hours' +%FT%T%:z)
  echo "validation $S $E round=$R $OWNER" > "$MARKER"
  log "marker written: $(cat "$MARKER")"
fi
MARKER_OURS=1
[ -e "$ROOT/status.json" ] && log "results root $ROOT before start: counts $(counts $ROOT)"
log "live registry $LIVE, window $(ctl_window): starting (no apply, no rollback)"

# ---- 1. campaign
log "campaign $SPEC"
bash $T/eval/runner/run_campaign.sh $SPEC >> $C/campaign-icse-ktrace-full-F-20261101.out 2>&1 < /dev/null &
RUNNER_PID=$!
i=0
while kill -0 $RUNNER_PID 2>/dev/null; do
  # STOP forward every 5 s (10-30 lesson: a slow forward let the runner start one more arm)
  [ -e "$C/STOP" ] && [ ! -e "$ROOT/STOP" ] && { touch "$ROOT/STOP"; log "STOP: forwarded to $ROOT"; }
  i=$((i + 1))
  if [ $((i % 60)) = 0 ]; then     # every ~5 min: window and registry must hold for the whole campaign
    w=$(ctl_window); [ "$w" = $WINDOW_EXPECT ] || log "WARNING: TRE_METRICS_WINDOW_MS is '$w' (want $WINDOW_EXPECT) during the campaign"
    r=$(reg_id); [ "$r" = $ID_EXPECT ] || log "WARNING: registry id is '$r' (want $ID_EXPECT) during the campaign"
  fi
  sleep 5
done
wait $RUNNER_PID; rc=$?; RUNNER_PID=""
log "campaign rc=$rc (0 done/stopped, 1 failed arms, 2 refused, 3 blocked); counts $(counts $ROOT); window at end $(ctl_window), registry $(reg_id)"
exit $rc
