#!/usr/bin/env bash
# [repo] Repo copy of the driver as it was run: only paths changed (CAMPAIGN_DATA_ROOT = data root, CAMPAIGN_WORK_DIR = log / STOP / runner-stdout dir,
# [repo] TRE_REPO = the tre/ tree; spec, guards, registry snapshot and trace sha256 list come from the repo). See tre/eval/runner/drivers/README.md.
# 2026-10-31 ICSE per-trace-k campaign driver (user 10-31: one trace at a time, all 4 arms). WRITES THE CLUSTER.
#   apply win10 (TSS window 10 s, EMA tau 0, theta x0.5) -> ONE campaign campaign-icse-ktrace-20261031.yaml
#   (7 traces x {TRE, TokenScale, PreServe, APA}, seed 1, all arms under win10) -> rollback win10.
# Modelled on campaign-contend-r3-20261030/run_all2.sh. The rollback also runs from the EXIT trap (failure,
# kill -TERM/INT, normal end) whenever the apply was started. kill -9 skips the trap: then run
#   bash $CAMPAIGN_DATA_ROOT/backups/tss-win10-20261030/rollback_win10.sh   by hand.
#
# Launch:
#   cd $CAMPAIGN_DATA_ROOT/campaign-icse-ktrace-20261031 && setsid nohup bash run_all3.sh >> run_all3.out 2>&1 < /dev/null &
# Clean stop (no race): touch $CAMPAIGN_DATA_ROOT/campaign-icse-ktrace-20261031-run/STOP
#   = the runner's own STOP file; the runner checks it before every arm and stops after the current one; then this
#   script rolls win10 back. (Touching $C/STOP also works: it is forwarded within 5 s, but a forward that lands just
#   after an arm boundary lets one more arm start - prefer the runner's file.)
# Resume: delete $ROOT/STOP (and $C/STOP), launch again (done arms are skipped by the runner).
# Logs: run_all3.log (this script), all.out (runner stdout); results + campaign.log + status.json in $ROOT.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # [repo] this driver directory (guards, trace sha256 lists)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
C="${CAMPAIGN_WORK_DIR:-$X/campaign-icse-ktrace-20261031}"                       # [repo] logs, STOP file, runner stdout (original: the campaign dir next to the spec)
mkdir -p "$C"
T="${TRE_REPO:-$(cd "$HERE/../../../.." && pwd)}"                      # [repo] the tre/ tree of this checkout
W="${WIN10_DIR:-$X/analysis-tss-params-20261030/win10}"       # [repo] NOT in the repo: apply_win10.sh, registry/ (drivers/README.md)
B="${WIN10_BACKUP_DIR:-$X/backups/tss-win10-20261030}"         # [repo] NOT in the repo: rollback_win10.sh, cm_replace.py, registry.before.yaml
SPEC=$T/eval/runner/campaigns/campaign-icse-ktrace-20261031.yaml
TRACES=$X/traces-icse-ktrace-20261031
R=campaign-icse-ktrace-20261031              # campaign name = round
ROOT=$X/campaign-icse-ktrace-20261031-run    # results_root
MAIN_SHA=9f2856a1                            # runner / client / images (spec header)
ID_STD=af9aef4701804743                      # canonical W20 registry
ID_WIN10=5028a1209a5fe38d                    # win10 registry (spec registry_expect)
NS=tre-v2; CTL=tre-v2-controller
OWNER=owner=icse-ktrace-run_all3
export MARKER=$X/TRE_EXCLUSIVE_WINDOW TRE_REPO=$T TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy
LOG=$C/run_all3.log
WIN10_LIVE=0; RUNNER_PID=""; MARKER_OURS=0
log() { echo "[$(date +%FT%T%z)] $*" | tee -a "$LOG"; }
reg_id() { python3 $T/eval/runner/arm_registry.py id; }
ctl_window() { kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_METRICS_WINDOW_MS")].value}'; }
counts() { python3 -c "import json;print(json.load(open('$ROOT/status.json'))['counts'])" 2>&1; }

stop_runner() {   # clean stop of a running campaign: STOP file, then wait for it (at most one arm, ~20 min)
  [ -n "$RUNNER_PID" ] && kill -0 "$RUNNER_PID" 2>/dev/null || return 0
  log "runner $RUNNER_PID still running: STOP file -> $ROOT, waiting for it to exit"
  touch "$ROOT/STOP"
  while kill -0 "$RUNNER_PID" 2>/dev/null; do sleep 10; done
}

rollback() {
  [ "$WIN10_LIVE" = 1 ] || return 0
  log "rollback win10: bash $B/rollback_win10.sh"
  if bash $B/rollback_win10.sh >> "$LOG" 2>&1 && [ "$(reg_id)" = $ID_STD ] && [ "$(ctl_window)" = 20000 ]; then
    WIN10_LIVE=0; log "rollback ok: registry id $ID_STD, TRE_METRICS_WINDOW_MS 20000"
  else
    log "ROLLBACK FAILED (registry id $(reg_id), window $(ctl_window)): run $B/rollback_win10.sh by hand"; return 1
  fi
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM
  stop_runner
  rollback || rc=3
  if [ "$MARKER_OURS" = 1 ] && grep -q "round=$R" "$MARKER" 2>/dev/null; then rm -f "$MARKER"; log "marker removed"; fi
  log "run_all3 exit rc=$rc"
  exit $rc
}
trap on_exit EXIT
trap 'log "signal: stopping"; exit 130' INT TERM
trap '' HUP

log "run_all3 start pid $$ host $(hostname) main $(git -C $T rev-parse --short HEAD) spec $SPEC (sha256 $(sha256sum < $SPEC | cut -c1-12))"

# ---- 0. guards (read-only, except the marker)
for p in $(pgrep -f '^bash ([^ ]*/)?run_all[0-9]*\.sh( |$)'); do
  [ "$p" = $$ ] && continue
  [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")" = $$ ] && continue     # our own command-substitution subshell
  kill -0 "$p" 2>/dev/null || continue
  log "ABORT: another run_all*.sh is running (pid $p)"; exit 1
done
pgrep -af '^python3 .*campaign\.py run' >/dev/null && { log "ABORT: a campaign runner is active"; exit 1; }
pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null && { log "ABORT: run_arm_pilot.sh is active"; exit 1; }
[ -e "$C/STOP" ] && { log "ABORT: $C/STOP present (remove it to start)"; exit 1; }
[ -e "$ROOT/STOP" ] && { log "ABORT: $ROOT/STOP present (remove it to start)"; exit 1; }
[ "$(git -C $T/.. status --porcelain | wc -l)" = 0 ] || { log "ABORT: main tree dirty"; exit 1; }
[ "$(git -C $T/.. rev-parse HEAD)" = "$(git -C $T/.. rev-parse origin/main)" ] || { log "ABORT: main != origin/main"; exit 1; }
[ "$(git -C $T/.. rev-parse --short=8 HEAD)" = $MAIN_SHA ] || { log "ABORT: main $(git -C $T/.. rev-parse --short=8 HEAD) != $MAIN_SHA"; exit 1; }
[ -s "$TRACES/SOURCES.md" ] || { log "ABORT: $TRACES/SOURCES.md missing (trace set incomplete)"; exit 1; }
[ -z "$(find $TRACES -name pre_t0.json -print -quit)" ] || { log "ABORT: pre_t0.json in $TRACES (would shadow preserve_history.json)"; exit 1; }
if [ -e "$MARKER" ]; then
  grep -q "round=$R" "$MARKER" || { log "ABORT: foreign exclusive-window marker: $(head -c 300 "$MARKER")"; exit 1; }
  log "marker already ours (adopted): $(head -1 "$MARKER")"
else
  S=$(date +%FT%T%:z); E=$(date -d '+12 hours' +%FT%T%:z)
  echo "validation $S $E round=$R $OWNER" > "$MARKER"
  log "marker written: $(cat "$MARKER")"
fi
MARKER_OURS=1
[ -e "$ROOT/status.json" ] && log "results root $ROOT before start: counts $(counts)"

# ---- 1. apply win10 (same procedure as run_all2.sh)
LIVE_SHA=$(kubectl -n $NS get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' | sha256sum | cut -c1-64)
BASE_SHA=$(sha256sum < $W/registry/live-registry.yaml | cut -c1-64)
[ "$(reg_id)" = $ID_STD ] || { log "ABORT: live registry id $(reg_id) != $ID_STD (win10 still live? old rollback failed?)"; exit 1; }
[ "$(ctl_window)" = 20000 ] || { log "ABORT: TRE_METRICS_WINDOW_MS $(ctl_window) != 20000"; exit 1; }
[ "$LIVE_SHA" = "$BASE_SHA" ] || { log "ABORT: live registry content changed since registry-win10.yaml was built ($LIVE_SHA != $BASE_SHA)"; exit 1; }
# RV lock = the live RV read now; the content lock (--expect-id $ID_STD) and the sha check above stay.
sed "s|^RV=.*|RV=\$(kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.metadata.resourceVersion}')|" \
    $W/apply_win10.sh > $C/apply_win10.ktrace.sh
[ "$(diff $W/apply_win10.sh $C/apply_win10.ktrace.sh | grep -c '^[<>]')" = 2 ] || { log "ABORT: apply copy differs by more than the RV line"; exit 1; }
log "apply win10 ($C/apply_win10.ktrace.sh)"
WIN10_LIVE=1                       # from here on the trap rolls back (a partial apply included)
bash $C/apply_win10.ktrace.sh >> "$LOG" 2>&1 || { log "apply_win10 failed"; exit 2; }
[ "$(reg_id)" = $ID_WIN10 ] && [ "$(ctl_window)" = 10000 ] || { log "win10 not live after apply (id $(reg_id), window $(ctl_window))"; exit 2; }
log "win10 live: registry id $ID_WIN10, TRE_METRICS_WINDOW_MS 10000"
if [ -e "$C/STOP" ] || [ -e "$ROOT/STOP" ]; then log "STOP present after apply: campaign not started"; exit 0; fi

# ---- 2. the campaign (all arms under win10)
log "campaign $SPEC"
bash $T/eval/runner/run_campaign.sh $SPEC >> $C/all.out 2>&1 < /dev/null &
RUNNER_PID=$!
i=0
while kill -0 $RUNNER_PID 2>/dev/null; do
  # STOP forward every 5 s (10-30 lesson: a slow forward let the runner start one more arm)
  [ -e "$C/STOP" ] && [ ! -e "$ROOT/STOP" ] && { touch "$ROOT/STOP"; log "STOP: forwarded to $ROOT"; }
  i=$((i + 1))
  if [ $((i % 60)) = 0 ]; then     # every ~5 min: the window must hold for the whole campaign
    w=$(ctl_window); [ "$w" = 10000 ] || log "WARNING: TRE_METRICS_WINDOW_MS is '$w' (want 10000) during the campaign"
  fi
  sleep 5
done
wait $RUNNER_PID; RC=$?; RUNNER_PID=""
log "campaign rc=$RC (0 done/stopped, 1 failed arms, 2 refused, 3 blocked); counts $(counts); window at end $(ctl_window), registry $(reg_id)"
exit 0
