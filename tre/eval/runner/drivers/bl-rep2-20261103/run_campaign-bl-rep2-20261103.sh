#!/usr/bin/env bash
# [repo] Repo copy of the driver as it was run: only paths changed (CAMPAIGN_DATA_ROOT = data root, CAMPAIGN_WORK_DIR = log / STOP / runner-stdout dir,
# [repo] TRE_REPO = the tre/ tree; spec, guards, registry snapshot and trace sha256 list come from the repo). See tre/eval/runner/drivers/README.md.
# 2026-11-03 baselines rep 2 driver: campaign-bl-rep2-20261103.yaml = 7 ICSE traces x {PreServe, TokenScale, APA}, seed 1,
# NO TRE arm, 21 arms, ~5.3 h (spec header: sources, differences, ETA).
#
# WRITES THE CLUSTER only through the runner (per arm: run mode, APA CR / baseline scaler, policy ConfigMaps, one controller
# restart, awake layout) and through the exclusive-window marker. It does NOT switch the registry, the controller image
# or the run mode by itself, and it never applies a registry (the baselines read no TRE parameter).
#
#   bash run_campaign-bl-rep2-20261103.sh --check
#       read-only: every guard below (git, processes, marker, traces, runner dry-run, registry, cluster); writes nothing,
#       starts nothing; exit 0 = a launch would pass the guards.
#   cd $CAMPAIGN_DATA_ROOT/campaign-bl-rep2-20261103 && setsid nohup bash run_campaign-bl-rep2-20261103.sh >> run_campaign-bl-rep2-20261103.out 2>&1 < /dev/null &
#       launch (only after the calibration window has ended and its marker file is gone).
#   Variant with the r3e Real_code / Real_conv traces (the 11-02 comparison): prefix the command with
#       NAME=campaign-bl-rep2-r3e-20261103
#
# Clean stop (no race): touch $CAMPAIGN_DATA_ROOT/campaign-bl-rep2-20261103-run/STOP  (the runner stops after the
#   current arm). Touching $C/STOP also works (forwarded within 5 s).
# Resume: delete both STOP files, launch again (done arms are skipped). SIGKILL of the runner: run reset_canonical.sh by hand.
# After arm 1 (PreServe x Alternating, ~16 min):
#   python3 $T/eval/runner/first_run_check.py <results_root>/Alternating_hot_model_periodic_A_s1/preserve
#   then the same for the first APA arm (the fastpath smoke of main d17bccea covered a TRE arm and TokenScale only), then every 50 min.
# Logs: run_<NAME>.log (this script), <NAME>.out (runner stdout); campaign.log + status.json in the results root.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # [repo] this driver directory (guards, trace sha256 lists)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
C="${CAMPAIGN_WORK_DIR:-$X/campaign-bl-rep2-20261103}"                       # [repo] logs, STOP file, runner stdout (original: the campaign dir next to the spec)
NAME="${NAME:-campaign-bl-rep2-20261103}"
T="${TRE_REPO:-$(cd "$HERE/../../../.." && pwd)}"                      # [repo] the tre/ tree of this checkout
R=$NAME                                      # round of the spec = marker token
SPEC=$T/eval/runner/campaigns/$NAME.yaml
ROOT=$X/$NAME-run
MAIN_SHA="${MAIN_SHA:-d17bccea}"             # runner / client (spec header); override only after a new dry-run
REF=$T/eval/registries/0ab5352747bdd70e.yaml   # registry of the first look (copy of the ConfigMap, 11-03 02:20)
OWNER=owner=bl-rep2-run_$NAME
export MARKER="${TEST_MARKER:-$X/TRE_EXCLUSIVE_WINDOW}"   # TEST_MARKER: only to test the guards with a fake marker file
export TRE_REPO=$T TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy PYTHONDONTWRITEBYTECODE=1
LOG=$C/run_$NAME.log
CHECK=0; [ "${1:-}" = --check ] && CHECK=1
[ "$CHECK" = 1 ] || mkdir -p "$C"            # [repo] --check writes nothing
RUNNER_PID=""; MARKER_OURS=0; FAIL=0

log() {
  if [ "$CHECK" = 1 ]; then echo "[$(date +%FT%T%z)] $*"; else echo "[$(date +%FT%T%z)] $*" | tee -a "$LOG"; fi
}
bad() { log "ABORT: $*"; FAIL=1; }
counts() { python3 -c "import json;print(json.load(open('$ROOT/status.json'))['counts'])" 2>&1; }

stop_runner() {   # clean stop of a running campaign: STOP file, then wait for it (at most one arm, ~17 min)
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
  log "driver exit rc=$rc; live registry $(python3 $HERE/registry_guard.py $REF --id 2>&1 | tail -1)"
  exit $rc
}
if [ "$CHECK" = 0 ]; then
  trap on_exit EXIT
  trap 'log "signal: stopping"; exit 130' INT TERM
  trap '' HUP
fi

log "driver start pid $$ host $(hostname) mode $([ $CHECK = 1 ] && echo CHECK-read-only || echo RUN) NAME $NAME main $(git -C $T rev-parse --short HEAD) spec $(sha256sum < $SPEC 2>/dev/null | cut -c1-12)"

# ---- 0. guards (read-only)
# 0a. no other driver / campaign / arm / calibration process
for p in $(pgrep -f '^bash ([^ ]*/)?run_[A-Za-z0-9_.-]*\.sh( |$)'); do
  [ "$p" = $$ ] && continue
  [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")" = $$ ] && continue     # our own command-substitution subshell
  kill -0 "$p" 2>/dev/null || continue
  case "$(ps -o args= -p "$p" 2>/dev/null)" in
    *run_campaign.sh*) ;;                                                 # the runner wrapper: checked below via campaign.py
    *) bad "another driver / arm script is running (pid $p: $(ps -o args= -p "$p" | head -c 160))" ;;
  esac
done
pgrep -af '^python3 .*campaign\.py run' >/dev/null && bad "a campaign runner is active"
pgrep -af '^bash .*run_arm_pilot\.sh' >/dev/null && bad "run_arm_pilot.sh is active"
pgrep -af '[c]alibration_campaign|[t]re_models\.sh' >/dev/null && bad "a calibration / tre_models.sh process is running: $(pgrep -af '[c]alibration_campaign|[t]re_models\.sh' | head -c 200)"
[ -e "$C/STOP" ] && bad "$C/STOP present (remove it to start)"
[ -e "$ROOT/STOP" ] && bad "$ROOT/STOP present (remove it to start)"
# 0b. runner = pushed main, clean tree
[ "$(git -C $T/.. status --porcelain | wc -l)" = 0 ] || bad "main tree dirty"
[ "$(git -C $T/.. rev-parse HEAD)" = "$(git -C $T/.. rev-parse origin/main)" ] || bad "main != origin/main"
[ "$(git -C $T/.. rev-parse --short=8 HEAD)" = "$MAIN_SHA" ] || bad "main $(git -C $T/.. rev-parse --short=8 HEAD) != $MAIN_SHA (re-run the dry-run, then MAIN_SHA=<sha> to accept)"
# 0c. exclusive-window marker: ours (adopted) or none; a foreign one (the calibration still running) aborts
if [ -e "$MARKER" ]; then
  if grep -qE "round=$R( |$)" "$MARKER"; then log "marker already ours (adopted): $(head -1 "$MARKER")"
  else bad "foreign exclusive-window marker (the calibration window has not ended?): $(head -c 300 "$MARKER")"; fi
else log "ok   no exclusive-window marker"; fi
# 0d. trace files = the files of the first sample (sha256), no pre_t0.json shadowing preserve_history.json
SHAOUT=$(cd "$X" && nice -n 19 sha256sum -c --quiet "$HERE/$NAME.traces.sha256" 2>&1); SHARC=$?
if [ $SHARC = 0 ]; then log "ok   $(wc -l < $HERE/$NAME.traces.sha256) trace files match the sha256 of the first sample"
else bad "trace sha256 mismatch (rc $SHARC): $(echo "$SHAOUT" | head -c 400)"; fi
for d in $(awk -v x="$X" '{print x "/" $2}' "$HERE/$NAME.traces.sha256" | xargs -n1 dirname); do
  [ -z "$(find "$d" -name pre_t0.json -print -quit)" ] || bad "pre_t0.json in $d (would shadow preserve_history.json)"
  [ -s "$d/preserve_history.json" ] || bad "no preserve_history.json in $d"
done
# 0e. runner plan + input validation (no cluster access)
DRY=$(bash $T/eval/runner/run_campaign.sh "$SPEC" --dry-run 2>&1); DRC=$?
if [ $DRC = 0 ]; then log "ok   runner dry-run: $(echo "$DRY" | grep -E '^total:|^dry-run:' | tr '\n' ' ')"
else bad "runner dry-run rc=$DRC: $(echo "$DRY" | grep -E 'PROBLEM' | head -c 600)"; fi
# 0f. registry: the baseline-relevant part must be the one of the first look (TRE parameters may differ)
RG=$(python3 $HERE/registry_guard.py "$REF" 2>&1); RGC=$?
log "registry guard rc=$RGC: $(echo "$RG" | tr '\n' ' ' | head -c 700)"
[ $RGC = 0 ] || bad "baseline-relevant part of the live registry differs from $REF (or unreadable)"
# 0g. cluster: nodes, tre-v2 components and images, 20/20 model pods, no APA CR / Failed pod, node headroom
CG=$(python3 $HERE/guard_cluster.py 2>&1); CGC=$?
echo "$CG" | while IFS= read -r l; do log "  $l"; done
[ $CGC = 0 ] || bad "cluster guard failed (lines FAIL above)"

if [ $FAIL != 0 ]; then log "guards FAILED: nothing started"; [ "$CHECK" = 1 ] && exit 1; exit 1; fi
if [ "$CHECK" = 1 ]; then log "CHECK OK: a launch now would pass every guard (run mode and awake layout are reset by the runner's first pre-check)"; exit 0; fi

# ---- 1. marker (the runner's pre-check wants `validation ... round=$R`; it extends the end by its ETA + margin)
if [ -e "$MARKER" ]; then
  log "marker already ours (adopted)"
else
  S=$(date +%FT%T%:z); E=$(date -d '+6 hours 30 minutes' +%FT%T%:z)
  echo "validation $S $E round=$R $OWNER" > "$MARKER"
  log "marker written: $(cat "$MARKER")"
fi
MARKER_OURS=1
[ -e "$ROOT/status.json" ] && log "results root $ROOT before start: counts $(counts)"

# ---- 2. campaign
log "campaign $SPEC"
bash $T/eval/runner/run_campaign.sh "$SPEC" >> $C/$NAME.out 2>&1 < /dev/null &
RUNNER_PID=$!
i=0
while kill -0 $RUNNER_PID 2>/dev/null; do
  # STOP forward every 5 s (10-30 lesson: a slow forward let the runner start one more arm)
  [ -e "$C/STOP" ] && [ ! -e "$ROOT/STOP" ] && { touch "$ROOT/STOP"; log "STOP: forwarded to $ROOT"; }
  i=$((i + 1))
  if [ $((i % 60)) = 0 ]; then     # every ~5 min: the baseline-relevant registry part must hold for the whole campaign
    python3 $HERE/registry_guard.py "$REF" >/dev/null 2>&1; rc=$?
    [ $rc = 0 ] || log "WARNING: baseline-relevant registry part differs from the reference during the campaign (rc $rc): python3 $HERE/registry_guard.py $REF"
  fi
  sleep 5
done
wait $RUNNER_PID; rc=$?; RUNNER_PID=""
log "campaign rc=$rc (0 done/stopped, 1 failed arms, 2 refused, 3 blocked); counts $(counts)"
exit $rc
