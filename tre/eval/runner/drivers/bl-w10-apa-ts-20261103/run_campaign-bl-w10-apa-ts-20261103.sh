#!/usr/bin/env bash
# 2026-11-03 baselines with a 10 s window: campaign-bl-w10-apa-ts-20261103.yaml = 7 ICSE-suite traces x {APA, TokenScale}, seed 1,
# NO TRE arm, 14 arms, ~3.5 h (spec header: what changed, sources, ETA).
#
# WRITES THE CLUSTER only through the runner (per arm: run mode, APA CRs / baseline scaler, policy ConfigMaps, one controller
# restart, awake layout) and through the exclusive-window marker. It does NOT switch the registry, the controller image or the
# run mode by itself, and it never applies a registry (the baselines read no TRE parameter; the spec pins the registry id).
#
# Parameters (environment):
#   CAMPAIGN_DATA_ROOT  required: the directory with the results roots, the trace sets and the exclusive-window marker
#   CAMPAIGN_WORK_DIR   log / STOP file / runner stdout          (default $CAMPAIGN_DATA_ROOT/campaign-bl-w10-apa-ts-20261103)
#   TRE_REPO            the tre/ tree of the checkout the runner, client, ConfigMap and APA CRs come from (default: this checkout)
#   MAIN_SHA            optional: pin the runner commit (8 hex digits); without it the pushed HEAD of this checkout is accepted
#   ALLOW_UNPUSHED      optional, 1 = accept HEAD != origin/main (explicit user decision; logged as WARN)
#   TEST_MARKER         only to test the guards with a fake marker file
#
#   bash run_campaign-bl-w10-apa-ts-20261103.sh --check
#       read-only: every guard below (git, processes, marker, traces, spec, runner dry-run, registry id, 10 s windows in the
#       checkout, cluster); writes nothing, starts nothing; exit 0 = a launch would pass the guards.
#   cd $CAMPAIGN_WORK_DIR && setsid nohup bash <this file> >> run_campaign-bl-w10-apa-ts-20261103.out 2>&1 < /dev/null &
#       launch (after --check passed and no foreign marker exists).
#
# Clean stop (no race): touch $CAMPAIGN_WORK_DIR/STOP  (forwarded to the results root within 5 s; the runner stops after the
#   current arm). Resume: delete both STOP files ($CAMPAIGN_WORK_DIR/STOP and <results_root>/STOP), launch again (done arms are
#   skipped). SIGKILL of the runner: run tre/eval/runner/reset_canonical.sh by hand.
# After arm 1 (APA x Alternating, ~16 min):
#   python3 $TRE_REPO/eval/runner/first_run_check.py <results_root>/Alternating_hot_model_periodic_A_s1/apa
#   then the same for the first TokenScale arm (trace 1, second position), then every 50 min.
# Logs: run_<NAME>.log (this script), <NAME>.out (runner stdout); campaign.log + status.json in the results root.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # this driver directory (guard, trace sha256 list)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
NAME=campaign-bl-w10-apa-ts-20261103
C="${CAMPAIGN_WORK_DIR:-$X/$NAME}"
T="${TRE_REPO:-$(cd "$HERE/../../../.." && pwd)}"             # the tre/ tree of this checkout
R=$NAME                                                       # round of the spec = marker token
SPEC=$T/eval/runner/campaigns/$NAME.yaml
SHAFILE=$HERE/$NAME.traces.sha256                             # paths relative to $X
ROOT=$X/$NAME-run
MAIN_SHA="${MAIN_SHA:-}"
OWNER=owner=bl-w10-run_$NAME
export CAMPAIGN_DATA_ROOT="$X"
export MARKER="${TEST_MARKER:-$X/TRE_EXCLUSIVE_WINDOW}"
export TRE_REPO=$T TRE_DIR=$T PYTHONPATH=$T/common:$T/deploy PYTHONDONTWRITEBYTECODE=1
LOG=$C/run_$NAME.log
CHECK=0; [ "${1:-}" = --check ] && CHECK=1
[ "$CHECK" = 1 ] || mkdir -p "$C"            # --check writes nothing
RUNNER_PID=""; MARKER_OURS=0; FAIL=0

log() {
  if [ "$CHECK" = 1 ]; then echo "[$(date +%FT%T%z)] $*"; else echo "[$(date +%FT%T%z)] $*" | tee -a "$LOG"; fi
}
bad() { log "ABORT: $*"; FAIL=1; }
counts() { python3 -c "import json;print(json.load(open('$ROOT/status.json'))['counts'])" 2>&1; }
live_id() { python3 $T/eval/runner/arm_registry.py id 2>&1 | tail -1; }
spec_id() { python3 -c "import yaml;print((yaml.safe_load(open('$SPEC')).get('registry_expect') or {}).get('registry_id'))" 2>&1; }

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
  log "driver exit rc=$rc; live registry $(live_id)"
  exit $rc
}
if [ "$CHECK" = 0 ]; then
  trap on_exit EXIT
  trap 'log "signal: stopping"; exit 130' INT TERM
  trap '' HUP
fi

log "driver start pid $$ host $(hostname) mode $([ $CHECK = 1 ] && echo CHECK-read-only || echo RUN) NAME $NAME main $(git -C $T rev-parse --short=8 HEAD) spec $(sha256sum < $SPEC 2>/dev/null | cut -c1-12)"

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
[ "$(git -C $T/.. status --porcelain | wc -l)" = 0 ] || bad "checkout dirty"
if [ "$(git -C $T/.. rev-parse HEAD)" != "$(git -C $T/.. rev-parse origin/main)" ]; then
  if [ "${ALLOW_UNPUSHED:-0}" = 1 ]; then
    log "WARN HEAD != origin/main: running an unpushed main by explicit ALLOW_UNPUSHED=1 (user decision; deviates from AGENTS.md rule 6 until pushed, a fast-forward push keeps the sha)"
  else
    bad "HEAD != origin/main (merge and push first; the runner must be pushed main; ALLOW_UNPUSHED=1 overrides by explicit decision)"
  fi
fi
HEAD8=$(git -C $T/.. rev-parse --short=8 HEAD)
if [ -n "$MAIN_SHA" ]; then
  [ "$HEAD8" = "$(echo "$MAIN_SHA" | cut -c1-8)" ] || bad "HEAD $HEAD8 != MAIN_SHA $MAIN_SHA"
else log "info MAIN_SHA not given: the pushed HEAD $HEAD8 is accepted (give MAIN_SHA=$HEAD8 to pin it)"; fi
# 0c. exclusive-window marker: ours (adopted) or none; a foreign one (calibration / another batch still running) aborts
if [ -e "$MARKER" ]; then
  if grep -qE "round=$R( |$)" "$MARKER"; then log "marker already ours (adopted): $(head -1 "$MARKER")"
  else bad "foreign exclusive-window marker (another batch still running?): $(head -c 300 "$MARKER")"; fi
else log "ok   no exclusive-window marker"; fi
# 0d. spec in the checkout; registry id of the spec = live registry id
[ -s "$SPEC" ] || bad "no spec $SPEC"
SID=$(spec_id); LID=$(live_id)
if echo "$SID" | grep -qE '^[0-9a-f]{16}$' && [ "$SID" = "$LID" ]; then log "ok   spec registry_id $SID = live registry id"
else bad "spec registry_id '$SID' != live registry id '$LID' (the baselines read no TRE parameter, but the pin is checked before every arm too)"; fi
# 0e. the 10 s windows are in this checkout: APA CRs and TokenScale ConfigMap (PreServe is not run here)
WIN=$(python3 - <<PY 2>&1
import glob, yaml
w = {f.rsplit("/", 1)[-1]: yaml.safe_load(open(f))["metadata"]["annotations"]["apa.autoscaling.aibrix.ai/window"]
     for f in sorted(glob.glob("$T/deploy/baselines/apa/*-apa.yaml"))}
cm = {d["metadata"]["name"]: d for d in yaml.safe_load_all(open("$T/deploy/baselines/tre/policy-configmaps.yaml")) if d and d.get("kind") == "ConfigMap"}
w["tokenscale"] = "%gs" % float(yaml.safe_load(cm["tre-v2-baseline-tokenscale"]["data"]["tokenscale.yaml"])["window_s"])
print("OK" if set(w.values()) == {"10s"} and len(w) == 4 else "BAD", w)
PY
)
case "$WIN" in OK*) log "ok   windows in the checkout: ${WIN#OK }" ;; *) bad "APA / TokenScale windows are not all 10 s in $T: $WIN" ;; esac
# 0f. trace files = the files of the earlier samples (sha256), paths relative to the data root
SHAOUT=$(cd "$X" && nice -n 19 sha256sum -c --quiet "$SHAFILE" 2>&1); SHARC=$?
if [ $SHARC = 0 ]; then log "ok   $(wc -l < $SHAFILE) trace files match the sha256 of the earlier samples"
else bad "trace sha256 mismatch (rc $SHARC): $(echo "$SHAOUT" | head -c 400)"; fi
# 0g. runner plan + input validation (no cluster access)
DRY=$(bash $T/eval/runner/run_campaign.sh "$SPEC" --dry-run 2>&1); DRC=$?
if [ $DRC = 0 ]; then log "ok   runner dry-run: $(echo "$DRY" | grep -E '^total:|^dry-run:' | tr '\n' ' ')"
else bad "runner dry-run rc=$DRC: $(echo "$DRY" | grep -E 'PROBLEM' | head -c 600)"; fi
# 0h. cluster: nodes, tre-v2 components and images, 20/20 model pods, no APA CR / Failed pod, node headroom
CG=$(python3 $HERE/guard_cluster.py 2>&1); CGC=$?
echo "$CG" | while IFS= read -r l; do log "  $l"; done
[ $CGC = 0 ] || bad "cluster guard failed (lines FAIL above)"

if [ $FAIL != 0 ]; then log "guards FAILED: nothing started"; exit 1; fi
if [ "$CHECK" = 1 ]; then log "CHECK OK: a launch now would pass every guard (run mode and awake layout are reset by the runner's first pre-check)"; exit 0; fi

# ---- 1. marker (the runner's pre-check wants `validation ... round=$R`; it extends the end by its ETA + margin)
if [ -e "$MARKER" ]; then
  log "marker already ours (adopted)"
else
  S=$(date +%FT%T%:z); E=$(date -d '+5 hours' +%FT%T%:z)
  echo "validation $S $E round=$R $OWNER" > "$MARKER"
  log "marker written: $(cat "$MARKER")"
fi
MARKER_OURS=1
[ -e "$ROOT/status.json" ] && log "results root $ROOT before start: counts $(counts)"

# ---- 2. campaign
log "campaign $SPEC (registry $SID)"
bash $T/eval/runner/run_campaign.sh "$SPEC" >> $C/$NAME.out 2>&1 < /dev/null &
RUNNER_PID=$!
i=0
while kill -0 $RUNNER_PID 2>/dev/null; do
  # STOP forward every 5 s (10-30 lesson: a slow forward let the runner start one more arm)
  [ -e "$C/STOP" ] && [ ! -e "$ROOT/STOP" ] && { touch "$ROOT/STOP"; log "STOP: forwarded to $ROOT"; }
  i=$((i + 1))
  if [ $((i % 60)) = 0 ]; then     # every ~5 min: the registry must stay the pinned one for the whole campaign
    r=$(live_id); [ "$r" = "$SID" ] || log "WARNING: registry id is '$r' (want $SID) during the campaign"
  fi
  sleep 5
done
wait $RUNNER_PID; rc=$?; RUNNER_PID=""
log "campaign rc=$rc (0 done/stopped, 1 failed arms, 2 refused, 3 blocked); counts $(counts)"
exit $rc
