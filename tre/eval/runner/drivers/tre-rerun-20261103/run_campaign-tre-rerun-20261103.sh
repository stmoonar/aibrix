#!/usr/bin/env bash
# [repo] Repo copy of the driver as it was run: only paths changed (CAMPAIGN_DATA_ROOT = data root, CAMPAIGN_WORK_DIR = log / STOP / runner-stdout dir,
# [repo] TRE_REPO = the tre/ tree; spec, guards, registry snapshot and trace sha256 list come from the repo). See tre/eval/runner/drivers/README.md.
# 2026-11-03 TRE rerun driver: campaign-tre-rerun-r3e-20261103.yaml = 7 traces x {TRE} with the recalibrated TSS parameters,
# seed 1, 7 arms, ~1.8 h per rep (spec header: sources, differences, ETA). Rep 2: NAME=campaign-tre-rerun-r3e-rep2-20261103.
#
# WRITES THE CLUSTER only through the runner (per arm: run mode, one controller rollout with the arm's switches, awake
# layout) and through the exclusive-window marker. It does NOT publish parameters, switch the registry, the controller
# image or the run mode by itself: publish first (CHECKLIST.md), then this driver only checks.
#
#   bash run_campaign-tre-rerun-20261103.sh --check
#       read-only: every guard below (git, processes, marker, spec placeholder, traces, runner dry-run, registry = TSS-only
#       change, params in sync, controller image / window, cluster); writes nothing; exit 0 = a launch would pass.
#   cd $CAMPAIGN_DATA_ROOT/campaign-tre-rerun-draft-20261103 && setsid nohup bash run_campaign-tre-rerun-20261103.sh >> run_campaign-tre-rerun-20261103.out 2>&1 < /dev/null &
#       launch rep 1 (only after the baselines rep 2 has ended and no foreign marker exists).
#   rep 2 alone: prefix NAME=campaign-tre-rerun-r3e-rep2-20261103 ; both reps back to back (rep 2 only if rep 1 ended done):
#       setsid nohup bash -c 'bash run_campaign-tre-rerun-20261103.sh && NAME=campaign-tre-rerun-r3e-rep2-20261103 bash run_campaign-tre-rerun-20261103.sh' >> run_campaign-tre-rerun-20261103.out 2>&1 < /dev/null &
#   Settings (environment): MAIN_SHA (runner / client main, default d17bccea), CTL_IMAGE (expected controller image, default
#   the live 20261102-420c69ef), ALLOW_SAME=1 (accept a registry equal to the reference in every TSS key), TSS_GUARD_EXTRA.
#
# Clean stop (no race): touch $CAMPAIGN_DATA_ROOT/campaign-tre-rerun-draft-20261103/STOP  (forwarded within 5 s; the
#   runner stops after the current arm; it also blocks a chained rep 2). Resume: delete $C/STOP and $ROOT/STOP, launch again
#   (done arms are skipped). SIGKILL of the runner: run reset_canonical.sh by hand.
# After arm 1 (TRE x Alternating, ~15 min):
#   python3 $T/eval/runner/first_run_check.py <results_root>/Alternating_hot_model_periodic_A_s1/tre
#   then every 50 min.
# Logs: run_<NAME>.log (this script), <NAME>.out (runner stdout); campaign.log + status.json in the results root.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # [repo] this driver directory (guards, trace sha256 lists)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
C="${CAMPAIGN_WORK_DIR:-$X/campaign-tre-rerun-draft-20261103}"                       # [repo] logs, STOP file, runner stdout (original: the campaign dir next to the spec)
REP1=campaign-tre-rerun-r3e-20261103
NAME="${NAME:-$REP1}"
T="${TRE_REPO:-$(cd "$HERE/../../../.." && pwd)}"                      # [repo] the tre/ tree of this checkout
R=$NAME                                      # round of the spec = marker token
SPEC=$T/eval/runner/campaigns/$NAME.yaml
ROOT=$X/$NAME-run
SHAFILE=$HERE/$REP1.traces.sha256               # the 7 trace files (= campaign-bl-rep2-r3e-20261103), both reps
MAIN_SHA="${MAIN_SHA:-d17bccea}"             # runner / client (spec header); override only after a new dry-run
CTL_IMAGE="${CTL_IMAGE:-tre-v2-controller:20261102-420c69ef}"   # expected controller image (eta-removal image: set it)
WINDOW_EXPECT=10000                          # TRE_METRICS_WINDOW_MS of the controller Deployment (first sample: 10 s)
REF=$T/eval/registries/0ab5352747bdd70e.yaml   # registry of the baselines rep 2 (tune F + 11-02 alt_thresholds)
OLD_ID=0ab5352747bdd70e
NS=tre-v2; CTL=tre-v2-controller
OWNER=owner=tre-rerun_$NAME
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
state() { python3 -c "import json;print(json.load(open('$ROOT/status.json'))['state'])" 2>/dev/null; }
live_id() { python3 $T/eval/runner/arm_registry.py id 2>&1 | tail -1; }
spec_id() { python3 -c "import yaml;print((yaml.safe_load(open('$SPEC')).get('registry_expect') or {}).get('registry_id'))" 2>&1; }
ctl_window() { kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_METRICS_WINDOW_MS")].value}'; }
ctl_image() { kubectl -n $NS get deploy $CTL -o jsonpath='{.spec.template.spec.containers[0].image}'; }

stop_runner() {   # clean stop of a running campaign: STOP file, then wait for it (at most one arm, ~16 min)
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
  log "driver exit rc=$rc; live registry $(live_id), controller $(ctl_image) window $(ctl_window)"
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
[ "$(git -C $T/.. rev-parse --short=8 HEAD)" = "$MAIN_SHA" ] || bad "main $(git -C $T/.. rev-parse --short=8 HEAD) != $MAIN_SHA (re-run the dry-run, update the spec header, then MAIN_SHA=<sha> to accept)"
# 0c. exclusive-window marker: ours (adopted) or none; a foreign one (calibration / baselines rep 2 still running) aborts
if [ -e "$MARKER" ]; then
  if grep -qE "round=$R( |$)" "$MARKER"; then log "marker already ours (adopted): $(head -1 "$MARKER")"
  else bad "foreign exclusive-window marker (calibration or baselines rep 2 not ended?): $(head -c 300 "$MARKER")"; fi
else log "ok   no exclusive-window marker"; fi
# 0d. spec: placeholder filled, id = live registry id, rep 2 = rep 1 except name / round / results_root / # REP
[ -s "$SPEC" ] || bad "no spec $SPEC"
grep -q 'FILL_REGISTRY_ID' "$SPEC" && bad "registry_expect.registry_id is still the placeholder FILL_REGISTRY_ID in $SPEC (CHECKLIST.md step 4)"
SID=$(spec_id); LID=$(live_id)
if echo "$SID" | grep -qE '^[0-9a-f]{16}$' && [ "$SID" = "$LID" ]; then log "ok   spec registry_id $SID = live registry id"
else bad "spec registry_id '$SID' != live registry id '$LID'"; fi
[ "$LID" = "$OLD_ID" ] && [ "${ALLOW_SAME:-0}" != 1 ] && bad "live registry is still $OLD_ID (tune F): the new parameters are not published (ALLOW_SAME=1 to accept)"
if [ "$NAME" != "$REP1" ]; then
  DR=$(diff <(grep -vE '^(name|round|results_root):|^# REP:' "$T/eval/runner/campaigns/$REP1.yaml") <(grep -vE '^(name|round|results_root):|^# REP:' "$SPEC") | head -c 400)
  [ -z "$DR" ] && log "ok   $NAME = $REP1 except name / round / results_root" || bad "$NAME drifted from $REP1 (bash $HERE/make_rep2.sh): $DR"
fi
# 0e. trace files = the files of the first sample and of both baseline samples (sha256)
SHAOUT=$(cd "$X" && nice -n 19 sha256sum -c --quiet "$SHAFILE" 2>&1); SHARC=$?
if [ $SHARC = 0 ]; then log "ok   $(wc -l < $SHAFILE) trace files match the sha256 of the first sample"
else bad "trace sha256 mismatch (rc $SHARC): $(echo "$SHAOUT" | head -c 400)"; fi
# 0f. runner plan + input validation (no cluster access)
DRY=$(bash $T/eval/runner/run_campaign.sh "$SPEC" --dry-run 2>&1); DRC=$?
if [ $DRC = 0 ]; then log "ok   runner dry-run: $(echo "$DRY" | grep -E '^total:|^dry-run:' | tr '\n' ' ')"
else bad "runner dry-run rc=$DRC: $(echo "$DRY" | grep -E 'PROBLEM' | head -c 600)"; fi
# 0g. registry: differs from the reference ONLY in the TSS parameters
RG=$(python3 $HERE/tss_only_guard.py "$REF" 2>&1); RGC=$?
echo "$RG" | while IFS= read -r l; do log "  $l"; done
[ $RGC = 0 ] || bad "registry is not a TSS-only change of $REF (tss_only_guard.py rc $RGC)"
# 0h. SM and controller restarted with the current ConfigMap (params_set.py --restart = SM first, then controller)
PS=$(python3 $T/deploy/scripts/release/params_set.py --status 2>&1)
echo "$PS" | grep -q '^in sync' && log "ok   params_set --status: in sync" || bad "params not applied: $(echo "$PS" | tail -1 | head -c 300)"
# 0i. cluster: nodes, tre-v2 components and images (controller HARD), window, 20/20 model pods, no APA CR / Failed pod, headroom
CG=$(EXPECT_CTL=$CTL_IMAGE EXPECT_WINDOW=$WINDOW_EXPECT python3 $HERE/guard_cluster_tre.py 2>&1); CGC=$?
echo "$CG" | while IFS= read -r l; do log "  $l"; done
[ $CGC = 0 ] || bad "cluster guard failed (lines FAIL above)"

if [ $FAIL != 0 ]; then log "guards FAILED: nothing started"; exit 1; fi
if [ "$CHECK" = 1 ]; then log "CHECK OK: a launch now would pass every guard (run mode and awake layout are reset by the runner's first pre-check)"; exit 0; fi

# ---- 1. marker (the runner's pre-check wants `validation ... round=$R`; it extends the end by its ETA + margin)
if [ -e "$MARKER" ]; then
  log "marker already ours (adopted)"
else
  S=$(date +%FT%T%:z); E=$(date -d '+2 hours 30 minutes' +%FT%T%:z)
  echo "validation $S $E round=$R $OWNER" > "$MARKER"
  log "marker written: $(cat "$MARKER")"
fi
MARKER_OURS=1
[ -e "$ROOT/status.json" ] && log "results root $ROOT before start: counts $(counts)"

# ---- 2. campaign
log "campaign $SPEC (registry $SID, controller $CTL_IMAGE, window $WINDOW_EXPECT)"
bash $T/eval/runner/run_campaign.sh "$SPEC" >> $C/$NAME.out 2>&1 < /dev/null &
RUNNER_PID=$!
i=0
while kill -0 $RUNNER_PID 2>/dev/null; do
  # STOP forward every 5 s (10-30 lesson: a slow forward let the runner start one more arm)
  [ -e "$C/STOP" ] && [ ! -e "$ROOT/STOP" ] && { touch "$ROOT/STOP"; log "STOP: forwarded to $ROOT"; }
  i=$((i + 1))
  if [ $((i % 60)) = 0 ]; then     # every ~5 min: registry, controller image and window must hold for the whole campaign
    r=$(live_id); [ "$r" = "$SID" ] || log "WARNING: registry id is '$r' (want $SID) during the campaign"
    w=$(ctl_window); [ "$w" = $WINDOW_EXPECT ] || log "WARNING: TRE_METRICS_WINDOW_MS is '$w' (want $WINDOW_EXPECT) during the campaign"
    m=$(ctl_image); [ "$m" = "$CTL_IMAGE" ] || log "WARNING: controller image is '$m' (want $CTL_IMAGE) during the campaign"
  fi
  sleep 5
done
wait $RUNNER_PID; rc=$?; RUNNER_PID=""
log "campaign rc=$rc (0 done/stopped, 1 failed arms, 2 refused, 3 blocked); state $(state); counts $(counts)"
# a chained rep 2 (`... && NAME=<rep2> ...`) must start only after a COMPLETE rep 1: rc 0 with state stopped -> 4
[ $rc = 0 ] && [ "$(state)" != done ] && { log "runner stopped before the end (state $(state)): exit 4 (no chained rep)"; exit 4; }
exit $rc
