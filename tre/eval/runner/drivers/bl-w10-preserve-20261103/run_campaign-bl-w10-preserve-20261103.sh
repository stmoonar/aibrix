#!/usr/bin/env bash
# 2026-11-03 PreServe with a 10 s window: campaign-bl-w10-preserve-20261103.yaml = 7 ICSE-suite traces x PreServe, then one APA
# rerun on Alternating (last), seed 1, NO TRE arm, 8 arms, ~2.0 h (spec header: what changed, sources, ETA).
#
# WRITES THE CLUSTER only through the runner (per arm: run mode, APA CRs / baseline scaler, policy ConfigMaps, one controller
# restart, awake layout) and through the exclusive-window marker. It does NOT switch the registry, the controller image, the
# baseline-scaler image or the run mode by itself, and it never applies a registry (the baselines read no TRE parameter; the spec
# pins the registry id).
#
# Parameters (environment):
#   CAMPAIGN_DATA_ROOT  required: the directory with the results roots, the trace sets and the exclusive-window marker
#   EXPECT_BL           required: the baseline-scaler image the live Deployment must run (the 10 s build,
#                       tre-v2-baseline-scaler:<YYYYMMDD>-<sha8>); <sha8> must be HEAD or its ancestor with the same tre/baselines
#   EXPECT_BL_ID        optional: the image ID (sha256:...) of EXPECT_BL, checked on this host (docker image inspect) and on
#                       every host of BL_IMAGE_SSH_HOSTS (space-separated ssh names of the other nodes; default none)
#   CAMPAIGN_WORK_DIR   log / STOP file / runner stdout          (default $CAMPAIGN_DATA_ROOT/campaign-bl-w10-preserve-20261103)
#   TRE_REPO            the tre/ tree of the checkout the runner, client, ConfigMap and APA CRs come from (default: this checkout)
#   MAIN_SHA            optional: pin the runner commit (8 hex digits); without it the pushed HEAD of this checkout is accepted
#   ALLOW_UNPUSHED      optional, 1 = accept HEAD != origin/main (the feature branch or an unpushed main; explicit user
#                       decision; logged as WARN)
#   TEST_MARKER         only to test the guards with a fake marker file
#   (guard_cluster.py also reads EXPECT_SM / EXPECT_GW / EXPECT_CTL)
#
#   EXPECT_BL=... bash run_campaign-bl-w10-preserve-20261103.sh --check
#       read-only: every guard below (git, processes, marker, traces, PreServe window / weights / histories in the checkout,
#       image, spec, runner dry-run, registry id, cluster); writes nothing, starts nothing; exit 0 = a launch would pass.
#   cd $CAMPAIGN_WORK_DIR && EXPECT_BL=... setsid nohup bash <this file> >> run_campaign-bl-w10-preserve-20261103.out 2>&1 < /dev/null &
#       launch (after --check passed and no foreign marker exists).
#
# Clean stop (no race): touch $CAMPAIGN_WORK_DIR/STOP  (forwarded to the results root within 5 s; the runner stops after the
#   current arm). Resume: delete both STOP files ($CAMPAIGN_WORK_DIR/STOP and <results_root>/STOP), launch again (done arms are
#   skipped). SIGKILL of the runner: run tre/eval/runner/reset_canonical.sh by hand.
# After arm 1 (PreServe x Alternating, ~16 min):
#   python3 $TRE_REPO/eval/runner/first_run_check.py <results_root>/Alternating_hot_model_periodic_A_s1/preserve
#   then every 50 min; the last arm (APA rerun) is <results_root>/Alternating_hot_model_periodic_A_s1/apa-rerun.
# Logs: run_<NAME>.log (this script), <NAME>.out (runner stdout); campaign.log + status.json in the results root.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # this driver directory (guard, trace sha256 list)
X="${CAMPAIGN_DATA_ROOT:?set CAMPAIGN_DATA_ROOT: the directory with the results roots, trace sets and the exclusive-window marker}"
NAME=campaign-bl-w10-preserve-20261103
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
export EXPECT_BL="${EXPECT_BL:-}"                            # guard_cluster.py: the image the Deployment must run
HISTFILE=$T/baselines/histories/preserve_history_w10-20261103.sha256   # paths relative to $X
WDIR=$T/baselines/weights/preserve-mlstm-20261103-paper-w10
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
    log "WARN HEAD != origin/main: running HEAD $(git -C $T/.. rev-parse --short=8 HEAD) ($(git -C $T/.. rev-parse --abbrev-ref HEAD)) by explicit ALLOW_UNPUSHED=1 (user decision; deviates from AGENTS.md rule 6 until merged and pushed; a fast-forward merge keeps the sha)"
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
# 0e. the 10 s files are in this checkout: PreServe window_s 10, its predictors = the w10 weights (paths, sha256 of the weights
#     manifest and of the files), the APA CRs at 10 s (the apa-rerun arm); the history list covers exactly the 7 trace dirs
WIN=$(python3 - <<PY 2>&1
import glob, hashlib, json, os, yaml
bad = []
cm = {d["metadata"]["name"]: d for d in yaml.safe_load_all(open("$T/deploy/baselines/tre/policy-configmaps.yaml")) if d and d.get("kind") == "ConfigMap"}
p = yaml.safe_load(cm["tre-v2-baseline-preserve"]["data"]["preserve.yaml"])
if float(p.get("window_s") or 0) != 10.0: bad.append(f"preserve window_s {p.get('window_s')}")
if str(p.get("tier1")) != "mlstm": bad.append(f"preserve tier1 {p.get('tier1')}")
man = json.load(open("$WDIR/manifest.json"))
if float(man.get("window_s") or 0) != 10.0: bad.append(f"weights manifest window_s {man.get('window_s')}")
wd = os.path.relpath("$WDIR", "$T/baselines")
pred = p.get("predictors") or {}
if set(pred) != set(man["files"]): bad.append(f"predictor sets {sorted(pred)} != manifest {sorted(man['files'])}")
for k, spec in sorted(pred.items()):
    f = man["files"].get(k) or {}
    if spec.get("path") != f"{wd}/{f.get('file')}": bad.append(f"predictor {k} path {spec.get('path')} (want {wd}/{f.get('file')})")
    if spec.get("sha256") != f.get("sha256"): bad.append(f"predictor {k} sha256 != manifest")
    fp = os.path.join("$T/baselines", str(spec.get("path")))
    if not os.path.isfile(fp) or hashlib.sha256(open(fp, "rb").read()).hexdigest() != spec.get("sha256"): bad.append(f"predictor {k}: {fp} missing or sha256 differs")
w = {f.rsplit("/", 1)[-1]: yaml.safe_load(open(f))["metadata"]["annotations"]["apa.autoscaling.aibrix.ai/window"]
     for f in sorted(glob.glob("$T/deploy/baselines/apa/*-apa.yaml"))}
if set(w.values()) != {"10s"} or len(w) != 3: bad.append(f"APA windows {w}")
dirs = lambda f: {os.path.dirname(l.split(None, 1)[1].strip()) for l in open(f) if l.strip()}
h = [l.split(None, 1)[1].strip() for l in open("$HISTFILE") if l.strip()]
if dirs("$HISTFILE") != dirs("$SHAFILE") or len(h) != 7: bad.append("history list != the 7 trace dirs")
if {os.path.basename(x) for x in h} != {"preserve_history_w10.json"}: bad.append("history names != preserve_history_w10.json")
print(("BAD " + "; ".join(bad)) if bad else f"OK preserve window_s 10, tier1 mlstm, predictors {sorted(pred)} = {wd} (sha256 ok), APA {sorted(set(w.values()))}, 7 histories listed")
PY
)
case "$WIN" in OK*) log "ok   10 s files in the checkout: ${WIN#OK }" ;; *) bad "10 s files in $T: $WIN" ;; esac
# 0e2. the 7 histories before t0 (10 s) exist and are the reviewed files (sha256)
HOUT=$(cd "$X" && nice -n 19 sha256sum -c --quiet "$HISTFILE" 2>&1); HRC=$?
if [ $HRC = 0 ]; then log "ok   $(wc -l < $HISTFILE) PreServe histories (10 s) match $HISTFILE"
else bad "PreServe history sha256 mismatch / missing (rc $HRC): $(echo "$HOUT" | head -c 400)"; fi
# 0e3. the baseline-scaler image: EXPECT_BL given, built from HEAD or an ancestor with the same tre/baselines tree; image IDs
if [ -z "$EXPECT_BL" ]; then bad "EXPECT_BL not set (the 10 s baseline-scaler image, tre-v2-baseline-scaler:<YYYYMMDD>-<sha8>)"
else
  BSHA=$(echo "$EXPECT_BL" | sed -nE 's/^tre-v2-baseline-scaler:[0-9]{8}-([0-9a-f]{8})$/\1/p')
  if [ -z "$BSHA" ]; then bad "EXPECT_BL '$EXPECT_BL' is not tre-v2-baseline-scaler:<YYYYMMDD>-<sha8>"
  elif ! git -C $T/.. merge-base --is-ancestor "$BSHA" HEAD 2>/dev/null; then bad "image commit $BSHA is not HEAD $HEAD8 or its ancestor"
  elif ! git -C $T/.. diff --quiet "$BSHA" HEAD -- tre/baselines; then bad "tre/baselines differs between the image commit $BSHA and HEAD $HEAD8"
  else log "ok   image $EXPECT_BL: commit $BSHA is HEAD or its ancestor, same tre/baselines"; fi
  if [ -n "${EXPECT_BL_ID:-}" ]; then
    for h in local ${BL_IMAGE_SSH_HOSTS:-}; do
      if [ "$h" = local ]; then id=$(docker image inspect --format '{{.Id}}' "$EXPECT_BL" 2>&1)
      else id=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" docker image inspect --format '{{.Id}}' "$EXPECT_BL" 2>&1); fi
      if [ "$id" = "$EXPECT_BL_ID" ]; then log "ok   image ID on $h = $EXPECT_BL_ID"; else bad "image $EXPECT_BL on $h: '$id' (want $EXPECT_BL_ID)"; fi
    done
  else log "info EXPECT_BL_ID not given: image IDs on the nodes are not checked"; fi
fi
# 0f. trace files = the files of the earlier samples (sha256), paths relative to the data root
SHAOUT=$(cd "$X" && nice -n 19 sha256sum -c --quiet "$SHAFILE" 2>&1); SHARC=$?
if [ $SHARC = 0 ]; then log "ok   $(wc -l < $SHAFILE) trace files match the sha256 of the earlier samples"
else bad "trace sha256 mismatch (rc $SHARC): $(echo "$SHAOUT" | head -c 400)"; fi
# 0g. runner plan + input validation (no cluster access)
DRY=$(bash $T/eval/runner/run_campaign.sh "$SPEC" --dry-run 2>&1); DRC=$?
if [ $DRC = 0 ]; then log "ok   runner dry-run: $(echo "$DRY" | grep -E '^total:|^dry-run:' | tr '\n' ' ')"
else bad "runner dry-run rc=$DRC: $(echo "$DRY" | grep -E 'PROBLEM' | head -c 600)"; fi
# 0h. cluster: nodes, tre-v2 components and images (baseline-scaler = EXPECT_BL), 20/20 model pods, no APA CR / Failed pod, node headroom
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
