#!/usr/bin/env bash
# The 2026-11-03 calibration supplement (local workspace docs/calib-supplement-design-20261103.md,
# blocks A / B / E; preregistration prereg-calib-supplement-20261103.md) from ONE process: per
# model A -> B -> E (run_stage3.sh --training-plan supp-{a,b,e}-20261103), the models in
# parallel, one driver each.
#
#   1. guards: CHAIN_LOG_DIR / CALIB_ROOT / BASE_RUN set; CALIB_ROOT holds nothing but
#      PREREG.md + PREREG.sha256 (verified), E_seal/ and the log dir; no STOP file; one
#      routable pod per model (its name and restart counts are kept for step 3). A real run
#      also needs the preregistration and a clean worktree (a dry run warns).
#   2. E seal, before any cell: per model the dry run of supp-e-20261103 into
#      $CALIB_ROOT/E_seal; its plan.json gets a sha256sum sidecar and both are made
#      read-only (a seal already there is verified and kept). The real E run gets
#      --sealed-plan and refuses any cell that is not in it.
#   3. per model, in the background, block by block: STOP check; the routable pod is still
#      the one pod of step 1 with the same restart counts; run_stage3.sh into
#      $CALIB_ROOT/<block>/<model> with --stop-file $CALIB_ROOT/STOP --stop-file
#      $CALIB_ROOT/STOP.<model> --cell-log $CHAIN_LOG_DIR/cells.<model>.jsonl (one line per
#      driven attempt) --instant-sample-ms $CHAIN_INSTANT_SAMPLE_MS (E: --sealed-plan too).
#      Exit 0 -> the next block; 4 -> a STOP file ended it and this model's chain; anything
#      else (e.g. a cell void twice ends the run) -> block_failed, the next block goes on and
#      the chain exits 1. A changed pod (step 3) stops this model's chain (the others go on).
#   4. wait for every model's chain.
#
# Stop gracefully: touch $CALIB_ROOT/STOP (every model) or $CALIB_ROOT/STOP.<model>. The cell
# being driven finishes; its run then ends "stopped" (exit 4) before the next cell and builds
# its dataset from the cells it drove; the chain starts no further block. A stopped block is
# partial (a run has no resume): keep it and drive what is missing into a NEW root.
#
#   run_chain_supp.sh [--dry-run] [model ...]        (default: the three models)
#
# --dry-run: every step with run_stage3.sh --dry-run (no load; lib.sh's read-only guards still
# run), the E seal included, into $CHAIN_DRYRUN_ROOT; CALIB_ROOT is only read.
#
# Exit: 0 every block of every model exited 0; 4 a STOP file ended a chain and none failed;
# 1 a guard refused or a block failed (CHAIN_STATUS.jsonl says which).
#
# Launch detached (survives the ssh session): mkdir -p "$CHAIN_LOG_DIR"; setsid nohup bash
# run_chain_supp.sh > "$CHAIN_LOG_DIR/chain.log" 2>&1 < /dev/null &
#
# Environment (nothing site-specific is written here):
#   CHAIN_LOG_DIR     (required, no default: a restart must never write over the logs of an
#                     earlier attempt) step logs (appended, never truncated),
#                     cells.<model>.jsonl and CHAIN_STATUS.jsonl
#   CALIB_ROOT        (required) the supplement's root
#   BASE_RUN          (required) round 2's root: A's rho*_run2 and the capacity prior's points
#   BASE_RUN_<model, '-' -> '_'>   per-model override, e.g. BASE_RUN_dsqwen_14b=<run2b>
#   DESIGN_SEED       (default 20261103)
#   CHAIN_INSTANT_SAMPLE_MS   queue sidecar and per-pod vLLM metrics cadence (default 500)
#   CHAIN_BLOCKS      (default "A B E")
#   CHAIN_LOCK_DIR    host-local directory of the status flock (default /run/lock)
#   CHAIN_DRYRUN_ROOT --dry-run outputs (default $CHAIN_LOG_DIR/dryrun)
#   TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE, CALIB_TASKSET, KUBECTL, ...
#                     as run_stage3.sh / lib.sh
set -euo pipefail
trap '' HUP

CAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$CAL/lib.sh"

DRY=0
MODELS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -*) echo "unknown argument '$a'" >&2; exit 2 ;;
    *) calib_require_model "$a"; MODELS+=("$a") ;;
  esac
done
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(dsqwen-7b dsllama-8b dsqwen-14b)

[[ -n "${CHAIN_LOG_DIR:-}" ]] || calib_die "set CHAIN_LOG_DIR (no default: a restart must not write over an earlier attempt's logs)"
calib_require_env CALIB_ROOT
calib_require_env BASE_RUN
: "${DESIGN_SEED:=20261103}"
: "${CHAIN_INSTANT_SAMPLE_MS:=500}"
: "${CHAIN_BLOCKS:=A B E}"
: "${CHAIN_LOCK_DIR:=/run/lock}"
: "${CHAIN_DRYRUN_ROOT:=$CHAIN_LOG_DIR/dryrun}"
export DESIGN_SEED

plan_of() {
  case "$1" in
    A) echo supp-a-20261103 ;;
    B) echo supp-b-20261103 ;;
    E) echo supp-e-20261103 ;;
    *) return 1 ;;
  esac
}
for b in $CHAIN_BLOCKS; do
  plan_of "$b" > /dev/null || calib_die "unknown block '$b' in CHAIN_BLOCKS (A B E)"
done

if [[ "$DRY" == 1 ]]; then
  WORK="$CHAIN_DRYRUN_ROOT"     # every output of a dry run; CALIB_ROOT is only read
  LOGS="$CHAIN_DRYRUN_ROOT/logs"
  DRY_FLAG=(--dry-run)
else
  WORK="$CALIB_ROOT"
  LOGS="$CHAIN_LOG_DIR"
  DRY_FLAG=()
fi
CHAIN_STATUS="$LOGS/CHAIN_STATUS.jsonl"
SEAL_DIR="$WORK/E_seal"
mkdir -p "$LOGS" "$CHAIN_LOCK_DIR"
STATUS_LOCK="$CHAIN_LOCK_DIR/tre-calib-supp-status.lock"

json_str() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# status <model|all> <step> <event> [rc] [detail]   (time = this host's clock)
status() {
  local line
  line="{\"ts\":\"$(date -Iseconds)\",\"host\":\"$(hostname)\",\"model\":\"$1\",\"step\":\"$2\",\"event\":\"$3\",\"rc\":${4:-null},\"pid\":$BASHPID,\"dry_run\":$DRY,\"detail\":$(json_str "${5:-}")}"
  flock "$STATUS_LOCK" bash -c 'printf "%s\n" "$1" >> "$2"' _ "$line" "$CHAIN_STATUS"
  echo "[supp] $line"
}

fail() {  # fail <step> <why>
  status all "$1" refused 1 "$2"
  exit 1
}

refuse() {  # refuse <step> <why>: a real run stops, a dry run warns and goes on
  if [[ "$DRY" == 1 ]]; then
    status all "$1" warning null "$2 (a real run refuses)"
  else
    fail "$1" "$2"
  fi
}

base_run() { local v="BASE_RUN_${1//-/_}"; echo "${!v:-$BASE_RUN}"; }

# "<pod> <container>=<restarts>,..." of each routable pod of the model, one line per pod
routable() {
  "$KUBECTL" -n "$MODEL_NS" get pods -l "model.aibrix.ai/name=$1,tre.aibrix.io/routable=true" \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .status.containerStatuses[*]}{.name}{"="}{.restartCount}{","}{end}{"\n"}{end}'
}

e_plan() { echo "$SEAL_DIR/calibration_stage3_dryrun/$1/plan.json"; }

# step <model> <block> <log> <run_stage3.sh flags...>: run_stage3.sh of the block's plan into
# $WORK/<block>, in a subshell, output appended to <log> (never truncated); returns its exit
# code. CALIB_DRYRUN_ROOT (a dry run's out-dir root) comes from the caller.
# Call it as `step ... || rc=$?`: a bare call under `set -e` ends the caller on a non-zero
# return before it can record the block's end (2026-11-03: an earlier version turned
# `set -e` back on before returning, so a failed block left no end / block_failed line).
step() {
  local m="$1" b="$2" log="$3" rc=0
  shift 3
  echo "===== $(date -Iseconds) $m $b ($(plan_of "$b")): run_stage3.sh $m $* =====" >> "$log"
  ( OUT_ROOT="$WORK/$b" BASE_RUN="$(base_run "$m")" SUPP_RUN="" TRAINING_PLAN="$(plan_of "$b")" \
      bash "$CAL/run_stage3.sh" "$m" "$@" ) >> "$log" 2>&1 < /dev/null || rc=$?
  return "$rc"
}

seal_e() {  # seal_e <model>
  local m="$1" plan log rc=0
  plan="$(e_plan "$m")"
  if [[ "$DRY" != 1 && -f "$plan.sha256" ]]; then
    ( cd "$(dirname "$plan")" && sha256sum --status -c plan.json.sha256 ) \
      || fail E_seal "$plan no longer matches its sidecar: the E seal changed (owner)"
    status "$m" E_seal reused 0 "$plan sha256 $(cut -d' ' -f1 "$plan.sha256")"
    return 0
  fi
  log="$LOGS/E_seal.$m.log"
  status "$m" E_seal start null "log=$log"
  CALIB_DRYRUN_ROOT="$SEAL_DIR" step "$m" E "$log" --dry-run -- \
    --instant-sample-ms "$CHAIN_INSTANT_SAMPLE_MS" || rc=$?
  [[ "$rc" == 0 && -s "$plan" ]] || fail E_seal "the dry run of E for $m exited $rc (see $log)"
  ( cd "$(dirname "$plan")" && sha256sum plan.json > plan.json.sha256 && chmod a-w plan.json plan.json.sha256 )
  status "$m" E_seal sealed 0 "$plan sha256 $(cut -d' ' -f1 "$plan.sha256")"
}

chain_model() {  # chain_model <model>: the blocks in turn; returns 0, 4 (STOP, none failed) or 1
  local m="$1" b now log rc extra failed=""
  status "$m" chain start null "blocks=$CHAIN_BLOCKS base_run=$(base_run "$m") pod=${POD0[$m]}"
  for b in $CHAIN_BLOCKS; do
    if [[ -e "$CALIB_ROOT/STOP" || -e "$CALIB_ROOT/STOP.$m" ]]; then
      status "$m" "$b" stopped 4 "a STOP file exists: block $b and the rest are not driven"
      [[ -z "$failed" ]] || return 1
      return 4
    fi
    now="$(routable "$m")" || now="(kubectl get failed)"
    if [[ "$now" != "${POD0[$m]}" ]]; then
      status "$m" "$b" chain_stopped 1 "the routable pod changed: '${POD0[$m]}' at the start, now '$now'"
      return 1
    fi
    extra=(--stop-file "$CALIB_ROOT/STOP" --stop-file "$CALIB_ROOT/STOP.$m"
           --cell-log "$LOGS/cells.$m.jsonl" --instant-sample-ms "$CHAIN_INSTANT_SAMPLE_MS")
    [[ "$b" != E ]] || extra+=(--sealed-plan "$(e_plan "$m")")
    log="$LOGS/$b.$m.log"
    status "$m" "$b" start null "log=$log out=$WORK/$b/$([[ "$DRY" == 1 ]] && echo calibration_stage3_dryrun/)$m plan=$(plan_of "$b")"
    rc=0
    CALIB_DRYRUN_ROOT="$WORK/$b" step "$m" "$b" "$log" ${DRY_FLAG[@]+"${DRY_FLAG[@]}"} -- "${extra[@]}" || rc=$?
    status "$m" "$b" end "$rc" "log=$log"
    case "$rc" in
      0) ;;
      4) status "$m" "$b" stopped 4 "a STOP file ended block $b before its next cell"
         [[ -z "$failed" ]] || return 1
         return 4 ;;
      *) status "$m" "$b" block_failed "$rc" "block $b exited $rc (see $log); the next block goes on"
         failed="${failed:+$failed }$b" ;;
    esac
  done
  if [[ -n "$failed" ]]; then
    status "$m" chain done 1 "block(s) $failed failed (block_failed lines); the other blocks ran"
    return 1
  fi
  status "$m" chain done 0 "every block exited 0"
}

# ================================================================= main
GIT="$(git -C "$CALIB_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
# no pipe and no bare failure here: under set -euo pipefail a failing git (not a repo, "dubious
# ownership") or a SIGPIPE from head ended the script before its first status line.
st="$(git -C "$CALIB_REPO" status --porcelain --untracked-files=no 2>&1)" || st="git status failed: $st"
DIRTY="$(sed -n 1,5p <<< "$st")"
status all chain start null "dry_run=$DRY root=$CALIB_ROOT logs=$LOGS models=${MODELS[*]} blocks=$CHAIN_BLOCKS seed=$DESIGN_SEED instant_sample_ms=$CHAIN_INSTANT_SAMPLE_MS repo=$CALIB_REPO git=$GIT dirty=$([[ -z "$DIRTY" ]] && echo no || echo yes)"

[[ -z "$DIRTY" ]] || refuse code "the worktree is not a clean commit (uncommitted changes, or git cannot read it; a data run needs a clean commit): $DIRTY"

allowed=(PREREG.md PREREG.sha256 E_seal)
case "$CHAIN_LOG_DIR" in
  "$CALIB_ROOT"/*) rel="${CHAIN_LOG_DIR#"$CALIB_ROOT"/}"; allowed+=("${rel%%/*}") ;;
esac
if [[ -d "$CALIB_ROOT" ]]; then
  while IFS= read -r entry; do
    ok=0
    for a in "${allowed[@]}"; do [[ "$entry" == "$a" ]] && ok=1; done
    [[ "$ok" == 1 ]] || refuse root "$CALIB_ROOT/$entry exists: the root may hold only ${allowed[*]} (a partial run is void evidence: keep it, use a NEW root)"
  done < <(ls -A "$CALIB_ROOT")
fi
for f in "$CALIB_ROOT/STOP" "${MODELS[@]/#/$CALIB_ROOT/STOP.}"; do
  [[ ! -e "$f" ]] || refuse stop_file "$f exists before the start (remove it or it stops the run)"
done
if [[ -s "$CALIB_ROOT/PREREG.md" && -s "$CALIB_ROOT/PREREG.sha256" ]]; then
  ( cd "$CALIB_ROOT" && sha256sum --status -c PREREG.sha256 ) \
    || refuse prereg "$CALIB_ROOT/PREREG.md does not match PREREG.sha256"
  status all prereg verified 0 "$(cat "$CALIB_ROOT/PREREG.sha256")"
else
  refuse prereg "no $CALIB_ROOT/PREREG.md + PREREG.sha256: freeze the preregistration first"
fi

declare -A POD0
for m in "${MODELS[@]}"; do
  POD0[$m]="$(routable "$m")" || fail pods "kubectl get pods failed for $m"
  [[ -n "${POD0[$m]}" && "$(grep -c . <<< "${POD0[$m]}")" == 1 ]] \
    || fail pods "$m: need exactly one routable pod, have '${POD0[$m]}'"
done

if [[ " $CHAIN_BLOCKS " == *" E "* ]]; then
  for m in "${MODELS[@]}"; do seal_e "$m"; done
fi

declare -A PID
for m in "${MODELS[@]}"; do
  chain_model "$m" &
  PID[$m]=$!
done
worst=0
for m in "${MODELS[@]}"; do
  set +e
  wait "${PID[$m]}"
  rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 4 ]]; then
    worst=1
  elif [[ "$rc" == 4 && "$worst" == 0 ]]; then
    worst=4
  fi
done
status all chain finished "$worst" "0 = every block of every model exited 0, 4 = stopped by a STOP file, 1 = a failure (see the model lines)"
exit "$worst"
