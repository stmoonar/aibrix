#!/usr/bin/env bash
# Unattended training-collection chain of a θ recalibration (RUN plan §D), ONE model per
# process. Start one per model (detached); the chains meet only at the D2 barrier and the
# P1 lock:
#
#   D1 run1 (run_primitives.sh)
#   -> barrier: every model in CHAIN_MODELS has finished D1 with exit 0
#   -> D2 priors (once, by whichever chain takes the D2 lock first): standard dataset of
#      RUN1_ROOT (if missing), calibration_priors -> PRIORS_DIR, SHA256SUMS, and an
#      addendum text file. No git commit here: the preregistration addendum is committed
#      by a person (the chain does not wait for it).
#   -> D3 run2 (run_ladder.sh)
#   -> D4 S3 supplement (run_supplement.sh); REPROBE_GRID from round 2's own D6' b50 table
#      (scripts.analysis.reprobe_grid), written to ATTEMPTS.md before the launch
#   -> D5 P1 (run_stage3.sh, TRAINING_PLAN); a host-local flock makes the chains run P1
#      one model at a time
#
# Any step exiting non-zero stops THIS model's chain (status file + CHAIN_STATUS line);
# the other chains go on, except at the D2 barrier, which needs every model's run1
# (the priors are computed over all of them): a failed run1 stops every chain there.
# Exit 3 of D4 / D5 (finished, a check failed) is a stop for the owner, like the plan says.
#
#   run_chain_train.sh <model> [--dry-run]
#
# --dry-run: every launcher with --dry-run (drives nothing), no barrier, no locks waited
# on; D2 / D4 compute into $CALIB_ROOT/dryrun. Point RUN1_ROOT / PRIORS_DIR / RUN2_ROOT /
# SUPP_ROOT at an earlier round to exercise the later steps before their inputs exist.
#
# Resume after D1 (CHAIN_REUSE_RUN1=1): D1 is not driven; the run1 already under
# RUN1_ROOT/<model> is accepted only if it is verifiably complete (run1_reuse_check:
# every planned cell and boundary search on disk, every dataset row valid, no guard
# voided, no backfill pending, clean code commit). Then run1.<model>.rc = 0, the evidence
# in $CHAIN_DIR/run1.<model>.reused.json and a CHAIN_STATUS line D1 "reused"; otherwise
# run1.<model>.rc = 1 and the chain stops. run1 is only read: its campaign_status.json
# stays as the run wrote it. Used on 2026-10-03: run1 of cc3964b6 drove every cell, then
# died in _drive_campaign (NameError static_cells) and recorded status "failed". Start
# every chain with the switch and no chain/run1.*.rc left from an earlier attempt.
#
# Environment (nothing site-specific is written here):
#   CALIB_ROOT        (required) this round's root; every output, log and status is under it
#   DESIGN_SEED       (required) passed to every launcher
#   TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE, CALIB_TASKSET   for the launchers
#   CHAIN_MODELS      models the D2 barrier waits for (default: the three calibration models)
#   CHAIN_REUSE_RUN1  1 = reuse a complete run1 instead of driving D1 (see above; default 0)
#   CHAIN_REUSE_RUN1_NOTE  free text recorded with the reuse (why D1 is not re-driven)
#   RUN1_ROOT PRIORS_DIR RUN2_ROOT SUPP_ROOT P1_ROOT
#                     default $CALIB_ROOT/{run1,prereg,run2,supp,p1}
#   TRAINING_PLAN     D5 plan (default p1-deep-overload)
#   RUN2_WINDOWS      windows.csv the D4 b50 table reads
#                     (default $RUN2_ROOT/<model>/dataset/windows.csv)
#   REPROBE_GRID_<model, '-' -> '_'>   e.g. REPROBE_GRID_dsqwen_7b=1.3,1.45,1.6,1.8:
#                     overrides the D4 grid rule for that model
#   CHAIN_DIR         markers, b50 tables, per-model state (default $CALIB_ROOT/chain)
#   CHAIN_LOG_DIR     step logs (default $CALIB_ROOT/logs)
#   CHAIN_STATUS      JSONL of every step start/end (default $CALIB_ROOT/CHAIN_STATUS.jsonl)
#   CHAIN_LOCK_DIR    HOST-LOCAL directory for the flock files (default /run/lock; flock on
#                     NFS is not reliable, so every chain must run on the same host)
#   CHAIN_POLL_S      barrier poll (default 60); CHAIN_BARRIER_TIMEOUT_S (default 86400)
#   CALIB_DRYRUN_ROOT launcher dry-run scratch (default $CALIB_ROOT/dryrun)
set -euo pipefail

CAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CALIB_REPO:=$(cd "$CAL/../../.." && pwd)}"

MODEL="${1:-}"
[[ "$MODEL" =~ ^(dsqwen-7b|dsllama-8b|dsqwen-14b)$ ]] \
  || { echo "usage: $(basename "$0") <dsqwen-7b|dsllama-8b|dsqwen-14b> [--dry-run]" >&2; exit 2; }
DRY=0
case "${2:-}" in
  "") ;;
  --dry-run) DRY=1 ;;
  *) echo "unknown argument '$2'" >&2; exit 2 ;;
esac

[[ -n "${CALIB_ROOT:-}" ]] || { echo "FATAL: set CALIB_ROOT" >&2; exit 2; }
[[ -n "${DESIGN_SEED:-}" ]] || { echo "FATAL: set DESIGN_SEED" >&2; exit 2; }
export DESIGN_SEED
: "${CHAIN_MODELS:=dsqwen-7b dsllama-8b dsqwen-14b}"
: "${RUN1_ROOT:=$CALIB_ROOT/run1}"
: "${PRIORS_DIR:=$CALIB_ROOT/prereg}"
: "${RUN2_ROOT:=$CALIB_ROOT/run2}"
: "${SUPP_ROOT:=$CALIB_ROOT/supp}"
: "${P1_ROOT:=$CALIB_ROOT/p1}"
: "${TRAINING_PLAN:=p1-deep-overload}"
: "${CHAIN_DIR:=$CALIB_ROOT/chain}"
: "${CHAIN_LOG_DIR:=$CALIB_ROOT/logs}"
: "${CHAIN_LOCK_DIR:=/run/lock}"
: "${CHAIN_POLL_S:=60}"
: "${CHAIN_BARRIER_TIMEOUT_S:=86400}"
: "${CALIB_DRYRUN_ROOT:=$CALIB_ROOT/dryrun}"
: "${CHAIN_REUSE_RUN1:=0}"
[[ "$CHAIN_REUSE_RUN1" =~ ^[01]$ ]] || { echo "FATAL: CHAIN_REUSE_RUN1 must be 0 or 1" >&2; exit 2; }
export CALIB_DRYRUN_ROOT
if [[ "$DRY" == 1 ]]; then
  CHAIN_DIR="$CALIB_DRYRUN_ROOT/chain"
  CHAIN_LOG_DIR="$CALIB_DRYRUN_ROOT/logs"
  : "${CHAIN_STATUS:=$CALIB_DRYRUN_ROOT/CHAIN_STATUS.jsonl}"
  D2_OUT="$CALIB_DRYRUN_ROOT/prereg"
  DRY_FLAG=(--dry-run)
else
  : "${CHAIN_STATUS:=$CALIB_ROOT/CHAIN_STATUS.jsonl}"
  D2_OUT="$PRIORS_DIR"
  DRY_FLAG=()
fi
mkdir -p "$CHAIN_DIR" "$CHAIN_LOG_DIR" "$CHAIN_LOCK_DIR"
STATUS_LOCK="$CHAIN_LOCK_DIR/tre-calib-chain-status.lock"
D2_LOCK="$CHAIN_LOCK_DIR/tre-calib-d2.lock"
P1_LOCK="$CHAIN_LOCK_DIR/tre-calib-p1.lock"
ATTEMPTS="$CALIB_ROOT/ATTEMPTS.md"
STATE_FILE="$CHAIN_DIR/$MODEL.state"

json_str() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# status <step> <event> [rc] [detail]   (time = this host's clock)
status() {
  local line
  line="{\"ts\":\"$(date -Iseconds)\",\"host\":\"$(hostname)\",\"model\":\"$MODEL\",\"step\":\"$1\",\"event\":\"$2\",\"rc\":${3:-null},\"pid\":$$,\"dry_run\":$DRY,\"detail\":$(json_str "${4:-}")}"
  flock "$STATUS_LOCK" bash -c 'printf "%s\n" "$1" >> "$2"' _ "$line" "$CHAIN_STATUS"
  printf '%s\n' "$line" > "$STATE_FILE"
  echo "[chain $MODEL] $line"
}

stop() {  # stop <step> <rc> <why>
  status "$1" chain_stopped "$2" "$3"
  exit "$2"
}

# run_step <step> <log> <command...>: the command runs in a subshell with the lock fds
# closed (a long-lived child must never hold a chain lock), output to <log>.
run_step() {
  local step="$1" log="$2" rc; shift 2
  status "$step" start null "log=$log"
  set +e
  ( exec 8>&- 9>&-; "$@" ) > "$log" 2>&1 < /dev/null
  rc=$?
  set -e
  status "$step" end "$rc" "log=$log"
  return "$rc"
}

step_log() { echo "$CHAIN_LOG_DIR/$1.$MODEL.log"; }

enter_deploy() {
  cd "$CALIB_REPO/deploy"
  export PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer:../ui"
}

# ---------------------------------------------------------------- D1
d1() { OUT_ROOT="$RUN1_ROOT" bash "$CAL/run_primitives.sh" "$MODEL" ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}; }

# ---------------------------------------------------------------- barrier
barrier_run1() {
  local start now m rc missing
  start=$(date +%s)
  status barrier wait null "run1 of: $CHAIN_MODELS"
  while :; do
    missing=""
    for m in $CHAIN_MODELS; do
      if [[ -f "$CHAIN_DIR/run1.$m.rc" ]]; then
        rc="$(cat "$CHAIN_DIR/run1.$m.rc")"
        [[ "$rc" == 0 ]] || stop barrier 1 "run1 of $m exited $rc: priors need every model's run1 (owner)"
      else
        missing+=" $m"
      fi
    done
    [[ -n "$missing" ]] || break
    now=$(date +%s)
    (( now - start < CHAIN_BARRIER_TIMEOUT_S )) \
      || stop barrier 1 "timed out after ${CHAIN_BARRIER_TIMEOUT_S}s waiting for run1 of$missing"
    sleep "$CHAIN_POLL_S"
  done
  status barrier passed 0 ""
}

# ---------------------------------------------------------------- D2
d2_compute() {
  set -euo pipefail
  enter_deploy
  if [[ "$DRY" != 1 && -e "$D2_OUT/rho_priors.json" ]]; then
    echo "FATAL: $D2_OUT/rho_priors.json exists; priors are computed once (register and use a new dir)" >&2
    exit 1
  fi
  if [[ ! -s "$RUN1_ROOT/dataset/windows.csv" ]]; then
    if [[ "$DRY" == 1 ]]; then echo "FATAL: $RUN1_ROOT/dataset/windows.csv missing" >&2; exit 1; fi
    echo "building the merged standard dataset of $RUN1_ROOT"
    python3 -m scripts.calibration_dataset "$RUN1_ROOT"
  fi
  mkdir -p "$D2_OUT"
  python3 -m scripts.analysis.calibration_priors "$RUN1_ROOT" --out-dir "$D2_OUT"
  ( cd "$D2_OUT" && sha256sum regime_groups.json rho_priors.json > SHA256SUMS )
  {
    echo "Preregistration addendum (D2): round-1 priors, frozen before round 2 collects anything"
    echo "written_at=$(date -Iseconds) host=$(hostname)"
    echo "calibration_code=$(git -C "$CALIB_REPO" rev-parse HEAD 2>/dev/null || echo unknown) ($(git -C "$CALIB_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?))"
    echo "run1_root=$RUN1_ROOT"
    echo "design_seed=$DESIGN_SEED"
    echo "run1 dataset inputs:"
    ( cd "$RUN1_ROOT/dataset" && sha256sum requests.csv windows.csv cells.csv )
    echo "priors (SHA256SUMS in $D2_OUT):"
    cat "$D2_OUT/SHA256SUMS"
    echo "No deviation from the plan. Commit this text into the preregistration addendum (by hand; the chain does not commit)."
  } > "$D2_OUT/ADDENDUM-D2.txt"
  cat "$D2_OUT/ADDENDUM-D2.txt"
}

d2_once() {
  local log rc
  log="$(step_log D2)"
  if [[ "$DRY" == 1 ]]; then
    rm -rf "$D2_OUT"
    # not `run_step ... || stop`: under || bash ignores set -e inside d2_compute
    set +e; run_step D2 "$log" d2_compute; rc=$?; set -e
    [[ "$rc" == 0 ]] || stop D2 "$rc" "priors failed (dry run)"
    return 0
  fi
  exec 9>"$D2_LOCK"
  flock 9
  if [[ -f "$CHAIN_DIR/d2.rc" ]]; then
    rc="$(cat "$CHAIN_DIR/d2.rc")"
    flock -u 9; exec 9>&-
    status D2 reused "$rc" "computed by another chain into $D2_OUT"
    [[ "$rc" == 0 ]] || stop D2 "$rc" "D2 failed in another chain"
    return 0
  fi
  set +e; run_step D2 "$log" d2_compute; rc=$?; set -e
  echo "$rc" > "$CHAIN_DIR/d2.rc"
  flock -u 9; exec 9>&-
  [[ "$rc" == 0 ]] || stop D2 "$rc" "priors failed"
}

# ---------------------------------------------------------------- D3
d3() {
  OUT_ROOT="$RUN2_ROOT" PRIORS_DIR="$1" bash "$CAL/run_ladder.sh" "$MODEL" ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

# ---------------------------------------------------------------- D4
# Sets GRID (and GRID_WHY) from the override or round 2's b50 table.
d4_grid() {
  local var="REPROBE_GRID_${MODEL//-/_}" table
  if [[ -n "${!var:-}" ]]; then
    GRID="${!var}"; GRID_WHY="override $var"
    return 0
  fi
  table="$CHAIN_DIR/b50_run2.$MODEL.csv"
  rm -f "$table"   # derived, rebuilt from the run2 dataset every time
  ( enter_deploy
    python3 -m scripts.analysis.boundary_b50_table --windows "${RUN2_WINDOWS:-$RUN2_ROOT/$MODEL/dataset/windows.csv}" \
      --base-run "$RUN2_ROOT" --out "$table" ) > "$(step_log D4grid)" 2>&1 \
    || { cat "$(step_log D4grid)" >&2; return 1; }
  GRID="$( enter_deploy; python3 -m scripts.analysis.reprobe_grid --b50-table "$table" --model "$MODEL" \
           2>>"$(step_log D4grid)" )" || return 1
  GRID_WHY="$(tail -n 1 "$(step_log D4grid)")"
}

d4() {
  OUT_ROOT="$SUPP_ROOT" BASE_RUN="$RUN2_ROOT" REPROBE_GRID="$GRID" \
    bash "$CAL/run_supplement.sh" "$MODEL" ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

# ---------------------------------------------------------------- D5
d5() {
  OUT_ROOT="$P1_ROOT" BASE_RUN="$RUN2_ROOT" SUPP_RUN="$SUPP_ROOT" TRAINING_PLAN="$TRAINING_PLAN" \
    bash "$CAL/run_stage3.sh" "$MODEL" ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
}

# ---------------------------------------------------------------- D1 reuse
# run1_reuse_check <evidence.json>: CHAIN_REUSE_RUN1=1 accepts the run1 already on disk
# instead of driving D1, only when it is verifiably complete: one campaign of this model,
# clean code commit, every cell plan.json scheduled present in the standard dataset
# (run1/<model>/dataset/cells.csv), every dataset row valid with its csv / raw / guard
# on disk and the guard not voided, every planned boundary search written without a stop
# reason, no capture backfill pending, no static grid planned. Reads run1 only; writes
# the evidence JSON and prints a one-line summary (exit 1 + reason on stderr otherwise).
run1_reuse_check() {
  ( enter_deploy
    python3 - "$RUN1_ROOT/$MODEL" "$MODEL" "$1" \
      "$(git -C "$CALIB_REPO" rev-parse HEAD 2>/dev/null || echo unknown)" \
      "${CHAIN_REUSE_RUN1_NOTE:-}" <<'PY'
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

from scripts import calibration_capture as capture

run, model, evidence_path, chain_git, note = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]),
                                              sys.argv[4], sys.argv[5])


def fail(why):
    sys.exit(f"run1 of {model} at {run} is not reusable: {why}")


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{path.name}: {exc}")


plan = load(run / "plan.json")
status = load(run / "campaign_status.json")
code = (plan.get("provenance") or {}).get("code") or {}
if plan.get("models") != [model]:
    fail(f"plan.json models {plan.get('models')}")
if not code.get("commit") or code.get("dirty") is not False:
    fail(f"plan.json code {code} is not a clean commit")
if (plan.get("static_grid") or {}).get("cells"):
    fail("plan.json has static-grid cells (not checked here)")
try:
    with open(run / "dataset" / "cells.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
except OSError as exc:
    fail(f"dataset/cells.csv: {exc}")
if not rows or {r["model"] for r in rows} != {model}:
    fail("dataset/cells.csv is empty or holds another model")
bad = [f"{r['shape']}/{r['primitive']}/{r['stage']}/{r['cell_id']}={r['status']}"
       for r in rows if r["status"] != "valid"]
if bad:
    fail(f"{len(bad)} dataset row(s) not valid: {', '.join(bad[:5])}")
for r in rows:
    for col in ("online_csv_path", "raw_path", "guard_path"):
        p = run / r[col] if r[col] else None
        if p is None or not p.is_file() or p.stat().st_size == 0:
            fail(f"{r['cell_id']} ({r['shape']}/{r['primitive']}): {col} {r[col]!r} missing or empty")
    if load(run / r["guard_path"]).get("voided") is not False:
        fail(f"guard {r['guard_path']} is voided")
planned = sorted((c["shape"], c["primitive"]) for c in plan.get("cells") or [])
driven = sorted((r["shape"], r["primitive"]) for r in rows if r["primitive"] != "hold")
if not planned or driven != planned:
    fail(f"planned cells {planned} != dataset cells {driven}")
searches = {}
for b in plan.get("boundary_cells") or []:
    doc = load(run / "boundary" / f"{model}_{b['shape']}.json")
    if doc.get("stopped_reason") or doc.get("rho_star") is None:
        fail(f"boundary {b['shape']}: rho_star {doc.get('rho_star')} stopped {doc.get('stopped_reason')!r}")
    if not any(r["shape"] == b["shape"] and r["primitive"] == "hold" for r in rows):
        fail(f"boundary {b['shape']}: no probe cell in the dataset")
    searches[b["shape"]] = {"rho_star": doc["rho_star"], "status": doc.get("rho_star_status")}
pending = capture.pending_cells(run, max_age_s=None)
if pending:
    fail(f"{len(pending)} capture backfill(s) pending: {[p.name for p in pending][:5]}")
evidence_path.write_text(json.dumps({
    "model": model, "run": str(run), "run_code_commit": code["commit"],
    "campaign_status": status, "planned_cells": len(planned),
    "boundary_searches": searches, "dataset_rows": len(rows),
    "reused_at_utc": datetime.now(timezone.utc).isoformat(), "chain_code_commit": chain_git,
    "note": note,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"run1 not re-driven: driven at git {code['commit'][:8]}, campaign_status "
      f"{status.get('status')} exit {status.get('exit_code')}; {len(planned)} planned cells + "
      f"{len(searches)} boundary searches complete, {len(rows)} dataset rows valid; "
      f"evidence {evidence_path}" + (f"; {note}" if note else ""))
PY
  )
}

# ================================================================= main
status chain start null "dry_run=$DRY repo=$CALIB_REPO git=$(git -C "$CALIB_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) run1=$RUN1_ROOT priors=$PRIORS_DIR run2=$RUN2_ROOT supp=$SUPP_ROOT p1=$P1_ROOT plan=$TRAINING_PLAN reuse_run1=$CHAIN_REUSE_RUN1"

# D1
if [[ "$CHAIN_REUSE_RUN1" == 1 ]]; then
  log="$(step_log D1reuse)"
  status D1 reuse_check null "run1=$RUN1_ROOT/$MODEL log=$log"
  set +e; detail="$(run1_reuse_check "$CHAIN_DIR/run1.$MODEL.reused.json" 2>"$log")"; rc=$?; set -e
  if [[ "$rc" != 0 ]]; then
    [[ "$DRY" == 1 ]] || echo 1 > "$CHAIN_DIR/run1.$MODEL.rc"
    stop D1 1 "CHAIN_REUSE_RUN1=1 but run1 is not verifiably complete: $(tail -n 1 "$log")"
  fi
  [[ "$DRY" == 1 ]] || echo 0 > "$CHAIN_DIR/run1.$MODEL.rc"
  status D1 reused 0 "$detail"
else
  set +e; run_step D1 "$(step_log D1)" d1; rc=$?; set -e
  [[ "$DRY" == 1 ]] || echo "$rc" > "$CHAIN_DIR/run1.$MODEL.rc"
  [[ "$rc" == 0 ]] || stop D1 "$rc" "run1 failed"
fi

# barrier + D2
[[ "$DRY" == 1 ]] || barrier_run1
d2_once

# D3 (a dry run reads PRIORS_DIR as given: D2's dry output has no frozen status)
set +e; run_step D3 "$(step_log D3)" d3 "$PRIORS_DIR"; rc=$?; set -e
[[ "$rc" == 0 ]] || stop D3 "$rc" "run2 failed"

# D4
GRID=""; GRID_WHY=""
d4_grid || stop D4 1 "could not derive REPROBE_GRID from $RUN2_ROOT (see $(step_log D4grid))"
status D4 grid 0 "REPROBE_GRID=$GRID ($GRID_WHY)"
if [[ "$DRY" != 1 ]]; then
  flock "$STATUS_LOCK" bash -c 'printf "%s\n" "$1" >> "$2"' _ \
    "- $(date -Iseconds) D4 $MODEL: REPROBE_GRID=$GRID (x rho*_run2 of S3), rule: $GRID_WHY; base run $RUN2_ROOT, out $SUPP_ROOT/$MODEL" \
    "$ATTEMPTS"
fi
set +e; run_step D4 "$(step_log D4)" d4; rc=$?; set -e
[[ "$rc" == 0 ]] || stop D4 "$rc" "$([[ $rc == 3 ]] && echo 'exit 3: a rho* is a bound or the smoke is out of band (owner)' || echo 'supplement failed')"

# D5 (one model at a time across the chains)
if [[ "$DRY" == 1 ]]; then
  set +e; run_step D5 "$(step_log D5)" d5; rc=$?; set -e
else
  exec 8>"$P1_LOCK"
  status D5 wait_lock null "$P1_LOCK"
  flock 8
  status D5 lock_acquired null "$P1_LOCK"
  set +e; run_step D5 "$(step_log D5)" d5; rc=$?; set -e
  flock -u 8; exec 8>&-
fi
[[ "$rc" == 0 ]] || stop D5 "$rc" "$([[ $rc == 3 ]] && echo 'exit 3: a P1 cell had no labelled window or a check failed (owner)' || echo 'P1 failed')"

status chain done 0 "all steps exited 0"
