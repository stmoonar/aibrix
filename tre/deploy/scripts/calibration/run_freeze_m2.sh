#!/usr/bin/env bash
# Offline theta freeze from already collected calibration runs (CPU only; no cluster access):
#
#   standard datasets (calibration_dataset, --queue-estimator)
#     -> dline_refit trainset --train-dynamic (H2 = run 1 + run 2)
#     -> alpha --publish-tau-s 10 -> wp --lambda-method v1 (lambda / w_p jointly)
#     -> final --no-holdout --fit-method <M2> -> freeze (+ verify-freeze)
#
# and, when ACCEPT_M_ROOT is set, the M datasets with the same queue estimator and
# `dline_refit accept` on them (DEV_OPENED_M set = a DEV evaluation of M sets sealed under
# another freeze: `accept --dev-opened-m`, never an acceptance).
#
# The defaults are the 2026-10-25 method (user decision): queue = raw_mean (the mean of every
# sidecar sample in the window), fit = m2_unique_grid_plateau_midpoint (eps .005, theta and
# bands), lambda / w_p by v1's selection. QUEUE_ESTIMATOR=live_grid_sum_over_expected with
# FIT_METHOD=ba_argmax_quantile_grid rebuilds the 2026-10-05 freeze's chain.
#
#   run_freeze_m2.sh            (configuration from the environment; nothing site-specific here)
#
# Environment:
#   CALIB_ROOT        required: root of the collected runs, <CALIB_ROOT>/<stage>/<model>/
#   OUT_DIR           required: new or empty output directory (datasets/, fit/, freeze/, logs/)
#   MODE              dev (default) | final. final refuses a dirty worktree; dev records it.
#   MODELS            "dsqwen-7b dsllama-8b dsqwen-14b"
#   H2_STAGES         "run1 run2"   stages whose sealed split joins H2 (D16)
#   TRAIN_STAGES      "supp p1"     further training stages
#   STAGE_OVERRIDES   "dsqwen-14b:run2=run2b dsqwen-7b:p1=p1r2"  per-model stage renames
#   QUEUE_ESTIMATOR   raw_mean      (scripts.rewindow_from_raw.QUEUE_ESTIMATORS)
#   FIT_METHOD        m2_unique_grid_plateau_midpoint  (tre_calibration.fit.FIT_METHODS)
#   LAMBDA_METHOD     v1            PUBLISH_TAU_S 10   ATTRIBUTION hybrid
#   LABEL_REGISTRY    the repo's deploy/registry.yaml (the label's idle TTFT fits)
#   JOBS              4 parallel dataset builds      NICE 10
#   ACCEPT_M_ROOT     optional: <ACCEPT_M_ROOT>/<model>/ holds M and its M_manifest.json
#   DEV_OPENED_M      optional reason text: run accept as a DEV evaluation (--dev-opened-m)
#   ACCEPT_NAME       M2   dataset name prefix of the M sets      ACCEPT_DWELL  1 (B' dwell)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRE="$(cd "$HERE/../../.." && pwd)"
: "${CALIB_ROOT:?CALIB_ROOT (root of the collected runs) is required}"
: "${OUT_DIR:?OUT_DIR (new output directory) is required}"
: "${MODE:=dev}"
: "${MODELS:=dsqwen-7b dsllama-8b dsqwen-14b}"
: "${H2_STAGES:=run1 run2}"
: "${TRAIN_STAGES:=supp p1}"
: "${STAGE_OVERRIDES:=dsqwen-14b:run2=run2b dsqwen-7b:p1=p1r2}"
: "${QUEUE_ESTIMATOR:=raw_mean}"
: "${FIT_METHOD:=m2_unique_grid_plateau_midpoint}"
: "${LAMBDA_METHOD:=v1}"
: "${PUBLISH_TAU_S:=10}"
: "${ATTRIBUTION:=hybrid}"
: "${LABEL_REGISTRY:=$TRE/deploy/registry.yaml}"
: "${JOBS:=4}"
: "${NICE:=10}"
: "${ACCEPT_M_ROOT:=}"
: "${DEV_OPENED_M:=}"
: "${ACCEPT_NAME:=M2}"
: "${ACCEPT_DWELL:=1}"

case "$MODE" in dev|final) ;; *) echo "MODE must be dev or final, got $MODE" >&2; exit 2 ;; esac
[ -d "$CALIB_ROOT" ] || { echo "CALIB_ROOT $CALIB_ROOT is not a directory" >&2; exit 2; }
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "OUT_DIR $OUT_DIR is not empty: a freeze run never reuses an output directory" >&2; exit 2
fi
DIRTY="$(git -C "$TRE" status --porcelain --untracked-files=no)"
if [ -n "$DIRTY" ] && [ "$MODE" = final ]; then
  echo "MODE=final refuses a dirty worktree (the freeze records the commit):" >&2; echo "$DIRTY" >&2; exit 2
fi

stage_for() {  # stage_for <model> <stage> -> the model's directory name of that stage
  local m=$1 st=$2 o
  for o in $STAGE_OVERRIDES; do
    if [ "${o%%:*}" = "$m" ] && [ "${o#*:}" != "${o#*:$st=}" ]; then echo "${o#*:$st=}"; return; fi
  done
  echo "$st"
}

mkdir -p "$OUT_DIR"/{datasets,logs,freeze}
DS="$OUT_DIR/datasets"; LOG="$OUT_DIR/logs"; FIT="$OUT_DIR/fit"
cd "$TRE/deploy"
export PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer:../ui"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=(nice -n "$NICE" python3)
{
  echo "mode=$MODE"; git -C "$TRE" rev-parse HEAD; echo "dirty:"; echo "$DIRTY"
  echo "calib_root=$CALIB_ROOT"; echo "queue_estimator=$QUEUE_ESTIMATOR fit_method=$FIT_METHOD"
  echo "lambda_method=$LAMBDA_METHOD publish_tau_s=$PUBLISH_TAU_S attribution=$ATTRIBUTION"
  echo "models=$MODELS h2=$H2_STAGES train=$TRAIN_STAGES overrides=$STAGE_OVERRIDES"
  sha256sum "$LABEL_REGISTRY"; date -u +%FT%TZ
} > "$OUT_DIR/run_record.txt"

# 1. standard datasets (training stages, plus M when it is evaluated)
jobs=()
for m in $MODELS; do
  for st in $H2_STAGES $TRAIN_STAGES; do jobs+=("$(stage_for "$m" "$st") $m $CALIB_ROOT/$(stage_for "$m" "$st")/$m"); done
  if [ -n "$ACCEPT_M_ROOT" ]; then jobs+=("M $m $ACCEPT_M_ROOT/$m"); fi
done
build() {  # build <stage> <model> <run dir>
  local out="$DS/$1/$2/dataset_$ATTRIBUTION"
  mkdir -p "$(dirname "$out")"
  if "${PY[@]}" -m scripts.calibration_dataset "$3" --out-dir "$out" --attribution "$ATTRIBUTION" \
       --label-registry "$LABEL_REGISTRY" --queue-estimator "$QUEUE_ESTIMATOR" > "$LOG/dataset.$1.$2.log" 2>&1
  then echo "dataset $1 $2 ok"; else echo "dataset $1 $2 FAILED"; return 1; fi
}
export -f build; export DS LOG ATTRIBUTION LABEL_REGISTRY QUEUE_ESTIMATOR NICE
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -L 1 bash -c 'PY=(nice -n "$NICE" python3); build "$@"' _ \
  || { echo "a dataset build failed (see $LOG)" >&2; exit 1; }

# 2. per model: trainset -> alpha -> wp -> final
fit_one() {  # fit_one <model>
  local m=$1 F="$FIT/$m/fit" O="$FIT/refit" st args=()
  for st in $H2_STAGES; do args+=(--h2-dataset "$(stage_for "$m" "$st")=$DS/$(stage_for "$m" "$st")/$m/dataset_$ATTRIBUTION"); done
  for st in $TRAIN_STAGES; do args+=(--dataset "$(stage_for "$m" "$st")=$DS/$(stage_for "$m" "$st")/$m/dataset_$ATTRIBUTION"); done
  local c=(--model "$m" --arm primary --fit-dir "$F" --registry "$LABEL_REGISTRY")
  "${PY[@]}" -m scripts.dline_refit trainset --train-dynamic --fit-dir "$F" --model "$m" \
      --queue-estimator "$QUEUE_ESTIMATOR" "${args[@]}" || return 1
  "${PY[@]}" -m scripts.dline_refit alpha "${c[@]}" --out-dir "$O" --publish-tau-s "$PUBLISH_TAU_S" || return 1
  "${PY[@]}" -m scripts.dline_refit wp "${c[@]}" --out-dir "$O" --lambda-method "$LAMBDA_METHOD" || return 1
  "${PY[@]}" -m scripts.dline_refit final "${c[@]}" --out-dir "$O" --no-holdout --fit-method "$FIT_METHOD" || return 1
}
pids=()
for m in $MODELS; do
  ( fit_one "$m"; echo "EXIT $?" ) > "$LOG/fit.$m.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
grep -H EXIT "$LOG"/fit.*.log
if grep -L '^EXIT 0$' "$LOG"/fit.*.log | grep -q .; then echo "a fit failed (see $LOG)" >&2; exit 1; fi

# 3. freeze
margs=(); for m in $MODELS; do margs+=(--model "$m"); done
FREEZE="$OUT_DIR/freeze/params_freeze.json"
"${PY[@]}" -m scripts.dline_refit freeze --out-dir "$FIT/refit" --fit-dir "$FIT/{model}/fit" "${margs[@]}" \
    --registry "$LABEL_REGISTRY" --freeze-file "$FREEZE" | tee "$LOG/freeze.log"
"${PY[@]}" -m scripts.dline_refit verify-freeze --freeze-file "$FREEZE"

# 4. optional: M (DEV when it was sealed under another freeze)
if [ -n "$ACCEPT_M_ROOT" ]; then
  aargs=()
  for m in $MODELS; do
    aargs+=(--dataset "$ACCEPT_NAME.$m=$DS/M/$m/dataset_$ATTRIBUTION" --m-manifest "$ACCEPT_M_ROOT/$m/M_manifest.json")
  done
  if [ -n "$DEV_OPENED_M" ]; then aargs+=(--dev-opened-m "$DEV_OPENED_M"); fi
  set +e
  "${PY[@]}" -m scripts.dline_refit accept --freeze-file "$FREEZE" "${aargs[@]}" --dwell-windows "$ACCEPT_DWELL" \
      > "$LOG/accept.log" 2>&1
  rc=$?
  set -e
  tail -40 "$LOG/accept.log"
  echo "accept exit $rc"
fi
echo ALLDONE > "$OUT_DIR/ALLDONE"
