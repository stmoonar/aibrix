#!/usr/bin/env bash
# PRIMARY tau fit at W = 10 s: the repo's D4' alpha_fit (main 9f2856a1, read-only) on the freeze
# calibration runs (calib_20261003, same stages / overrides as freeze-ctr20-20261028), re-windowed
# from raw at WINDOW_MS (default 10000) / STEP_MS, label = fixed SLO p95 TTFT 500 / TPOT 75 + unserved.
# CPU only, nice; outputs under OUT (no cluster access, inputs not modified).
# Step 1 (datasets) is what the fit tools read as the 'old' 10 s root (supp_lib.OLD_ROOTS[10000] =
# <data root>/analysis-tss-params-20261030/win10/alphafit-w10000-s10000/datasets): the 10-30 run had
# WINDOW_MS=10000 STEP_MS=10000 DT_REF_S=10 DWELL=1 and TRE = main at 9f2856a1 (its run_record.txt).
#   OUT=<new dir> [WINDOW_MS= STEP_MS= DT_REF_S= DWELL=] run_alphafit_w10.sh
# Environment: fit_env.sh (TRE = the builder tree, default this repo; TRE_FIT_*; TRE_FIT_NICE default 15 here),
#   CALIB_ROOT_OLD (default $TRE_FIT_DATA_ROOT/calib_20261003), LABEL_REGISTRY (default: registry.yaml of e891d29b).
# 2026-11-03 repo copy of <data root>/analysis-tss-params-20261030/win10/run_alphafit_w10.sh: the paths are
# configuration, OUT has no default (it named the original output directory), the unused FREEZE path is gone.
set -euo pipefail
: "${TRE_FIT_NICE:=15}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fit_env.sh"
fit_require_root
CALIB_ROOT=${CALIB_ROOT_OLD:-$TRE_FIT_DATA_ROOT/calib_20261003}
WINDOW_MS=${WINDOW_MS:-10000}; STEP_MS=${STEP_MS:-5000}; DT_REF_S=${DT_REF_S:-5}; DWELL=${DWELL:-1}
OUT=${OUT:?set OUT (a new directory)}
fit_marker_check
TAUS="0 1 2.5 5 7.5 10 15 20 30"
MODELS="dsqwen-7b dsllama-8b dsqwen-14b"
stage_for() { case "$1:$2" in dsqwen-14b:run2) echo run2b;; dsqwen-7b:p1) echo p1r2;; *) echo "$2";; esac; }
mkdir -p "$OUT"/{datasets,fit,logs}
OUT=$(cd "$OUT" && pwd)
LABEL_REGISTRY=$(fit_label_registry "$OUT")
cd "$TRE/deploy"
export PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer${PYTHONPATH:+:$PYTHONPATH}"
echo "tre=$(git -C $TRE rev-parse --short HEAD) window=$WINDOW_MS step=$STEP_MS dt_ref=$DT_REF_S dwell=$DWELL taus=$TAUS" > "$OUT/run_record.txt"
# 1. datasets
build() {
  local st=$1 m=$2 out="$OUT/datasets/$1/$2/dataset_hybrid"
  [ -f "$out/windows.csv" ] && return 0
  mkdir -p "$(dirname "$out")"
  "${PY[@]}" -m scripts.calibration_dataset "$CALIB_ROOT/$st/$m" --out-dir "$out" --attribution hybrid \
     --label-registry "$LABEL_REGISTRY" --queue-estimator raw_mean --numerator vllm_counter \
     --window-ms "$WINDOW_MS" --step-ms "$STEP_MS" > "$OUT/logs/dataset.$st.$m.log" 2>&1 && echo "dataset $st $m ok" || { echo "dataset $st $m FAILED"; return 1; }
}
pids=()
for m in $MODELS; do for st in run1 run2 supp p1; do s=$(stage_for $m $st); build "$s" "$m" & pids+=($!); done; done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done; [ $fail = 0 ] || exit 1
# 2. trainset + 3. alpha_fit per model (fixed label)
fit_one() {
  local m=$1 F="$OUT/fit/$m" args=() lg=()
  for st in run1 run2; do s=$(stage_for $m $st); args+=(--h2-dataset "$s=$OUT/datasets/$s/$m/dataset_hybrid"); done
  for st in supp p1; do s=$(stage_for $m $st); args+=(--dataset "$s=$OUT/datasets/$s/$m/dataset_hybrid"); done
  "${PY[@]}" -m scripts.dline_refit trainset --train-dynamic --fit-dir "$F" --model "$m" --queue-estimator raw_mean \
      --window-ms "$WINDOW_MS" --step-ms "$STEP_MS" "${args[@]}" > "$OUT/logs/trainset.$m.log" 2>&1
  for st in run1 run2 supp p1; do s=$(stage_for $m $st); [ -f "$CALIB_ROOT/$s/$m/cells.jsonl" ] && lg+=(--ledger "$CALIB_ROOT/$s/$m/cells.jsonl"); done
  case $m in dsqwen-7b) W=0.025 L=2.125;; dsllama-8b) W=0.0125 L=2.0;; dsqwen-14b) W=0.01 L=2.0;; esac   # freeze = live
  "${PY[@]}" -m scripts.alpha_fit --model "$m" --fitting-csv "$F/${m}_fitting.csv" --w-p "$W" --lambda-wait "$L" \
      --ttft-slo-mode fixed --ttft-p95-ms 500 --tpot-p95-ms 75 --label-registry "$LABEL_REGISTRY" \
      --tau-grid-s $TAUS --dt-ref-s "$DT_REF_S" --dwell-windows "$DWELL" --window-ms "$WINDOW_MS" --step-ms "$STEP_MS" \
      --bootstrap 300 "${lg[@]}" --output "$OUT/alpha.$m.json" > "$OUT/logs/alpha.$m.log" 2>&1
  echo "alpha $m exit $?"
}
for m in $MODELS; do fit_one "$m" & done; wait
grep -h "tau=" "$OUT"/logs/alpha.*.log || true
