#!/usr/bin/env bash
# Build calibration datasets for the fit tools with the recipe of the old roots
# (run_alphafit_w10.sh: --attribution hybrid --queue-estimator raw_mean
#  --numerator vllm_counter --label-registry <freeze-tree registry> --window-ms W --step-ms 10000).
# CPU only; writes only OUT_ROOT/<SRC>/<model>/dataset_hybrid (+ OUT_ROOT/build-logs/, the extracted label
# registry).  Builder = the repo at $TRE (its HEAD is recorded in every manifest; supp_lib checks window / step /
# numerator / queue estimator on load).
#
#   build_datasets.sh RAW_DIR SRC OUT_ROOT WINDOW_MS [--i-am-the-final-evaluation]
#       RAW_DIR = a campaign root holding one calibration_campaign --out-dir per model (RAW_DIR/<model>/plan.json)
#   build_datasets.sh --old OUT_ROOT WINDOW_MS
#       the 10-03 training sources ($CALIB_ROOT_OLD/<stage>/<model>, stages as supp_lib.OLD_SOURCES), e.g. the 30 s root
# A sealed RAW_DIR (calib_20261003/M, the E set, ...) is refused unless --i-am-the-final-evaluation (final step only).
# Environment: fit_env.sh (TRE, TRE_FIT_*, the exclusive-window marker, GIT_CONFIG_GLOBAL) and
#   CALIB_ROOT_OLD   the 10-03 campaign root (default $TRE_FIT_DATA_ROOT/calib_20261003)
#   LABEL_REGISTRY   the --label-registry file (default: tre/deploy/registry.yaml of commit e891d29b, fit_env.sh)
#   MODELS           default "dsqwen-7b dsllama-8b dsqwen-14b"
# 2026-11-03 repo copy of <data root>/calib_supp_20261103_fit_tools/build_datasets.sh: the paths, the host test
# (worker host -> nice 19 + two pinned cores + a bundled site-packages + a gitconfig) and the log directory are
# configuration now.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fit_env.sh"
MODELS=${MODELS:-"dsqwen-7b dsllama-8b dsqwen-14b"}
fit_marker_check
guard() {  # $1 path, $2 final flag
  PYTHONPATH="$FIT_TSS_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -B -c '
import sys; sys.dont_write_bytecode = True
from supp_lib import sealed_reason
r = sealed_reason(sys.argv[1])
if r and sys.argv[2] != "--i-am-the-final-evaluation":
    sys.exit("REFUSED %s: %s (sealed; only the final step builds it, with --i-am-the-final-evaluation)" % (sys.argv[1], r))
print("sealed (final step): %s" % r if r else "not sealed")' "$1" "${2:-}"
}
one() {  # raw_model_dir out_dir window
  local raw=$1 out=$2 w=$3 lg
  [ -f "$out/windows.csv" ] && { echo "exists: $out"; return 0; }
  mkdir -p "$(dirname "$out")" "$LOG_DIR"
  lg="$LOG_DIR/build.$(echo "$out" | tr '/' '_').log"
  ( cd "$TRE/deploy" && PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer${PYTHONPATH:+:$PYTHONPATH}" \
      "${PY[@]}" -m scripts.calibration_dataset "$raw" --out-dir "$out" --attribution hybrid \
      --label-registry "$LABEL_REG" --queue-estimator raw_mean --numerator vllm_counter \
      --window-ms "$w" --step-ms 10000 ) > "$lg" 2>&1 && echo "ok $out" || { echo "FAILED $out (log $lg)"; return 1; }
}
sha=$(git -C "$TRE" rev-parse HEAD) || { echo "cannot read the builder commit of $TRE (git safe.directory? set GIT_CONFIG_GLOBAL): refusing, the manifests would record code_commit null" >&2; exit 5; }
dirty=$(git -C "$TRE" status --porcelain) || { echo "cannot read the work-tree status of $TRE: refusing" >&2; exit 5; }
echo "builder: $TRE @ ${sha:0:8} (dirty: $(printf '%s' "$dirty" | grep -c . || true) files)"
if [ "${1:-}" = "--old" ]; then
  OUT_ROOT=$2 W=$3
  fit_require_root
  CALIB_ROOT_OLD=${CALIB_ROOT_OLD:-$TRE_FIT_DATA_ROOT/calib_20261003}
  LOG_DIR=${TRE_FIT_LOG_DIR:-$OUT_ROOT/build-logs}
  LABEL_REG=$(fit_label_registry "$OUT_ROOT")
  pids=()
  for m in $MODELS; do
    case $m in dsqwen-7b) st="run1 run2 supp p1r2";; dsllama-8b) st="run1 run2 supp p1";; dsqwen-14b) st="run1 run2b supp p1";; esac
    for s in $st; do one "$CALIB_ROOT_OLD/$s/$m" "$OUT_ROOT/$s/$m/dataset_hybrid" "$W" & pids+=($!); done
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done; exit $fail
fi
RAW=$1 SRC=$2 OUT_ROOT=$3 W=$4 FINAL=${5:-}
guard "$RAW" "$FINAL"
guard "$OUT_ROOT/$SRC" "$FINAL"
LOG_DIR=${TRE_FIT_LOG_DIR:-$OUT_ROOT/build-logs}
LABEL_REG=$(fit_label_registry "$OUT_ROOT")
pids=()
for m in $MODELS; do
  [ -d "$RAW/$m" ] || { echo "no $RAW/$m - skipped"; continue; }
  one "$RAW/$m" "$OUT_ROOT/$SRC/$m/dataset_hybrid" "$W" & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done; exit $fail
