#!/usr/bin/env bash
# 2026-11-03 "before" reference: self-checks a-c and the prereg analyses on the OLD (10-03) training data only.
# CPU only, read-only on the data; writes under OUT_DIR (selfcheck/, before/, logs/).
#   run_before.sh OUT_DIR        environment: fit_env.sh (TRE_FIT_DATA_ROOT required)
# Repo copy of <data root>/calib_supp_20261103_fit_tools/run_before.sh (it wrote into the tools directory).
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fit_env.sh"
fit_require_root
OUT=${1:?usage: run_before.sh OUT_DIR}
mkdir -p "$OUT/logs"
fit_marker_check; "${PY[@]}" "$FIT_TSS_DIR/selfcheck.py" --out "$OUT/selfcheck" > "$OUT/logs/selfcheck.log" 2>&1          # a-c
fit_marker_check; "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" --variant all10 --window-ms 10000 --out "$OUT/before" \
    > "$OUT/logs/before_w10.log" 2>&1                                                                               # primary, S1-S4
fit_marker_check; "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" --variant S5 --window-ms 20000 --out "$OUT/before" \
    > "$OUT/logs/before_S5_w20.log" 2>&1                                                                            # S5 20 s
# S5 30 s: no old 30 s root (build_datasets.sh --old ROOT 30000 builds one; then --data old=ROOT --window-ms 30000)
# dev test of validate_supp.py on a TRAINING subset (40 random run2 / run2b cells per model; in-sample, not acceptance)
fit_marker_check; "${PY[@]}" "$FIT_TSS_DIR/validate_supp.py" --candidate "$OUT/before/primary/candidate.json" --data old \
    --only-src run2,run2b --sample-cells 40 --out "$OUT/before/primary/dev_validate" > "$OUT/logs/dev_validate.log" 2>&1
