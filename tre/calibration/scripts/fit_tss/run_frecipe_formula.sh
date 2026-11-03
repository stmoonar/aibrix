#!/usr/bin/env bash
# Frecf_* fits (2026-11-03): tune F's recipe (completion attribution, r* 5 %, 10 s windows) with lambda / w_p from the
# SLO-budget formula (lambda = 0.15 L_out, w_p = 6.67 / L_in, training means). Training data only (no E / M).
# f_all = Frecf_all = the TSS parameters published 2026-11-03 13:49 (registry f75155148eba1e61); f_old and f_le1100
# are its two controls (old data only; input <= 1100).
#
#   run_frecipe_formula.sh OUT_DIR [f_all] [f_old] [f_le1100]     (default: f_all; OUT_DIR/<variant> must not exist)
#
# Inputs: the old 10 s root (supp_lib.OLD_ROOTS[10000], under TRE_FIT_DATA_ROOT) and the supplement's A / B datasets
# ($SUPP_DS/{A,B}, default $TRE_FIT_DATA_ROOT/calib_supp_20261103/datasets/w10000). Environment: fit_env.sh.
# Outputs per variant: OUT_DIR/<variant>/<tag>/{fit.json, candidate.json, summary.txt, per_shape.csv,
# tau_high_by_shape.csv}, log OUT_DIR/logs/<variant>.log. Exit 0 iff every variant exited 0.
# Repo copy of <data root>/calib_supp_20261103/fit/adhoc/frecipe-20261103/run_frecipe_formula.sh (one host only, nice 19,
# two pinned cores, a bundled site-packages; it ran all three variants): host rules and paths are configuration.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fit_env.sh"
fit_require_root
fit_marker_check
O=${1:?usage: run_frecipe_formula.sh OUT_DIR [f_all] [f_old] [f_le1100]}
shift
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -gt 0 ]] || VARIANTS=(f_all)
DS=${SUPP_DS:-$TRE_FIT_DATA_ROOT/calib_supp_20261103/datasets/w10000}
for v in "${VARIANTS[@]}"; do
  case "$v" in f_all|f_old|f_le1100) ;; *) echo "unknown variant '$v' (f_all f_old f_le1100)" >&2; exit 2 ;; esac
  [[ ! -e "$O/$v" ]] || { echo "$O/$v exists: refusing to write over an earlier fit (use a new OUT_DIR)" >&2; exit 2; }
done
mkdir -p "$O/logs"
F=(--variant custom --attribution completion --lam-wp formula --r-star 0.05 --window-ms 10000)
NEW=(--data old --data "A=$DS/A" --data "B=$DS/B")
worst=0
for v in "${VARIANTS[@]}"; do
  case "$v" in
    f_old)    args=(--tag Frecf_old --out "$O/f_old" --data old) ;;
    f_all)    args=(--tag Frecf_all --out "$O/f_all" "${NEW[@]}") ;;
    f_le1100) args=(--tag Frecf_le1100 --shapes le1100 --out "$O/f_le1100" "${NEW[@]}") ;;
  esac
  "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" "${F[@]}" "${args[@]}" > "$O/logs/$v.log" 2>&1
  rc=$?
  echo "$v exit $rc"
  [[ "$rc" == 0 ]] || worst=1
done
exit "$worst"
