#!/usr/bin/env bash
# After the supplementary campaign (A, B, E): build datasets -> refit (primary + S1-S5) -> dev check -> (the user picks
# the candidate from primary / S1-S3, prereg §3, BEFORE any E / M number exists) -> one-time final validation on E, M.
# CPU only, read-only on the campaign data.
#   run_after_campaign.sh build          new A / B datasets at 10 / 20 / 30 s + the old roots at 30 s
#   run_after_campaign.sh fit            primary, S1-S4 (10 s), S5 (20 s, 30 s) on old + A + B  -> $FIT/<variant>/
#   run_after_campaign.sh dev            validate_supp.py on a TRAINING subset (sanity only, in-sample)
#   run_after_campaign.sh final-build    E and M datasets at 10 s (reads the sealed raw data: final step only)
#   run_after_campaign.sh final CAND     prereg §4 on E and on M, once per candidate (ledger $TRE_FIT_LEDGER)
# Environment: fit_env.sh (TRE_FIT_DATA_ROOT, TRE = the dataset builder, TRE_FIT_*) and
#   CALIB_ROOT      (required) the supplement's root (2026-11-03: <data root>/calib_supp_20261103)
#   RAW_A / RAW_B / RAW_E   the parts' campaign roots (default $CALIB_ROOT/A, /B, /E)
#   RAW_M           the sealed M set (default $TRE_FIT_DATA_ROOT/calib_20261003/M: MP / MD / G800x240 / U512x512)
#   DS / FIT        dataset roots $DS/w<W>/<src>/<model>/dataset_hybrid and fit results (default $CALIB_ROOT/datasets,
#                   $CALIB_ROOT/fit: prereg §5)
#   TRE_FIT_LEDGER  (final only, required) the one-time ledger (11-03: <data root>/calib_supp_20261103_fit_tools/
#                   final_eval_ledger.jsonl)
# 2026-11-03 run: TRE = the campaign's worktree at 97a4d4ab (main then lacked ES / EL in is_held_out).
# Repo copy of <data root>/calib_supp_20261103_fit_tools/run_after_campaign.sh: paths and host rules are configuration.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fit_env.sh"
export TRE
fit_require_root
C=${CALIB_ROOT:?set CALIB_ROOT (the supplement root)}
RAW_A=${RAW_A:-$C/A}
RAW_B=${RAW_B:-$C/B}
RAW_E=${RAW_E:-$C/E}
RAW_M=${RAW_M:-$TRE_FIT_DATA_ROOT/calib_20261003/M}
DS=${DS:-$C/datasets}
FIT=${FIT:-$C/fit}
fit_marker_check
NEW10=(--data old --data "A=$DS/w10000/A" --data "B=$DS/w10000/B")
case "${1:-}" in
build)
  for W in 10000 20000 30000; do
    "$FIT_TSS_DIR/build_datasets.sh" "$RAW_A" A "$DS/w$W" "$W"
    "$FIT_TSS_DIR/build_datasets.sh" "$RAW_B" B "$DS/w$W" "$W"
  done
  "$FIT_TSS_DIR/build_datasets.sh" --old "$DS/w30000-old" 30000   # the 10 s and 20 s old roots exist (supp_lib.OLD_ROOTS)
  ;;
fit)
  mkdir -p "$FIT/logs"
  "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" --variant all10 --window-ms 10000 --out "$FIT" "${NEW10[@]}" > "$FIT/logs/fit_w10.log" 2>&1
  "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" --variant S5 --window-ms 20000 --out "$FIT" --data old \
      --data "A=$DS/w20000/A" --data "B=$DS/w20000/B" > "$FIT/logs/fit_S5_w20.log" 2>&1
  "${PY[@]}" "$FIT_TSS_DIR/refit_supp.py" --variant S5 --window-ms 30000 --out "$FIT" --data "old=$DS/w30000-old" \
      --data "A=$DS/w30000/A" --data "B=$DS/w30000/B" > "$FIT/logs/fit_S5_w30.log" 2>&1
  cat "$FIT"/{primary,S1,S2,S3,S4,S5_w20000,S5_w30000}/summary.txt
  ;;
dev)
  "${PY[@]}" "$FIT_TSS_DIR/validate_supp.py" --candidate "$FIT/primary/candidate.json" "${NEW10[@]}" --sample-cells 60 \
      --out "$FIT/primary/dev_validate"
  ;;
final-build)
  "$FIT_TSS_DIR/build_datasets.sh" "$RAW_E" E "$DS/w10000" 10000 --i-am-the-final-evaluation
  "$FIT_TSS_DIR/build_datasets.sh" "$RAW_M" M "$DS/w10000" 10000 --i-am-the-final-evaluation
  ;;
final)
  CAND=${2:?usage: final CANDIDATE_JSON  (the set the user chose, e.g. $FIT/primary/candidate.json)}
  LEDGER=${TRE_FIT_LEDGER:?set TRE_FIT_LEDGER (the one-time ledger of the final evaluation)}
  # the prereg must be frozen (CALIB_ROOT/PREREG.md + PREREG.sha256) before E / M are opened
  h=$(sha256sum "$C/PREREG.md" | cut -d' ' -f1); grep -q "$h" "$C/PREREG.sha256" || { echo "PREREG not frozen / changed" >&2; exit 4; }
  "${PY[@]}" "$FIT_TSS_DIR/validate_supp.py" --candidate "$CAND" --data "E=$DS/w10000/E" --out "$FIT/final" \
      --i-am-the-final-evaluation --set-name E --ledger "$LEDGER"
  "${PY[@]}" "$FIT_TSS_DIR/validate_supp.py" --candidate "$CAND" --data "M=$DS/w10000/M" --out "$FIT/final" \
      --i-am-the-final-evaluation --set-name M --ledger "$LEDGER"
  ;;
*) sed -n '2,9p' "$0"; exit 2 ;;
esac
