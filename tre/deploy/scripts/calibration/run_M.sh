#!/usr/bin/env bash
# The acceptance set M (D22): collected AFTER the parameters are frozen, sealed as soon as
# it is collected (M_SHA256SUMS + M_manifest.json), read once by `dline_refit accept`.
#
#   run_M.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Exit codes: 0 = every probe measured, every cell driven, M sealed; 3 = a new shape's rho*
# did not come out measured (no M cell driven - stop for the user); 1 = stopped.
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   M_COMPOSITION     m1-20260923 (default: stage G's M) or m2-20261005 (the next round's M2)
#   FREEZE_FILE       the frozen parameters; must verify (`dline_refit verify-freeze`)
#   DESIGN_SEED       (optional; the campaign's default is per composition)
# m1-20260923 only:
#   BASE_RUN          round 2's root            SUPP_RUN   the boundary supplement's root
#   BOUNDARY_TABLE    round 2's D6' b50 table (scripts/analysis/boundary_b50_table.py)
#   RETAINED_DATASET  round 1's dataset directory (M's three retained mixture cells)
# m2-20261005 only (no probes, no retained cells):
#   RHO_STAR_RUN      the sealed M root whose measured rho* M2 reuses (<root>/<model>/boundary)
#   LEDGER_ROOTS      (optional) space-separated roots whose ledgers M2's ids / seeds must not
#                     appear in; RHO_STAR_RUN's parent is always scanned
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
M_COMPOSITION="${M_COMPOSITION:-m1-20260923}"
calib_require_env FREEZE_FILE
case "$M_COMPOSITION" in
  m1-20260923)
    for v in BASE_RUN SUPP_RUN BOUNDARY_TABLE RETAINED_DATASET; do calib_require_env "$v"; done
    calib_require_file "$BASE_RUN/$MODEL/design_result.json" "$BASE_RUN/$MODEL/run_manifest.json" \
                       "$SUPP_RUN/$MODEL/boundary/${MODEL}_S3.json" "$BOUNDARY_TABLE" \
                       "$RETAINED_DATASET/windows.csv" "$RETAINED_DATASET/cells.csv"
    COMPOSITION_ARGS=(
      --base-run "$BASE_RUN"
      --boundary-supplement-run "$SUPP_RUN"
      --boundary-table "$BOUNDARY_TABLE"
      --retained-dataset "$RETAINED_DATASET")
    ;;
  m2-20261005)
    calib_require_env RHO_STAR_RUN
    calib_require_file "$RHO_STAR_RUN/$MODEL/M_manifest.json" "$RHO_STAR_RUN/$MODEL/M_SHA256SUMS"
    COMPOSITION_ARGS=(--composition "$M_COMPOSITION" --rho-star-run "$RHO_STAR_RUN")
    for r in ${LEDGER_ROOTS:-}; do COMPOSITION_ARGS+=(--ledger-root "$r"); done
    ;;
  *) calib_die "unknown M_COMPOSITION '$M_COMPOSITION' (m1-20260923 | m2-20261005)" ;;
esac
# D22: the parameters are frozen before M exists. A dry run only warns.
if [[ -f "$FREEZE_FILE" ]]; then
  ( calib_enter_deploy && python3 -m scripts.dline_refit verify-freeze --freeze-file "$FREEZE_FILE" ) \
    || calib_die "the freeze file $FREEZE_FILE does not verify"
elif [[ "$DRY_RUN" == 1 ]]; then
  echo "WARNING: no freeze file at $FREEZE_FILE; the real run would refuse to start (D22)" >&2
else
  calib_die "no freeze file at $FREEZE_FILE - M is collected only after 'dline_refit freeze' (D22)"
fi
calib_launch M \
  --acceptance-set \
  "${COMPOSITION_ARGS[@]}" \
  --freeze-file "$FREEZE_FILE" \
  ${DESIGN_SEED:+--design-seed "$DESIGN_SEED"}
