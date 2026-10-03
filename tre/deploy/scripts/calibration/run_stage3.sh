#!/usr/bin/env bash
# Training supplement (stage 3): constant-load hold cells placed from round 2's rho*. The
# 2026-10-03 round runs only the P1 deep-overload plan (--training-plan p1-deep-overload:
# S2/S3/T8/S4/S5 x {1.5, 2.0, 3.0} rho*_run2, 150 s, plus 3 S2 drift sentinels; each cell
# waits for the engine to drain, up to 300 s). legacy-20260923 is the 09-23 plan.
#
#   run_stage3.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Exit codes: 0 = every cell driven and the plan's checks passed; 3 = finished but a check
# failed (design_result.json says which); 1 = stopped. Then refit offline and freeze.
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   BASE_RUN        round 2's root (read only)
#   TRAINING_PLAN   p1-deep-overload | legacy-20260923
#   SUPP_RUN        the boundary supplement's root: required by legacy-20260923, optional
#                   for p1 (when given, its load path is checked and recorded)
#   DESIGN_SEED     (optional)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_require_env BASE_RUN
calib_require_env TRAINING_PLAN
calib_require_file "$BASE_RUN/$MODEL/design_result.json" "$BASE_RUN/$MODEL/run_manifest.json"
[[ -z "${SUPP_RUN:-}" ]] || calib_require_file "$SUPP_RUN/$MODEL/boundary/${MODEL}_S3.json"
calib_launch stage3 \
  --training-supplement \
  --training-plan "$TRAINING_PLAN" \
  --base-run "$BASE_RUN" \
  ${SUPP_RUN:+--boundary-supplement-run "$SUPP_RUN"} \
  ${DESIGN_SEED:+--design-seed "$DESIGN_SEED"}
