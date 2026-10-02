#!/usr/bin/env bash
# Training supplement (stage 3): constant-load hold cells placed from round 2's rho* and
# the boundary supplement. The 2026-10-03 round runs only the P1 deep-overload plan
# (S2/S3/T8/S4/S5 x {1.5, 2.0, 3.0} rho*, 150 s): select it with TRAINING_PLAN.
#
#   run_stage3.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Exit codes: 0 = every cell driven and the plan's checks passed; 3 = finished but a check
# failed (design_result.json says which); 1 = stopped. Then refit offline and freeze.
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   BASE_RUN        round 2's root (read only)
#   SUPP_RUN        the boundary supplement's root (read only)
#   TRAINING_PLAN   value of --training-plan (this round: p1-deep-overload)
#   DESIGN_SEED     (optional)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_require_env BASE_RUN
calib_require_env SUPP_RUN
calib_require_env TRAINING_PLAN
calib_require_file "$BASE_RUN/$MODEL/design_result.json" "$BASE_RUN/$MODEL/run_manifest.json" \
                   "$SUPP_RUN/$MODEL/boundary/${MODEL}_S3.json"
calib_launch stage3 \
  --training-supplement \
  --training-plan "$TRAINING_PLAN" \
  --base-run "$BASE_RUN" \
  --boundary-supplement-run "$SUPP_RUN" \
  ${DESIGN_SEED:+--design-seed "$DESIGN_SEED"}
