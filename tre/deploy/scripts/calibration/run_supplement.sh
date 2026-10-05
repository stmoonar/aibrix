#!/usr/bin/env bash
# Boundary supplement (D19): S3 re-probed above round 2's rho* on a per-model grid, then
# one 300 s smoke hold at the located rho* (violating windows must be 20-70 %). M, the
# stage-3 supplement and the T14 capacity prior read its S3 anchor.
#
#   run_supplement.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Exit codes: 0 = every rho* measured and every smoke inside the band; 3 = finished but a
# rho* is a bound or a smoke is outside the band (stop for the user); 1 = stopped.
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   BASE_RUN        round 2's root (read only)
#   REPROBE_GRID    multiples of rho*_run2 for S3, e.g. 1.3,1.45,1.6,1.8. Set it from the
#                   NEW round 2: the 2026-09-23 grids (7b 1.3-1.8, 8b 1.5-2.1, 14b 1.8-2.8)
#                   came from the old engine and are not reusable as they are.
#   DESIGN_SEED     (optional)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_require_env BASE_RUN
calib_require_env REPROBE_GRID
calib_require_file "$BASE_RUN/$MODEL/design_result.json" "$BASE_RUN/$MODEL/run_manifest.json" \
                   "$BASE_RUN/$MODEL/boundary/${MODEL}_S3.json"
calib_launch supplement \
  --reprobe-shapes "$MODEL:S3" \
  --reprobe-base "$BASE_RUN" \
  --reprobe-grid "$MODEL:$REPROBE_GRID" \
  --smoke-at-rho-star \
  ${DESIGN_SEED:+--design-seed "$DESIGN_SEED"}
