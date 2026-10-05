#!/usr/bin/env bash
# Round 2: the ladder design (prior-guided boundary search + hold ladders on the 7
# training shapes), one model per run. Replaces the host-side run_calibration.sh.
#
#   run_ladder.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   PRIORS_DIR      rho_priors.json + regime_groups.json, computed from round 1 and frozen
#                   (sha256 in the preregistration) before this round collects anything
#   DESIGN_SEED     seed of every order and per-cell seed (preregistered)
#   PREREG_MD       (optional) the preregistration document whose commit the manifest records
#   CALIB_TASKSET   (optional)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_require_env PRIORS_DIR
calib_require_env DESIGN_SEED
calib_require_file "$PRIORS_DIR/rho_priors.json" "$PRIORS_DIR/regime_groups.json"
calib_launch ladder \
  --rho-priors "$PRIORS_DIR/rho_priors.json" \
  --regime-groups "$PRIORS_DIR/regime_groups.json" \
  --design-seed "$DESIGN_SEED" \
  ${PREREG_MD:+--preregistration "$PREREG_MD"}
