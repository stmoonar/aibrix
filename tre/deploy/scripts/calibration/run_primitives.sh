#!/usr/bin/env bash
# Round 1 of a fresh calibration (no priors): the first round's primitives (steps /
# boundary / ramp / bursts) on the committed capacity surface (replayer INDEX.json), one
# model per run. Its dataset feeds the priors of round 2 (analysis/calibration_priors.py),
# the H2 training pool and M's three retained mixture cells.
#
#   run_primitives.sh <model> [--dry-run] [-- <extra campaign args>]
#
# Environment (no site defaults on purpose; see lib.sh for the cluster-name overrides):
#   OUT_ROOT                      new, empty root; the run writes $OUT_ROOT/<model>
#   TRE_CALIBRATION_GATEWAY_URL   http://<gateway>/v1/chat/completions
#   TRE_EXCLUSIVE_WINDOW_FILE     the exclusive-window marker (must exist for a real run)
#   DESIGN_SEED                   (optional) recorded in the run manifest
#   CALIB_TASKSET                 (optional) CPU list the load driver is pinned to
#
# Launch detached, one per model (the three may run in parallel):
#   nohup bash run_primitives.sh dsqwen-14b > $OUT_ROOT/dsqwen-14b.log 2>&1 &
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_launch primitives --design primitives ${DESIGN_SEED:+--design-seed "$DESIGN_SEED"}
