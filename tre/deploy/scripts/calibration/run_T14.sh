#!/usr/bin/env bash
# T14 (dsqwen-14b only): the held-out / extrapolation test set, hold cells placed by the
# preregistered capacity prior. Evaluated once per parameter set, by the preregistered rule.
#
#   run_T14.sh dsqwen-14b [--dry-run] [-- <extra campaign args>]
#
# Guards on top of lib.sh's: exactly ONE routable dsqwen-14b pod; an EnvoyPatchPolicy in the
# TRE namespace (the least-gpu-cache ext_proc route); the preregistration verifies against
# its .sha256 sidecar; every parameter file verifies. The campaign itself checks the
# preregistration's bindings (capacity prior sha, seeds, factors, hold, shapes, api, gateway
# url, parameter-set shas) and isolation roots (t14.forbidden_roots + --forbidden-root).
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   PREREG_JSON          the T14 preregistration (+ PREREG_JSON.sha256)
#   CAPACITY_PRIOR       the capacity prior it binds (calibration_t14 capacity-prior)
#   FREEZE_FILE          the frozen parameters
#   REFIT_PARAMS_FILE    the second parameter set the preregistration names (the campaign
#                        requires one; with a single frozen set, name the freeze file again)
#   DESIGN_SEED          must equal the preregistration's t14.design_seed
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
[[ "$MODEL" == dsqwen-14b ]] || calib_die "T14 is dsqwen-14b only"
for v in PREREG_JSON CAPACITY_PRIOR FREEZE_FILE REFIT_PARAMS_FILE DESIGN_SEED; do calib_require_env "$v"; done
calib_require_file "$PREREG_JSON" "$PREREG_JSON.sha256" "$CAPACITY_PRIOR" "$FREEZE_FILE" "$REFIT_PARAMS_FILE"
( cd "$(dirname "$PREREG_JSON")" && sha256sum -c "$(basename "$PREREG_JSON").sha256" ) \
  || calib_die "$PREREG_JSON does not match its sha256 sidecar"
for f in "$FREEZE_FILE" "$REFIT_PARAMS_FILE"; do
  ( calib_enter_deploy && python3 -m scripts.dline_refit verify-freeze --freeze-file "$f" ) \
    || calib_die "$f does not verify"
done
calib_require_routable "$MODEL" 1
"$KUBECTL" -n "$TRE_NS" get envoypatchpolicy --no-headers 2>/dev/null | grep -q . \
  || calib_die "no EnvoyPatchPolicy in $TRE_NS - the least-gpu-cache ext_proc route is not deployed"
calib_launch T14 \
  --t14-set \
  --api chat \
  --routing-strategy least-gpu-cache \
  --capacity-prior-file "$CAPACITY_PRIOR" \
  --freeze-file "$FREEZE_FILE" \
  --refit-params-file "$REFIT_PARAMS_FILE" \
  --preregistration-json "$PREREG_JSON" \
  --design-seed "$DESIGN_SEED"
