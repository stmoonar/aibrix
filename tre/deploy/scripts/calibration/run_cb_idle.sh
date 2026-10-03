#!/usr/bin/env bash
# Idle TTFT(L) capture for the D6' label's c/b (scripts.ttft_idle_capture): one routable
# replica, strictly one request in flight (deterministic gap, 16 output tokens, ignore_eos),
# L in {128 ... 4096} in seeded blocks, fixed labels. Must run BEFORE any labelled
# collection; the fit (scripts.ttft_idle_fit) then writes c/b into deploy/registry.yaml.
#
#   run_cb_idle.sh <model> [--dry-run] [-- <extra ttft_idle_capture args>]
#
# Environment:
#   OUT_ROOT, TRE_CALIBRATION_GATEWAY_URL, TRE_EXCLUSIVE_WINDOW_FILE   as run_primitives.sh
#   CALIB_TASKSET   (optional)
#
# The three models may run in parallel (~10 min each). Then:
#   cd <tre>/deploy && PYTHONPATH=../common:. python3 -m scripts.ttft_idle_fit --root $OUT_ROOT \
#       --models dsqwen-7b,dsllama-8b,dsqwen-14b --min-gap-ms 500 --out $OUT_ROOT/ttft_idle_fit.json --registry-patch
#   ... and with --write-registry <tre>/deploy/registry.yaml once the numbers are accepted.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
calib_parse_args "$@"
calib_require_env OUT_ROOT
calib_require_env TRE_CALIBRATION_GATEWAY_URL
calib_preflight "$MODEL" "$OUT_ROOT/$MODEL" "$DRY_RUN"
calib_require_routable "$MODEL" 1
dry_flag=()
[[ "$DRY_RUN" == 1 ]] && dry_flag=(--dry-run)
calib_enter_deploy
exec ${CALIB_DRIVER[@]+"${CALIB_DRIVER[@]}"} python3 -m scripts.ttft_idle_capture \
  --models "$MODEL" \
  --gateway-url "$TRE_CALIBRATION_GATEWAY_URL" \
  --out-dir "$OUT_ROOT" \
  --redis-url "redis://$REDIS_IP:6379/0" \
  --envoy-stats-url "$ENVOY_STATS" \
  --envoy-cluster-filter "$MODEL" \
  --registry "${CALIB_REGISTRY:-$CALIB_REPO/deploy/registry.yaml}" \
  --model-namespace "$MODEL_NS" \
  --controller-namespace "$TRE_NS" \
  --pod-metrics-port "$ENGINE_PORT" \
  ${dry_flag[@]+"${dry_flag[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
