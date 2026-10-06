# Sourced by the runner scripts: site configuration (runner.env, else runner.env.example).
# Values already in the environment win (every line of the env file is VAR="${VAR:-default}").
RUNNER_DIR="${RUNNER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUNNER_ENV="${RUNNER_ENV:-$RUNNER_DIR/runner.env}"
TRE_REPO="${TRE_REPO:-$(cd "$RUNNER_DIR/../.." && pwd)}"   # the tree the runner runs from (runner.env defaults refer to it)
if [ -f "$RUNNER_ENV" ]; then
  # shellcheck disable=SC1090
  . "$RUNNER_ENV"
else
  echo "[runner] $RUNNER_ENV not found: using the defaults of $RUNNER_DIR/runner.env.example" >&2
  # shellcheck disable=SC1091
  . "$RUNNER_DIR/runner.env.example"
fi
# exported: sampler.py / snap.py / clock_probe.py / components.py read them
export TRE_NS MODEL_NS APA_NS AIBRIX_NS ENVOY_NS MODEL_SELECTOR SIDECAR_CONTAINER ENGINE_PORT MODELS \
  CLOCK_NODES CLOCK_SSH_OPTS SAMPLE_LAYOUT_S SAMPLE_PODS_S SAMPLE_GAUGES_S SAMPLE_APA_S SAMPLE_GPU_TRUTH_S \
  SAMPLE_RESOURCES_S POD_LIST_REFRESH_S BL_NS GW_NS
need_env() {   # need_env VAR ... : refuse when a required setting is empty
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "[runner] $v is not set (runner.env / environment)" >&2; exit 2; }
  done
}
# runner provenance (recorded per arm): this directory's git sha + dirty state
runner_sha() {
  echo "dir=$RUNNER_DIR"
  echo "sha=$(git -C "$RUNNER_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  echo "branch=$(git -C "$RUNNER_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)"
  echo "dirty_files=$(git -C "$RUNNER_DIR" status --porcelain -- . 2>/dev/null | wc -l)"
  echo "env_file=$( [ -f "$RUNNER_ENV" ] && echo "$RUNNER_ENV" || echo "$RUNNER_DIR/runner.env.example")"
}
