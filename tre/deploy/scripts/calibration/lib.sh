# shellcheck shell=bash
# Shared guards of the calibration launchers (sourced, never executed).
#
# Every guard is the residue of a burnt run:
#
#   already running   two campaigns drove the same model at once for an hour. Their
#                     load added up and neither dataset means anything. Worse, clearing
#                     the output directory for the new one pulled the floor out from
#                     under the old one.
#   non-empty out-dir refuse rather than clobber. A partial run is void evidence: keep
#                     it, register it, and use a NEW directory.
#   engines idle      otherwise the first cells measure someone else's backlog.
#   redis / envoy     both addresses are resolved here and PROBED here. Neither is
#                     written down: the redis ClusterIP survives pod restarts but a
#                     literal would go stale silently, and the envoy admin endpoint is a
#                     POD IP that changes on every restart. An unreachable sentinel does
#                     not fail loudly -- it reports "not measured" forever, which is how
#                     the overflow counter came to bless every cell it never read.
#   run mode          controller observe (a scale action mid-cell changes the
#                     denominator of everything the cell records) and SM observe (the
#                     supervisor's self-heal / admission may wake or sleep a pod).
#   exclusive window  the marker file says the cluster is ours; parallel sessions check
#                     it before they drive load.
#   --stop-on-failure without it the campaign logs "cell failed" and moves on. A run
#                     where all 24 cells failed still printed "campaign complete" and
#                     left zero CSVs.
#
# Nothing site-specific is written here: every name has an environment override.
#
#   CALIB_REPO              the tre/ directory (default: derived from this file)
#   TRE_NS / MODEL_NS       tre-v2 / default
#   REDIS_SVC / REDIS_DEPLOY  tre-v2-redis / tre-v2-redis
#   ENVOY_NS / ENVOY_POD_PATTERN / ENVOY_ADMIN_PORT
#                           envoy-gateway-system / tre-v2-tre-aibrix-eg / 19001
#   ENGINE_CONTAINER / ENGINE_PORT   vllm-openai / 8000
#   TRE_EXCLUSIVE_WINDOW_FILE   marker file; required to exist for a real run
#                           (set CALIB_REQUIRE_EXCLUSIVE_WINDOW=0 to skip, e.g. a dry run)
#   CALIB_REQUIRE_SM_OBSERVE    1 (default) = refuse unless tre:v2:sm:actuation is observe
#   CALIB_TASKSET           CPU list for the load driver (taskset -c); empty = no pinning
#   KUBECTL                 kubectl
set -o pipefail

CALIB_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CALIB_REPO:=$(cd "$CALIB_LIB_DIR/../../.." && pwd)}"
: "${TRE_NS:=tre-v2}"
: "${MODEL_NS:=default}"
: "${REDIS_SVC:=tre-v2-redis}"
: "${REDIS_DEPLOY:=tre-v2-redis}"
: "${ENVOY_NS:=envoy-gateway-system}"
: "${ENVOY_POD_PATTERN:=tre-v2-tre-aibrix-eg}"
: "${ENVOY_ADMIN_PORT:=19001}"
: "${ENGINE_CONTAINER:=vllm-openai}"
: "${ENGINE_PORT:=8000}"
: "${KUBECTL:=kubectl}"
: "${CALIB_REQUIRE_SM_OBSERVE:=1}"
: "${CALIB_REQUIRE_EXCLUSIVE_WINDOW:=1}"
: "${CALIB_TASKSET:=}"

CALIB_MODELS_RE='^(dsqwen-7b|dsllama-8b|dsqwen-14b)$'

calib_die() { echo "FATAL: $*" >&2; exit 1; }

calib_require_model() {
  [[ "$1" =~ $CALIB_MODELS_RE ]] || { echo "FATAL: unknown model '$1'" >&2; exit 2; }
}

# A required input path given through the environment (no site default on purpose).
calib_require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || calib_die "set $name (see the header of $(basename "$0"))"
}

calib_require_file() {
  local f
  for f in "$@"; do [[ -s "$f" ]] || calib_die "input $f is missing or empty"; done
}

# One driver per model. Anchored on the command line's start (`python3 -m scripts.`, or
# an absolute python3, which is how the campaign starts its r3_grid children): a bare
# module-name match makes pgrep -f match any shell whose own command line mentions it,
# including the one launching this script. taskset exec()s, so a pinned driver still
# shows as `python3 -m ...`.
calib_require_no_driver() {
  local model="$1" running
  running="$(pgrep -af '^(\S*/)?python3 -m scripts\.(calibration_campaign|r3_grid)' \
             | grep -- "$model" | grep -v "^$$ " || true)"
  [[ -z "$running" ]] || { echo "FATAL: a load driver is already running for $model" >&2
                           echo "$running" >&2; exit 1; }
}

calib_require_empty_dir() {
  local d="$1"
  if [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
    calib_die "$d is not empty (a partial run is void evidence: keep it, register it, use a NEW directory)"
  fi
}

calib_require_exclusive_window() {
  [[ "$CALIB_REQUIRE_EXCLUSIVE_WINDOW" == 1 ]] || return 0
  calib_require_env TRE_EXCLUSIVE_WINDOW_FILE
  [[ -f "$TRE_EXCLUSIVE_WINDOW_FILE" ]] \
    || calib_die "exclusive-window marker $TRE_EXCLUSIVE_WINDOW_FILE does not exist (create it first; see the RUN plan)"
}

calib_rcli() { "$KUBECTL" -n "$TRE_NS" exec "deploy/$REDIS_DEPLOY" -- redis-cli --raw "$@"; }

# Sets REDIS_IP.
calib_resolve_redis() {
  REDIS_IP="$("$KUBECTL" -n "$TRE_NS" get svc "$REDIS_SVC" -o jsonpath='{.spec.clusterIP}')"
  [[ -n "$REDIS_IP" ]] || calib_die "could not resolve $REDIS_SVC ClusterIP"
  timeout 5 bash -c "printf 'PING\r\n' | nc $REDIS_IP 6379 | grep -q PONG" \
    || calib_die "redis at $REDIS_IP:6379 did not answer PING from this host"
}

# Sets ENVOY_IP and ENVOY_STATS.
calib_resolve_envoy() {
  ENVOY_IP="$("$KUBECTL" -n "$ENVOY_NS" get pods \
              -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}' \
              | awk -v p="$ENVOY_POD_PATTERN" '$0 ~ p {print $2; exit}')"
  [[ -n "$ENVOY_IP" ]] || calib_die "could not find the envoy pod matching $ENVOY_POD_PATTERN"
  ENVOY_STATS="http://$ENVOY_IP:$ENVOY_ADMIN_PORT/stats/prometheus"
  # Not `curl | grep -q`: grep exits on the first match, curl takes SIGPIPE, and under
  # pipefail a healthy endpoint reads as a failure.
  local body
  body="$(curl -sf --max-time 5 "$ENVOY_STATS" || true)"
  case "$body" in
    *upstream_rq_pending_overflow*) ;;
    *) calib_die "$ENVOY_STATS did not serve the overflow counter (the shed sentinel would report 'not measured' for the whole run)" ;;
  esac
}

# Sets RUN_MODE_CONTROLLER / RUN_MODE_SM. A missing key reads as observe (its readers
# fail closed), but we want the key written: refuse when it is missing.
calib_require_run_modes() {
  RUN_MODE_CONTROLLER="$(calib_rcli GET tre:v2:controller:mode | tr -d '\r')"
  RUN_MODE_SM="$(calib_rcli GET tre:v2:sm:actuation | tr -d '\r')"
  [[ "$RUN_MODE_CONTROLLER" == observe ]] \
    || calib_die "controller mode is '$RUN_MODE_CONTROLLER', expected 'observe' (set_run_mode.sh observe observe)"
  if [[ "$CALIB_REQUIRE_SM_OBSERVE" == 1 ]]; then
    [[ "$RUN_MODE_SM" == observe ]] \
      || calib_die "SM actuation is '$RUN_MODE_SM', expected 'observe' (set_run_mode.sh observe observe)"
  fi
}

calib_model_pods() {
  "$KUBECTL" -n "$MODEL_NS" get pods -l "model.aibrix.ai/name=$1" --no-headers \
    -o custom-columns=:.metadata.name
}

# Scoped to the model: a sibling model under calibration is expected to be busy.
calib_require_engines_idle() {
  local model="$1" busy p r
  busy=""
  for p in $(calib_model_pods "$model"); do
    r="$("$KUBECTL" -n "$MODEL_NS" exec "$p" -c "$ENGINE_CONTAINER" -- \
          curl -s --max-time 5 "localhost:$ENGINE_PORT/metrics" 2>/dev/null \
          | awk '/^vllm:num_requests_(running|waiting)/ {s += $2} END {print s + 0}')"
    case "$r" in ''|0|0.0) ;; *) busy+="$p running+waiting=$r"$'\n' ;; esac
  done
  [[ -z "$busy" ]] || { echo "FATAL: engines of $model are not idle:" >&2; echo "$busy" >&2; exit 1; }
}

# Exactly N routable pods of the model (calibration is single-replica per model).
calib_require_routable() {
  local model="$1" want="${2:-1}" n
  n="$("$KUBECTL" -n "$MODEL_NS" get pods -l "model.aibrix.ai/name=$model,tre.aibrix.io/routable=true" \
        --no-headers -o custom-columns=:.metadata.name | grep -c . || true)"
  [[ "$n" == "$want" ]] || calib_die "$n routable $model pods, need exactly $want"
}

calib_enter_deploy() {
  cd "$CALIB_REPO/deploy" || calib_die "no $CALIB_REPO/deploy"
  export PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer:../ui"
}

# The prefix that runs the load driver: pinned to CALIB_TASKSET when set.
calib_driver_prefix() {
  CALIB_DRIVER=()
  if [[ -n "$CALIB_TASKSET" ]]; then
    command -v taskset >/dev/null || calib_die "CALIB_TASKSET is set but taskset is not installed"
    CALIB_DRIVER=(taskset -c "$CALIB_TASKSET")
  fi
}

# All the run-time guards of a real (or dry) run against the cluster, in one call.
#   calib_preflight <model> <out-dir> <dry_run 0|1>
calib_preflight() {
  local model="$1" out="$2" dry="$3"
  calib_require_no_driver "$model"
  if [[ "$dry" == 1 ]]; then
    CALIB_REQUIRE_EXCLUSIVE_WINDOW=0
  else
    calib_require_empty_dir "$out"
  fi
  calib_require_exclusive_window
  calib_resolve_redis
  calib_resolve_envoy
  calib_require_run_modes
  calib_require_engines_idle "$model"
  calib_driver_prefix
  echo "model=$model redis=$REDIS_IP envoy=$ENVOY_IP controller=$RUN_MODE_CONTROLLER sm=$RUN_MODE_SM" \
       "taskset=${CALIB_TASKSET:-none} out=$out dry_run=$dry repo=$CALIB_REPO" \
       "git=$(git -C "$CALIB_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

# --- the launcher skeleton ----------------------------------------------------------
# calib_parse_args "$@"  ->  MODEL, DRY_RUN (0|1), EXTRA_ARGS (everything after `--`,
# passed to the campaign verbatim, e.g. `-- --training-plan p1-deep-overload`).
calib_parse_args() {
  MODEL="${1:-}"; shift || true
  [[ -n "$MODEL" ]] || { echo "usage: $(basename "$0") <model> [--dry-run] [-- <campaign args>]" >&2; exit 2; }
  calib_require_model "$MODEL"
  DRY_RUN=0; EXTRA_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift ;;
      --) shift; EXTRA_ARGS=("$@"); break ;;
      *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
  done
}

# calib_launch <stage-name> <campaign args...>
# Real run: OUT = $OUT_ROOT/$MODEL (must be empty). Dry run: a scratch directory under
# ${CALIB_DRYRUN_ROOT:-/tmp}, cleared so it can be repeated. The campaign creates OUT/raw.
calib_launch() {
  local stage="$1"; shift
  calib_require_env OUT_ROOT
  calib_require_env TRE_CALIBRATION_GATEWAY_URL
  local out dry_flag=()
  if [[ "$DRY_RUN" == 1 ]]; then
    out="${CALIB_DRYRUN_ROOT:-/tmp}/calibration_${stage}_dryrun/$MODEL"
    rm -rf "$out"
    dry_flag=(--dry-run)
  else
    out="$OUT_ROOT/$MODEL"
  fi
  calib_preflight "$MODEL" "$out" "$DRY_RUN"
  export TRE_CALIBRATION_GATEWAY_URL
  calib_enter_deploy
  echo "stage=$stage gateway=$TRE_CALIBRATION_GATEWAY_URL registry=${CALIB_REGISTRY:-$CALIB_REPO/deploy/registry.yaml}"
  exec ${CALIB_DRIVER[@]+"${CALIB_DRIVER[@]}"} python3 -m scripts.calibration_campaign \
    --models "$MODEL" \
    --out-dir "$out" \
    --raw-dir "$out/raw" \
    --redis-url "redis://$REDIS_IP:6379/0" \
    --envoy-stats-url "$ENVOY_STATS" \
    --envoy-cluster-filter "$MODEL" \
    --registry "${CALIB_REGISTRY:-$CALIB_REPO/deploy/registry.yaml}" \
    --model-namespace "$MODEL_NS" \
    --controller-namespace "$TRE_NS" \
    --stop-on-failure \
    "$@" \
    ${dry_flag[@]+"${dry_flag[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}
