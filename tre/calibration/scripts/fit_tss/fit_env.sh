# shellcheck shell=bash
# Shared environment of the fit_tss shell entry points (sourced, never executed). CPU only.
# Nothing site-specific is written here: every location comes from the environment.
#
#   TRE_FIT_DATA_ROOT   directory that holds the calibration data roots (calib_20261003/,
#                       calib_supp_20261103/, analysis-tss-params-20261030/, ...); read by
#                       supp_lib.py too. Required by the entry points that name a data root.
#   TRE                 the tre/ tree whose scripts.calibration_dataset builds datasets
#                       (default: this repo's tre/). Its HEAD is recorded in every manifest.
#   TRE_FIT_PYLIB       extra PYTHONPATH entries put first, e.g. a site-packages copy with the
#                       pinned numpy / pandas (README.md, requirements.txt) [empty]
#   TRE_FIT_NICE        nice level [19]
#   TRE_FIT_CPUS        CPU list for taskset -c [empty = no pinning]
#   TRE_EXCLUSIVE_WINDOW_FILE        exclusive-window marker [$TRE_FIT_DATA_ROOT/TRE_EXCLUSIVE_WINDOW]
#   TRE_EXCLUSIVE_WINDOW_DENY_HOSTS  comma list of hostname substrings that refuse to run while
#                       the marker exists [empty]
#   GIT_CONFIG_GLOBAL   passed through: a host whose git refuses the repo ("dubious ownership",
#                       e.g. an NFS checkout owned by another uid) needs a gitconfig with
#                       [safe] directory = <repo>, or the manifests record code_commit null.
FIT_TSS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TRE:=$(cd "$FIT_TSS_DIR/../../.." && pwd)}"
: "${TRE_FIT_NICE:=19}"
: "${TRE_FIT_CPUS:=}"
: "${TRE_FIT_PYLIB:=}"
PY=(nice -n "$TRE_FIT_NICE")
[[ -z "$TRE_FIT_CPUS" ]] || PY+=(taskset -c "$TRE_FIT_CPUS")
PY+=(python3 -B)
[[ -z "$TRE_FIT_PYLIB" ]] || export PYTHONPATH="$TRE_FIT_PYLIB${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

fit_require_root() {
  [[ -n "${TRE_FIT_DATA_ROOT:-}" ]] || { echo "FATAL: set TRE_FIT_DATA_ROOT (see fit_env.sh)" >&2; exit 2; }
}

# Exit 3 on a denied host while the exclusive-window marker exists.
fit_marker_check() {
  local mk="${TRE_EXCLUSIVE_WINDOW_FILE:-${TRE_FIT_DATA_ROOT:+$TRE_FIT_DATA_ROOT/TRE_EXCLUSIVE_WINDOW}}" host h deny=()
  [[ -n "$mk" && -e "$mk" ]] || return 0
  host="$(hostname)"
  IFS=, read -ra deny <<< "${TRE_EXCLUSIVE_WINDOW_DENY_HOSTS:-}"
  for h in ${deny[@]+"${deny[@]}"}; do
    if [[ -n "$h" && "$host" == *"$h"* ]]; then
      echo "exclusive window ($mk): not on $host" >&2
      exit 3
    fi
  done
}

# fit_label_registry <dir>: the --label-registry of the dataset builds. LABEL_REGISTRY if set;
# else tre/deploy/registry.yaml of commit LABEL_REGISTRY_COMMIT (default e891d29b = the freeze
# build tree the 10-30 / 11-03 builds read) written to <dir>/label-registry-<commit>.yaml.
fit_label_registry() {
  if [[ -n "${LABEL_REGISTRY:-}" ]]; then echo "$LABEL_REGISTRY"; return 0; fi
  local c="${LABEL_REGISTRY_COMMIT:-e891d29b}" f="$1/label-registry-${LABEL_REGISTRY_COMMIT:-e891d29b}.yaml"
  mkdir -p "$1"
  if [[ ! -s "$f" ]]; then
    git -C "$TRE" show "$c:tre/deploy/registry.yaml" > "$f.tmp" && mv "$f.tmp" "$f" \
      || { rm -f "$f.tmp"; echo "FATAL: cannot read tre/deploy/registry.yaml at $c from $TRE (set LABEL_REGISTRY)" >&2; return 1; }
  fi
  echo "$f"
}
