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
#                       the marker exists. Fail closed: while the marker exists, an entry point
#                       refuses unless this AND TRE_FIT_CPUS are set (a site env file outside the
#                       repo sets the site's values) and the host is not denied.
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

# While the exclusive-window marker exists: exit 3 unless TRE_EXCLUSIVE_WINDOW_DENY_HOSTS and
# TRE_FIT_CPUS are set (then PY pins the tools) and this host is not denied.
fit_marker_check() {
  local mk="${TRE_EXCLUSIVE_WINDOW_FILE:-${TRE_FIT_DATA_ROOT:+$TRE_FIT_DATA_ROOT/TRE_EXCLUSIVE_WINDOW}}" host h deny=()
  local names="${TRE_EXCLUSIVE_WINDOW_DENY_HOSTS:-}"
  [[ -n "$mk" && -e "$mk" ]] || return 0
  host="$(hostname)"
  IFS=, read -ra deny <<< "$names"
  if [[ -z "${names//[, ]/}" || -z "${TRE_FIT_CPUS// /}" ]]; then
    echo "exclusive window ($mk): refusing - set TRE_EXCLUSIVE_WINDOW_DENY_HOSTS and TRE_FIT_CPUS (the site env file)" >&2
    exit 3
  fi
  for h in "${deny[@]}"; do
    h="${h// /}"
    if [[ -n "$h" && "$host" == *"$h"* ]]; then
      echo "exclusive window ($mk): not on $host" >&2
      exit 3
    fi
  done
}

# fit_label_registry <dir>: the --label-registry of the dataset builds. LABEL_REGISTRY if set;
# else tre/deploy/registry.yaml of commit LABEL_REGISTRY_COMMIT (default e891d29b = the freeze
# build tree the 10-30 / 11-03 builds read) written to <dir>/label-registry-<commit>.yaml. A file
# already there is used only if its sha256 equals the commit's (else: refuse).
fit_label_registry() {
  if [[ -n "${LABEL_REGISTRY:-}" ]]; then echo "$LABEL_REGISTRY"; return 0; fi
  local c="${LABEL_REGISTRY_COMMIT:-e891d29b}" f want have
  f="$1/label-registry-$c.yaml"
  mkdir -p "$1"
  git -C "$TRE" show "$c:tre/deploy/registry.yaml" > "$f.tmp.$$" \
    || { rm -f "$f.tmp.$$"; echo "FATAL: cannot read tre/deploy/registry.yaml at $c from $TRE (set LABEL_REGISTRY)" >&2; return 1; }
  want="$(sha256sum < "$f.tmp.$$" | cut -d' ' -f1)"
  if [[ -e "$f" ]]; then
    have="$(sha256sum < "$f" | cut -d' ' -f1)"
    rm -f "$f.tmp.$$"
    [[ "$have" == "$want" ]] \
      || { echo "FATAL: $f (sha256 $have) is not tre/deploy/registry.yaml at $c ($want)" >&2; return 1; }
  else
    mv "$f.tmp.$$" "$f"
  fi
  echo "$f"
}
