#!/usr/bin/env bash
# One-click E1 campaign: traces in order, each trace's arms back to back (order rotated per trace
# and recorded), pre-checks + reset + validity + reports after every arm. See campaign.py.
#
#   run_campaign.sh <campaign.yaml> --dry-run [--probe]   plan, ETAs, input checks (no cluster access;
#                                                         --probe adds the read-only pre-checks)
#   nohup run_campaign.sh <campaign.yaml> > <log> 2>&1 &  run (CHANGES CLUSTER STATE; needs the marker)
#   run_campaign.sh <campaign.yaml> [--retry-failed]      resume: done arms are skipped
#   run_campaign.sh <campaign.yaml> --status              status.json summary
#   touch <results_root>/STOP                             clean stop after the current arm
#
# Settings: the campaign file's env: block (wins), then runner.env / the caller's environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 1 ] || { sed -n 2,13p "$0"; exit 2; }
CAMPAIGN="$1"; shift
export TRE_REPO="${TRE_REPO:-$(cd "$HERE/../.." && pwd)}"
EXPORTS=$(python3 "$HERE/campaign.py" env "$CAMPAIGN")
eval "$EXPORTS"
set -a
# shellcheck source=lib_env.sh
. "$HERE/lib_env.sh"
set +a
eval "$EXPORTS"   # the campaign's values win over runner.env
if [ "${1:-}" = --status ]; then exec python3 "$HERE/campaign.py" status "$CAMPAIGN"; fi
exec python3 "$HERE/campaign.py" run "$CAMPAIGN" "$@"
