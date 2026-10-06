#!/usr/bin/env bash
# Generates the v2 trace set (docs: trace-design-v2.md, round 2) from a clean checkout; CPU only, nice 19.
# Usage: generate_traces_v2.sh <tre dir> <azure 2024 csv dir> <out root> <v1 traces_v9 dir> [spec ...]
# (Real-*: the v1 traces_v9 per-model files, named by sha256 in the specs; no spec args = all specs)
# Writes <out root>/<trace>/seed<k>/{design.json,manifest.json,traces_tre.effective.json},
# <out root>/by-name/<trace>_s<k> links (runner ICSE root), audit.{md,json}.
set -euo pipefail
TRE="$1"; AZ="$2"; OUT="$3"; V1="$4"; shift 4
cd "$TRE/replayer"
export PYTHONPATH="$PWD:$TRE/loadgen_v1"
test -z "$(git status --porcelain -- . ../loadgen_v1/configs/traces_v2)" || { echo "dirty tree; commit first" >&2; exit 2; }
G="nice -n 19 python3 -m tre_replayer.tracegen"
CSV="--azure-csv conv2024=$AZ/AzureLLMInferenceTrace_conv_1week.csv --azure-csv code2024=$AZ/AzureLLMInferenceTrace_code_1week.csv --source v1_traces_v9=$V1"
SPECS=("$@"); [ ${#SPECS[@]} -gt 0 ] || SPECS=(tre_replayer/tracegen/specs/*.json)
for spec in "${SPECS[@]}"; do $G generate --spec "$spec" --out-root "$OUT" $CSV; done
RUNS=(); for spec in "${SPECS[@]}"; do t=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['trace'])" "$spec"); RUNS+=("$OUT/$t"/seed*); done
$G link-names "$OUT"
$G materialize "${RUNS[@]}" --processes 4
$G audit $(ls -d "$OUT"/*/seed* | grep -v /_) --json "$OUT/audit.json" --md "$OUT/audit.md" > /dev/null
