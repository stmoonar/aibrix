#!/usr/bin/env bash
# Generates the v2 trace set (docs: trace-design-v2.md, round 2) from a clean checkout; CPU only, nice 19.
# Usage: generate_traces_v2.sh <tre dir> <azure 2024 csv dir> <out root>
# Writes <out root>/<trace>/seed<k>/{design.json,manifest.json,traces_tre.effective.json},
# <out root>/by-name/<trace>_s<k> links (runner ICSE root), audit.{md,json}.
set -euo pipefail
TRE="$1"; AZ="$2"; OUT="$3"
cd "$TRE/replayer"
export PYTHONPATH="$PWD:$TRE/loadgen_v1"
test -z "$(git status --porcelain -- . ../loadgen_v1/configs/traces_v2)" || { echo "dirty tree; commit first" >&2; exit 2; }
G="nice -n 19 python3 -m tre_replayer.tracegen"
CSV="--azure-csv conv2024=$AZ/AzureLLMInferenceTrace_conv_1week.csv --azure-csv code2024=$AZ/AzureLLMInferenceTrace_code_1week.csv"
for spec in tre_replayer/tracegen/specs/*.json; do $G generate --spec "$spec" --out-root "$OUT" $CSV; done
$G link-names "$OUT"
$G audit "$OUT"/*/seed* --json "$OUT/audit.json" --md "$OUT/audit.md" > /dev/null
$G materialize "$OUT"/*/seed* --processes 4
