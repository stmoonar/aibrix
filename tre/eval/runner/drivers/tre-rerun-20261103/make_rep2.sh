#!/usr/bin/env bash
# [repo] Repo copy of the driver as it was run: only paths changed (CAMPAIGN_DATA_ROOT = data root, CAMPAIGN_WORK_DIR = log / STOP / runner-stdout dir,
# [repo] TRE_REPO = the tre/ tree; spec, guards, registry snapshot and trace sha256 list come from the repo). See tre/eval/runner/drivers/README.md.
# Make rep 2 of the TRE rerun from rep 1: same file, only name / round / results_root and the "# REP:" line differ.
# Run it again after every edit of rep 1 (e.g. after filling registry_expect.registry_id); the driver refuses a rep 2
# that drifted from rep 1.   bash make_rep2.sh        (writes campaign-tre-rerun-r3e-rep2-20261103.yaml next to it)
set -euo pipefail
C="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../campaigns" && pwd)"   # [repo] the specs live in tre/eval/runner/campaigns
R1=campaign-tre-rerun-r3e-20261103
R2=campaign-tre-rerun-r3e-rep2-20261103
sed -e "s|^name: $R1\$|name: $R2|" \
    -e "s|^round: $R1\$|round: $R2|" \
    -e "s|^results_root: \${CAMPAIGN_DATA_ROOT}/$R1-run\$|results_root: \${CAMPAIGN_DATA_ROOT}/$R2-run|" \
    -e "s|^# REP: 1 of 2 .*|# REP: 2 of 2 (made by make_rep2.sh from $R1.yaml: edit rep 1, then rerun make_rep2.sh)|" \
    "$C/$R1.yaml" > "$C/.$R2.yaml.tmp"
n=$(diff "$C/$R1.yaml" "$C/.$R2.yaml.tmp" | grep -c '^>' || true)
[ "$n" = 4 ] || { rm -f "$C/.$R2.yaml.tmp"; echo "make_rep2: expected 4 changed lines, got $n (rep 1 layout changed?)" >&2; exit 1; }
mv "$C/.$R2.yaml.tmp" "$C/$R2.yaml"
echo "make_rep2: wrote $C/$R2.yaml (4 lines differ from rep 1: name, round, results_root, # REP)"
