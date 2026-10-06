#!/usr/bin/env bash
# Offline harness scenarios for run_arm_pilot.sh / run_pilot.sh (fake kubectl, curl, redis; no cluster).
# Usage: SRC=<runner dir> bash scen.sh      (Linux; GNU date / sed). Prints per scenario rc and
# the calls of interest; compare by eye (no assertion). 2026-10-07 additions at the end: no
# controller env switch is ever set, every arm writes runner_sha / trace_manifest.json /
# clock_offsets.json, TRE arms dump gateway_events/ when the stream is on.
H=${H:-/tmp/bl-harness}; HERE=$(cd "$(dirname "$0")" && pwd); export SRC=${SRC:-$(dirname "$HERE")} H
run() { env PATH=$H/bin:$PATH PYTHONPATH=$H/pyfake TRE_DIR=$H/tre LOADGEN_TRE_DIR=$H/tre TOOLS=$H/tools PILOT_ROOT=$H/pilot REQUIRE_MARKER=0 MARKER=/nonexistent ICSE=$H/icse IDLE_S=0 POST_S=0 CLOCK_NODES="fake-node=local" RUNNER_ENV=/nonexistent "$@"; }
reset() { bash "$HERE/make_harness.sh" >/dev/null; cd /tmp; }
echo "== tre GW_PARITY=1"; reset; run nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep -E "TRE_BL_REQ_EVENTS|^loadgen|^arm " $H/state/calls.log | grep -v "get deploy" | sed "s/--config.*--max-retries 0//" | cut -c1-150; ls $H/state/events 2>/dev/null || echo "events file gone (restored UNSET)"
python3 -c "import json;s=json.load(open(\"$H/pilot/Alt/tre/score.json\"));print(\"valid\" in s)"
echo "== tre GW_PARITY=0"; reset; run env GW_PARITY=0 nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep -E "TRE_BL_REQ_EVENTS|^loadgen|^arm " $H/state/calls.log | grep -v "get deploy" | sed "s/--config.*--max-retries 0//" | cut -c1-150
echo "== apa GW_PARITY=0, events pre-set true (left alone)"; reset; echo -n true > $H/state/events; run env GW_PARITY=0 nice bash $H/tools/run_arm_pilot.sh apa Alt >/dev/null 2>&1; echo rc=$?; grep -c "set env deploy/tre-gateway-plugins" $H/state/calls.log; cat $H/state/events; echo
echo "== preserve, trace not in pod -> exit 4, trap"; reset; run nice bash $H/tools/run_arm_pilot.sh preserve Alt >/dev/null 2>&1; echo rc=$?; grep -E "TRAP|ERROR" $H/pilot/Alt/preserve/runner.log; echo "scaler=$(cat $H/state/scaler_replicas) events=$(cat $H/state/events 2>/dev/null || echo UNSET)"
echo "== preserve, trace tail mismatch -> exit 2 before cluster"; reset; sed -i "s#/Alt/#/Other/#" $H/state/policy-preserve.yaml; run nice bash $H/tools/run_arm_pilot.sh preserve Alt >/dev/null 2>&1; echo rc=$?; grep -c "set env" $H/state/calls.log; grep ERROR $H/pilot/Alt/preserve/runner.log | cut -c1-200
echo "== preserve ok"; reset; touch $H/state/trace_ok; run nice bash $H/tools/run_arm_pilot.sh preserve Alt >/dev/null 2>&1; echo rc=$?; grep "preserve trace in pod" $H/pilot/Alt/preserve/runner.log | cut -c1-120
echo "== tokenscale invalid events"; reset; touch $H/state/invalid; run nice bash $H/tools/run_arm_pilot.sh tokenscale Alt >/dev/null 2>&1; echo rc=$?; python3 -c "import json;s=json.load(open(\"$H/pilot/Alt/tokenscale/score.json\"));print(s[\"valid\"], s[\"invalid_because\"])"
echo "== chiron enable fails -> trap"; reset; touch $H/state/fail_enable; run nice bash $H/tools/run_arm_pilot.sh chiron Alt >/dev/null 2>&1; echo rc=$?; grep -E "TRAP|ERROR" $H/pilot/Alt/chiron/runner.log | cut -c1-150; echo "scaler=$(cat $H/state/scaler_replicas) events=$(cat $H/state/events 2>/dev/null || echo UNSET)"
echo "== chiron placeholder (theta null)"; reset; printf "theta: {dsqwen-7b: null}\n" > $H/state/policy-chiron.yaml; run nice bash $H/tools/run_arm_pilot.sh chiron Alt >/dev/null 2>&1; echo rc=$?; grep ERROR $H/pilot/Alt/chiron/runner.log | cut -c1-160
echo "== chiron frozen differs, BL_CM_APPLY=0 -> refuse"; reset; printf "apiVersion: v1\nkind: ConfigMap\nmetadata: {name: tre-v2-baseline-chiron}\ndata:\n  chiron.yaml: |\n    theta: {dsqwen-7b: 0.5, dsllama-8b: 0.5, dsqwen-14b: 0.5}\n" > $H/cm.yaml; run env BL_CM_FILE=$H/cm.yaml nice bash $H/tools/run_arm_pilot.sh chiron Alt >/dev/null 2>&1; echo rc=$?; grep ERROR $H/pilot/Alt/chiron/runner.log | cut -c1-120
echo "== run_pilot PLAN with baseline arms"; reset; touch $H/state/trace_ok; ln -s $H/tools $H/pilot/tools; run env GAP_S=0 PLAN="Alt:chiron,tokenscale Alt:tre:drift1" nice bash $H/tools/run_pilot.sh 2>&1 | tail -3; ls $H/pilot/Alt
echo "== 2026-10-07: no controller env switch; provenance files; gateway events for tre"; reset; run nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep -c "set env deploy/tre-v2-controller" $H/state/calls.log
for f in runner_sha trace_manifest.json clock_offsets.json arm_start_redis_ms; do [ -s "$H/pilot/Alt/tre/$f" ] && echo "have $f" || echo "MISSING $f"; done
ls $H/pilot/Alt/tre/gateway_events 2>/dev/null | head -3
echo "== 2026-10-07: chiron waits for no open probe (fake redis: none) and starts"; reset; run nice bash $H/tools/run_arm_pilot.sh chiron Alt >/dev/null 2>&1; echo rc=$?; grep -E "decision source|baseline shell enabled" $H/pilot/Alt/chiron/runner.log | cut -c1-120
echo "== 2026-10-07: seeded trace -> Alt_s7 directory"; reset; echo '{"seed": 7}' > $H/icse/Alt/manifest.json; run nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?; ls $H/pilot; python3 -c "import json;print(json.load(open('$H/pilot/Alt_s7/tre/trace_manifest.json'))['seed'])"
echo "== 2026-10-07: apa arm: decision source off (0 APA CRs) before every restore (asserted)"; reset; run nice bash $H/tools/run_arm_pilot.sh apa Alt >/dev/null 2>&1; echo rc=$?
grep -E "^toggle (tre|apa)|^awake_ctl" $H/state/calls.log
if ! grep -q "^awake_ctl restore-ids" $H/state/calls.log; then echo "FAIL: no restore-ids call"
elif grep -q "^awake_ctl restore-ids apa_crs_live=1" $H/state/calls.log; then echo "FAIL: restore-ids ran with APA CRs live"
elif [ "$(grep -c '^awake_ctl restore-ids' $H/state/calls.log)" != 2 ]; then echo "FAIL: want 2 restores (arm start + end)"
else echo "PASS: both restores with 0 APA CRs"; fi
