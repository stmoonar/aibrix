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
echo "== 2026-10-08: ablation arm (ARM_CONTROLLER_ENV) sets the switch, checks the startup line, puts it back to false"; reset
run env ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SAFESCALE=true" nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep "set env deploy/tre-v2-controller" $H/state/calls.log | sed 's/.*set env/set env/'
python3 -c "import json;a=json.load(open('$H/pilot/Alt/tre/ablation_switches.json'));m=json.load(open('$H/pilot/Alt/tre/arm_meta.json'));print(a['match'], a['logged'], m['controller_env'])"
[ "$(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE)" = false ] && [ -s $H/pilot/Alt/tre/probe_gate.json ] && echo "PASS: switch back to false, probe gate recorded" || echo "FAIL: switch=$(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE) gate=$(cat $H/pilot/Alt/tre/probe_gate.json 2>/dev/null)"
echo "== 2026-10-08: unresolved (probing) probe record in Redis -> exit 4 before any controller change"; reset; echo probing > $H/state/probes
run env FAKE_STATE=$H/state ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SAFESCALE=true" nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
N=$(grep -c "set env deploy/tre-v2-controller" $H/state/calls.log); grep -E "ERROR" $H/pilot/Alt/tre/runner.log | cut -c1-140; cat $H/pilot/Alt/tre/probe_gate.json
[ "$N" = 0 ] && echo "PASS: no controller env change" || echo "FAIL: $N controller set env"
echo "== 2026-10-08: only resolved probe records (earlier arm + this arm) -> gate passes; safescale.json keeps this arm's probe only"; reset; echo resolved > $H/state/probes
run env FAKE_STATE=$H/state nice bash $H/tools/run_arm_pilot.sh apa Alt >/dev/null 2>&1; echo rc=$?
cat $H/pilot/Alt/apa/probe_gate.json
python3 -c "
import json; s = json.load(open('$H/pilot/Alt/apa/safescale.json'))
ok = len(s['probes_all']) == 2 and list(s['probes']) != [] and 'dsqwen-7b-1000' not in s['probes']
print('PASS' if ok else 'FAIL', 'probes', sorted(s['probes']), 'all', len(s['probes_all']))"
echo "== 2026-10-08: startup line missing with an explicit switch -> exit 4, trap puts the switch back"; reset; touch $H/state/no_switch_line
run env ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SAFESCALE=true" nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep -E "ERROR|TRAP" $H/pilot/Alt/tre/runner.log | cut -c1-140
[ "$(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE)" = false ] && echo "PASS: switch back to false" || echo "FAIL: switch left $(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE)"
echo "== 2026-10-08: ARM_CONTROLLER_ENV on a non-tre arm / unknown key -> exit 2"; reset
run env ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SAFESCALE=true" nice bash $H/tools/run_arm_pilot.sh apa Alt >/dev/null 2>&1; echo rc=$?
run env ARM_CONTROLLER_ENV="TRE_FOO=true" nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
echo "== 2026-10-08: plain tre arm, switch left true by a crash -> set back to false before the arm"; reset; echo -n true > $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE
run nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep "set env deploy/tre-v2-controller" $H/state/calls.log | sed 's/.*set env/set env/'
python3 -c "import json;a=json.load(open('$H/pilot/Alt/tre/ablation_switches.json'));print(a['match'], a['logged'])"
echo "== 2026-10-08: trap puts run mode observe/observe before the switch"; reset; touch $H/state/no_switch_line
run env ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SAFESCALE=true" nice bash $H/tools/run_arm_pilot.sh tre Alt >/dev/null 2>&1; echo rc=$?
grep -n -E "TRAP rc=4: (run mode|controller ablation)" $H/pilot/Alt/tre/runner.log | cut -c1-120; echo "mode now: $(cat $H/state/mode)"
echo "== 2026-10-08: reset_canonical.sh puts a switch left true back to false"; reset; echo -n true > $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE
run env CONTROLLER_DEPLOY=tre-v2-controller bash $H/tools/reset_canonical.sh --check >/dev/null 2>&1; echo "check rc=$? (want 1)"
run env CONTROLLER_DEPLOY=tre-v2-controller bash $H/tools/reset_canonical.sh 2>&1 | grep -E "switch|canonical" | cut -c1-140
[ "$(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE)" = false ] && echo "PASS: switch false after reset" || echo "FAIL: switch $(cat $H/state/ctl_TRE_ABLATION_DISABLE_SAFESCALE)"
