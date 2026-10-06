#!/usr/bin/env bash
# Builds a fake cluster environment under $H and runs run_arm_pilot.sh / baseline_sanity.sh against it.
set -euo pipefail
H=${H:-/tmp/bl-harness}; SRC=${SRC:-/tmp/bl-selftest}
rm -rf "$H"; mkdir -p "$H"/{bin,pyfake,tools,state,pilot,tre,lg}
LOGK=$H/state/calls.log
# ---------- fake kubectl
cat > "$H/bin/kubectl" <<'EOF'
#!/usr/bin/env bash
S=__H__/state; echo "kubectl $*" >> $S/calls.log
a="$*"
case "$a" in
  *"get svc"*clusterIP*) echo 127.0.0.1 ;;
  *"get deploy tre-v2-baseline-scaler"*replicas*) cat $S/scaler_replicas 2>/dev/null || echo 0 ;;
  *"get deploy tre-gateway-plugins"*TRE_BL_REQ_EVENTS*) cat $S/events 2>/dev/null || true ;;
  *"set env deploy/tre-gateway-plugins TRE_BL_REQ_EVENTS-"*) rm -f $S/events ;;
  *"set env deploy/tre-gateway-plugins TRE_BL_REQ_EVENTS="*) echo -n "${a##*TRE_BL_REQ_EVENTS=}" > $S/events ;;
  *"get deploy tre-v2-controller"*ENABLE_TRE_SCALING*) cat $S/tre_scaling 2>/dev/null || echo true ;;
  *"set env deploy/tre-v2-controller ENABLE_TRE_SCALING="*) echo -n "${a##*ENABLE_TRE_SCALING=}" > $S/tre_scaling ;;
  *"get cm tre-v2-registry"*) cat __H__/state/registry.yaml ;;
  *"get cm tre-v2-baseline-"*) p=${a#*tre-v2-baseline-}; p=${p%% *}; cat __H__/state/policy-$p.yaml ;;
  *"patch cm tre-v2-baseline-preserve"*) echo "$a" >> $S/cm_patches ;;
  *"get --raw"*) printf 'tre_gateway_bl_req_events_written_total 5\n' ;;
  *"get podautoscalers"*-o\ name*) ;;
  *"get podautoscalers"*) echo "items: []" ;;
  *"get pods"*"-o name"*) echo "pod/dsqwen-7b-nscc-ds-4a100-node9-gpu-0-x" ;;
  *"get pods"*"podIP"*) echo 10.0.0.9 ;;
  *"get pods"*) echo "p1 1/1 Running" ;;
  *"get deploy -o jsonpath"*) echo "tre-v2-controller img:1" ;;
  *"scale deploy/tre-v2-baseline-scaler --replicas="*) echo -n "${a##*--replicas=}" > $S/scaler_replicas ;;
  *"exec deploy/tre-v2-baseline-scaler -- test -r"*) [ -e $S/trace_ok ] ;;
  *) : ;;
esac
EOF
# ---------- fake curl (SM state + engine metrics)
cat > "$H/bin/curl" <<'EOF'
#!/usr/bin/env bash
echo "curl $*" >> __H__/state/calls.log
case "$*" in
  *"/v2/state"*) cat __H__/state/sm_state.json ;;
  *"/metrics"*) printf 'vllm:num_requests_running{a="b"} 0\nvllm:num_requests_waiting{a="b"} 0\n' ;;
esac
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$H/bin/docker"
printf '#!/usr/bin/env bash\necho "sleep $*" >> __H__/state/calls.log\nexit 0\n' > "$H/bin/sleep"
sed -i "s#__H__#$H#g" "$H/bin/"*; chmod +x "$H/bin/"*
# ---------- fake redis-py
cat > "$H/pyfake/redis.py" <<'EOF'
import time, json, os
__version__ = "fake"
_S = os.environ.get("FAKE_STATE", "/tmp")
class Redis:
    def __init__(self, *a, **k): pass
    @classmethod
    def from_url(cls, *a, **k): return cls()
    def time(self):
        t = time.time(); return (int(t), int((t % 1) * 1e6))
    def delete(self, k): return 1
    def get(self, k): return None
    def set(self, k, v, **kw): return True
    def hgetall(self, k): return {}
    def scan_iter(self, **k): return iter(())
    def lrange(self, *a): return []
    def xrange(self, key, min="-", max="+", count=None):
        if key.startswith("tre:v2:bl:req:"):
            return [(f"{int(time.time()*1000)}-0", {"kind": "arr", "req_id": "r1"})]
        return []
EOF
# ---------- fake TRE tree (git repo)
T=$H/tre; mkdir -p $T/deploy/scripts/release $T/deploy/scripts/analysis $T/baselines/tre_baselines/tools $T/common
cat > $T/deploy/scripts/set_run_mode.sh <<'EOF'
#!/usr/bin/env bash
S=__H__/state; if [ "$1" = status ]; then cat $S/mode 2>/dev/null || echo "observe active"; else echo "$1 $2" > $S/mode; fi
EOF
cat > $T/deploy/scripts/toggle_tre_apa.sh <<'EOF'
#!/usr/bin/env bash
S=__H__/state; echo "toggle $*" >> $S/calls.log
case "$1" in tre) echo -n true > $S/tre_scaling ;; apa) echo -n false > $S/tre_scaling ;; status)
  v=$(cat $S/tre_scaling 2>/dev/null || echo true); if [ "$v" = true ]; then echo "active decision source: TRE"; else echo "active decision source: NONE (both stopped)"; fi ;; esac
EOF
printf 'import sys\nprint("awake_ctl", sys.argv[1:])\n' > $T/deploy/scripts/release/awake_ctl.py
touch $T/deploy/scripts/__init__.py $T/deploy/scripts/analysis/__init__.py
printf 'print("{}")\n' > $T/deploy/scripts/analysis/safescale_summary.py
touch $T/baselines/tre_baselines/__init__.py $T/baselines/tre_baselines/tools/__init__.py
cat > $T/baselines/tre_baselines/tools/arm.py <<'EOF'
# fake arm tool; the real flag name below satisfies the runner's grep: client-sent-in-tokens
import json, os, sys
S = "__H__/state"
a = sys.argv[1:]
open(f"{S}/calls.log", "a").write("arm " + " ".join(a) + "\n")
cmd = a[0]
if cmd == "enable":
    open(f"{S}/scaler_replicas", "w").write("1")
    if os.path.exists(f"{S}/fail_enable"): sys.exit(1)
elif cmd == "disable":
    open(f"{S}/scaler_replicas", "w").write("0")
    if "--collect-dir" in a:
        d = a[a.index("--collect-dir") + 1]; os.makedirs(d + "/pod-x", exist_ok=True)
        ev = not os.path.exists(f"{S}/invalid")
        json.dump({"events_valid": ev, "invalid_because": [] if ev else ["gateway dropped events: 2.0"],
                   "replay_marker": {"t0_ms": 1}, "gw_bl_dropped_delta": 0.0, "client_sent_in_tokens": "--client-sent-in-tokens" in a},
                  open(d + "/run_validity.json", "w"))
        dec = os.path.join(S, "decisions.jsonl")
        if os.path.exists(dec):
            import shutil; shutil.copy(dec, d + "/pod-x/decisions-p-x.jsonl")
EOF
sed -i "s#__H__#$H#g" $T/deploy/scripts/*.sh $T/baselines/tre_baselines/tools/arm.py; chmod +x $T/deploy/scripts/*.sh
# fake loadgen + replayer
mkdir -p $T/loadgen_v1/tre_loadgen_v1 $T/loadgen_v1/configs/traces_v14/Alt $T/replayer/tre_replayer
cat > $T/loadgen_v1/tre_loadgen_v1/cli.py <<'EOF'
# flags: --ignore-eos --send-in-tokens
EOF
cat > $T/loadgen_v1/tre_loadgen_v1/__main__.py <<'EOF'
import json, os, sys
a = sys.argv[1:]; out = a[a.index("--output") + 1]; os.makedirs(out, exist_ok=True)
open("__H__/state/calls.log", "a").write("loadgen " + " ".join(a) + "\n")
with open(out + "/performance_metrics.json", "w") as fh:
    for i in range(40):
        fh.write(json.dumps({"request_id": f"r{i}", "model_name": "dsqwen-7b", "start_time": 100.0 + i, "success": True, "ttft": 0.1,
                             "e2e_latency": 10.0, "output_tokens": 400, "input_tokens": 492}) + "\n")
json.dump({"in_tokens_header": {"counted": 40, "omitted": 0} if "--send-in-tokens" in a else None}, open(out + "/loadgen_run_meta.json", "w"))
EOF
touch $T/loadgen_v1/tre_loadgen_v1/__init__.py $T/loadgen_v1/configs/traces_v14/Alt/config.yaml
cat > $T/replayer/tre_replayer/run_trace.py <<'EOF'
# --send-in-tokens
import sys, json
open("__H__/state/calls.log", "a").write("replayer " + " ".join(sys.argv[1:]) + "\n")
EOF
touch $T/replayer/tre_replayer/__init__.py
sed -i "s#__H__#$H#g" $T/loadgen_v1/tre_loadgen_v1/__main__.py $T/replayer/tre_replayer/run_trace.py
( cd $T && git init -q && git add -A && git -c user.email=x@x -c user.name=x commit -qm fake )
# ---------- state
cat > $H/state/registry.yaml <<'EOF'
models:
  - {name: dsqwen-7b, max_awake_replicas: 4, slo: {ttft_p95_ms: 2000, tpot_p95_ms: 75, e2e_p95_ms: 15000}}
  - {name: dsllama-8b, max_awake_replicas: 4, slo: {ttft_p95_ms: 2000, tpot_p95_ms: 75, e2e_p95_ms: 15000}}
  - {name: dsqwen-14b, max_awake_replicas: 4, slo: {ttft_p95_ms: 2000, tpot_p95_ms: 75, e2e_p95_ms: 15000}}
EOF
printf 'theta: {dsqwen-7b: 0.46, dsllama-8b: 0.534, dsqwen-14b: 0.46}\nalpha: 0.5\n' > $H/state/policy-chiron.yaml
printf 'velocity: {dsqwen-7b: {buckets: [[12000]], v_prefill: 30000}, dsllama-8b: {buckets: [[10000]], v_prefill: 25000}, dsqwen-14b: {buckets: [[8000]], v_prefill: 20000}}\nwindow_s: 10\n' > $H/state/policy-tokenscale.yaml
printf 'trace_path: /etc/tre-baselines-traces/Alt/traces_tre.effective.json\nwindow_s: 600\nmu: {dsqwen-7b: {p: 1, d: 1, t: 8920}, dsllama-8b: {p: 1, d: 1, t: 7136}, dsqwen-14b: {p: 1, d: 1, t: 6000}}\n' > $H/state/policy-preserve.yaml
python3 - "$H/state/sm_state.json" <<'EOF'
import json, sys
ids = ["dsqwen-7b/nscc-ds-4a100-node9/0", "dsllama-8b/nscc-ds-4a100-node9/1", "dsqwen-14b/nscc-ds-4a100-node10/0,1"]
json.dump({"version": 1, "models": {}, "bindings": [{"binding_id": i, "awake": True, "hidden": False} for i in ids]}, open(sys.argv[1], "w"))
EOF
# ---------- tools
cp $SRC/run_arm_pilot.sh $SRC/run_pilot.sh $SRC/baseline_sanity.sh $SRC/bl_tools.py $SRC/score_pilot.py $H/tools/
printf 'import os,sys,time\nd=sys.argv[1]\nfor _ in range(600):\n    if os.path.exists(d+"/STOP"): break\n    time.sleep(0.05)\n' > $H/tools/sampler.py
printf 'import json,sys\njson.dump({}, open(sys.argv[1],"w"))\n' > $H/tools/snap.py
printf 'print("analyze ok")\n' > $H/tools/analyze.py
mkdir -p $H/icse/Alt; echo '[]' > $H/icse/Alt/traces_tre.effective.json
echo "built $H"
