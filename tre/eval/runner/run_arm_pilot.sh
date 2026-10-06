#!/usr/bin/env bash
# Pilot (informal, not paper data): one arm of one ICSE-final v1 trace through the
# tre-v2 gateway ($GW). Derived from smoke-e1-20260930/tools/run_arm.sh. Changes vs that file:
#   1. trace NAME argument: TRACE + loadgen CONFIG are chosen per trace (was: Alternating config
#      hard-coded even when another trace file was passed);
#   2. per-trace output dir under $PILOT_ROOT ($RUN_TAG suffix for repeats; never overwrites);
#   3. baseline = explicit binding ids via tre/deploy/scripts/release/awake_ctl.py restore-ids
#      ($BASELINE in runner.env: one binding per model at the start);
#      the awake set is VERIFIED after every restore, the runner stops if it differs
#      (the 99_restore.sh failure mode: a 409 on a swap that is reported as success);
#   4. wait until every routable engine is idle (running+waiting = 0) before the controller restart;
#   5. score with score_pilot.py (V_req, output tokens, ignore_eos check) + smoke analyze.py;
#   6. (2026-10-05) exclusive-window marker: REQUIRED and must be this pilot's (first word
#      `validation`, a token `round=$PILOT_ROUND`); any other marker (calibration, other round) refuses;
#   7. (2026-10-05) loadgen_v1 runs from $LOADGEN_TRE_DIR (default: main tree), its git sha + dirty
#      state are recorded; `--ignore-eos` is sent when IGNORE_EOS=1 (default) and the tree must have it;
#   8. (2026-10-05) sidecar `tre_reissue_nofile` startup line recorded per pod at run start
#      (sidecar_nofile.tsv; expect after>=65535: 65535 where the runtime soft limit is 1024, else the hard limit); record only.
#   9. (2026-10-06) baseline arms chiron | tokenscale | preserve (labels Chiron-global /
#      TokenScale-colocated / PreServe-oracle, in arm_label): run mode observe/active, TRE scaling
#      off (2026-10-07: controller run mode observe, see 11) and 0 APA CRs (decision source NONE), policy ConfigMap
#      checked (non-empty, no placeholder, = $BL_CM_FILE when given), `arm enable --policy --execute`
#      (waits for /healthz + owner lock), `arm mark-replay` at the first send (MARK_AT), loadgen
#      with --ignore-eos --send-in-tokens, `arm disable --collect-dir $D/baseline
#      --client-sent-in-tokens`, run_validity.json folded into score.json (valid=false when
#      events_valid is false). The scaler is back at 0 replicas at the end; a trap disables it
#      when the runner dies mid-arm.
#  10. (2026-10-06) GW_PARITY (default 1): EVERY arm, TRE and APA too, runs with the gateway
#      request-event stream on (TRE_BL_REQ_EVENTS=true on the gateway plugins) and sends
#      x-tre-bl-in-tokens (loadgen --send-in-tokens), so all arms put identical work on the
#      gateway and its Redis; TRE/APA do not read either. GW_PARITY=0 = the 10-05/10-06 TRE/APA
#      runs exactly (stream left as found, no header); baseline arms need both and always get
#      them (gw_parity is recorded in arm_meta.json). The stream is put back to its pre-arm
#      state at the end of every arm.
#  11. (2026-10-07, in-repo tre/eval/runner) site settings come from runner.env (see
#      runner.env.example; no IP / NodePort / node name / NFS path in the scripts); the runner's
#      git sha is recorded (runner_sha). Evaluation data gaps (docs eval-metrics-spec §7):
#      G1 APA and baseline arms run the controller in run mode observe with its decision pipeline
#         ON (the scaling env switch is gone): signal_log + decision snapshots for every arm;
#         baseline arms wait until no SafeScale probe is open before the shell starts;
#      G2/G3/G10/G12 sampler: every awake pod incl. hidden at 1 Hz with its routable label and
#         counters (pod_metrics_1s.jsonl), SM routable view, per-GPU map on change (gpu_map.jsonl),
#         APA status 1 Hz, gpu-truth 1 Hz;  G8 kubelet /stats/summary -> resource_usage.jsonl (5 s);
#      G4b gateway per-request events dumped for every arm with the stream on (gateway_events/);
#      G5/G13 trace phase table + trace manifest (sha256, seed) copied into the arm dir; a seeded
#         trace gets a _s<seed> trace directory;  G6/G7 clock_offsets.json + components.json at arm
#         start and end (node of every TRE component, image IDs from pod status on all nodes);
#      G9 sm.ts.log = SM log with kubelet timestamps (uvicorn access lines get a time).
#  12. (2026-10-07, run_campaign.sh) optional per-arm overrides set by the campaign runner (unset =
#      unchanged behaviour): ARM_TRACE_DIR (trace directory, else $ICSE/<NAME>), ARM_LOADGEN_CONFIG
#      (loadgen config, else $LOADGEN_CONFIG_ROOT/<NAME>/config.yaml), ARM_OUT_DIR (result directory,
#      else $PILOT_ROOT/<trace>[_s<seed>]/<arm>[-tag]), ARM_LABEL_OVERRIDE (arm_label, e.g. a PreServe
#      variant) and ARM_VARIANT (the campaign arm id, recorded in arm_meta.json).
#  13. (2026-10-07) post-arm cleanup: decision source off (toggle tre, 0 APA CRs verified) BEFORE the
#      restore to $BASELINE, as reset_canonical.sh (APA scaled the restored replica down again: 409).
# CHANGES CLUSTER STATE (run mode, APA CRs/anchors, controller restarts, SM power, load,
# gateway plugin env (+ rollout), baseline-scaler env + replicas).
# Usage: run_arm_pilot.sh <tre|apa|chiron|tokenscale|preserve> <TRACE_NAME>   (nohup it; progress in <dir>/runner.log)
set -euo pipefail
ARM="$1"; NAME="$2"
IS_BL=0
case "$ARM" in
  tre) ARM_LABEL=TRE ;;
  apa) ARM_LABEL=APA-kvcache ;;
  chiron) ARM_LABEL=Chiron-global; IS_BL=1 ;;
  tokenscale) ARM_LABEL=TokenScale-colocated; IS_BL=1 ;;
  preserve) ARM_LABEL=PreServe-oracle; IS_BL=1 ;;
  *) echo "arm must be tre|apa|chiron|tokenscale|preserve" >&2; exit 2;;
esac
ARM_LABEL="${ARM_LABEL_OVERRIDE:-$ARM_LABEL}"
# shellcheck source=lib_env.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_env.sh"   # runner.env: every site setting below
need_env PILOT_ROOT TRE_DIR LOADGEN_TRE_DIR ICSE TRACE_FILE LOADGEN_CONFIG_ROOT GW MODELS BASELINE TRE_NS MODEL_NS APA_NS \
  AIBRIX_NS CONTROLLER_DEPLOY SM_DEPLOY SM_SVC SM_PORT REDIS_SVC REDIS_PORT GW_NS GW_DEPLOY GW_SELECTOR BL_NS BL_DEPLOY
PILOT_ROUND="${PILOT_ROUND:-$(basename "$PILOT_ROOT")}"
TOOLS="${TOOLS:-$RUNNER_DIR}"                       # this directory (sampler, snap, scoring, bl_tools)
TRE="$TRE_DIR"                                      # deploy scripts (toggle, run mode, awake_ctl), baselines arm tool
LG="$LOADGEN_TRE_DIR"                               # loadgen_v1 + replayer (the client); sha recorded
IGNORE_EOS="${IGNORE_EOS:-1}"
TRACE_DIR="${ARM_TRACE_DIR:-$ICSE/$NAME}"
TRACE="$TRACE_DIR/$TRACE_FILE"                      # same request plan for every arm (as smoke-e1)
CFG="${ARM_LOADGEN_CONFIG:-$LOADGEN_CONFIG_ROOT/$NAME/config.yaml}"
# G13: the trace seed (TRACE_SEED, else the v2 generator manifest next to the trace); a seeded
# trace's results go to <trace>_s<seed>/ (unless the trace directory name already ends so).
if [ -z "${TRACE_SEED:-}" ] && [ -f "$TRACE_DIR/${TRACE_SOURCE_MANIFEST:-manifest.json}" ]; then
  TRACE_SEED=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("seed");print("" if v is None else v)' "$TRACE_DIR/${TRACE_SOURCE_MANIFEST:-manifest.json}" 2>/dev/null || true)
fi
TRACE_SEED="${TRACE_SEED:-}"
OUT_NAME="$NAME"
if [ -n "$TRACE_SEED" ]; then case "$NAME" in *_s"$TRACE_SEED") ;; *) OUT_NAME="${NAME}_s$TRACE_SEED" ;; esac; fi
D="${ARM_OUT_DIR:-$PILOT_ROOT/$OUT_NAME/$ARM${RUN_TAG:+-$RUN_TAG}}"
REQUIRE_MARKER="${REQUIRE_MARKER:-1}"
IDLE_S="${IDLE_S:-60}"; POST_S="${POST_S:-30}"
SCORE_REGISTRY="${SCORE_REGISTRY:-$D/live-registry.yaml}"   # default: the live registry recorded at arm start
PROBE_WAIT_S="${PROBE_WAIT_S:-120}"              # baseline arms: max wait for open SafeScale probes to resolve
# ---- gateway parity + baseline arms (2026-10-06)
GW_PARITY="${GW_PARITY:-1}"                      # 1: every arm gets event stream on + x-tre-bl-in-tokens
BL_CM_FILE="${BL_CM_FILE:-}"                     # frozen policy-configmaps.yaml (feat/baseline-params-20261006); empty = check live only
BL_CM_APPLY="${BL_CM_APPLY:-0}"                  # 1: kubectl apply BL_CM_FILE when the live ConfigMap differs
BL_SEED="${BL_SEED:-0}"                          # TRE_BL_SEED of the shell + the replay marker's seed (policy RNGs)
MARK_AT="${MARK_AT:-first-send}"                 # first-send: marker after the first gateway `arr`; pre-load: just before loadgen
WANT_EVENTS=0; SEND_IN_TOKENS=0
if [ "$IS_BL" = 1 ] || [ "$GW_PARITY" = 1 ]; then WANT_EVENTS=1; SEND_IN_TOKENS=1; fi

# ---- checks before anything touches the cluster
[ -f "$TRACE" ] && [ -f "$CFG" ] || { echo "missing $TRACE or $CFG" >&2; exit 2; }
if [ -e "$MARKER" ]; then
  MK_TYPE=""; read -r MK_TYPE _ < "$MARKER" || true
  if [ "$MK_TYPE" != validation ] || ! awk -v want="round=$PILOT_ROUND" '{sub(/\r$/, ""); for (i = 1; i <= NF; i++) if ($i == want) f = 1} END {exit !f}' "$MARKER"; then
    echo "exclusive window marker is not this pilot's (want first word 'validation' + 'round=$PILOT_ROUND'): $(head -c 300 "$MARKER"); refusing" >&2; exit 2
  fi
elif [ "$REQUIRE_MARKER" = 1 ]; then
  echo "no exclusive window marker $MARKER (pilot needs 'validation <start> <end> round=$PILOT_ROUND'; REQUIRE_MARKER=0 to skip); refusing" >&2; exit 2
fi
LG_FLAGS=()
if [ "$IGNORE_EOS" = 1 ]; then
  grep -q -- "--ignore-eos" "$LG/loadgen_v1/tre_loadgen_v1/cli.py" \
    || { echo "IGNORE_EOS=1 but $LG has no loadgen_v1 --ignore-eos (merge feat/loadgen-ignore-eos-20261005 or set LOADGEN_TRE_DIR)" >&2; exit 2; }
  LG_FLAGS+=(--ignore-eos)
fi
if [ "$SEND_IN_TOKENS" = 1 ]; then
  grep -q -- "--send-in-tokens" "$LG/loadgen_v1/tre_loadgen_v1/cli.py" \
    || { echo "--send-in-tokens needed (arm=$ARM GW_PARITY=$GW_PARITY) but $LG has no loadgen_v1 --send-in-tokens" >&2; exit 2; }
  LG_FLAGS+=(--send-in-tokens)
fi
if [ -e "$D/client/performance_metrics.json" ]; then
  echo "$D already has client results; set RUN_TAG for a repeat (never overwritten)" >&2; exit 2
fi
if [ "$IS_BL" = 1 ]; then
  [ "$IGNORE_EOS" = 1 ] || { echo "baseline arms run with --ignore-eos (IGNORE_EOS=1)" >&2; exit 2; }
  case "$MARK_AT" in first-send|pre-load) ;; *) echo "MARK_AT must be first-send|pre-load" >&2; exit 2;; esac
  grep -q -- "client-sent-in-tokens" "$TRE/baselines/tre_baselines/tools/arm.py" \
    || { echo "$TRE has no baselines arm tool with --client-sent-in-tokens" >&2; exit 2; }
  R=$(kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}') \
    || { echo "no deployment $BL_NS/$BL_DEPLOY" >&2; exit 2; }
  [ "$R" = 0 ] || { echo "$BL_DEPLOY has $R replicas (another baseline arm running?); refusing" >&2; exit 2; }
fi

SMIP=$(kubectl -n "$TRE_NS" get svc "$SM_SVC" -o jsonpath='{.spec.clusterIP}')
SM=http://$SMIP:$SM_PORT
export SM_URL=$SM
REDIS_HOST=$(kubectl -n "$TRE_NS" get svc "$REDIS_SVC" -o jsonpath='{.spec.clusterIP}')
export REDIS_URL="redis://$REDIS_HOST:$REDIS_PORT/0"
mkdir -p "$D"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$D/runner.log"; }
mode() { bash "$TRE/deploy/scripts/set_run_mode.sh" "$1" "$2" >/dev/null; bash "$TRE/deploy/scripts/set_run_mode.sh" status | tr '\n' ' '; }
awake_ids() { curl -s "$SM/v2/state" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(" ".join(sorted(b["binding_id"] for b in d["bindings"] if b["awake"])))'; }
awake_set() { curl -s "$SM/v2/state" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(" ".join(sorted(b["binding_id"]+("(H)" if b["hidden"] else "") for b in d["bindings"] if b["awake"] or b["hidden"])))'; }
wait_no_hidden() {
  for i in $(seq 1 60); do
    H=$(curl -s "$SM/v2/state" | python3 -c 'import sys,json;print(sum(b["hidden"] for b in json.load(sys.stdin)["bindings"]))')
    [ "$H" = 0 ] && return 0; sleep 5
  done
  log "WARN hidden bindings remain after 300 s"; return 1
}
wait_engines_idle() {   # every Ready model pod: vllm running+waiting == 0 (max 300 s; sidecar :8000 proxies /metrics)
  for i in $(seq 1 60); do
    BUSY=$(kubectl -n "$MODEL_NS" get pods -l "$MODEL_SELECTOR" -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}' | while read -r IP; do
      [ -n "$IP" ] || continue
      curl -s --max-time 3 "http://$IP:$ENGINE_PORT/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}'
    done | awk '{s+=$1} END {print s+0}')
    [ "$BUSY" = 0 ] && return 0; sleep 5
  done
  log "WARN engines still busy after 300 s"; return 1
}
reset_baseline() {
  log "reset: observe/observe -> $(mode observe observe)"
  wait_no_hidden || true
  log "reset: awake before restore: $(awake_set)"
  # shellcheck disable=SC2086
  python3 "$TRE/deploy/scripts/release/awake_ctl.py" restore-ids $BASELINE >> "$D/runner.log" 2>&1 \
    || { log "ERROR restore-ids failed (see runner.log; 99_restore-type swap conflict? use the two-step manual reset in RUN.md)"; exit 3; }
  local want got; want=$(echo $BASELINE | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//'); got=$(awake_ids)
  [ "$want" = "$got" ] || { log "ERROR awake set after restore = [$got], want [$want]"; exit 3; }
  log "reset: awake after restore: $(awake_set)"
}
restart_controller() {
  kubectl -n "$TRE_NS" rollout restart "deploy/$CONTROLLER_DEPLOY" >/dev/null
  kubectl -n "$TRE_NS" rollout status "deploy/$CONTROLLER_DEPLOY" --timeout=180s >/dev/null
  date +%s > "$D/controller_restart_epoch"
  log "controller restarted: $(kubectl -n "$TRE_NS" get pods -l "$CONTROLLER_SELECTOR" -o name | tr '\n' ' ')"
}
record_sidecar_nofile() {   # startup line of the (fixed) sidecar; record only, never fails the run
  printf 'pod\tnode\tbefore\tafter\thard\ttarget\n' > "$D/sidecar_nofile.tsv"
  local P N LINE
  kubectl -n "$MODEL_NS" get pods -l "$MODEL_SELECTOR" -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName}{"\n"}{end}' | while read -r P N; do
    [ -n "$P" ] || continue
    LINE=$(kubectl -n "$MODEL_NS" logs "$P" -c "$SIDECAR_CONTAINER" --limit-bytes=2000000 2>/dev/null | grep -m1 '"tre_reissue_nofile"' || true)
    printf '%s\t%s\t%s\n' "$P" "${N:-?}" "${LINE:-MISSING}" | python3 -c '
import sys, json
for l in sys.stdin:
    pod, node, raw = l.rstrip("\n").split("\t", 2)
    try: r = json.loads(raw)
    except ValueError: r = {}
    print("\t".join(str(x) for x in (pod, node, r.get("before"), r.get("after"), r.get("hard"), r.get("target"))))' >> "$D/sidecar_nofile.tsv"
  done
  local total ok
  total=$(($(wc -l < "$D/sidecar_nofile.tsv") - 1))
  ok=$(awk -F'\t' 'NR>1 && $4 ~ /^[0-9]+$/ && $4 >= 65535' "$D/sidecar_nofile.tsv" | wc -l)
  log "sidecar nofile: $ok/$total pods with after>=65535 (65535 where the runtime soft limit is 1024, else the hard limit); see sidecar_nofile.tsv"
}
# ---- gateway request-event stream (TRE_BL_REQ_EVENTS on the plugin deployment; "" = unset = off)
events_get() { kubectl -n "$GW_NS" get deploy "$GW_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_BL_REQ_EVENTS")].value}'; }
events_set() {   # events_set <value> | events_set UNSET ; rollout only when it changes
  local want="$1" cur; cur=$(events_get)
  [ "$want" = UNSET ] && [ -z "$cur" ] && return 0
  [ "$want" = "$cur" ] && return 0
  if [ "$want" = UNSET ]; then
    kubectl -n "$GW_NS" set env "deploy/$GW_DEPLOY" TRE_BL_REQ_EVENTS- >> "$D/runner.log" 2>&1
  else
    kubectl -n "$GW_NS" set env "deploy/$GW_DEPLOY" "TRE_BL_REQ_EVENTS=$want" >> "$D/runner.log" 2>&1
  fi
  kubectl -n "$GW_NS" rollout status "deploy/$GW_DEPLOY" --timeout=180s >> "$D/runner.log" 2>&1
  log "gateway event stream: '${cur:-UNSET}' -> '$(events_get || true)' (plugins: $(kubectl -n "$GW_NS" get pods -l "$GW_SELECTOR" --no-headers 2>/dev/null | awk '{print $1":"$2":"$3}' | tr '\n' ' '))"
}
ARM_TOOL() { env PYTHONPATH="$TRE/common:$TRE/deploy:$TRE/baselines" python3 -m tre_baselines.tools.arm "$@" \
               --namespace "$BL_NS" --deployment "$BL_DEPLOY" --redis-url "$REDIS_URL" --gw-namespace "$GW_NS" --gw-selector "$GW_SELECTOR"; }
# G1: a baseline shell starts only when the observe controller has no open SafeScale probe. The
# observe controller's single SM write is the rollback unhide of its OWN open probes, and that
# PUT .../routable replaces the model's hidden set; with none open it never writes to the SM.
wait_no_open_probes() {
  local i n
  for i in $(seq 1 $(( (PROBE_WAIT_S + 4) / 5 ))); do
    n=$(BLT open-probes --redis "$REDIS_HOST" 2>/dev/null || echo "?")
    [ "$n" = 0 ] && return 0; sleep 5
  done
  log "ERROR $n open SafeScale probe(s) after ${PROBE_WAIT_S} s in observe mode; refusing to start the baseline shell"; exit 4
}
# G6/G7: clock offsets of every node vs this host, node + image IDs of every component
probe_clocks() { python3 "$TOOLS/clock_probe.py" "$D/clock_offsets.json" "$1" >> "$D/runner.log" 2>&1 || log "WARN clock probe ($1) failed"; }
record_components() { python3 "$TOOLS/components.py" "$D/components.json" "$1" >> "$D/runner.log" 2>&1 || log "WARN components ($1) failed"; }
BLT() { python3 "$TOOLS/bl_tools.py" "$@"; }
BL_ENABLED=0; EVENTS_CHANGED=0; EV_BEFORE=UNSET
on_exit() {   # safety net: never leave a baseline shell actuating or the stream flipped after a crash
  local rc=$?
  if [ "$BL_ENABLED" = 1 ]; then
    log "TRAP rc=$rc: disabling the baseline shell"
    ARM_TOOL disable --collect-dir "$D/baseline.trap" --execute >> "$D/runner.log" 2>&1 \
      || ARM_TOOL disable --skip-collect --execute >> "$D/runner.log" 2>&1 || log "TRAP ERROR: arm disable failed; scale $BL_DEPLOY to 0 by hand"
  fi
  if [ "$EVENTS_CHANGED" = 1 ]; then
    log "TRAP rc=$rc: gateway event stream back to '$EV_BEFORE'"; events_set "$EV_BEFORE" || log "TRAP ERROR: event stream restore failed"
  fi
  [ "$rc" = 0 ] || log "TRAP: runner exited rc=$rc mid-arm; cluster NOT reset (see RUN.md 'stop')"
}
trap on_exit EXIT

log "=== arm=$ARM ($ARM_LABEL) trace=$NAME file=$TRACE cfg=$CFG round=$PILOT_ROUND dir=$D"
echo "$ARM_LABEL" > "$D/arm_label"
log "marker: $(head -c 300 "$MARKER" 2>/dev/null || echo none)"
sha256sum "$TRACE" "$CFG" | tee -a "$D/runner.log"
runner_sha > "$D/runner_sha"
log "runner: $(tr '\n' ' ' < "$D/runner_sha")"
# G5/G13: phase table + trace manifest (sha256, seed) next to the results
SEG_SRC="$TRACE_DIR/${TRACE_SEGMENTS_FILE:-trace.json}"; SRC_MAN="$TRACE_DIR/${TRACE_SOURCE_MANIFEST:-manifest.json}"
[ -f "$SEG_SRC" ] && cp "$SEG_SRC" "$D/trace_segments.json"
[ -f "$SRC_MAN" ] && cp "$SRC_MAN" "$D/trace_source_manifest.json"
BLT trace-manifest --out "$D/trace_manifest.json" --name "$NAME" --trace "$TRACE" --config "$CFG" \
  --segments "$SEG_SRC" --source-manifest "$SRC_MAN" --seed "$TRACE_SEED" >> "$D/runner.log" 2>&1 || log "WARN trace manifest failed"
git -C "$TRE" rev-parse HEAD > "$D/tre_sha"
{ echo "dir=$LG"; echo "sha=$(git -C "$LG" rev-parse HEAD)"; echo "branch=$(git -C "$LG" rev-parse --abbrev-ref HEAD)";
  echo "dirty_files=$(git -C "$LG" status --porcelain -- loadgen_v1 replayer | wc -l)"; echo "ignore_eos=$IGNORE_EOS"; echo "send_in_tokens=$SEND_IN_TOKENS"; } > "$D/loadgen_sha"
log "loadgen: $(tr '\n' ' ' < "$D/loadgen_sha")"
kubectl -n "$TRE_NS" get deploy -o jsonpath='{range .items[*]}{.metadata.name} {.spec.template.spec.containers[0].image}{"\n"}{end}' > "$D/images.txt"
# this host's docker only (as before); components.json has the image IDs on every node (G7)
docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null | grep -E 'tre-v2-|gateway-plugins|vllm-openai-tre' > "$D/image_ids_$(hostname -s).txt" || true
kubectl -n "$MODEL_NS" get pods -l "$MODEL_SELECTOR" -o jsonpath='{range .items[*]}{.metadata.name} {.spec.containers[0].image} {.status.containerStatuses[0].imageID}{"\n"}{end}' > "$D/model_images.txt"
kubectl -n "$TRE_NS" get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > "$D/live-registry.yaml"
EV_BEFORE=$(events_get); EV_BEFORE=${EV_BEFORE:-UNSET}
python3 -c 'import json,sys; json.dump(dict(arm=sys.argv[1], label=sys.argv[2], gw_parity=int(sys.argv[3]), events_wanted=int(sys.argv[4]),
  send_in_tokens=int(sys.argv[5]), events_before=sys.argv[6], bl_seed=int(sys.argv[7]), mark_at=sys.argv[8], bl_cm_file=sys.argv[9] or None,
  trace_seed=sys.argv[11] or None, controller_mode="active" if sys.argv[1] == "tre" else "observe", variant=sys.argv[12] or None),
  open(sys.argv[10], "w"), indent=1)' "$ARM" "$ARM_LABEL" "$GW_PARITY" "$WANT_EVENTS" "$SEND_IN_TOKENS" "$EV_BEFORE" "$BL_SEED" "$MARK_AT" "$BL_CM_FILE" "$D/arm_meta.json" "$TRACE_SEED" "${ARM_VARIANT:-}"
log "gateway parity: GW_PARITY=$GW_PARITY events_wanted=$WANT_EVENTS send_in_tokens=$SEND_IN_TOKENS (stream before: $EV_BEFORE)"
if [ "$IS_BL" = 1 ]; then
  # policy ConfigMap: live content must be the frozen parameters (BL_CM_FILE) and no placeholder
  kubectl -n "$BL_NS" get cm "tre-v2-baseline-$ARM" -o jsonpath="{.data.$ARM\.yaml}" > "$D/policy-$ARM.yaml"
  if ! BLT check-policy --policy "$ARM" --live "$D/policy-$ARM.yaml" --registry "$D/live-registry.yaml" \
        ${BL_CM_FILE:+--frozen "$BL_CM_FILE"} --trace "$TRACE" > "$D/policy_check.json" 2>&1; then
    if [ -n "$BL_CM_FILE" ] && [ "$BL_CM_APPLY" = 1 ] && grep -q '"frozen_equal": false' "$D/policy_check.json"; then
      log "policy ConfigMap differs from $BL_CM_FILE: applying it (BL_CM_APPLY=1)"
      kubectl apply -f "$BL_CM_FILE" >> "$D/runner.log" 2>&1
      kubectl -n "$BL_NS" get cm "tre-v2-baseline-$ARM" -o jsonpath="{.data.$ARM\.yaml}" > "$D/policy-$ARM.yaml"
      BLT check-policy --policy "$ARM" --live "$D/policy-$ARM.yaml" --registry "$D/live-registry.yaml" \
        --frozen "$BL_CM_FILE" --trace "$TRACE" > "$D/policy_check.json" 2>&1 \
        || { log "ERROR policy check after apply: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["problems"])' "$D/policy_check.json" 2>/dev/null || head -c 600 "$D/policy_check.json")"; exit 2; }
    else
      log "ERROR policy check: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["problems"])' "$D/policy_check.json" 2>/dev/null || head -c 600 "$D/policy_check.json")"; exit 2
    fi
  fi
  if [ -n "$BL_CM_FILE" ]; then
    { echo "file=$BL_CM_FILE"; echo "sha=$(git -C "$(dirname "$BL_CM_FILE")" rev-parse HEAD 2>/dev/null || echo none)";
      echo "dirty=$(git -C "$(dirname "$BL_CM_FILE")" status --porcelain -- "$(basename "$BL_CM_FILE")" 2>/dev/null | wc -l)"; } > "$D/policy_cm_sha"
  fi
  log "policy ConfigMap ok: sha256 $(sha256sum "$D/policy-$ARM.yaml" | cut -c1-16) (frozen file: ${BL_CM_FILE:-none})"
fi
record_sidecar_nofile
if [ "$WANT_EVENTS" = 1 ]; then
  [ "$EV_BEFORE" = true ] || EVENTS_CHANGED=1
  events_set true
fi
reset_baseline
wait_engines_idle || true
# Decision source = run mode + APA CRs + baseline owner lock (toggle_tre_apa.sh status). The
# controller computes and records its signals and decisions in every arm; it actuates only in
# run mode active (TRE arm). APA / baseline arms: controller observe (counterfactual log, G1).
if [ "$ARM" = tre ]; then
  bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
elif [ "$ARM" = apa ]; then
  bash "$TRE/deploy/scripts/toggle_tre_apa.sh" apa --keep-run-mode >> "$D/runner.log" 2>&1
else
  # decision source NONE before the shell: 0 APA CRs (toggle tre removes them), controller observe
  N_APA=$(kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o name 2>/dev/null | grep -c . || true)
  [ "$N_APA" = 0 ] || bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
  kubectl -n "$BL_NS" set env "deploy/$BL_DEPLOY" "TRE_BL_SEED=$BL_SEED" >> "$D/runner.log" 2>&1   # 0 replicas: template only
fi
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" status --keep-run-mode >> "$D/runner.log" 2>&1 || true
if [ "$IS_BL" = 1 ]; then
  ST=$(bash "$TRE/deploy/scripts/toggle_tre_apa.sh" status --keep-run-mode 2>/dev/null | grep 'active decision source' || true)
  case "$ST" in *NONE*) log "decision source before the shell: $ST" ;; *) log "ERROR decision source is not NONE: $ST"; exit 4;; esac
fi
kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o yaml > "$D/apa_crs_before.yaml" 2>&1 || true
python3 "$TOOLS/snap.py" "$D/snap_before.json" "$SM" >> "$D/runner.log" 2>&1
log "awake at start: $(awake_set)"
if [ "$ARM" = tre ]; then log "run mode -> $(mode active active)"; else log "run mode -> $(mode observe active)"; fi
restart_controller
T_RESTART=$(cat "$D/controller_restart_epoch"); START_ISO=$(date -u -d @"$T_RESTART" +%FT%TZ); echo "$START_ISO" > "$D/start_iso"
T_IDLE0=$T_RESTART
probe_clocks start
record_components start
BLT redis-ms --redis "$REDIS_HOST" > "$D/arm_start_redis_ms" 2>/dev/null || true
if [ "$IS_BL" = 1 ]; then
  wait_no_open_probes
  BLT del-key --redis "$REDIS_HOST" --key tre:v2:bl:replay_t0 >> "$D/runner.log" 2>&1   # no stale marker from an earlier run
  BL_ENABLED=1
  ARM_TOOL enable --policy "$ARM" --execute --timeout-s 180 >> "$D/runner.log" 2>&1 \
    || { log "ERROR arm enable failed (see runner.log)"; exit 4; }
  T_IDLE0=$(date +%s); echo "$T_IDLE0" > "$D/bl_enabled_epoch"
  BLT redis-ms --redis "$REDIS_HOST" > "$D/bl_enabled_redis_ms"
  log "baseline shell enabled: $(kubectl -n "$BL_NS" get pods -l app.kubernetes.io/name=$BL_DEPLOY --no-headers | awk '{print $1":"$2":"$3}' | tr '\n' ' ')"
  if [ "$ARM" = preserve ]; then
    TP=$(python3 -c 'import yaml,sys;print((yaml.safe_load(open(sys.argv[1])) or {}).get("trace_path",""))' "$D/policy-$ARM.yaml")
    kubectl -n "$BL_NS" exec "deploy/$BL_DEPLOY" -- test -r "$TP" \
      && log "preserve trace in pod: $TP" || { log "ERROR preserve trace_path $TP not readable in the pod"; exit 4; }
  fi
fi
rm -f "$D/STOP"
nohup python3 "$TOOLS/sampler.py" "$D" "$SM" "$REDIS_URL" > "$D/sampler.log" 2>&1 &
SAMPLER=$!
if [ "$ARM" = apa ]; then
  N=0
  for i in $(seq 1 36); do
    N=$(kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="AbleToScale")].status}{"\n"}{end}' | grep -c True || true)
    [ "$N" = 3 ] && break; sleep 5
  done
  log "APA AbleToScale=True count: $N"
  [ "$N" = 3 ] || { log "ERROR APA not ready"; touch "$D/STOP"; exit 4; }
fi
NOW=$(date +%s); W=$(( T_IDLE0 + IDLE_S - NOW )); [ $W -gt 0 ] && sleep $W
log "idle done ($(( $(date +%s) - T_RESTART )) s after controller restart); awake: $(awake_set); mode: $(bash $TRE/deploy/scripts/set_run_mode.sh status | tr '\n' ' ')"
if [ "$IS_BL" = 1 ] && [ "$MARK_AT" = pre-load ]; then
  ARM_TOOL mark-replay --trace "$TRACE" --seed "$BL_SEED" --execute >> "$D/runner.log" 2>&1 || { log "ERROR mark-replay failed"; exit 4; }
fi
if [ "$IS_BL" = 1 ]; then BLT redis-ms --redis "$REDIS_HOST" > "$D/load_start_redis_ms"; fi
T_LOAD=$(date +%s); echo "$T_LOAD" > "$D/load_start_epoch"; log "load start (loadgen flags: ${LG_FLAGS[*]:-none})"
set +e
run_loadgen() {
  ( cd "$LG/loadgen_v1" && PYTHONPATH="$LG/loadgen_v1" python3 -m tre_loadgen_v1 --stage all \
      --config "$CFG" --trace-file "$TRACE" --base-url "$GW" --max-retries 0 ${LG_FLAGS[@]+"${LG_FLAGS[@]}"} --output "$D/client" ) > "$D/client.log" 2>&1
}
if [ "$IS_BL" = 1 ] && [ "$MARK_AT" = first-send ]; then
  run_loadgen &   # baseline arms: the replay marker is written at the first send (gateway arr event)
  LGPID=$!
  FIRST=$(BLT wait-first-arr --redis "$REDIS_HOST" --since-ms "$(cat "$D/load_start_redis_ms")" --models "$MODELS" --timeout-s 300)
  [ -n "$FIRST" ] || log "WARN no gateway arr event within 300 s of load start; marking now"
  ARM_TOOL mark-replay --trace "$TRACE" --seed "$BL_SEED" --execute >> "$D/runner.log" 2>&1 || log "ERROR mark-replay failed (PreServe Tier-1 inactive)"
  echo "${FIRST:-null}" > "$D/first_arr_redis_ms"
  log "replay marker written after the first arr (first arr redis ms ${FIRST:-none})"
  wait $LGPID; RC=$?
else
  run_loadgen; RC=$?   # tre / apa: foreground, as before
fi
set -e
T_END=$(date +%s); echo "$T_END" > "$D/load_end_epoch"; log "load end rc=$RC ($(( T_END - T_LOAD )) s)"
if [ "$SEND_IN_TOKENS" = 1 ]; then
  log "x-tre-bl-in-tokens precount: $(BLT client-header --meta "$D/client/loadgen_run_meta.json" 2>&1 || echo 'WARN header omitted for some requests')"
fi
sleep "$POST_S"
if [ "$IS_BL" = 1 ]; then
  kubectl -n "$BL_NS" logs "deploy/$BL_DEPLOY" > "$D/baseline-scaler.log" 2>&1 || true
  DIS_FLAGS=(); [ "$SEND_IN_TOKENS" = 1 ] && DIS_FLAGS=(--client-sent-in-tokens)
  ARM_TOOL disable --collect-dir "$D/baseline" "${DIS_FLAGS[@]}" --execute >> "$D/runner.log" 2>&1 \
    || { log "ERROR arm disable failed (see runner.log)"; exit 4; }
  BL_ENABLED=0
  log "baseline shell disabled; validity: $(python3 -c 'import json,sys;v=json.load(open(sys.argv[1]));print({k:v.get(k) for k in ("events_valid","invalid_because","gw_bl_dropped_delta")})' "$D/baseline/run_validity.json" 2>&1)"
fi
touch "$D/STOP"; wait $SAMPLER || true
probe_clocks end
record_components end
python3 "$TOOLS/snap.py" "$D/snap_after.json" "$SM" >> "$D/runner.log" 2>&1
kubectl -n "$TRE_NS" logs "deploy/$CONTROLLER_DEPLOY" --since-time="$START_ISO" > "$D/controller.log" 2>&1 || true
kubectl -n "$TRE_NS" logs "deploy/$SM_DEPLOY" --since-time="$START_ISO" > "$D/sm.log" 2>&1 || true
# G9: the same SM log with kubelet timestamps (uvicorn access lines carry no time of their own)
kubectl -n "$TRE_NS" logs "deploy/$SM_DEPLOY" --since-time="$START_ISO" --timestamps > "$D/sm.ts.log" 2>&1 || true
kubectl -n "$GW_NS" logs "deploy/$GW_DEPLOY" --since-time="$START_ISO" > "$D/gateway-plugins.log" 2>&1 || true
mkdir -p "$D/sidecar"
for P in $(kubectl -n "$MODEL_NS" get pods -l "$MODEL_SELECTOR" -o name); do
  kubectl -n "$MODEL_NS" logs "$P" -c "$SIDECAR_CONTAINER" --since-time="$START_ISO" 2>/dev/null > "$D/sidecar/${P#pod/}.log" || true
done
grep -l -i "too many open files" "$D"/sidecar/*.log > "$D/EMFILE_PODS" 2>/dev/null || true
APA_GREP="podautoscal|apa|scale|$(echo "$MODELS" | tr ',' '|')"
kubectl -n "$AIBRIX_NS" logs "deploy/$APA_CTRL_DEPLOY" --since-time="$START_ISO" 2>/dev/null \
  | grep -i -E "$APA_GREP" > "$D/aibrix-controller-manager.log" || true
kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o yaml > "$D/apa_crs_after.yaml" 2>&1 || true
python3 - "$D" "$T_RESTART" "$REDIS_URL" <<'PY'
import json, sys, redis
d, t0 = sys.argv[1], int(sys.argv[2]) * 1000
r = redis.Redis.from_url(sys.argv[3], decode_responses=True)
probes = dict(r.hgetall("tre:v2:controller:safescale:probes"))
journals = {}
for k in r.scan_iter(match="tre:v2:controller:safescale:probe:*:journal", count=1000):
    try: ts = int(k.split(":")[-2].rsplit("-", 1)[-1])
    except ValueError: ts = t0
    if ts >= t0 - 60000: journals[k] = r.lrange(k, 0, -1)
json.dump({"probes": probes, "journals": journals}, open(f"{d}/safescale.json", "w"), indent=1)
with open(f"{d}/signal_log.jsonl", "w") as f:
    for eid, fields in r.xrange("tre:v2:controller:signal_log", min=f"{t0}-0"):
        f.write(json.dumps({"id": eid, **fields}) + "\n")
print("probes", len(probes), "journals", len(journals))
PY
if [ "$IS_BL" = 1 ]; then
  log "bl streams: $(BLT dump-streams --redis "$REDIS_HOST" --since-ms "$(cat "$D/bl_enabled_redis_ms")" --out-dir "$D/baseline" --models "$MODELS" 2>&1)"
elif [ "$WANT_EVENTS" = 1 ] && [ -s "$D/arm_start_redis_ms" ]; then
  # G4b: the gateway per-request events of TRE / APA arms too (the stream was on)
  log "gateway events: $(BLT dump-streams --redis "$REDIS_HOST" --since-ms "$(cat "$D/arm_start_redis_ms")" --out-dir "$D/gateway_events" --models "$MODELS" 2>&1)"
fi
( cd "$TRE/deploy" && PYTHONPATH="$TRE/deploy:$TRE/common" python3 -m scripts.analysis.safescale_summary "$D/safescale.json" > "$D/safescale_summary.json" 2>&1 ) || true
log "collected; awake at end: $(awake_set); EMFILE pods: $(wc -l < "$D/EMFILE_PODS")"
# Decision source off BEFORE any restore / wake (same order as reset_canonical.sh): the APA
# controller polls about every 1 s and scales a replica that restore-ids just woke back down
# (SM scale_service path=apa), and restore-ids then fails with 409 slot_occupied (exit 3, arm
# INVALID after load and collection). observe/observe first: the controller restart inside
# toggle tre comes up observing. Everything measured is collected above.
log "restore: observe/observe -> $(mode observe observe)"
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
N_APA=$(kubectl -n "$APA_NS" get podautoscalers.autoscaling.aibrix.ai -o name 2>/dev/null | grep -c . || true)
log "restore: APA CRs live: $N_APA"
[ "$N_APA" = 0 ] || { log "ERROR $N_APA APA CR(s) still live after toggle tre; not restoring"; exit 3; }
reset_baseline
log "restore: run mode -> $(mode observe active)"
if [ "$EVENTS_CHANGED" = 1 ]; then events_set "$EV_BEFORE"; EVENTS_CHANGED=0; log "restore: gateway event stream -> '$(events_get)' (pre-arm '$EV_BEFORE')"; fi
if [ "$IS_BL" = 1 ]; then
  R=$(kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}')
  [ "$R" = 0 ] || { kubectl -n "$BL_NS" scale "deploy/$BL_DEPLOY" --replicas=0 >> "$D/runner.log" 2>&1; log "WARN scaler had $R replicas at the end: scaled to 0"; }
  log "restore: $BL_DEPLOY replicas $(kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}')"
fi
python3 "$TOOLS/analyze.py" "$D" > "$D/analyze.out" 2>&1 || true
SCORE_EXTRA=()
if [ "$IS_BL" = 1 ]; then SCORE_EXTRA=(--validity "$D/baseline/run_validity.json" --arm-label "$ARM_LABEL"); fi
python3 "$TOOLS/score_pilot.py" "$D/client" --registry "$SCORE_REGISTRY" ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"} --out "$D/score.json" > "$D/score.out" 2>&1 || log "WARN score_pilot failed (see score.out)"
[ -f "$D/score.json" ] && log "score: $(python3 -c 'import json,sys;s=json.load(open(sys.argv[1]));a=s["models"]["ALL"];print({k:a.get(k) for k in ("n","fail","V_req_pct","output_tokens_sum","max_tokens_hit_frac")}, "valid=%s" % s.get("valid", "n/a"))' "$D/score.json")"
log "=== arm=$ARM trace=$NAME done rc=$RC"
