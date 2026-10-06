#!/usr/bin/env bash
# Pilot (informal, not paper data): one arm of one ICSE-final v1 trace through the
# tre-v2 gateway 31094. Derived from smoke-e1-20260930/tools/run_arm.sh. Changes vs that file:
#   1. trace NAME argument: TRACE + loadgen CONFIG are chosen per trace (was: Alternating config
#      hard-coded even when another trace file was passed);
#   2. per-trace output dir under $PILOT_ROOT ($RUN_TAG suffix for repeats; never overwrites);
#   3. baseline = explicit binding ids via tre/deploy/scripts/release/awake_ctl.py restore-ids
#      (2026-10-05 layout after the sidecar redeploy: 7b node9/0, 8b node9/1, 14b node10/0,1);
#      the awake set is VERIFIED after every restore, the runner stops if it differs
#      (the 99_restore.sh failure mode: a 409 on a swap that is reported as success);
#   4. wait until every routable engine is idle (running+waiting = 0) before the controller restart;
#   5. score with score_pilot.py (V_req, output tokens, ignore_eos check) + smoke analyze.py;
#   6. (2026-10-05) exclusive-window marker: REQUIRED and must be this pilot's (first word
#      `validation`, a token `round=$PILOT_ROUND`); any other marker (calibration, other round) refuses;
#   7. (2026-10-05) loadgen_v1 runs from $LOADGEN_TRE_DIR (default: main tree), its git sha + dirty
#      state are recorded; `--ignore-eos` is sent when IGNORE_EOS=1 (default) and the tree must have it;
#   8. (2026-10-05) sidecar `tre_reissue_nofile` startup line recorded per pod at run start
#      (sidecar_nofile.tsv; expect after>=65535: 65535 on node10, 1048576 on node9); record only.
#   9. (2026-10-06) baseline arms chiron | tokenscale | preserve (labels Chiron-global /
#      TokenScale-colocated / PreServe-oracle, in arm_label): run mode observe/active, TRE scaling
#      off (ENABLE_TRE_SCALING=false) and 0 APA CRs (decision source NONE), policy ConfigMap
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
# CHANGES CLUSTER STATE (run mode, APA CRs/anchors, controller env + restarts, SM power, load,
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
PILOT_ROOT="${PILOT_ROOT:-/data/nfs_shared_data/xxy/pilot-e1-20261005}"
PILOT_ROUND="${PILOT_ROUND:-$(basename "$PILOT_ROOT")}"
TOOLS="${TOOLS:-$PILOT_ROOT/tools}"                 # copy of smoke-e1 tools + this script + score_pilot.py
D=$PILOT_ROOT/$NAME/$ARM${RUN_TAG:+-$RUN_TAG}
TRE="${TRE_DIR:-/data/nfs_shared_data/xxy/aibrix/tre}"          # deploy scripts (toggle, run mode, awake_ctl), baselines arm tool
LG="${LOADGEN_TRE_DIR:-/data/nfs_shared_data/xxy/aibrix/tre}"   # loadgen_v1 + replayer (the client); sha recorded
IGNORE_EOS="${IGNORE_EOS:-1}"
ICSE="${ICSE:-/root/aibrix-main/CustomTraceGenerator/config/icse_final}"
TRACE="$ICSE/$NAME/traces_tre.effective.json"       # same request plan for both arms (as smoke-e1)
CFG="$LG/loadgen_v1/configs/traces_v14/$NAME/config.yaml"
GW="${GW:-http://192.168.223.76:31094}"
MARKER="${MARKER:-/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW}"
REQUIRE_MARKER="${REQUIRE_MARKER:-1}"
IDLE_S="${IDLE_S:-60}"; POST_S="${POST_S:-30}"
BASELINE="${BASELINE:-dsqwen-7b/nscc-ds-4a100-node9/0 dsllama-8b/nscc-ds-4a100-node9/1 dsqwen-14b/nscc-ds-4a100-node10/0,1}"
SCORE_REGISTRY="${SCORE_REGISTRY:-$D/live-registry.yaml}"   # default: the live registry recorded at arm start
# ---- gateway parity + baseline arms (2026-10-06)
GW_PARITY="${GW_PARITY:-1}"                      # 1: every arm gets event stream on + x-tre-bl-in-tokens
GW_NS="${GW_NS:-tre-v2}"; GW_DEPLOY="${GW_DEPLOY:-tre-gateway-plugins}"; GW_SELECTOR="${GW_SELECTOR:-app=tre-gateway-plugins}"
BL_NS="${BL_NS:-tre-v2}"; BL_DEPLOY="${BL_DEPLOY:-tre-v2-baseline-scaler}"
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

SMIP=$(kubectl -n tre-v2 get svc tre-v2-service-manager -o jsonpath='{.spec.clusterIP}')
SM=http://$SMIP:8000
export SM_URL=$SM
REDIS_HOST=$(kubectl -n tre-v2 get svc tre-v2-redis -o jsonpath='{.spec.clusterIP}')
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
    BUSY=$(kubectl -n default get pods -l tre.aibrix.io/managed=true -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}' | while read -r IP; do
      [ -n "$IP" ] || continue
      curl -s --max-time 3 "http://$IP:8000/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}'
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
  kubectl -n tre-v2 rollout restart deploy/tre-v2-controller >/dev/null
  kubectl -n tre-v2 rollout status deploy/tre-v2-controller --timeout=180s >/dev/null
  date +%s > "$D/controller_restart_epoch"
  log "controller restarted: $(kubectl -n tre-v2 get pods -l app.kubernetes.io/name=tre-v2-controller -o name | tr '\n' ' ')"
}
record_sidecar_nofile() {   # startup line of the (fixed) sidecar; record only, never fails the run
  printf 'pod\tnode\tbefore\tafter\thard\ttarget\n' > "$D/sidecar_nofile.tsv"
  local P LINE
  for P in $(kubectl -n default get pods -l tre.aibrix.io/managed=true -o name); do
    LINE=$(kubectl -n default logs "$P" -c tre-reissue-sidecar --limit-bytes=2000000 2>/dev/null | grep -m1 '"tre_reissue_nofile"' || true)
    printf '%s\t%s\n' "${P#pod/}" "${LINE:-MISSING}" | python3 -c '
import sys, json
for l in sys.stdin:
    pod, raw = l.rstrip("\n").split("\t", 1)
    node = "node10" if "node10" in pod else ("node9" if "node9" in pod else "?")
    try: r = json.loads(raw)
    except ValueError: r = {}
    print("\t".join(str(x) for x in (pod, node, r.get("before"), r.get("after"), r.get("hard"), r.get("target"))))' >> "$D/sidecar_nofile.tsv"
  done
  local total ok
  total=$(($(wc -l < "$D/sidecar_nofile.tsv") - 1))
  ok=$(awk -F'\t' 'NR>1 && $4 ~ /^[0-9]+$/ && $4 >= 65535' "$D/sidecar_nofile.tsv" | wc -l)
  log "sidecar nofile: $ok/$total pods with after>=65535 (node10 expect 65535, node9 1048576); see sidecar_nofile.tsv"
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
               --namespace "$BL_NS" --deployment "$BL_DEPLOY" --redis-url "redis://$REDIS_HOST:6379/0" --gw-namespace "$GW_NS" --gw-selector "$GW_SELECTOR"; }
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
git -C "$TRE" rev-parse HEAD > "$D/tre_sha"
{ echo "dir=$LG"; echo "sha=$(git -C "$LG" rev-parse HEAD)"; echo "branch=$(git -C "$LG" rev-parse --abbrev-ref HEAD)";
  echo "dirty_files=$(git -C "$LG" status --porcelain -- loadgen_v1 replayer | wc -l)"; echo "ignore_eos=$IGNORE_EOS"; echo "send_in_tokens=$SEND_IN_TOKENS"; } > "$D/loadgen_sha"
log "loadgen: $(tr '\n' ' ' < "$D/loadgen_sha")"
kubectl -n tre-v2 get deploy -o jsonpath='{range .items[*]}{.metadata.name} {.spec.template.spec.containers[0].image}{"\n"}{end}' > "$D/images.txt"
docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null | grep -E 'tre-v2-|gateway-plugins|vllm-openai-tre' > "$D/image_ids_node10.txt" || true
kubectl -n default get pods -l tre.aibrix.io/managed=true -o jsonpath='{range .items[*]}{.metadata.name} {.spec.containers[0].image} {.status.containerStatuses[0].imageID}{"\n"}{end}' > "$D/model_images.txt"
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > "$D/live-registry.yaml"
EV_BEFORE=$(events_get); EV_BEFORE=${EV_BEFORE:-UNSET}
python3 -c 'import json,sys; json.dump(dict(arm=sys.argv[1], label=sys.argv[2], gw_parity=int(sys.argv[3]), events_wanted=int(sys.argv[4]),
  send_in_tokens=int(sys.argv[5]), events_before=sys.argv[6], bl_seed=int(sys.argv[7]), mark_at=sys.argv[8], bl_cm_file=sys.argv[9] or None),
  open(sys.argv[10], "w"), indent=1)' "$ARM" "$ARM_LABEL" "$GW_PARITY" "$WANT_EVENTS" "$SEND_IN_TOKENS" "$EV_BEFORE" "$BL_SEED" "$MARK_AT" "$BL_CM_FILE" "$D/arm_meta.json"
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
if [ "$ARM" = tre ]; then
  bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
elif [ "$ARM" = apa ]; then
  bash "$TRE/deploy/scripts/toggle_tre_apa.sh" apa --keep-run-mode >> "$D/runner.log" 2>&1
  kubectl -n tre-v2 set env deploy/tre-v2-controller ENABLE_TRE_SCALING=true >> "$D/runner.log" 2>&1   # counterfactual logging only
else
  # decision source NONE before the shell: 0 APA CRs (toggle tre removes them) and TRE scaling off
  N_APA=$(kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o name 2>/dev/null | grep -c . || true)
  [ "$N_APA" = 0 ] || bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
  kubectl -n tre-v2 set env deploy/tre-v2-controller ENABLE_TRE_SCALING=false >> "$D/runner.log" 2>&1
  kubectl -n tre-v2 rollout status deploy/tre-v2-controller --timeout=180s >> "$D/runner.log" 2>&1
  kubectl -n "$BL_NS" set env "deploy/$BL_DEPLOY" "TRE_BL_SEED=$BL_SEED" >> "$D/runner.log" 2>&1   # 0 replicas: template only
fi
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" status --keep-run-mode >> "$D/runner.log" 2>&1 || true
if [ "$IS_BL" = 1 ]; then
  ST=$(bash "$TRE/deploy/scripts/toggle_tre_apa.sh" status --keep-run-mode 2>/dev/null | grep 'active decision source' || true)
  case "$ST" in *NONE*) log "decision source before the shell: $ST" ;; *) log "ERROR decision source is not NONE: $ST"; exit 4;; esac
fi
kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o yaml > "$D/apa_crs_before.yaml" 2>&1 || true
python3 "$TOOLS/snap.py" "$D/snap_before.json" "$SM" >> "$D/runner.log" 2>&1
log "awake at start: $(awake_set)"
if [ "$ARM" = tre ]; then log "run mode -> $(mode active active)"; else log "run mode -> $(mode observe active)"; fi
restart_controller
T_RESTART=$(cat "$D/controller_restart_epoch"); START_ISO=$(date -u -d @"$T_RESTART" +%FT%TZ); echo "$START_ISO" > "$D/start_iso"
T_IDLE0=$T_RESTART
if [ "$IS_BL" = 1 ]; then
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
nohup python3 "$TOOLS/sampler.py" "$D" "$SM" > "$D/sampler.log" 2>&1 &
SAMPLER=$!
if [ "$ARM" = apa ]; then
  N=0
  for i in $(seq 1 36); do
    N=$(kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="AbleToScale")].status}{"\n"}{end}' | grep -c True || true)
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
  FIRST=$(BLT wait-first-arr --redis "$REDIS_HOST" --since-ms "$(cat "$D/load_start_redis_ms")" --models dsqwen-7b,dsllama-8b,dsqwen-14b --timeout-s 300)
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
python3 "$TOOLS/snap.py" "$D/snap_after.json" "$SM" >> "$D/runner.log" 2>&1
kubectl -n tre-v2 logs deploy/tre-v2-controller --since-time="$START_ISO" > "$D/controller.log" 2>&1 || true
kubectl -n tre-v2 logs deploy/tre-v2-service-manager --since-time="$START_ISO" > "$D/sm.log" 2>&1 || true
kubectl -n tre-v2 logs deploy/tre-gateway-plugins --since-time="$START_ISO" > "$D/gateway-plugins.log" 2>&1 || true
mkdir -p "$D/sidecar"
for P in $(kubectl -n default get pods -l tre.aibrix.io/managed=true -o name); do
  kubectl -n default logs "$P" -c tre-reissue-sidecar --since-time="$START_ISO" 2>/dev/null > "$D/sidecar/${P#pod/}.log" || true
done
grep -l -i "too many open files" "$D"/sidecar/*.log > "$D/EMFILE_PODS" 2>/dev/null || true
kubectl -n aibrix-system logs deploy/aibrix-controller-manager --since-time="$START_ISO" 2>/dev/null \
  | grep -i -E "podautoscal|apa|scale|dsqwen|dsllama" > "$D/aibrix-controller-manager.log" || true
kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o yaml > "$D/apa_crs_after.yaml" 2>&1 || true
python3 - "$D" "$T_RESTART" <<'PY'
import json, sys, redis, subprocess
d, t0 = sys.argv[1], int(sys.argv[2]) * 1000
r = redis.Redis(host=subprocess.check_output(["kubectl","-n","tre-v2","get","svc","tre-v2-redis","-o","jsonpath={.spec.clusterIP}"],text=True).strip(), port=6379, decode_responses=True)
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
  log "bl streams: $(BLT dump-streams --redis "$REDIS_HOST" --since-ms "$(cat "$D/bl_enabled_redis_ms")" --out-dir "$D/baseline" --models dsqwen-7b,dsllama-8b,dsqwen-14b 2>&1)"
fi
( cd "$TRE/deploy" && PYTHONPATH="$TRE/deploy:$TRE/common" python3 -m scripts.analysis.safescale_summary "$D/safescale.json" > "$D/safescale_summary.json" 2>&1 ) || true
log "collected; awake at end: $(awake_set); EMFILE pods: $(wc -l < "$D/EMFILE_PODS")"
reset_baseline
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$D/runner.log" 2>&1
log "restore: APA CRs live: $(kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o name | wc -l)"
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
