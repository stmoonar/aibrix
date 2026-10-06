#!/usr/bin/env bash
# Baseline-arm sanity S1-S4 (docs/baselines-methodology-20261005.md, "比较前的步骤" 2) for ONE arm,
# with a synthetic single-model load (tre_replayer.run_trace, replay profile: /v1/completions,
# stream, ignore_eos, in 492 / max_tokens 400 = the hot alt trace shape, x-tre-bl-in-tokens sent).
# Every part: dry-run shell first (TRE_BL_DRY_RUN=true), then the actuating shell; SM active,
# controller observe, TRE scaling off, 0 APA CRs, gateway event stream on. CHANGES CLUSTER STATE.
#
# Usage: baseline_sanity.sh <chiron|tokenscale|preserve> [PART ...]      (nohup it; log in $SD/sanity.log)
#   PARTS (default "S1 S23 S3B S4"):
#   S1   constant 0.5x / 1.0x / 1.5x one-replica capacity, LEVEL_S (300) each. dry: one run with the
#        three levels in sequence (DRY_FRAC of the length); act: three runs S1a/S1b/S1c from 1 replica.
#        (A dry shell never adds replicas, so above 1.0x - and after the S23 3x step - the one replica is
#        overloaded and its queue grows: those dry phases are informational, the dry gate uses S1 <= 1.0x,
#        the S2 scale-up decision, decisions present, no guard, no tick stall.)
#        PASS per level after min(SETTLE_S, 0.4 x level length) (so DRY_FRAC 0.3 has samples): target constant (<= MAX_CHANGES changes, no +-1 flip-flop), no
#        ratchet to the cap; tokenscale/preserve also target 1 / 1-2 / 2 (Chiron: constant only, the
#        expected value is ceil(busy_eff/theta), see last_inputs in result.json).
#   S23  1x (S23_PRE_S) -> S23_HIGH x (S23_HIGH_S) -> 1x (S23_LOW_S), 10 min. S2: first scale-up within
#        UP_DEADLINE_S of the step, time-to-wake logged. Chiron (methodology 10-06): S23_HIGH 4.0
#        (sum_q at 3 awake ~1.4 B, at 4 awake ~1.2 B: the target moves 3 -> 4 unambiguously; at 3x
#        sum_q ~0.9 B sat on the busy edge); S2 = decision within one tick of the evidence (first
#        tick with sum_q > busy0 x B_mean), wake within UP_DEADLINE_S, <= 1 direction reversal
#        over the plateau, < 2 % of the phase's requests continued / retried (aborted by a sleep). S3: first scale-down within DOWN_DEADLINE_S of
#        the drop (TokenScale immediate, Chiron next tick, PreServe once per window: <= 1 down per
#        window and tier2_below_t1 == 0).
#   S3B  3x (S3B_HIGH_S) then 1x (S3B_LOW_S) with the scraper BLOCKED from S3B_SWITCH_LEAD_S before
#        the drop: the actuating shell is restarted with TRE_BL_METRICS_PORT=$BLOCK_PORT (a closed
#        port: every pod scrape fails = unknown). PASS: no scale-down and awake never decreases
#        (unknown-hold). Shell-only fault, nothing else in the cluster changes. act only (a dry shell
#        never scaled up, nothing to hold). Event-stream-blocked variant: manual (RUN.md), because
#        switching the gateway stream mid-load restarts the plugin pod under load.
#   S4   MODEL at S4_M1 x and MODEL2 at S4_M2 x for S4_CONT_S (demand > free GPUs: the last GPU is
#        contended), then MODEL2 drops to S4_M2_REL x for S4_REL_S. Logged: SM refusals (409 or 200
#        with unfilled/refusals), backoff lines; PASS: no tick stall (no deadlock) and, once MODEL2
#        slept (SM state change), MODEL reaches its target within RELEASE_DEADLINE_S.
# Output: $SD/<part>/<dry|act>/{result.json,part_meta.json,trace.json,records.jsonl,baseline/...},
#         $SD/summary.json (per part pass/fail + key timings).  $SD = $SANITY_ROOT/<arm>${SANITY_TAG:+-tag}
set -euo pipefail
ARM="${1:-}"; shift || true
case "$ARM" in chiron|tokenscale|preserve) ;; *) echo "usage: $0 <chiron|tokenscale|preserve> [S1 S23 S3B S4]" >&2; exit 2;; esac
PARTS="${*:-${PARTS:-S1 S23 S3B S4}}"
PILOT_ROOT="${PILOT_ROOT:-/data/nfs_shared_data/xxy/pilot-e1-20261005}"
PILOT_ROUND="${PILOT_ROUND:-$(basename "$PILOT_ROOT")}"
TOOLS="${TOOLS:-$PILOT_ROOT/tools}"
SANITY_ROOT="${SANITY_ROOT:-$PILOT_ROOT/sanity}"
SD="$SANITY_ROOT/$ARM${SANITY_TAG:+-$SANITY_TAG}"
TRE="${TRE_DIR:-/data/nfs_shared_data/xxy/aibrix/tre}"
LG="${LOADGEN_TRE_DIR:-/data/nfs_shared_data/xxy/aibrix/tre}"       # replayer (client)
GW="${GW:-http://192.168.223.76:31094}"
MARKER="${MARKER:-/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW}"
REQUIRE_MARKER="${REQUIRE_MARKER:-1}"
BASELINE="${BASELINE:-dsqwen-7b/nscc-ds-4a100-node9/0 dsllama-8b/nscc-ds-4a100-node9/1 dsqwen-14b/nscc-ds-4a100-node10/0,1}"
GW_NS="${GW_NS:-tre-v2}"; GW_DEPLOY="${GW_DEPLOY:-tre-gateway-plugins}"; GW_SELECTOR="${GW_SELECTOR:-app=tre-gateway-plugins}"
BL_NS="${BL_NS:-tre-v2}"; BL_DEPLOY="${BL_DEPLOY:-tre-v2-baseline-scaler}"
MODEL="${MODEL:-dsqwen-7b}"; MODEL2="${MODEL2:-dsllama-8b}"
IN_TOK="${IN_TOK:-492}"; OUT_TOK="${OUT_TOK:-400}"
case "$ARM" in tokenscale) DEF_CAP=vb ;; *) DEF_CAP=mu ;; esac
CAP_SRC="${CAP_SRC:-$DEF_CAP}"                    # mu (PreServe mu_t) | vb (TokenScale V_b) | auto
CAP_RPS="${CAP_RPS:-}"; CAP_RPS2="${CAP_RPS2:-}"  # explicit 1.0x req/s for MODEL / MODEL2 (wins)
LEVEL_S="${LEVEL_S:-300}"; SETTLE_S="${SETTLE_S:-60}"; MAX_CHANGES="${MAX_CHANGES:-1}"
DRY_FIRST="${DRY_FIRST:-1}"; DRY_FRAC="${DRY_FRAC:-0.5}"; DRY_GATE="${DRY_GATE:-1}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-2048}"; TICK_S="${TICK_S:-2}"; WARM_S="${WARM_S:-20}"; POST_S="${POST_S:-20}"; BL_SEED="${BL_SEED:-0}"
S23_PRE_S="${S23_PRE_S:-120}"; S23_HIGH_S="${S23_HIGH_S:-240}"; S23_LOW_S="${S23_LOW_S:-240}"   # S23_HIGH: per arm below
S3B_HIGH_S="${S3B_HIGH_S:-180}"; S3B_LOW_S="${S3B_LOW_S:-240}"; S3B_SWITCH_LEAD_S="${S3B_SWITCH_LEAD_S:-45}"; BLOCK_PORT="${BLOCK_PORT:-9}"
S4_PRE_S="${S4_PRE_S:-60}"; S4_CONT_S="${S4_CONT_S:-240}"; S4_REL_S="${S4_REL_S:-150}"
S4_M1="${S4_M1:-3.5}"; S4_M2="${S4_M2:-2.5}"; S4_M2_REL="${S4_M2_REL:-0.3}"; RELEASE_DEADLINE_S="${RELEASE_DEADLINE_S:-30}"
PRESERVE_TRACE_HOST_ROOT="${PRESERVE_TRACE_HOST_ROOT:-}"   # host dir that the scaler pod mounts at PRESERVE_TRACE_POD_ROOT
PRESERVE_TRACE_POD_ROOT="${PRESERVE_TRACE_POD_ROOT:-/etc/tre-baselines-traces}"
PRESERVE_SANITY_WINDOW_S="${PRESERVE_SANITY_WINDOW_S:-60}"  # sanity only (paper/main runs: 600)
case "$ARM" in
  chiron)     UP_DEADLINE_S="${UP_DEADLINE_S:-10}"; DOWN_DEADLINE_S="${DOWN_DEADLINE_S:-20}"; S23_HIGH="${S23_HIGH:-4.0}" ;;
  tokenscale) UP_DEADLINE_S="${UP_DEADLINE_S:-25}"; DOWN_DEADLINE_S="${DOWN_DEADLINE_S:-25}" ;;
  preserve)   UP_DEADLINE_S="${UP_DEADLINE_S:-30}"; DOWN_DEADLINE_S="${DOWN_DEADLINE_S:-$((PRESERVE_SANITY_WINDOW_S + 30))}" ;;
esac
S23_HIGH="${S23_HIGH:-3.0}"

# ---- checks before anything touches the cluster
if [ -e "$MARKER" ]; then
  MK_TYPE=""; read -r MK_TYPE _ < "$MARKER" || true
  if [ "$MK_TYPE" != validation ] || ! awk -v want="round=$PILOT_ROUND" '{for (i = 1; i <= NF; i++) if ($i == want) f = 1} END {exit !f}' "$MARKER"; then
    echo "exclusive window marker is not this pilot's (want 'validation ... round=$PILOT_ROUND'): $(head -c 300 "$MARKER"); refusing" >&2; exit 2
  fi
elif [ "$REQUIRE_MARKER" = 1 ]; then
  echo "no exclusive window marker $MARKER (REQUIRE_MARKER=0 to skip); refusing" >&2; exit 2
fi
grep -q -- "--send-in-tokens" "$LG/replayer/tre_replayer/run_trace.py" || { echo "$LG replayer has no --send-in-tokens" >&2; exit 2; }
R=$(kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}')
[ "$R" = 0 ] || { echo "$BL_DEPLOY has $R replicas (an arm is running?); refusing" >&2; exit 2; }
SMIP=$(kubectl -n tre-v2 get svc tre-v2-service-manager -o jsonpath='{.spec.clusterIP}'); SM=http://$SMIP:8000; export SM_URL=$SM
REDIS_HOST=$(kubectl -n tre-v2 get svc tre-v2-redis -o jsonpath='{.spec.clusterIP}')
mkdir -p "$SD/policies"
LOGF="$SD/sanity.log"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGF"; }
BLT() { python3 "$TOOLS/bl_tools.py" "$@"; }
ARM_TOOL() { env PYTHONPATH="$TRE/common:$TRE/deploy:$TRE/baselines" python3 -m tre_baselines.tools.arm "$@" \
               --namespace "$BL_NS" --deployment "$BL_DEPLOY" --redis-url "redis://$REDIS_HOST:6379/0" \
               --gw-namespace "$GW_NS" --gw-selector "$GW_SELECTOR"; }
mode() { bash "$TRE/deploy/scripts/set_run_mode.sh" "$1" "$2" >/dev/null; bash "$TRE/deploy/scripts/set_run_mode.sh" status | tr '\n' ' '; }
awake_ids() { curl -s "$SM/v2/state" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(" ".join(sorted(b["binding_id"] for b in d["bindings"] if b["awake"])))'; }
wait_no_hidden() {
  for i in $(seq 1 60); do
    H=$(curl -s "$SM/v2/state" | python3 -c 'import sys,json;print(sum(b["hidden"] for b in json.load(sys.stdin)["bindings"]))')
    [ "$H" = 0 ] && return 0; sleep 5
  done; log "WARN hidden bindings remain after 300 s"; return 1
}
wait_engines_idle() {
  for i in $(seq 1 60); do
    BUSY=$(kubectl -n default get pods -l tre.aibrix.io/managed=true -o jsonpath='{range .items[*]}{.status.podIP}{"\n"}{end}' | while read -r IP; do
      [ -n "$IP" ] || continue
      curl -s --max-time 3 "http://$IP:8000/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}'
    done | awk '{s+=$1} END {print s+0}')
    [ "$BUSY" = 0 ] && return 0; sleep 5
  done; log "WARN engines still busy after 300 s"; return 1
}
reset_baseline() {
  log "reset: observe/observe -> $(mode observe observe)"
  wait_no_hidden || true
  # shellcheck disable=SC2086
  python3 "$TRE/deploy/scripts/release/awake_ctl.py" restore-ids $BASELINE >> "$LOGF" 2>&1 \
    || { log "ERROR restore-ids failed (two-step manual reset in RUN.md)"; exit 3; }
  local want got; want=$(echo $BASELINE | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//'); got=$(awake_ids)
  [ "$want" = "$got" ] || { log "ERROR awake set after restore = [$got], want [$want]"; exit 3; }
}
events_get() { kubectl -n "$GW_NS" get deploy "$GW_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TRE_BL_REQ_EVENTS")].value}'; }
events_set() {
  local want="$1" cur; cur=$(events_get)
  { [ "$want" = UNSET ] && [ -z "$cur" ]; } && return 0
  [ "$want" = "$cur" ] && return 0
  if [ "$want" = UNSET ]; then kubectl -n "$GW_NS" set env "deploy/$GW_DEPLOY" TRE_BL_REQ_EVENTS- >> "$LOGF" 2>&1
  else kubectl -n "$GW_NS" set env "deploy/$GW_DEPLOY" "TRE_BL_REQ_EVENTS=$want" >> "$LOGF" 2>&1; fi
  kubectl -n "$GW_NS" rollout status "deploy/$GW_DEPLOY" --timeout=180s >> "$LOGF" 2>&1
  log "gateway event stream: '${cur:-UNSET}' -> '$(events_get || true)'"
}
shell_env() { kubectl -n "$BL_NS" set env "deploy/$BL_DEPLOY" "$@" >> "$LOGF" 2>&1; }   # scaler at 0 replicas: template only
cm_preserve_set() {   # cm_preserve_set <file>: data["preserve.yaml"] = file content
  kubectl -n "$BL_NS" patch cm tre-v2-baseline-preserve --type merge \
    -p "$(python3 -c 'import json,sys;print(json.dumps({"data":{"preserve.yaml":open(sys.argv[1]).read()}}))' "$1")" >> "$LOGF" 2>&1
}
BL_ENABLED=0; EVENTS_CHANGED=0; EV_BEFORE=UNSET; CM_PATCHED=0; PORT_SET=0
on_exit() {
  local rc=$?
  [ "$BL_ENABLED" = 1 ] && { ARM_TOOL disable --skip-collect --execute >> "$LOGF" 2>&1 || log "TRAP ERROR: scale $BL_DEPLOY to 0 by hand"; }
  [ "$PORT_SET" = 1 ] && { shell_env TRE_BL_METRICS_PORT- || log "TRAP ERROR: remove TRE_BL_METRICS_PORT from $BL_DEPLOY by hand"; }
  [ "$CM_PATCHED" = 1 ] && { cm_preserve_set "$SD/policies/preserve.yaml" && log "TRAP: preserve ConfigMap restored" || log "TRAP ERROR: restore tre-v2-baseline-preserve from $SD/policies/preserve.yaml"; }
  [ "$EVENTS_CHANGED" = 1 ] && { events_set "$EV_BEFORE" || log "TRAP ERROR: gateway event stream restore"; }
  [ "$rc" = 0 ] || log "TRAP: sanity exited rc=$rc; cluster NOT reset (RUN.md 'stop')"
}
trap on_exit EXIT

log "=== sanity arm=$ARM parts=[$PARTS] model=$MODEL model2=$MODEL2 cap_src=$CAP_SRC dir=$SD"
git -C "$TRE" rev-parse HEAD > "$SD/tre_sha"; git -C "$LG" rev-parse HEAD > "$SD/client_sha"
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > "$SD/live-registry.yaml"
for P in chiron tokenscale preserve; do
  kubectl -n "$BL_NS" get cm "tre-v2-baseline-$P" -o jsonpath="{.data.$P\.yaml}" > "$SD/policies/$P.yaml"
done
BLT check-policy --policy "$ARM" --live "$SD/policies/$ARM.yaml" --registry "$SD/live-registry.yaml" > "$SD/policy_check.json" 2>&1 \
  || { log "ERROR policy ConfigMap: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["problems"])' "$SD/policy_check.json" 2>/dev/null || head -c 600 "$SD/policy_check.json")"; exit 2; }
CAP_ARGS=(); [ -n "$CAP_RPS" ] && CAP_ARGS=(--rps "$CAP_RPS")
CAP2_ARGS=(); [ -n "$CAP_RPS2" ] && CAP2_ARGS=(--rps "$CAP_RPS2")
C1=$(BLT capacity --policy-dir "$SD/policies" --model "$MODEL" --src "$CAP_SRC" --in "$IN_TOK" --out "$OUT_TOK" ${CAP_ARGS[@]+"${CAP_ARGS[@]}"})
C2=$(BLT capacity --policy-dir "$SD/policies" --model "$MODEL2" --src "$CAP_SRC" --in "$IN_TOK" --out "$OUT_TOK" ${CAP2_ARGS[@]+"${CAP2_ARGS[@]}"})
echo "{\"$MODEL\": $C1, \"$MODEL2\": $C2}" > "$SD/capacity.json"
log "1.0x capacity: $(tr '\n' ' ' < "$SD/capacity.json")"
CAP_AWAKE=$(python3 -c 'import yaml,sys;m=[x for x in yaml.safe_load(open(sys.argv[1]))["models"] if x["name"]==sys.argv[2]][0];print(m.get("max_awake_replicas") or m.get("max_replicas"))' "$SD/live-registry.yaml" "$MODEL")
PRESERVE_WIN=$(python3 -c 'import yaml,sys;print((yaml.safe_load(open(sys.argv[1])) or {}).get("window_s",600))' "$SD/policies/preserve.yaml")
[ "$ARM" = preserve ] && PRESERVE_WIN=$PRESERVE_SANITY_WINDOW_S

# ---- decision source NONE, event stream on, shell seed
N_APA=$(kubectl -n default get podautoscalers.autoscaling.aibrix.ai -o name 2>/dev/null | grep -c . || true)
[ "$N_APA" = 0 ] || bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$LOGF" 2>&1
kubectl -n tre-v2 set env deploy/tre-v2-controller ENABLE_TRE_SCALING=false >> "$LOGF" 2>&1
kubectl -n tre-v2 rollout status deploy/tre-v2-controller --timeout=180s >> "$LOGF" 2>&1
ST=$(bash "$TRE/deploy/scripts/toggle_tre_apa.sh" status --keep-run-mode 2>/dev/null | grep 'active decision source' || true)
case "$ST" in *NONE*) log "decision source: $ST" ;; *) log "ERROR decision source is not NONE: $ST"; exit 4;; esac
EV_BEFORE=$(events_get); EV_BEFORE=${EV_BEFORE:-UNSET}
[ "$EV_BEFORE" = true ] || EVENTS_CHANGED=1
events_set true
shell_env "TRE_BL_SEED=$BL_SEED"

# ---- one part run
# run_part <part-id> <dry|act> <duration_s> <phases-json> <seg>...   (segments model:start:end:mult)
run_part() {
  local PART="$1" MODE="$2" DUR="$3" PHASES="$4"; shift 4
  local PD="$SD/$PART/$MODE"
  [ -e "$PD/result.json" ] && { log "skip $PART/$MODE: result exists (SANITY_TAG for a repeat)"; return 0; }
  mkdir -p "$PD"
  log "--- $PART/$MODE (${DUR}s) segments: $*"
  local SEGS=(); local s; for s in "$@"; do SEGS+=(--seg "$s"); done
  BLT mktrace --out "$PD/trace.json" --cap-json "$SD/capacity.json" --in "$IN_TOK" --max "$OUT_TOK" "${SEGS[@]}" > /dev/null
  local MARK_TRACE="$PD/trace.json"
  reset_baseline; wait_engines_idle || true
  log "run mode -> $(mode observe active)"
  if [ "$ARM" = preserve ]; then   # sanity copy of the frozen params: shorter window (+ the synthetic trace when mountable)
    local TP=""
    if [ -n "$PRESERVE_TRACE_HOST_ROOT" ]; then
      mkdir -p "$PRESERVE_TRACE_HOST_ROOT/sanity-$ARM/$PART-$MODE"
      cp "$PD/trace.json" "$PRESERVE_TRACE_HOST_ROOT/sanity-$ARM/$PART-$MODE/trace.json"
      TP="$PRESERVE_TRACE_POD_ROOT/sanity-$ARM/$PART-$MODE/trace.json"; MARK_TRACE="$PRESERVE_TRACE_HOST_ROOT/sanity-$ARM/$PART-$MODE/trace.json"
    fi
    python3 -c 'import yaml,sys
p=yaml.safe_load(open(sys.argv[1])) or {}
p["window_s"]=float(sys.argv[2])
if sys.argv[3]: p["trace_path"]=sys.argv[3]; p["trace_seed"]=int(sys.argv[4])
yaml.safe_dump(p, open(sys.argv[5],"w"), sort_keys=False)' "$SD/policies/preserve.yaml" "$PRESERVE_SANITY_WINDOW_S" "$TP" "$BL_SEED" "$PD/preserve.sanity.yaml"
    CM_PATCHED=1; cm_preserve_set "$PD/preserve.sanity.yaml"
    log "preserve sanity params: window_s=$PRESERVE_SANITY_WINDOW_S trace_path=${TP:-<frozen, Tier-1 inactive for the synthetic load>}"
  fi
  BLT del-key --redis "$REDIS_HOST" --key tre:v2:bl:replay_t0 >> "$LOGF" 2>&1
  local DRYF=(); [ "$MODE" = dry ] && DRYF=(--dry-run-shell)
  BLT redis-ms --redis "$REDIS_HOST" > "$PD/enabled_redis_ms"
  BL_ENABLED=1
  ARM_TOOL enable --policy "$ARM" ${DRYF[@]+"${DRYF[@]}"} --execute --timeout-s 180 >> "$LOGF" 2>&1 || { log "ERROR arm enable"; exit 4; }
  sleep "$WARM_S"
  local T0; T0=$(BLT redis-ms --redis "$REDIS_HOST")
  ( cd "$LG" && PYTHONPATH="$LG/common:$LG/deploy:$LG/replayer" python3 -m tre_replayer.run_trace --trace "$PD/trace.json" \
      --gateway-url "$GW/v1/completions" --out "$PD/records.jsonl" --registry "$SD/live-registry.yaml" --seed "$BL_SEED" \
      --prompt-file "$PD/prompts.jsonl" --rps-timeline "$PD/rps.csv" --max-in-flight "$MAX_IN_FLIGHT" --send-in-tokens ) > "$PD/replayer.log" 2>&1 &
  local RPID=$!
  local FIRST; FIRST=$(BLT wait-first-arr --redis "$REDIS_HOST" --since-ms "$T0" --models "$MODEL,$MODEL2" --timeout-s 300 || true)
  [ -n "$FIRST" ] || { log "WARN no arr event within 300 s (event stream off? replayer failed?)"; FIRST=$(BLT redis-ms --redis "$REDIS_HOST"); }
  ARM_TOOL mark-replay --trace "$MARK_TRACE" --seed "$BL_SEED" --execute >> "$LOGF" 2>&1 || log "WARN mark-replay failed"
  if [ "$PART" = S3B ] && [ "$MODE" = act ]; then   # restart the actuating shell with the scraper blocked
    local AT=$(( FIRST + (S3B_HIGH_S - S3B_SWITCH_LEAD_S) * 1000 ))
    while [ "$(BLT redis-ms --redis "$REDIS_HOST")" -lt "$AT" ]; do sleep 1; done
    log "S3B: blocking the scraper (TRE_BL_METRICS_PORT=$BLOCK_PORT): shell restart under load"
    ARM_TOOL disable --collect-dir "$PD/baseline-pre" --execute >> "$LOGF" 2>&1 || { log "ERROR arm disable (pre)"; exit 4; }
    BL_ENABLED=0; PORT_SET=1; shell_env "TRE_BL_METRICS_PORT=$BLOCK_PORT"
    BL_ENABLED=1; ARM_TOOL enable --policy "$ARM" --execute --timeout-s 180 >> "$LOGF" 2>&1 || { log "ERROR arm enable (blocked)"; exit 4; }
    BLT redis-ms --redis "$REDIS_HOST" > "$PD/blocked_redis_ms"
    log "S3B: blocked shell up $(( ($(cat "$PD/blocked_redis_ms") - FIRST) / 1000 )) s after the first send (drop at ${S3B_HIGH_S}s)"
  fi
  wait $RPID || log "WARN replayer rc=$? (see replayer.log)"
  sleep "$POST_S"
  kubectl -n "$BL_NS" logs "deploy/$BL_DEPLOY" > "$PD/scaler.log" 2>&1 || true
  ARM_TOOL disable --collect-dir "$PD/baseline" --client-sent-in-tokens --execute >> "$LOGF" 2>&1 || { log "ERROR arm disable"; exit 4; }
  BL_ENABLED=0
  if [ "$PORT_SET" = 1 ]; then shell_env TRE_BL_METRICS_PORT-; PORT_SET=0; fi
  if [ "$CM_PATCHED" = 1 ]; then cm_preserve_set "$SD/policies/preserve.yaml"; CM_PATCHED=0; fi
  BLT dump-streams --redis "$REDIS_HOST" --since-ms "$(cat "$PD/enabled_redis_ms")" --out-dir "$PD/baseline" --models "$MODEL,$MODEL2" >> "$LOGF" 2>&1 || true
  local T0R; T0R=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1]));print(((v.get("replay_marker") or {}).get("t0_ms")) or "")' "$PD/baseline/run_validity.json" 2>/dev/null || true)
  local PH="$PHASES"
  if [ "$PART" = S3B ] && [ -f "$PD/blocked_redis_ms" ]; then   # the hold phase starts once the blocked shell runs
    PH=$(python3 -c 'import json,sys;p=json.loads(sys.argv[1]);b=(int(sys.argv[2])-int(sys.argv[3]))/1000.0
for x in p:
    if x.get("check")=="hold": x["start_s"]=max(x["start_s"], b)
print(json.dumps(p))' "$PHASES" "$(cat "$PD/blocked_redis_ms")" "$FIRST")
  fi
  python3 -c 'import json,sys
a=sys.argv
json.dump(dict(part=a[1], policy=a[2], mode=a[3], t_load_redis_ms=int(a[4]), tick_s=float(a[5]), settle_s=float(a[6]),
  max_changes=int(a[7]), cap_awake=int(a[8]), replay_t0_ms=(int(a[9]) if a[9] else None), phases=json.loads(a[10]),
  model=a[11], model2=a[12], capacity=json.load(open(a[13]))), open(a[14],"w"), indent=1)' \
    "$PART" "$ARM" "$MODE" "$FIRST" "$TICK_S" "$SETTLE_S" "$MAX_CHANGES" "$CAP_AWAKE" "$T0R" "$PH" "$MODEL" "$MODEL2" "$SD/capacity.json" "$PD/part_meta.json"
  local DIRS=("$PD/baseline"); [ -d "$PD/baseline-pre" ] && DIRS+=("$PD/baseline-pre")
  log "$PART/$MODE: $(BLT analyze --part "$PART" --policy "$ARM" --mode "$MODE" --decisions "${DIRS[@]}" \
        --meta "$PD/part_meta.json" --validity "$PD/baseline/run_validity.json" --records "$PD/records.jsonl" --out "$PD/result.json" 2>&1 | tail -1)"
}
part_pass() { python3 -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$SD/$1/$2/result.json" 2>/dev/null; }
fdur() { python3 -c "print(int(round(float('$1')*float('$2'))))"; }

# ---- parts
for PART in $PARTS; do
  case "$PART" in
  S1)
    if [ "$ARM" = chiron ]; then R05=null; R10=null; R15=null; else R05='[1,1]'; R10='[1,2]'; R15='[2,2]'; fi
    if [ "$DRY_FIRST" = 1 ]; then
      L=$(fdur "$LEVEL_S" "$DRY_FRAC")
      run_part S1 dry $((3*L)) "[{\"name\":\"0.5x\",\"model\":\"$MODEL\",\"mult\":0.5,\"start_s\":0,\"end_s\":$L,\"check\":\"steady\",\"expect_range\":$R05},
        {\"name\":\"1.0x\",\"model\":\"$MODEL\",\"mult\":1.0,\"start_s\":$L,\"end_s\":$((2*L)),\"check\":\"steady\",\"expect_range\":$R10},
        {\"name\":\"1.5x\",\"model\":\"$MODEL\",\"mult\":1.5,\"start_s\":$((2*L)),\"end_s\":$((3*L)),\"check\":\"steady\",\"expect_range\":$R15}]" \
        "$MODEL:0:$L:0.5" "$MODEL:$L:$((2*L)):1.0" "$MODEL:$((2*L)):$((3*L)):1.5"
      if [ "$DRY_GATE" = 1 ] && ! part_pass S1 dry; then log "S1 dry FAILED: actuating S1 skipped (DRY_GATE=0 to force)"; continue; fi
    fi
    for LV in "a 0.5 $R05" "b 1.0 $R10" "c 1.5 $R15"; do
      read -r SUF MULT RNG <<< "$LV"
      run_part "S1$SUF" act "$LEVEL_S" "[{\"name\":\"${MULT}x\",\"model\":\"$MODEL\",\"mult\":$MULT,\"start_s\":0,\"end_s\":$LEVEL_S,\"check\":\"steady\",\"expect_range\":$RNG}]" \
        "$MODEL:0:$LEVEL_S:$MULT"
    done ;;
  S23)
    for MODE in dry act; do
      [ "$MODE" = dry ] && [ "$DRY_FIRST" != 1 ] && continue
      F=1; [ "$MODE" = dry ] && F=$DRY_FRAC
      A=$(fdur "$S23_PRE_S" "$F"); B=$(( A + $(fdur "$S23_HIGH_S" "$F") )); C=$(( B + $(fdur "$S23_LOW_S" "$F") ))
      WIN=null; [ "$ARM" = preserve ] && WIN=$PRESERVE_WIN
      run_part S23 "$MODE" "$C" "[{\"name\":\"pre-1x\",\"model\":\"$MODEL\",\"mult\":1.0,\"start_s\":0,\"end_s\":$A},
        {\"name\":\"S2-up-${S23_HIGH}x\",\"model\":\"$MODEL\",\"mult\":$S23_HIGH,\"start_s\":$A,\"end_s\":$B,\"check\":\"up\",\"deadline_s\":$UP_DEADLINE_S},
        {\"name\":\"S3-down-1x\",\"model\":\"$MODEL\",\"mult\":1.0,\"start_s\":$B,\"end_s\":$C,\"check\":\"down\",\"deadline_s\":$DOWN_DEADLINE_S,\"once_per_window_s\":$WIN}]" \
        "$MODEL:0:$A:1.0" "$MODEL:$A:$B:$S23_HIGH" "$MODEL:$B:$C:1.0"
      if [ "$MODE" = dry ] && [ "$DRY_GATE" = 1 ] && ! part_pass S23 dry; then log "S23 dry FAILED: actuating S23 skipped"; break; fi
    done ;;
  S3B)
    A=$S3B_HIGH_S; C=$(( S3B_HIGH_S + S3B_LOW_S ))
    run_part S3B act "$C" "[{\"name\":\"3x\",\"model\":\"$MODEL\",\"mult\":3.0,\"start_s\":0,\"end_s\":$A},
      {\"name\":\"S3B-hold-1x-scrape-blocked\",\"model\":\"$MODEL\",\"mult\":1.0,\"start_s\":$A,\"end_s\":$C,\"check\":\"hold\"}]" \
      "$MODEL:0:$A:3.0" "$MODEL:$A:$C:1.0" ;;
  S4)
    for MODE in dry act; do
      [ "$MODE" = dry ] && [ "$DRY_FIRST" != 1 ] && continue
      F=1; [ "$MODE" = dry ] && F=$DRY_FRAC
      A=$(fdur "$S4_PRE_S" "$F"); B=$(( A + $(fdur "$S4_CONT_S" "$F") )); C=$(( B + $(fdur "$S4_REL_S" "$F") ))
      run_part S4 "$MODE" "$C" "[{\"name\":\"contend\",\"model\":\"$MODEL\",\"mult\":$S4_M1,\"start_s\":$A,\"end_s\":$B,\"check\":\"contend\"},
        {\"name\":\"release\",\"model\":\"$MODEL\",\"donor\":\"$MODEL2\",\"mult\":$S4_M1,\"start_s\":$B,\"end_s\":$C,\"check\":\"release\",\"deadline_s\":$RELEASE_DEADLINE_S}]" \
        "$MODEL:0:$A:1.0" "$MODEL:$A:$C:$S4_M1" "$MODEL2:0:$A:1.0" "$MODEL2:$A:$B:$S4_M2" "$MODEL2:$B:$C:$S4_M2_REL"
      if [ "$MODE" = dry ] && [ "$DRY_GATE" = 1 ] && ! part_pass S4 dry; then log "S4 dry FAILED: actuating S4 skipped"; break; fi
    done ;;
  *) log "unknown part $PART (S1 S23 S3B S4)";;
  esac
done

# ---- restore (what run_arm_pilot.sh leaves) + summary
reset_baseline
bash "$TRE/deploy/scripts/toggle_tre_apa.sh" tre --keep-run-mode >> "$LOGF" 2>&1
log "restore: run mode -> $(mode observe active)"
if [ "$EVENTS_CHANGED" = 1 ]; then events_set "$EV_BEFORE"; EVENTS_CHANGED=0; fi
log "restore: $BL_DEPLOY replicas $(kubectl -n "$BL_NS" get deploy "$BL_DEPLOY" -o jsonpath='{.spec.replicas}'), event stream '$(events_get)'"
log "summary: $(BLT summary --root "$SD" --arm "$ARM" --out "$SD/summary.json")"
log "=== sanity arm=$ARM done"
