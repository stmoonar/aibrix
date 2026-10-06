#!/usr/bin/env bash
# Pilot sequence: traces x arms, run one after another. CHANGES CLUSTER STATE.
# Usage: PILOT_ROOT=... nohup bash tools/run_pilot.sh > $PILOT_ROOT/pilot.out 2>&1 &
#
# PLAN = space-separated items  <TRACE>:<arm>,<arm>[:<run_tag>]
#   default (batch 1, 2026-10-05): Alternating_hot_model_periodic_A, TRE then APA.
#   output: $PILOT_ROOT/<TRACE>/<arm>[-<run_tag>]; an existing result dir is never overwritten
#   (run_arm_pilot.sh refuses), so a repeat of the same trace/arm needs a run_tag.
# ABBA later (spreads drift over the order): run the reverse pair with a tag, e.g.
#   PLAN="Alternating_hot_model_periodic_A:apa,tre:r2"              # B A  -> .../apa-r2, .../tre-r2
#   PLAN="Sinusoidal_demand:apa,tre Decode_heavy_burst:tre,apa"     # the other two batch-1 traces
#   (old default of the draft: Alternating:tre,apa Sinusoidal_demand:apa,tre Decode_heavy_burst:tre,apa)
# Baseline arms (2026-10-06): arms chiron | tokenscale | preserve are accepted too (run_arm_pilot.sh
#   change 9); the environment is passed through (GW_PARITY, BL_CM_FILE, BL_CM_APPLY, BL_SEED, MARK_AT), e.g.
#   PLAN="Alternating_hot_model_periodic_A:chiron,tokenscale,preserve Alternating_hot_model_periodic_A:tre:drift1"
# Stop after the current arm: touch $PILOT_ROOT/STOP_PILOT
set -euo pipefail
PILOT_ROOT="${PILOT_ROOT:-/data/nfs_shared_data/xxy/pilot-e1-20261005}"
export PILOT_ROOT
GAP_S="${GAP_S:-120}"     # extra idle between runs (engines already drained + 60 s warm-up inside run_arm)
PLAN="${PLAN:-Alternating_hot_model_periodic_A:tre,apa}"
echo "[$(date +%F' '%T)] pilot start PLAN=[$PLAN] GAP_S=$GAP_S"
first=1
for item in $PLAN; do
  IFS=: read -r NAME ARMS TAG <<< "$item"
  [ -n "$NAME" ] && [ -n "$ARMS" ] || { echo "bad PLAN item '$item' (want TRACE:arm,arm[:tag])"; exit 2; }
  for ARM in ${ARMS//,/ }; do
    [ -e "$PILOT_ROOT/STOP_PILOT" ] && { echo "[$(date +%T)] STOP_PILOT present; stopping"; exit 0; }
    [ "$first" = 1 ] || sleep "$GAP_S"
    first=0
    echo "[$(date +%F' '%T)] start $NAME/$ARM${TAG:+ tag=$TAG}"
    RC=0; RUN_TAG="$TAG" bash "$PILOT_ROOT/tools/run_arm_pilot.sh" "$ARM" "$NAME" || RC=$?
    [ "$RC" = 0 ] || { echo "[$(date +%F' '%T)] FAILED $NAME/$ARM rc=$RC (cluster may be mid-arm: see RUN.md 'stop')"; exit 1; }
    echo "[$(date +%F' '%T)] done $NAME/$ARM"
  done
done
echo "[$(date +%F' '%T)] pilot done"
