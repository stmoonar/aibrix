# Experiment 3 — TRE vs APA full 7×2 comparison (traceset-v2)

Evidence bundle for the collect/analysis pass of experiment 3. Small artifacts are in-repo;
the large per-request JSONL (~8.5 MB each × 14) stays on the run host.

## Contents (this dir)

- `final_report.json` — all numbers (per-run per-model scoring, system-level, oracle-norm, behavior, metadata).
- `final_report.md` — human-readable: comparison table + per-model p95/503 + oracle-norm + behavior analysis + conclusions.
- `summary_<arm>_<t>.json` — compact per-run summary (14 files): counts, V_sys, 503 rate, per-model p95, behavior (peak awake / first scale-up / final awake).
- `timeline_tre.csv`, `timeline_apa.csv` — per-arm 15s controller samples (awake dist + actions). NB: the `awake` column is faithful for TRE only; for APA it is the TRE-observer view (1/1/1 + `NA`) and does NOT reflect APA replica count (APA scales AIBrix PodAutoscaler CRs on separate anchor deployments).

## Large files (on run host 192.168.223.76)

- `/root/tre-experiments/comparison_v2/{tre,apa}/{t1..t7}/requests.jsonl` — authoritative per-request records (all scoring derives from these).
- `/root/tre-experiments/comparison_v2/{tre,apa}/{t1..t7}/run.log` — run_trace self-summary (per-model V_sys); recomputed values match exactly.
- `/root/tre-experiments/comparison_v2/{tre,apa}/{t1,t5,t7}/envoy_5xx_permodel_minute.csv` — supplementary per-minute 5xx (partial captures; JSONL is authoritative).
- `/root/tre-experiments/comparison_v2/orchestrator.log`, `STATUS` — run driver log + DONE marker.

## Provenance

- **Traceset**: traceset-v2, commit `04de30a1` (integer-feasible GPU occupancy, total_slots=8; dsqwen-7b=1, dsllama-8b=1, dsqwen-14b=2). Design/feasibility: `tre/replayer/traces_v2/INDEX.json`.
- **Three fixes** that let all 14 runs pass the infra gate (UF=0, SOCKFAIL=0):
  1. `bde7376f` (+ roll `f2699d84`) — controller: suppress proactive SafeScale probe on hot HIGH donors (t1 hot-proactive guard).
  2. `04de30a1` — replayer: traceset-v2 with integer-feasible GPU occupancy.
  3. `e017cffb` — gateway/envoy: harden against upstream socket exhaustion → zero UF (t1 503 fix).
- **SLO** (registry): ttft_p95 500 ms, tpot_p95 75 ms, e2e_p95 12000 ms (7b/llama) / 15000 ms (14b).
- **Scoring**: `tre_replayer.scoring.compute_v_sys` (30s window / 5s step / min_samples 5).
- **Status**: DONE 2026-07-09T01:49:47Z. Terminal state: controller=observe, signal_source=zm, 0 PodAutoscaler CRs, routable/awake 1/1/1, console 200.
