# TRE v2 Experiments Log (N5)

Per endgame plan §6. One entry per R-item run, appended in execution order.
Dependency order: R1 → R3 → R7 → R2 → R4 → R5 (R6 any gap).
All experiment output goes to 76 **local disk** (`/root/tre-experiments/`,
`/root/tre-n4b-soak/`), not NFS; only this log + final summary tables live in git.

> Status 2026-07-06: N5 is GATED on F4 (ADR-0008 isolated data plane) reaching a
> stable cutover (Phase B) so experiments run on the reproducible, traffic-isolated
> TRE data plane. Entries below are added as each R-item executes.

Entry template:

```markdown
## R<x> <name>  <YYYY-MM-DD>
- System version: git <sha> / images <digest list> / traceset <tag>
- Command(s) and params: <verbatim>
- Output dir: <path on 76 local disk>
- Result summary: <key numbers — oracle-normalized score, SLO viol %, switches, avg awake replicas>
- Anomalies / handling: <none | ...>
```

---

## Pending R-items (scaffold)

| # | Name | Depends on | Est | Driver |
|---|---|---|---|---|
| R1 | Old-system baseline (V_baseline, V_static) | isolated plane stable | ~2h+切换 | old system run_experiment.sh (secondary/prior-work) |
| R3 | Real re-fit (theta_m + capacity面), 2 percentile口径 | isolated plane | ~10h/model ×3 | `tre/deploy/scripts/r3_grid.py` + calibration CLI |
| R7 | Trace regenerate + freeze (design/lint/oracle) | R3 capacity面 | ~1h | `tre_replayer` design/lint/oracle; tag `traceset-v1` |
| R2 | New-system 7-trace regression (bucket_upper) | R7 | ~8h | `tre_replayer orchestrate`; reset_between_traces.sh |
| R4 | interpolated口径 re-run | R2 | ~8h | overlays/ablation-interpolated |
| R5 | Ablation matrix (no-fastloop / no-safescale / queue_len) × 3 traces | R4 | ~6h | overlays/ablation-* |
| R6 | Replayer timing precision (real vs dry-run) | any gap | ~0.5h | replayer dual-mode |

N5 gate: this log has R1–R7 entries, each reproducible; `git tag results-v1`;
main comparison table {old, new bucket_upper, interpolated, ablation arms} × oracle-normalized score.

## R3 grid driver — built + validated  2026-07-06
- Tool: `tre/deploy/scripts/r3_grid.py` (reuses MetricsStore window aggregation +
  TRSComputer for the trs column; per-cell checkpoint/resume; make check 289 with
  4 unit tests in `deploy/tests/test_r3_grid.py`).
- End-to-end smoke (1 cell, dsqwen-7b, 40s @ concurrency 4, direct model Service
  10.105.5.99:8000, controller paused): produced a calibration-ready CSV row —
  prompt_tokens=16512, gen_tokens=8192, p95_ttft=60ms, p95_tpot=25ms, **trs=4756.48**
  (Z_m~=6.4, healthy/high under load — the signal behaves correctly under real load;
  the earlier idle-critical was the zero-load degenerate case).
- Pipeline validated: drive -> tre-gateway-plugins scrape -> tre-v2-redis ->
  MetricsStore -> TRSComputer -> CSV. R3 can now run the full grid (i x o x concurrency,
  ~10h/model) with this driver; output feeds `calibration fit --input <csv> --signal trs`.
- R3 full-run reminder: recreate the fleet at 0.85 (ADR-0010) first for a consistent
  capacity baseline; drive each model's Service directly (bypass gateway) per plan 6.2.

## R3 dsqwen-7b grid — LAUNCHED (running)  2026-07-06
- System: git HEAD (isolated plane, SM fix a1d21c00, dsqwen-7b awake=1 @ util 0.9 live);
  traceset n/a (calibration grid).
- Command: `r3_grid.py --model dsqwen-7b --gateway-url http://10.105.5.99:8000/v1/completions
  --input-buckets 128,512,1024 --output-buckets 128,512 --concurrency 1,2,4,8,16,32
  --cell-seconds 300 --window-ms 60000 --metrics-schema v1` (controller paused; direct
  model Service per plan 6.2).
- Output: `/root/tre-experiments/r3/dsqwen-7b_util09.csv` (local disk, checkpointed).
- Baseline: util 0.9 (LIVE operational awake pod). registry 0.85 (ADR-0010) is
  recreate-hygiene only and does not affect r3_grid (reads trs params, not util).
- Est ~3h (36 cells x 300s). Feeds `calibration fit --input <csv> --signal trs`.
- Status: LAUNCHED PID 1008477. Result + theta fit to follow.
