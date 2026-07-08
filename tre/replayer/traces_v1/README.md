# traceset-v1 — experiment-3 (TRE vs APA) frozen trace set

Seven capacity-calibrated traces covering the six mechanism axes A1..A6 of
REFACTOR_PLAN.md section 12.4 (A2 has a medium + a tight variant). Generated from the R3
single-pod capacity surfaces by `tre_replayer.gen_traces`. This directory is the frozen,
git-tagged (`traceset-v1`) input to `run_comparison.py`; per section 12.7 it must not be
edited in place — iterate on scratch traces and cut a new version instead.

## Generation command

```bash
cd tre
PYTHONPATH=common:controller:service-manager:replayer:deploy:calibration   python -m tre_replayer.gen_traces     --capacity capacity/capacity_dsqwen-7b.json     --capacity capacity/capacity_dsllama-8b.json     --capacity capacity/capacity_dsqwen-14b.json     --out-dir replayer/traces_v1/
```

## Capacity source

R3 recalibration (2026-07-09), single-pod SLO-safe capacity surfaces C_m(i,o), archived
under `capacity/`. Origin: /root/tre-experiments/capacity_dsqwen-7b.json,
capacity_dsllama-8b.json, capacity_dsqwen-14b.json (post cap-supplement; 7b C(512,o) monotone
in o). theta_m at freeze: dsqwen-7b 715.27 / dsllama-8b 209.65 / dsqwen-14b 247.42.
Cluster: 2x4xA100(40G) = 8 GPU slots; slot widths dsqwen-7b/dsllama-8b=1 (tp1),
dsqwen-14b=2 (tp2).

## Traces (axis / headroom tier / peak occupancy / mechanism)

| trace | axis | tier | peak H | lint |
| --- | --- | --- | --- | --- |
| t1_a1_demand_shift | A1 demand-shift speed | tight 0.90 | 0.90 | PASS |
| t2_a2_anticorrelated | A2 inter-model anti-correlation | medium 0.75 | 0.75 | PASS |
| t3_a3_io_drift | A3 i/o mix drift (metric superiority) | medium 0.75 | 0.75 | PASS |
| t4_a4_spike_vs_burst | A4 spike vs burst | medium 0.75 | 0.75 | PASS |
| t5_a5_tp_pressure | A5 TP-heterogeneous pressure | tight 0.90 | 0.90 | PASS |
| t6_a6_control | A6 fairness control | loose 0.575 | 0.575 | C2 waived (by design) |
| t7_a2b_anticorrelated_hot | A2 (tight variant) | tight 0.90 | 0.90 | PASS |

## Lint (`lint_report.json`)

Constraints per section 12.3: C1 feasibility (oracle violation < 1% AND peak headroom
<= 0.95), C2 non-triviality (some model rho > 1.2 for >= 3 slow loops = 30s), C3 headroom
tier (peak H within +/-0.05 of the declared loose/medium/tight target).

**6 of 7 pass all three constraints.** The one exception is by design:

- **t6_a6_control fails C2** intentionally. A6 is the fairness/control arm whose entire
  purpose is a load every system handles without scaling; it deliberately keeps every model
  rho <= 1.2 (unit-tested invariant `test_a6_control_stays_below_non_triviality_threshold`),
  so the C2 non-triviality guard — which asks the opposite — does not apply to it. C2 is
  waived for A6; C1 and C3 still hold.

A3's peak headroom is capacity-derived (`headroom_is_capacity_dependent: true`). A3_RPS_MULT
was calibrated to 4.8 so the drift's peak occupancy (7b at the heaviest output + the constant
0.4-rho 8b/14b floor = 4.8 + 1.2 = 6.0 slots) lands on the medium tier (6.0/8 = 0.75). RPS is
held constant across the drift while output grows 128->512, so the rate/KVCache signal lags
while TSS's weighted throughput tracks the rising decode load — the A3 metric-superiority
scenario, now entirely in the saturated (rho > 1) regime.

`capacity_low_confidence: true` on every row reflects the sparse R3 grid (few i/o points per
model), not a lint failure.
