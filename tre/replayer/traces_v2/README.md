# traceset-v2 — experiment-3 (TRE vs APA) frozen trace set

Nine workloads: seven capacity-calibrated synthetic traces covering the six mechanism axes
A1..A6 of REFACTOR_PLAN.md section 12.4 (A2 has a medium + a tight variant), plus the two
production-derived workloads t8/t9 required by the 2026-07-10 architecture and experiment
plan. The seven synthetic traces were generated from the R3 single-pod capacity surfaces by
`tre_replayer.gen_traces` and remain byte-unchanged from the frozen `traceset-v2` tag.

## Why v2 exists: v1's occupancy口径 was physically infeasible

traceset-v1 sized every trace on **fractional** occupancy `Σ_m rho_m · slot_width_m / 8`.
That is a physics bug. Real deployments run an **integer** number of pods, every model that
receives traffic needs **≥1 awake replica** (serving floor), and `dsqwen-14b` is **tp2**
(2 GPU per replica). The true GPU requirement of a phase is

```
integer_gpu_occupancy = Σ_m ceil(rho_m) · slot_width_m      (ceil ≥ 1 for any rho_m > 0)
```

Under that (correct)口径, three of v1's seven traces were **physically unservable** — no
controller, however perfect, could have kept them off the overload floor. This was a root
cause of the **t1 TRE-arm 503 storm** (the TRE arm was asked to fit 9 GPU of work into 8).

| trace | tier (v1) | v1 rho(hot) | v1 fractional | **v1 integer GPU** | feasible? | v2 rho(hot) | **v2 integer GPU** | v2 fractional |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_a1_demand_shift | tight | 6.0 | 7.2 (0.90) | **9** | ❌ INFEASIBLE | 4.8 | **8** (1.00) | 6.0 |
| t2_a2_anticorrelated | medium | 4.4 | 6.0 (0.75) | 8 | ok but really *tight* | 2.8 | **6** (0.75) | 4.4 |
| t3_a3_io_drift | medium | 4.8 | 6.0 (0.75) | 8 | ok but really *tight* | 2.8 | **6** (0.75) | 4.0 |
| t4_a4_spike_vs_burst | medium | 4.8 | 6.0 (0.75) | 8 | ok but really *tight* | 2.8 | **6** (0.75) | 4.0 |
| t5_a5_tp_pressure | tight | 3.1 (14b) | 7.2 (0.90) | **10** | ❌ INFEASIBLE | 2.8 (14b) | **8** (1.00) | 6.6 |
| t6_a6_control | loose | 1.15 | 4.6 (0.575) | 8 | *full cluster* — not loose | 0.95 | **4** (0.50) | 3.8 |
| t7_a2b_anticorrelated_hot | tight | 5.6 | 7.2 (0.90) | **9** | ❌ INFEASIBLE | 4.8 | **8** (1.00) | 6.4 |

v1's A6 "control" (declared *loose* 0.575) actually demanded the **entire 8-GPU cluster**
once every model crossed rho 1.0 (ceil 2 each, ×2 for the tp2 14b): a fairness control that
silently saturated the cluster. v2 makes it genuinely loose.

## What v2 changes

1. **Integer feasibility口径.** Occupancy and the headroom tiers are now the *integer* GPU
   requirement `/8`:

   | tier | integer GPU | headroom | meaning |
   | --- | --- | --- | --- |
   | loose | 4/8 = 0.50 | 50% | resting serving floor: one replica per model (14b = 2) |
   | medium | 6/8 = 0.75 | 25% | 25% GPU headroom |
   | tight | 8/8 = 1.00 | 0% | hugging the physical limit, still integer-feasible |

   Design rho were re-tuned so `Σ ceil(rho)·width ≤ 8` in **every phase**. `gen_traces.assert_feasible`
   enforces this at generation time and lint's **C1** now checks the same integer occupancy
   (`≤ 1.0`); `lint._HEADROOM_TARGETS` moved to the integer basis above.

2. **Measured-grid token shapes.** v1 sent 8b/14b saturation at `(512,256)`, which is *not*
   a measured grid point for those models — nearest-neighbour borrowed the lighter `(512,128)`
   capacity and so **over-estimated** how much real load a pod could take (another t1 failure
   mode). v2 pins every shape to a measured grid point: baseline `(128,128)` and saturation
   `(512,512)` are measured for all three models; the A3 7b output ladder
   `(512,{128,256,384,512})` is measured for dsqwen-7b. Result: **every lint row now reports
   `capacity_low_confidence: false`** (v1 was `true` everywhere).

3. **INDEX feasibility proof.** Each design in `INDEX.json` carries `integer_gpu_occupancy`,
   `peak_integer_headroom`, the reference `peak_fractional_occupancy_slots`, and a
   `feasibility` block — a per-phase, per-model replica-requirement table
   (`{rho, replicas, slot_width, gpu}` + `gpu_total`) proving `≤ 8` at all times.

## Production-derived extension (t8/t9)

A4 explicitly requires t8/t9 to live beside t1-t7 so `run_comparison.py` discovers one
canonical trace root. `INDEX.json` therefore lists them under `workloads` and records their
provenance separately under `real_trace_derivations`; they are not synthetic `designs` and
do not claim an R3 capacity-lint tier. No t1-t7 bytes or design metadata were changed.

Both traces are deterministic 1120-second, 5-second-binned derivatives of public CC-BY-4.0
production traces. Their aggregate peak is rescaled to the observed t4 peak (29.426667 RPS),
and sessions are assigned to models with weights 0.40/0.35/0.25. Full source hashes,
selection rules, token bounds, exact regeneration command, and derived checksums are in
`docs/refactor/p11_evidence/real_traces_20260713/`.

## Generation command

```bash
cd tre
PYTHONPATH=common:controller:service-manager:replayer:deploy:calibration \
  python3 -m tre_replayer.gen_traces \
    --capacity replayer/traces_v2/capacity/capacity_dsqwen-7b.json \
    --capacity replayer/traces_v2/capacity/capacity_dsllama-8b.json \
    --capacity replayer/traces_v2/capacity/capacity_dsqwen-14b.json \
    --out-dir replayer/traces_v2/ --version experiment3-v2
```

## Capacity source

Same R3 recalibration surfaces as traceset-v1 (archived under `capacity/`); v2 is a
**design** fix, not a re-measurement. Cluster: 2×4×A100(40G) = 8 GPU slots; slot widths
dsqwen-7b/dsllama-8b = 1 (tp1), dsqwen-14b = 2 (tp2).

## Traces (axis / tier / integer GPU / mechanism)

| trace | axis | tier | integer GPU (peak) | lint |
| --- | --- | --- | --- | --- |
| t1_a1_demand_shift | A1 demand-shift speed | tight 1.00 | 8/8 (hot 5 + 1 + 2) | PASS |
| t2_a2_anticorrelated | A2 inter-model anti-correlation | medium 0.75 | 6/8 (3 + 1 + 2) | PASS |
| t3_a3_io_drift | A3 i/o mix drift (metric superiority) | medium 0.75 | 6/8 (rises 5→6) | PASS |
| t4_a4_spike_vs_burst | A4 spike vs burst | medium 0.75 | 6/8 (3 + 1 + 2) | PASS |
| t5_a5_tp_pressure | A5 TP-heterogeneous pressure | tight 1.00 | 8/8 (3×2 + 1 + 1) | PASS |
| t6_a6_control | A6 fairness control | loose 0.50 | 4/8 (flat floor) | C2 waived (by design) |
| t7_a2b_anticorrelated_hot | A2 (tight variant) | tight 1.00 | 8/8 (5 + 1 + 2) | PASS |
| t8_azure_conv | production temporal structure | n/a | peak-scaled | schema/schedule validated |
| t9_burstgpt | production burst/session structure | n/a | peak-scaled | schema/schedule validated |

## Lint (`lint_report.json`)

Constraints per section 12.3: **C1** feasibility (integer GPU occupancy ≤ 8 AND fractional
oracle violation < 1%), **C2** non-triviality (some model rho > 1.2 for ≥ 3 slow loops = 30s),
**C3** headroom tier (peak *integer* headroom within ±0.05 of the declared loose/medium/tight
target). Each row reports `max_headroom` (integer basis, the feasibility metric) and
`max_fractional_headroom` (the old `Σ rho·width/8`, kept for reference).

**6 of 7 pass all three constraints.** The one exception is by design:

- **t6_a6_control fails C2** intentionally. A6 is the fairness/control arm whose entire
  purpose is a load every system handles without scaling; every model stays at rho ≤ 0.95
  (unit-tested invariant `test_a6_control_stays_below_non_triviality_threshold`), so it never
  needs a second replica and its integer occupancy holds flat at the 4/8 floor. The C2
  non-triviality guard — which asks the opposite — is waived for A6; C1 and C3 still hold.

A3's peak occupancy is capacity-derived (`headroom_is_capacity_dependent: true`):
`A3_RPS_MULT` was set to 2.8 so the heaviest drift phase needs ceil(2.8)=3 7b replicas
(3 + 1 + 2 = 6/8, medium). RPS is held constant across the drift while output grows 128→512,
so the rate/KVCache signal lags while TSS's weighted throughput tracks the rising decode load
— the A3 metric-superiority scenario, entirely in the saturated (rho > 1) regime, and now
integer-feasible.
